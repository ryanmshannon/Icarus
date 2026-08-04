#!/usr/bin/env python3
"""
download_btsettl.py -- fetch high-resolution BT-Settl (CIFIST2011_2015) spectra
and lay them out exactly the way Starfish's ``CIFISTGridInterface`` expects.
 
Why this script exists
----------------------
Starfish ships ``download_PHOENIX_models`` for the Goettingen PHOENIX-ACES grid,
but there is no equivalent for BT-Settl. ``CIFISTGridInterface`` simply assumes
the raw files are already on disk, named:
 
    lte{Teff/100:0>5.1f}-{logg:.1f}-0.0a+0.0.BT-Settl.spec.fits.gz
 
(e.g. Teff=2800 K, logg=5.0  ->  ``lte028.0-5.0-0.0a+0.0.BT-Settl.spec.fits.gz``)
 
all sitting flat in one directory. This script downloads them from the Lyon
PHOENIX server into precisely that layout, then optionally runs ``HDF5Creator``
to produce the processed HDF5 grid that Starfish actually fits against.
 
Usage
-----
    # See what a selection would pull down, without downloading
    python download_btsettl.py --out ~/libraries/BTSettl --teff 2600 3400 --list
 
    # Download an M-dwarf slice
    python download_btsettl.py --out ~/libraries/BTSettl \
        --teff 2600 3400 --logg 4.5 5.5
 
    # Download and immediately build an IGRINS-H HDF5 grid
    python download_btsettl.py --out ~/libraries/BTSettl \
        --teff 2600 3400 --logg 4.5 5.5 \
        --hdf5 btsettl_igrins_H.hdf5 --instrument IGRINS_H
 
    # Re-check integrity of everything already downloaded, fetching nothing new
    python download_btsettl.py --out ~/libraries/BTSettl --verify-only
 
Requirements
------------
Downloading needs only the Python standard library. ``--verify`` (on by default)
and ``--hdf5`` additionally need ``astropy`` and ``Starfish`` respectively, both
of which you already have if you are using Starfish.
 
Please be considerate: the Lyon server is a shared academic resource. The
default of 3 concurrent workers with a small inter-request delay is deliberate.
"""
 
from __future__ import annotations
 
import argparse
import concurrent.futures as cf
import gzip
import json
import logging
import os
import random
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
 
__version__ = "1.0.0"
 
log = logging.getLogger("btsettl")
 
# --------------------------------------------------------------------------- #
# Grid definition
# --------------------------------------------------------------------------- #
 
DEFAULT_BASE_URL = "https://phoenix.ens-lyon.fr/Grids/BT-Settl/CIFIST2011_2015/FITS/"
 
# Must stay byte-identical to Starfish.grid_tools.CIFISTGridInterface.rname
RNAME = "lte{0:0>5.1f}-{1:.1f}-0.0a+0.0.BT-Settl.spec.fits.gz"
 
# Matches CIFISTGridInterface.points:
#   np.concatenate((np.arange(1200, 2351, 50), np.arange(2400, 7001, 100)))
#   np.arange(2.5, 5.6, 0.5)
NOMINAL_TEFF: tuple[int, ...] = tuple(range(1200, 2351, 50)) + tuple(
    range(2400, 7001, 100)
)
NOMINAL_LOGG: tuple[float, ...] = tuple(round(2.5 + 0.5 * i, 1) for i in range(7))
 
USER_AGENT = f"download_btsettl.py/{__version__} (Starfish BT-Settl grid fetcher)"
 
# Parses filenames out of an Apache-style directory index, tolerating the
# percent-encoding the server applies to the '+' in "-0.0a+0.0".
_HREF_RE = re.compile(r'href\s*=\s*["\']([^"\'>]+)["\']', re.IGNORECASE)
_FNAME_RE = re.compile(
    r"^lte(?P<t>\d{3}\.\d)-(?P<g>\d\.\d)-0\.0a\+0\.0\.BT-Settl\.spec\.fits\.gz$"
)
 
 
def starfish_filename(teff: float, logg: float) -> str:
    """Canonical on-disk name for a (Teff [K], logg) pair."""
    return RNAME.format(0.01 * teff, logg)
 
 
def encode_url(base_url: str, filename: str) -> str:
    """Join base URL and filename, percent-encoding the '+' in the name."""
    return urllib.parse.urljoin(base_url, urllib.parse.quote(filename, safe="/.-"))
 
 
# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
 
 
@dataclass(frozen=True)
class Target:
    teff: int
    logg: float
 
    @property
    def filename(self) -> str:
        return starfish_filename(self.teff, self.logg)
 
    def __str__(self) -> str:
        return f"Teff={self.teff:>5d} K  logg={self.logg:.1f}"
 
 
def _select_axis(
    nominal: Sequence[float],
    rng: tuple[float, float] | None,
    values: Sequence[float] | None,
    label: str,
    tol: float = 1e-6,
) -> list[float]:
    """Resolve an axis selection to actual grid points, warning about misses."""
    if values:
        chosen: list[float] = []
        for v in values:
            hit = next((p for p in nominal if abs(p - v) < tol), None)
            if hit is None:
                nearest = min(nominal, key=lambda p: abs(p - v))
                log.warning(
                    "%s=%g is not a BT-Settl grid point; nearest is %g. Skipping.",
                    label,
                    v,
                    nearest,
                )
            else:
                chosen.append(hit)
        return sorted(set(chosen))
 
    if rng:
        lo, hi = min(rng), max(rng)
        return [p for p in nominal if lo - tol <= p <= hi + tol]
 
    return list(nominal)
 
 
def build_targets(args: argparse.Namespace) -> list[Target]:
    teffs = _select_axis(NOMINAL_TEFF, args.teff, args.teff_values, "Teff")
    loggs = _select_axis(NOMINAL_LOGG, args.logg, args.logg_values, "logg")
    if not teffs or not loggs:
        raise SystemExit(
            "Selection is empty. The CIFIST grid covers Teff 1200-7000 K "
            "(50 K steps to 2350 K, then 100 K) and logg 2.5-5.5 in 0.5 dex steps."
        )
    return [Target(int(t), float(g)) for t in teffs for g in loggs]
 
 
# --------------------------------------------------------------------------- #
# Remote index
# --------------------------------------------------------------------------- #
 
 
def _open(url: str, headers: dict[str, str] | None = None, timeout: float = 60.0):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)
 
 
def fetch_index(base_url: str, cache: Path | None = None, timeout: float = 60.0) -> set[str] | None:
    """
    Return the set of spectrum filenames the server actually offers.
 
    The published grid has holes -- not every (Teff, logg) combination in the
    nominal rectangle was computed -- so reading the index first lets us skip
    those quietly instead of hammering the server with requests that 404.
 
    Returns None if the index cannot be read, in which case the caller should
    fall back to attempting each URL directly.
    """
    if cache and cache.exists():
        try:
            names = set(json.loads(cache.read_text())["files"])
            log.info("Using cached remote index (%d files) from %s", len(names), cache)
            return names
        except Exception:
            log.debug("Ignoring unreadable index cache %s", cache, exc_info=True)
 
    log.info("Reading remote index: %s", base_url)
    try:
        with _open(base_url, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        log.warning(
            "Could not read the directory index (%s). Falling back to direct "
            "per-file requests; missing models will show up as 404s.",
            exc,
        )
        return None
 
    names = set()
    for href in _HREF_RE.findall(html):
        name = urllib.parse.unquote(href.rsplit("/", 1)[-1])
        if _FNAME_RE.match(name):
            names.add(name)
 
    if not names:
        log.warning(
            "The index at %s parsed to zero spectrum filenames. The server "
            "layout may have changed; falling back to direct requests.",
            base_url,
        )
        return None
 
    log.info("Index lists %d spectra", len(names))
    if cache:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"base_url": base_url, "files": sorted(names)}))
        except OSError:
            log.debug("Could not write index cache", exc_info=True)
    return names
 
 
# --------------------------------------------------------------------------- #
# Integrity checking
# --------------------------------------------------------------------------- #
 
 
def verify_spectrum(path: Path, deep: bool = True) -> tuple[bool, str]:
    """
    Confirm a downloaded file is a complete, readable BT-Settl spectrum.
 
    A truncated download is the single most common failure mode here, and it is
    insidious: the file looks fine until Starfish tries to spline it. We check
    the gzip stream decompresses cleanly, and (when astropy is available) that
    HDU 1 carries the 'Wavelength' and 'Flux' columns CIFISTGridInterface reads.
    """
    if not path.exists():
        return False, "missing"
    if path.stat().st_size == 0:
        return False, "empty file"
 
    try:
        with gzip.open(path, "rb") as fh:
            while fh.read(1 << 20):
                pass
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        return False, f"corrupt gzip stream ({exc})"
 
    if not deep:
        return True, "gzip ok"
 
    try:
        from astropy.io import fits
    except ImportError:
        return True, "gzip ok (astropy unavailable, skipped FITS check)"
 
    try:
        with fits.open(path) as hdul:
            if len(hdul) < 2:
                return False, "FITS has no table extension"
            cols = {c.upper() for c in (hdul[1].columns.names or [])}
            missing = {"WAVELENGTH", "FLUX"} - cols
            if missing:
                return False, f"HDU 1 missing column(s): {', '.join(sorted(missing))}"
            if hdul[1].data is None or len(hdul[1].data) == 0:
                return False, "HDU 1 table is empty"
    except Exception as exc:
        return False, f"unreadable FITS ({exc})"
 
    return True, "ok"
 
 
# --------------------------------------------------------------------------- #
# Downloading
# --------------------------------------------------------------------------- #
 
 
@dataclass
class Stats:
    downloaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    absent: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    bytes_fetched: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
 
    def record(self, kind: str, name: str, detail: str = "", nbytes: int = 0) -> None:
        with self._lock:
            if kind == "downloaded":
                self.downloaded.append(name)
                self.bytes_fetched += nbytes
            elif kind == "skipped":
                self.skipped.append(name)
            elif kind == "absent":
                self.absent.append(name)
            else:
                self.failed.append((name, detail))
 
 
def _stream_to_disk(url: str, part: Path, resume_from: int, timeout: float) -> int:
    """Fetch `url` into `part`, resuming at `resume_from` bytes when possible."""
    headers = {}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
 
    with _open(url, headers=headers, timeout=timeout) as resp:
        # A server that ignores Range replies 200 with the whole file; in that
        # case we must start over rather than append and silently corrupt.
        resuming = resume_from > 0 and resp.status == 206
        mode = "ab" if resuming else "wb"
        if resume_from and not resuming:
            log.debug("%s: server ignored Range, restarting download", part.name)
 
        written = 0
        with open(part, mode) as fh:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                fh.write(chunk)
                written += len(chunk)
    return written
 
 
def download_one(
    target: Target,
    base_url: str,
    out_dir: Path,
    stats: Stats,
    *,
    overwrite: bool = False,
    verify: bool = True,
    retries: int = 4,
    timeout: float = 60.0,
    delay: float = 0.25,
) -> None:
    name = target.filename
    dest = out_dir / name
    part = out_dir / (name + ".part")
    url = encode_url(base_url, name)
 
    if dest.exists() and not overwrite:
        ok, why = verify_spectrum(dest, deep=verify)
        if ok:
            log.debug("%s already present (%s)", name, why)
            stats.record("skipped", name)
            return
        log.warning("%s is present but %s -- re-downloading", name, why)
        dest.unlink(missing_ok=True)
 
    for attempt in range(1, retries + 1):
        try:
            resume_from = part.stat().st_size if part.exists() else 0
            if delay:
                time.sleep(delay * random.uniform(0.5, 1.5))
            nbytes = _stream_to_disk(url, part, resume_from, timeout)
 
            ok, why = verify_spectrum(part, deep=verify)
            if not ok:
                part.unlink(missing_ok=True)
                raise OSError(f"downloaded file failed verification: {why}")
 
            part.replace(dest)
            log.info("%s  <-  %s (%.1f MiB)", target, name, dest.stat().st_size / 2**20)
            stats.record("downloaded", name, nbytes=nbytes)
            return
 
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                # Genuine hole in the published grid, not an error worth retrying.
                log.info("%s  --  not published on the server", target)
                part.unlink(missing_ok=True)
                stats.record("absent", name)
                return
            if exc.code == 416:  # stale .part longer than the real file
                part.unlink(missing_ok=True)
            detail = f"HTTP {exc.code} {exc.reason}"
        except Exception as exc:  # noqa: BLE001 - retry on anything transient
            detail = f"{type(exc).__name__}: {exc}"
 
        if attempt < retries:
            backoff = min(2**attempt, 30) * random.uniform(0.8, 1.2)
            log.warning(
                "%s failed (%s); retry %d/%d in %.1fs", name, detail, attempt, retries, backoff
            )
            time.sleep(backoff)
        else:
            log.error("%s failed after %d attempts: %s", name, retries, detail)
            stats.record("failed", name, detail)
 
 
def download_grid(
    targets: Sequence[Target],
    base_url: str,
    out_dir: Path,
    *,
    workers: int = 3,
    **kwargs,
) -> Stats:
    stats = Stats()
    out_dir.mkdir(parents=True, exist_ok=True)
 
    if workers <= 1:
        for t in targets:
            download_one(t, base_url, out_dir, stats, **kwargs)
        return stats
 
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(download_one, t, base_url, out_dir, stats, **kwargs) for t in targets
        ]
        for fut in cf.as_completed(futures):
            fut.result()  # surface unexpected programmer errors
    return stats
 
 
# --------------------------------------------------------------------------- #
# Optional HDF5 build
# --------------------------------------------------------------------------- #
 
 
def _guard_missing_spectra(grid) -> None:
    """
    Make missing spectra survivable during ``HDF5Creator.process_grid``.
 
    The published BT-Settl grid is not a full rectangle -- some (Teff, logg)
    combinations were never computed. ``HDF5Creator`` iterates the Cartesian
    product of the parameter ranges, so it will inevitably ask for a file that
    is not there. It catches ``ValueError`` and drops that point cleanly, but
    ``CIFISTGridInterface.load_flux`` tries to raise ``Starfish.constants
    .GridError``, which does not exist -- so you get an ``AttributeError`` that
    aborts the entire build instead.
 
    (Upstream bug as of Starfish master: ``C.GridError`` is referenced in
    interfaces.py but never defined in constants.py.)
 
    We check for the file ourselves first and raise the ``ValueError`` that
    ``process_grid`` is already written to handle.
    """
    original = grid.load_flux
 
    def load_flux(parameters, *args, **kwargs):
        fname = grid.full_rname.format(0.01 * parameters[0], parameters[1])
        if not os.path.exists(fname):
            raise ValueError(f"{fname} is not on disk")
        return original(parameters, *args, **kwargs)
 
    grid.load_flux = load_flux
 
 
def build_hdf5(
    raw_dir: Path,
    hdf5_path: Path,
    targets: Sequence[Target],
    *,
    instrument_name: str | None = None,
    wl_range: tuple[float, float] | None = None,
    air: bool = True,
) -> None:
    """Run Starfish's HDF5Creator over the freshly downloaded raw grid."""
    try:
        from Starfish.grid_tools import CIFISTGridInterface, HDF5Creator
        from Starfish.grid_tools import instruments as sf_instruments
    except ImportError as exc:
        raise SystemExit(
            f"--hdf5 needs Starfish installed ({exc}). Try: pip install astrostarfish"
        ) from exc
 
    instrument = None
    if instrument_name:
        cls = getattr(sf_instruments, instrument_name, None)
        if cls is None:
            available = sorted(
                n
                for n, o in vars(sf_instruments).items()
                if isinstance(o, type)
                and issubclass(o, sf_instruments.Instrument)
                and o is not sf_instruments.Instrument
            )
            raise SystemExit(
                f"Unknown instrument {instrument_name!r}. Available: {', '.join(available)}"
            )
        instrument = cls()
        log.info("Convolving to %s (FWHM %.1f km/s)", instrument.name, instrument.FWHM)
 
    teffs = sorted({t.teff for t in targets})
    loggs = sorted({t.logg for t in targets})
    ranges = [[min(teffs), max(teffs)], [min(loggs), max(loggs)]]
 
    # The interface needs a wl_range at construction; default to the full grid
    # span so we never truncate more than the user asked for.
    grid_kwargs = {"air": air}
    if wl_range:
        grid_kwargs["wl_range"] = tuple(wl_range)
 
    log.info("Opening raw grid at %s", raw_dir)
    grid = CIFISTGridInterface(str(raw_dir), **grid_kwargs)
    _guard_missing_spectra(grid)
 
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Building HDF5 grid -> %s (this is slow; each spectrum is splined)", hdf5_path)
    creator = HDF5Creator(
        grid,
        str(hdf5_path),
        instrument=instrument,
        wl_range=list(wl_range) if wl_range else None,
        ranges=ranges,
    )
    creator.process_grid()
    log.info("HDF5 grid written: %s (%.1f MiB)", hdf5_path, hdf5_path.stat().st_size / 2**20)
 
 
# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
 
 
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="download_btsettl.py",
        description=(
            "Download high-resolution BT-Settl (CIFIST2011_2015) spectra into the "
            "layout Starfish's CIFISTGridInterface expects."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Grid coverage: Teff 1200-2350 K in 50 K steps, 2400-7000 K in 100 K steps;\n"
            "logg 2.5-5.5 in 0.5 dex steps; solar metallicity only. Some combinations\n"
            "were never computed and are reported as 'not published'."
        ),
    )
    p.add_argument(
        "-o", "--out", required=True, type=Path,
        help="Directory for the raw .fits.gz spectra (created if absent).",
    )
 
    sel = p.add_argument_group("grid selection")
    sel.add_argument(
        "--teff", nargs=2, type=float, metavar=("MIN", "MAX"),
        help="Inclusive effective-temperature range in K. Default: the whole grid.",
    )
    sel.add_argument(
        "--teff-values", nargs="+", type=float, metavar="T",
        help="Explicit Teff values instead of a range.",
    )
    sel.add_argument(
        "--logg", nargs=2, type=float, metavar=("MIN", "MAX"),
        help="Inclusive surface-gravity range. Default: the whole grid.",
    )
    sel.add_argument(
        "--logg-values", nargs="+", type=float, metavar="G",
        help="Explicit logg values instead of a range.",
    )
 
    net = p.add_argument_group("download behaviour")
    net.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Override the source URL.")
    net.add_argument(
        "-j", "--workers", type=int, default=3,
        help="Concurrent downloads. Keep this modest; the server is shared. Default: 3.",
    )
    net.add_argument("--retries", type=int, default=4, help="Attempts per file. Default: 4.")
    net.add_argument("--timeout", type=float, default=60.0, help="Socket timeout, seconds.")
    net.add_argument(
        "--delay", type=float, default=0.25,
        help="Jittered pause before each request, seconds. Default: 0.25.",
    )
    net.add_argument("--overwrite", action="store_true", help="Re-download files already present.")
    net.add_argument(
        "--no-verify", dest="verify", action="store_false",
        help="Skip the FITS structure check (gzip integrity is still checked).",
    )
    net.add_argument(
        "--no-index", dest="use_index", action="store_false",
        help="Do not read the remote directory listing; request each file directly.",
    )
    net.add_argument("--refresh-index", action="store_true", help="Ignore any cached index.")
 
    mode = p.add_argument_group("modes")
    mode.add_argument("-n", "--list", action="store_true", help="List the selection and exit.")
    mode.add_argument(
        "--verify-only", action="store_true",
        help="Check files already on disk and exit without downloading.",
    )
 
    h5 = p.add_argument_group("HDF5 grid (optional)")
    h5.add_argument(
        "--hdf5", type=Path, metavar="PATH",
        help="After downloading, build a Starfish HDF5 grid at PATH.",
    )
    h5.add_argument(
        "--instrument", metavar="NAME",
        help="Starfish instrument to convolve to, e.g. IGRINS_H, TRES, SPEX.",
    )
    h5.add_argument(
        "--wl-range", nargs=2, type=float, metavar=("MIN", "MAX"),
        help="Wavelength range in Angstroms for the HDF5 grid, e.g. 14250 18400.",
    )
    h5.add_argument(
        "--vacuum", action="store_true",
        help="Keep vacuum wavelengths (default converts to air).",
    )
 
    p.add_argument("-v", "--verbose", action="store_true", help="Debug-level logging.")
    p.add_argument("-q", "--quiet", action="store_true", help="Warnings and errors only.")
    p.add_argument("--version", action="version", version=__version__)
 
    args = p.parse_args(argv)
    if args.teff and args.teff_values:
        p.error("--teff and --teff-values are mutually exclusive")
    if args.logg and args.logg_values:
        p.error("--logg and --logg-values are mutually exclusive")
    return args
 
 
def human(nbytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(nbytes) < 1024 or unit == "TiB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TiB"
 
 
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
 
    out_dir: Path = args.out.expanduser().resolve()
    targets = build_targets(args)
 
    # ----- verify-only ----------------------------------------------------- #
    if args.verify_only:
        bad = 0
        for t in targets:
            path = out_dir / t.filename
            if not path.exists():
                continue
            ok, why = verify_spectrum(path, deep=args.verify)
            if ok:
                log.debug("%s  ok", t.filename)
            else:
                bad += 1
                log.error("%s  %s", t.filename, why)
        present = sum((out_dir / t.filename).exists() for t in targets)
        print(f"\nChecked {present} file(s) on disk; {bad} bad.")
        if bad:
            print("Re-run without --verify-only to repair them.")
        return 1 if bad else 0
 
    # ----- prune the selection against the server index --------------------- #
    index = None
    if args.use_index:
        cache = out_dir / ".remote_index.json"
        if args.refresh_index:
            cache.unlink(missing_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        index = fetch_index(args.base_url, cache=cache, timeout=args.timeout)
 
    if index is not None:
        known, unpublished = [], []
        for t in targets:
            (known if t.filename in index else unpublished).append(t)
        if unpublished:
            log.info(
                "%d of %d requested models are not published in this grid "
                "(most often high logg at low Teff, or vice versa).",
                len(unpublished), len(targets),
            )
            for t in unpublished:
                log.debug("  not published: %s", t)
        targets = known
 
    if not targets:
        print("Nothing to download: every requested model is absent from the grid.")
        return 1
 
    # ----- list mode -------------------------------------------------------- #
    if args.list:
        print(f"\n{len(targets)} spectra selected from {args.base_url}\n")
        for t in targets:
            marker = "have" if (out_dir / t.filename).exists() else "    "
            print(f"  [{marker}] {t}   {t.filename}")
        have = sum((out_dir / t.filename).exists() for t in targets)
        print(f"\n{have} already on disk, {len(targets) - have} to fetch.")
        print("Individual spectra run roughly 5-40 MiB compressed.")
        return 0
 
    # ----- download --------------------------------------------------------- #
    free = shutil.disk_usage(out_dir).free
    log.info(
        "Fetching up to %d spectra into %s (%s free)", len(targets), out_dir, human(free)
    )
    t0 = time.time()
    stats = download_grid(
        targets,
        args.base_url,
        out_dir,
        workers=max(1, args.workers),
        overwrite=args.overwrite,
        verify=args.verify,
        retries=args.retries,
        timeout=args.timeout,
        delay=args.delay,
    )
    elapsed = time.time() - t0
 
    print(
        f"\nDone in {elapsed:.0f}s: {len(stats.downloaded)} downloaded "
        f"({human(stats.bytes_fetched)}), {len(stats.skipped)} already present, "
        f"{len(stats.absent)} not published, {len(stats.failed)} failed."
    )
    for name, why in stats.failed:
        print(f"  FAILED  {name}: {why}")
    if stats.failed:
        print("Re-run the same command to retry only the failures.")
 
    # ----- optional HDF5 ---------------------------------------------------- #
    if args.hdf5:
        if stats.failed:
            print("\nSkipping HDF5 build because some downloads failed.")
            return 1
        usable = [t for t in targets if (out_dir / t.filename).exists()]
        if not usable:
            print("\nNo spectra on disk; nothing to build.")
            return 1
        build_hdf5(
            out_dir,
            args.hdf5.expanduser().resolve(),
            usable,
            instrument_name=args.instrument,
            wl_range=tuple(args.wl_range) if args.wl_range else None,
            air=not args.vacuum,
        )
        print(
            "\nLoad it in Starfish with:\n"
            "    from Starfish.grid_tools import HDF5Interface\n"
            f"    grid = HDF5Interface({str(args.hdf5)!r})"
        )
 
    return 1 if stats.failed else 0
 
 
if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Partial downloads are kept as .part files and "
              "will resume on the next run.", file=sys.stderr)
        sys.exit(130)
 