#!/usr/bin/env python3
"""
download_phoenix.py -- fetch the Husser (2013) PHOENIX-ACES-AGSS-COND-2011
high-resolution grid from Goettingen and lay it out exactly the way Starfish's
``PHOENIXGridInterface`` / ``PHOENIXGridInterfaceNoAlpha`` expect.

Relationship to Starfish's own downloader
-----------------------------------------
Starfish ships ``download_PHOENIX_models``, and for a small pull it is fine.
This script covers what that one does not: it is parallel, it resumes partial
transfers, it retries with backoff, it verifies every file actually decodes as
a PHOENIX spectrum of the right length rather than trusting the transfer, it
prunes your request against the server's real directory listings so absent
models never become 404 storms, and it re-runs idempotently so an interrupted
multi-hundred-gigabyte download picks up where it stopped.

Layout produced
---------------
    <out>/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits
    <out>/Z-0.0/lte05800-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits
    <out>/Z-1.0.Alpha=+0.40/lte05800-4.50-1.0.Alpha=+0.40.PHOENIX-...fits

The WAVE file is mandatory -- ``PHOENIXGridInterface.__init__`` raises
``ValueError("Wavelength file improperly specified.")`` without it -- so it is
always fetched first.

Usage
-----
    # Preview a solar-neighbourhood slice
    python download_phoenix.py -o ~/libraries/PHOENIX \\
        --teff 5000 6500 --logg 3.5 5.0 --Z 0.0 --list

    # Download it
    python download_phoenix.py -o ~/libraries/PHOENIX \\
        --teff 5000 6500 --logg 3.5 5.0 --Z 0.0

    # Download and build a TRES-convolved HDF5 grid in one go
    python download_phoenix.py -o ~/libraries/PHOENIX \\
        --teff 5000 6500 --logg 3.5 5.0 --Z 0.0 \\
        --hdf5 phoenix_tres.hdf5 --instrument TRES --wl-range 5000 5200

    # Alpha-enhanced models (adds the 4th grid dimension)
    python download_phoenix.py -o ~/libraries/PHOENIX \\
        --teff 4000 6000 --logg 4.0 5.0 --Z -1.0 -0.5 --alpha 0.0 0.4

    # Re-check integrity of everything already on disk
    python download_phoenix.py -o ~/libraries/PHOENIX --verify-only

Requirements
------------
Downloading needs only the standard library plus astropy for verification.
``--hdf5`` additionally needs Starfish. Both are already present in any working
Starfish install.

Size warning
------------
These files are ~11 MiB each and the full grid runs to several terabytes.
Always use ``--list`` first. The script refuses selections over ``--max-gb``
(default 50 GiB) unless you pass ``--yes``.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
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

log = logging.getLogger("phoenix")

# --------------------------------------------------------------------------- #
# Grid definition -- kept in lockstep with Starfish.grid_tools.PHOENIXGridInterface
# --------------------------------------------------------------------------- #

DEFAULT_BASE_URL = "https://phoenix.astro.physik.uni-goettingen.de/data/HiResFITS/"
GRID_SUBDIR = "PHOENIX-ACES-AGSS-COND-2011"
WAVE_FILE = "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
WAVE_BYTES = 12_556_800  # published size; used only as a sanity hint
NPIX = 1_569_128  # every HiRes spectrum has exactly this many pixels

# Starfish rname, alpha form:
#   "Z{2:}{3:}/lte{0:0>5.0f}-{1:.2f}{2:}{3:}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
# and no-alpha form, which is the same thing with the alpha string empty.
RNAME = "Z{Z}{A}/lte{T:0>5.0f}-{logg:.2f}{Z}{A}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"

# PHOENIXGridInterface.points
NOMINAL_TEFF: tuple[int, ...] = tuple(range(2300, 7000, 100)) + tuple(
    range(7000, 12001, 200)
)
NOMINAL_LOGG: tuple[float, ...] = tuple(round(0.5 + 0.5 * i, 2) for i in range(12))
NOMINAL_Z: tuple[float, ...] = (-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0)
NOMINAL_ALPHA: tuple[float, ...] = (-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2)

# Alpha-enhanced models were only computed over this sub-box. This mirrors
# PHOENIXGridInterface.check_params, and matches which Z.Alpha=* directories
# actually exist on the server.
ALPHA_TEFF_RANGE = (3500, 8000)
ALPHA_Z_RANGE = (-3.0, 0.0)

# Starfish's par_dicts are incomplete relative to its own `points` (see
# _extend_par_dicts): these values exist on the server and are downloadable,
# but PHOENIXGridInterface.load_flux would KeyError on them unfixed.
PAR_DICT_GAPS_Z = (-4.0, -3.0)
PAR_DICT_GAPS_ALPHA = (1.0, 1.2)

APPROX_BYTES_PER_FILE = 11 * 2**20

USER_AGENT = f"download_phoenix.py/{__version__} (Starfish PHOENIX grid fetcher)"

_HREF_RE = re.compile(r'href\s*=\s*["\']([^"\'>]+)["\']', re.IGNORECASE)
_FNAME_RE = re.compile(
    r"^lte(?P<t>\d{5})-(?P<g>\d\.\d\d)(?P<z>[-+]\d\.\d)"
    r"(?P<a>\.Alpha=[-+]\d\.\d\d)?"
    r"\.PHOENIX-ACES-AGSS-COND-2011-HiRes\.fits$"
)


def z_string(Z: float) -> str:
    """PHOENIX writes solar metallicity as '-0.0', not '+0.0'."""
    return "-0.0" if Z == 0 else f"{Z:+.1f}"


def alpha_string(alpha: float) -> str:
    """Alpha is omitted entirely from the path when it is zero."""
    return "" if alpha == 0 else f".Alpha={alpha:+.2f}"


def phoenix_relpath(teff: float, logg: float, Z: float, alpha: float = 0.0) -> str:
    """Path relative to the library root, matching Starfish's rname exactly."""
    return RNAME.format(T=teff, logg=logg, Z=z_string(Z), A=alpha_string(alpha))


def encode_url(*parts: str) -> str:
    """Join URL parts, percent-encoding the '+' and '=' in alpha directories."""
    head, *tail = parts
    url = head if head.endswith("/") else head + "/"
    for part in tail:
        for seg in part.split("/"):
            url += urllib.parse.quote(seg, safe=".-_") + "/"
    return url.rstrip("/")


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Target:
    teff: int
    logg: float
    Z: float
    alpha: float = 0.0

    @property
    def relpath(self) -> str:
        return phoenix_relpath(self.teff, self.logg, self.Z, self.alpha)

    @property
    def subdir(self) -> str:
        return f"Z{z_string(self.Z)}{alpha_string(self.alpha)}"

    def __str__(self) -> str:
        # Show the PHOENIX string form of Z so it lines up with the directory name.
        base = f"Teff={self.teff:>5d} K  logg={self.logg:.2f}  [Fe/H]={z_string(self.Z):>4s}"
        return base + (f"  [a/Fe]={self.alpha:+.1f}" if self.alpha else "")


def _select_axis(
    nominal: Sequence[float],
    rng: tuple[float, float] | None,
    values: Sequence[float] | None,
    label: str,
    tol: float = 1e-6,
) -> list[float]:
    if values:
        chosen = []
        for v in values:
            hit = next((p for p in nominal if abs(p - v) < tol), None)
            if hit is None:
                nearest = min(nominal, key=lambda p: abs(p - v))
                log.warning(
                    "%s=%g is not a PHOENIX grid point; nearest is %g. Skipping.",
                    label, v, nearest,
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
    zs = _select_axis(NOMINAL_Z, args.Z_range, args.Z, "[Fe/H]")
    alphas = _select_axis(NOMINAL_ALPHA, args.alpha_range, args.alpha, "[alpha/Fe]")
    if args.alpha is None and args.alpha_range is None:
        alphas = [0.0]  # solar alpha only unless asked otherwise

    if not (teffs and loggs and zs and alphas):
        raise SystemExit(
            "Selection is empty. PHOENIX covers Teff 2300-7000 K (100 K steps) and "
            "7000-12000 K (200 K steps); logg 0.5-6.0 in 0.5 dex; [Fe/H] in "
            "-4,-3,-2,-1.5,-1,-0.5,0,+0.5,+1."
        )

    targets, skipped_alpha = [], 0
    for Z in zs:
        for a in alphas:
            for T in teffs:
                for g in loggs:
                    if a != 0 and not (
                        ALPHA_TEFF_RANGE[0] <= T <= ALPHA_TEFF_RANGE[1]
                        and ALPHA_Z_RANGE[0] <= Z <= ALPHA_Z_RANGE[1]
                    ):
                        skipped_alpha += 1
                        continue
                    targets.append(Target(int(T), float(g), float(Z), float(a)))

    if skipped_alpha:
        log.info(
            "Skipped %d alpha-enhanced combinations: those models only exist for "
            "%g <= Teff <= %g and %+.1f <= [Fe/H] <= %+.1f.",
            skipped_alpha, *ALPHA_TEFF_RANGE, *ALPHA_Z_RANGE,
        )

    gaps_z = sorted({t.Z for t in targets} & set(PAR_DICT_GAPS_Z))
    gaps_a = sorted({t.alpha for t in targets} & set(PAR_DICT_GAPS_ALPHA))
    if gaps_z or gaps_a:
        log.warning(
            "Your selection includes values Starfish can download but cannot "
            "currently load: %s%s%s. These are in PHOENIXGridInterface.points but "
            "missing from its par_dicts, so load_flux raises KeyError. --hdf5 "
            "patches this at runtime; see _extend_par_dicts if you load the grid "
            "yourself.",
            f"[Fe/H]={gaps_z}" if gaps_z else "",
            " and " if gaps_z and gaps_a else "",
            f"[alpha/Fe]={gaps_a}" if gaps_a else "",
        )

    return targets


# --------------------------------------------------------------------------- #
# Remote index
# --------------------------------------------------------------------------- #


def _open(url: str, headers: dict[str, str] | None = None, timeout: float = 60.0):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def fetch_subdir_index(base_url: str, subdir: str, timeout: float = 60.0) -> set[str] | None:
    """Return the filenames the server lists inside one Z/alpha directory."""
    url = encode_url(base_url, GRID_SUBDIR, subdir) + "/"
    try:
        with _open(url, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        log.warning("Could not list %s (%s); will request its files directly.", subdir, exc)
        return None

    names = {
        n
        for href in _HREF_RE.findall(html)
        if _FNAME_RE.match(n := urllib.parse.unquote(href.rsplit("/", 1)[-1]))
    }
    if not names:
        log.warning("Listing for %s parsed to zero spectra; requesting directly.", subdir)
        return None
    log.debug("%s lists %d spectra", subdir, len(names))
    return names


def load_index(
    base_url: str, subdirs: Sequence[str], cache: Path | None, timeout: float, workers: int
) -> dict[str, set[str] | None]:
    """Read (and cache) the directory listings for every subdir we need."""
    index: dict[str, set[str] | None] = {}
    if cache and cache.exists():
        try:
            raw = json.loads(cache.read_text())
            if raw.get("base_url") == base_url:
                index = {k: set(v) for k, v in raw["dirs"].items()}
                log.info("Loaded cached listings for %d directories", len(index))
        except Exception:
            log.debug("Ignoring unreadable index cache", exc_info=True)

    todo = [d for d in subdirs if d not in index]
    if todo:
        log.info("Reading %d remote directory listing(s)", len(todo))
        with cf.ThreadPoolExecutor(max_workers=min(workers, len(todo))) as pool:
            futs = {pool.submit(fetch_subdir_index, base_url, d, timeout): d for d in todo}
            for fut in cf.as_completed(futs):
                index[futs[fut]] = fut.result()

    if cache:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(
                json.dumps(
                    {
                        "base_url": base_url,
                        "dirs": {k: sorted(v) for k, v in index.items() if v is not None},
                    }
                )
            )
        except OSError:
            log.debug("Could not write index cache", exc_info=True)
    return index


# --------------------------------------------------------------------------- #
# Integrity checking
# --------------------------------------------------------------------------- #

_WAVE_LEN: int | None = None


def verify_spectrum(path: Path, deep: bool = True, expect_npix: int | None = NPIX) -> tuple[bool, str]:
    """
    Confirm a file is a complete PHOENIX HiRes spectrum.

    A half-transferred FITS is the failure mode that actually bites: astropy
    will often still open it, and you only find out when the flux array turns
    out to be the wrong length halfway through an HDF5 build. So we check the
    pixel count, not just that the file parses.
    """
    if not path.exists():
        return False, "missing"
    size = path.stat().st_size
    if size == 0:
        return False, "empty file"
    if deep and expect_npix and size < expect_npix * 4:
        return False, f"truncated ({size} bytes, too small for {expect_npix} pixels)"
    if not deep:
        return True, "size ok"

    try:
        from astropy.io import fits
    except ImportError:
        return True, "size ok (astropy unavailable, skipped FITS check)"

    try:
        with fits.open(path) as hdul:
            data = hdul[0].data
            if data is None:
                return False, "primary HDU has no data"
            if data.ndim != 1:
                return False, f"expected a 1-D spectrum, got shape {data.shape}"
            if expect_npix and data.size != expect_npix:
                return False, f"expected {expect_npix} pixels, got {data.size}"
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
    headers = {}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
    with _open(url, headers=headers, timeout=timeout) as resp:
        # If the server ignores Range it replies 200 with the whole body; append
        # in that case would silently corrupt the file, so start over instead.
        resuming = resume_from > 0 and resp.status == 206
        if resume_from and not resuming:
            log.debug("%s: server ignored Range, restarting", part.name)
        written = 0
        with open(part, "ab" if resuming else "wb") as fh:
            while chunk := resp.read(1 << 16):
                fh.write(chunk)
                written += len(chunk)
    return written


def _fetch(
    url: str,
    dest: Path,
    label: str,
    stats: Stats,
    *,
    overwrite: bool,
    verify: bool,
    expect_npix: int | None,
    retries: int,
    timeout: float,
    delay: float,
) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not overwrite:
        ok, why = verify_spectrum(dest, deep=verify, expect_npix=expect_npix)
        if ok:
            stats.record("skipped", dest.name)
            return
        log.warning("%s is present but %s -- re-downloading", dest.name, why)
        dest.unlink(missing_ok=True)

    for attempt in range(1, retries + 1):
        try:
            resume_from = part.stat().st_size if part.exists() else 0
            if delay:
                time.sleep(delay * random.uniform(0.5, 1.5))
            nbytes = _stream_to_disk(url, part, resume_from, timeout)

            ok, why = verify_spectrum(part, deep=verify, expect_npix=expect_npix)
            if not ok:
                part.unlink(missing_ok=True)
                raise OSError(f"downloaded file failed verification: {why}")

            part.replace(dest)
            log.info("%s  <-  %.1f MiB", label, dest.stat().st_size / 2**20)
            stats.record("downloaded", dest.name, nbytes=nbytes)
            return

        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                log.info("%s  --  not present on the server", label)
                part.unlink(missing_ok=True)
                stats.record("absent", dest.name)
                return
            if exc.code == 416:
                part.unlink(missing_ok=True)
            detail = f"HTTP {exc.code} {exc.reason}"
        except Exception as exc:  # noqa: BLE001
            detail = f"{type(exc).__name__}: {exc}"

        if attempt < retries:
            backoff = min(2**attempt, 30) * random.uniform(0.8, 1.2)
            log.warning(
                "%s failed (%s); retry %d/%d in %.1fs",
                dest.name, detail, attempt, retries, backoff,
            )
            time.sleep(backoff)
        else:
            log.error("%s failed after %d attempts: %s", dest.name, retries, detail)
            stats.record("failed", dest.name, detail)


def download_wave_file(base_url: str, out_dir: Path, stats: Stats, **kwargs) -> Path:
    """
    Fetch the shared wavelength solution.

    PHOENIXGridInterface.__init__ reads this before anything else and raises
    ValueError if it is absent, so there is no point downloading spectra without
    it. All HiRes spectra share this one wavelength array.
    """
    dest = out_dir / WAVE_FILE
    _fetch(
        encode_url(base_url, WAVE_FILE),
        dest,
        "wavelength solution",
        stats,
        expect_npix=NPIX,
        **kwargs,
    )
    global _WAVE_LEN
    if dest.exists():
        try:
            from astropy.io import fits

            _WAVE_LEN = len(fits.getdata(dest))
            log.info("Wavelength grid: %d pixels", _WAVE_LEN)
        except Exception:
            log.debug("Could not read wavelength file length", exc_info=True)
    return dest


def download_grid(
    targets: Sequence[Target], base_url: str, out_dir: Path, *, workers: int = 4, **kwargs
) -> Stats:
    stats = Stats()
    out_dir.mkdir(parents=True, exist_ok=True)

    def job(t: Target) -> None:
        _fetch(
            encode_url(base_url, GRID_SUBDIR, t.relpath),
            out_dir / t.relpath,
            str(t),
            stats,
            expect_npix=_WAVE_LEN or NPIX,
            **kwargs,
        )

    if workers <= 1:
        for t in targets:
            job(t)
        return stats

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in cf.as_completed([pool.submit(job, t) for t in targets]):
            fut.result()
    return stats


# --------------------------------------------------------------------------- #
# Optional HDF5 build
# --------------------------------------------------------------------------- #


class _TolerantDict(dict):
    """
    A par_dict that tolerates floating-point noise in its keys.

    ``PHOENIXGridInterface`` builds its alpha axis with
    ``np.arange(-0.2, 1.21, 0.2)``, which yields 0.4000000000000001 and
    1.0000000000000002 rather than 0.4 and 1.0. ``load_flux`` then does a plain
    ``par_dict[param]`` lookup against a dict keyed on exact values, so those
    points raise KeyError even though they are legitimate grid points. We fall
    back to a rounded, then nearest-within-tolerance, match.
    """

    _TOL = 1e-6

    def __missing__(self, key):
        try:
            rounded = round(float(key), 2)
        except (TypeError, ValueError):
            raise KeyError(key) from None
        if rounded in self.keys():
            return dict.__getitem__(self, rounded)
        for k in self.keys():
            if isinstance(k, (int, float)) and abs(k - rounded) < self._TOL:
                return dict.__getitem__(self, k)
        raise KeyError(key)


def _extend_par_dicts(grid) -> None:
    """
    Repair Starfish's numeric->string lookup tables for this grid instance.

    Two independent problems, both upstream as of Starfish master:

    1. Missing entries. ``points`` advertises [Fe/H] down to -4.0 and
       [alpha/Fe] up to +1.2, and those models genuinely exist on the
       Goettingen server, but ``par_dicts`` only maps [Fe/H] >= -2.0 and
       [alpha/Fe] <= +0.8. The parameter passes ``check_params`` and then dies
       with a bare KeyError while building the filename.

    2. Floating-point keys. The alpha axis comes from ``np.arange`` with a 0.2
       step, so four of its eight values are not exactly representable and miss
       their dict entries even when those entries exist.

    Left alone, an alpha-enhanced build either crashes or -- if something is
    catching the KeyError -- silently drops most of the alpha dimension and
    hands you a grid that looks fine but is not.
    """
    added = []

    z_dict = _TolerantDict(grid.par_dicts[2])
    for Z in PAR_DICT_GAPS_Z:
        if Z not in z_dict.keys():
            z_dict[Z] = f"{Z:+.1f}"
            added.append(f"[Fe/H]={Z:+.1f}")
    grid.par_dicts[2] = z_dict

    if len(grid.par_dicts) > 3 and grid.par_dicts[3] is not None:
        a_dict = _TolerantDict(grid.par_dicts[3])
        for a in PAR_DICT_GAPS_ALPHA:
            if a not in a_dict.keys():
                a_dict[a] = f".Alpha={a:+.2f}"
                added.append(f"[alpha/Fe]={a:+.1f}")
        grid.par_dicts[3] = a_dict

    if added:
        log.info("Added missing par_dicts entries: %s", ", ".join(added))
    log.debug("par_dicts lookups are now tolerant of arange floating-point noise")


def _guard_missing_spectra(grid) -> None:
    """
    Make absent models survivable during ``HDF5Creator.process_grid``.

    process_grid walks the full Cartesian product of the parameter ranges, so
    with any non-rectangular selection it will ask for files that were never
    downloaded. It catches ValueError and drops those points cleanly, which
    ``load_flux`` already raises for a missing file.

    A KeyError reaching here means a parameter had no par_dicts entry even
    after ``_extend_par_dicts``, which would quietly shrink your grid. We
    convert it so the build survives, but shout about it -- a silently smaller
    grid is worse than a loud one.
    """
    original = grid.load_flux

    def load_flux(parameters, *args, **kwargs):
        try:
            return original(parameters, *args, **kwargs)
        except ValueError:
            raise
        except KeyError as exc:
            log.warning(
                "Dropping %s: no par_dicts mapping for %s. This point will be "
                "absent from the HDF5 grid.",
                parameters, exc,
            )
            raise ValueError(f"{parameters}: unmapped grid value {exc}") from exc

    grid.load_flux = load_flux


def build_hdf5(
    raw_dir: Path,
    hdf5_path: Path,
    targets: Sequence[Target],
    *,
    use_alpha: bool,
    instrument_name: str | None = None,
    wl_range: tuple[float, float] | None = None,
    air: bool = True,
    legacy_keys: bool = False,
) -> None:
    try:
        from Starfish.grid_tools import (
            HDF5Creator,
            PHOENIXGridInterface,
            PHOENIXGridInterfaceNoAlpha,
        )
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

    kwargs = {"air": air}
    if wl_range:
        kwargs["wl_range"] = tuple(wl_range)

    cls = PHOENIXGridInterface if use_alpha else PHOENIXGridInterfaceNoAlpha
    log.info("Opening raw grid at %s with %s", raw_dir, cls.__name__)
    grid = cls(str(raw_dir), **kwargs)
    _extend_par_dicts(grid)
    _guard_missing_spectra(grid)

    axes = [
        sorted({t.teff for t in targets}),
        sorted({t.logg for t in targets}),
        sorted({t.Z for t in targets}),
    ]
    if use_alpha:
        axes.append(sorted({t.alpha for t in targets}))

    # HDF5Creator filters grid.points with `>= low` and `<= high`. Those points
    # come from np.arange, so e.g. the alpha axis holds 0.4000000000000001 --
    # an exact bound of 0.4 would silently exclude it. Pad outward.
    eps = 1e-6
    ranges = [[min(a) - eps, max(a) + eps] for a in axes]

    # Starfish's default key_name is the raw rname with the numeric parameters
    # substituted unformatted, so a grid point of 0.4000000000000001 becomes a
    # literal 19-character key that only matches if you pass back the identical
    # float. Pinning the precision makes keys stable and lookups forgiving.
    key_name = None
    if not legacy_keys:
        key_name = "T{0:.0f}_logg{1:.2f}_Z{2:+.1f}"
        if use_alpha:
            key_name += "_alpha{3:+.2f}"

    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Building HDF5 grid -> %s", hdf5_path)
    creator = HDF5Creator(
        grid,
        str(hdf5_path),
        instrument=instrument,
        wl_range=list(wl_range) if wl_range else None,
        ranges=ranges,
        key_name=key_name,
    )
    creator.process_grid()
    log.info("HDF5 grid written: %s (%.1f MiB)", hdf5_path, hdf5_path.stat().st_size / 2**20)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="download_phoenix.py",
        description=(
            "Download Husser (2013) PHOENIX-ACES-AGSS-COND-2011 high-resolution "
            "spectra into the layout Starfish's PHOENIXGridInterface expects."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Grid coverage: Teff 2300-7000 K (100 K steps), 7000-12000 K (200 K steps);\n"
            "logg 0.5-6.0 in 0.5 dex; [Fe/H] -4,-3,-2,-1.5,-1,-0.5,0,+0.5,+1.\n"
            "Alpha-enhanced models exist only for 3500-8000 K and -3 <= [Fe/H] <= 0.\n"
            "Files are ~11 MiB each -- run with --list before committing to a pull."
        ),
    )
    p.add_argument("-o", "--out", required=True, type=Path,
                   help="Library root. The WAVE file and Z*/ subdirectories go here.")

    sel = p.add_argument_group("grid selection")
    sel.add_argument("--teff", nargs=2, type=float, metavar=("MIN", "MAX"),
                     help="Inclusive Teff range in K.")
    sel.add_argument("--teff-values", nargs="+", type=float, metavar="T",
                     help="Explicit Teff values instead of a range.")
    sel.add_argument("--logg", nargs=2, type=float, metavar=("MIN", "MAX"),
                     help="Inclusive logg range.")
    sel.add_argument("--logg-values", nargs="+", type=float, metavar="G",
                     help="Explicit logg values instead of a range.")
    sel.add_argument("--Z", nargs="+", type=float, metavar="FEH", dest="Z",
                     help="Explicit [Fe/H] values. Default: 0.0 only.")
    sel.add_argument("--Z-range", nargs=2, type=float, metavar=("MIN", "MAX"),
                     help="Inclusive [Fe/H] range instead of explicit values.")
    sel.add_argument("--alpha", nargs="+", type=float, metavar="A",
                     help="Explicit [alpha/Fe] values. Omit for solar alpha only. "
                          "Any non-zero value switches the grid to 4 dimensions.")
    sel.add_argument("--alpha-range", nargs=2, type=float, metavar=("MIN", "MAX"),
                     help="Inclusive [alpha/Fe] range.")

    net = p.add_argument_group("download behaviour")
    net.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Override the source URL.")
    net.add_argument("-j", "--workers", type=int, default=4,
                     help="Concurrent downloads. Default: 4.")
    net.add_argument("--retries", type=int, default=4, help="Attempts per file. Default: 4.")
    net.add_argument("--timeout", type=float, default=120.0, help="Socket timeout, seconds.")
    net.add_argument("--delay", type=float, default=0.1,
                     help="Jittered pause before each request. Default: 0.1 s.")
    net.add_argument("--overwrite", action="store_true", help="Re-download existing files.")
    net.add_argument("--no-verify", dest="verify", action="store_false",
                     help="Skip the FITS pixel-count check (size check still applies).")
    net.add_argument("--no-index", dest="use_index", action="store_false",
                     help="Skip directory listings; request every file directly.")
    net.add_argument("--refresh-index", action="store_true", help="Ignore any cached listings.")
    net.add_argument("--max-gb", type=float, default=50.0,
                     help="Refuse selections larger than this without --yes. Default: 50.")
    net.add_argument("-y", "--yes", action="store_true", help="Skip the size confirmation.")

    mode = p.add_argument_group("modes")
    mode.add_argument("-n", "--list", action="store_true", help="List the selection and exit.")
    mode.add_argument("--verify-only", action="store_true",
                      help="Check files already on disk and exit.")

    h5 = p.add_argument_group("HDF5 grid (optional)")
    h5.add_argument("--hdf5", type=Path, metavar="PATH",
                    help="After downloading, build a Starfish HDF5 grid at PATH.")
    h5.add_argument("--instrument", metavar="NAME",
                    help="Instrument to convolve to, e.g. TRES, IGRINS_H, SPEX.")
    h5.add_argument("--wl-range", nargs=2, type=float, metavar=("MIN", "MAX"),
                    help="Wavelength range in Angstroms for the HDF5 grid.")
    h5.add_argument("--vacuum", action="store_true",
                    help="Keep vacuum wavelengths (default converts to air).")
    h5.add_argument("--legacy-keys", action="store_true",
                    help="Use Starfish's default HDF5 dataset key format instead of "
                         "the precision-pinned one. Only needed for compatibility "
                         "with grids built by older tooling.")

    p.add_argument("-v", "--verbose", action="store_true", help="Debug-level logging.")
    p.add_argument("-q", "--quiet", action="store_true", help="Warnings and errors only.")
    p.add_argument("--version", action="version", version=__version__)

    args = p.parse_args(argv)
    for a, b in (("teff", "teff_values"), ("logg", "logg_values")):
        if getattr(args, a) and getattr(args, b):
            p.error(f"--{a.replace('_','-')} and --{b.replace('_','-')} are mutually exclusive")
    if args.Z and args.Z_range:
        p.error("--Z and --Z-range are mutually exclusive")
    if args.alpha and args.alpha_range:
        p.error("--alpha and --alpha-range are mutually exclusive")
    if args.Z is None and args.Z_range is None:
        args.Z = [0.0]  # solar metallicity is the sane default for a huge grid
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
    use_alpha = any(t.alpha != 0 for t in targets)

    # ----- verify-only ----------------------------------------------------- #
    if args.verify_only:
        bad = present = 0
        wave = out_dir / WAVE_FILE
        if wave.exists():
            ok, why = verify_spectrum(wave, deep=args.verify)
            present += 1
            if not ok:
                bad += 1
                log.error("%s  %s", WAVE_FILE, why)
        else:
            log.error("%s is missing -- PHOENIXGridInterface cannot open the grid.", WAVE_FILE)
            bad += 1
        for t in targets:
            path = out_dir / t.relpath
            if not path.exists():
                continue
            present += 1
            ok, why = verify_spectrum(path, deep=args.verify)
            if not ok:
                bad += 1
                log.error("%s  %s", t.relpath, why)
        print(f"\nChecked {present} file(s) on disk; {bad} bad.")
        if bad:
            print("Re-run without --verify-only to repair them.")
        return 1 if bad else 0

    # ----- prune against the server's listings ------------------------------ #
    if args.use_index:
        cache = out_dir / ".remote_index.json"
        if args.refresh_index:
            cache.unlink(missing_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        index = load_index(
            args.base_url,
            sorted({t.subdir for t in targets}),
            cache,
            args.timeout,
            args.workers,
        )
        kept, absent = [], 0
        for t in targets:
            listing = index.get(t.subdir)
            if listing is None or Path(t.relpath).name in listing:
                kept.append(t)  # unknown listing -> try anyway
            else:
                absent += 1
                log.debug("not published: %s", t)
        if absent:
            log.info("%d of %d requested models are absent from the server.",
                     absent, len(targets))
        targets = kept

    if not targets:
        print("Nothing to download: every requested model is absent from the grid.")
        return 1

    missing = [t for t in targets if not (out_dir / t.relpath).exists()]
    est = len(missing) * APPROX_BYTES_PER_FILE

    # ----- list mode -------------------------------------------------------- #
    if args.list:
        print(f"\n{len(targets)} spectra selected from {args.base_url}")
        print(f"grid dimensions: {'T, logg, Z, alpha' if use_alpha else 'T, logg, Z'}\n")
        for t in targets[:200]:
            marker = "have" if (out_dir / t.relpath).exists() else "    "
            print(f"  [{marker}] {t}")
        if len(targets) > 200:
            print(f"  ... and {len(targets) - 200} more")
        print(f"\n{len(targets) - len(missing)} already on disk, {len(missing)} to fetch.")
        print(f"Estimated download: ~{human(est)} (plus {human(WAVE_BYTES)} for the WAVE file).")
        return 0

    # ----- size guard ------------------------------------------------------- #
    if est > args.max_gb * 2**30 and not args.yes:
        print(
            f"\nThis selection is about {human(est)} across {len(missing)} files, "
            f"over the --max-gb limit of {args.max_gb} GiB.\n"
            "Narrow the selection, raise --max-gb, or pass --yes to proceed."
        )
        return 1
    free = shutil.disk_usage(out_dir).free
    if est > free:
        print(f"\nNeed about {human(est)} but only {human(free)} is free at {out_dir}.")
        return 1

    fetch_kwargs = dict(
        overwrite=args.overwrite,
        verify=args.verify,
        retries=args.retries,
        timeout=args.timeout,
        delay=args.delay,
    )

    # ----- wavelength file first -------------------------------------------- #
    log.info("Fetching up to %d spectra into %s (%s free)", len(targets), out_dir, human(free))
    wave_stats = Stats()
    download_wave_file(args.base_url, out_dir, wave_stats, **fetch_kwargs)
    if wave_stats.failed:
        print("\nCould not fetch the wavelength file; Starfish cannot open a grid "
              "without it. Aborting.")
        return 1

    # ----- spectra ---------------------------------------------------------- #
    t0 = time.time()
    stats = download_grid(
        targets, args.base_url, out_dir, workers=max(1, args.workers), **fetch_kwargs
    )
    elapsed = time.time() - t0

    print(
        f"\nDone in {elapsed:.0f}s: {len(stats.downloaded)} downloaded "
        f"({human(stats.bytes_fetched)}), {len(stats.skipped)} already present, "
        f"{len(stats.absent)} absent upstream, {len(stats.failed)} failed."
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
        usable = [t for t in targets if (out_dir / t.relpath).exists()]
        if not usable:
            print("\nNo spectra on disk; nothing to build.")
            return 1
        build_hdf5(
            out_dir,
            args.hdf5.expanduser().resolve(),
            usable,
            use_alpha=use_alpha,
            instrument_name=args.instrument,
            wl_range=tuple(args.wl_range) if args.wl_range else None,
            air=not args.vacuum,
            legacy_keys=args.legacy_keys,
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
        print("\nInterrupted. Partial transfers are kept as .part files and "
              "resume on the next run.", file=sys.stderr)
        sys.exit(130)