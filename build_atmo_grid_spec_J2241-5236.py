# Licensed under a 3-clause BSD style license - see LICENSE

"""
build_atmo_grid_spec_J2241-5236.py

Builds the SPECTROSCOPIC atmosphere grid needed to compute a model spectrum
in model_J2241-5236.py:

    atmo_grid_spec_J2241-5236.h5    (Icarus AtmoGridSpec HDF5 grid, with
                                     axes (logtemp, logg, mu, wav))

This is the spectroscopic counterpart of build_atmo_grid_J2241-5236.py,
which builds the band-integrated photometric grids. Both are derived from
the same PHOENIX/BT-Settl model spectra.

Two model sources are supported (--source):

  cdbs           (default) PHOENIX/BT-Settl (Allard et al.) synthetic stellar
                 spectra, [M/H] = +0.5, in the CDBS/pysynphot FITS grid format
                 read from FITS_DIR below. Each file holds one effective
                 temperature and provides F_lambda (FLAM, i.e.
                 erg/s/cm^2/Angstrom) for a set of log(g) values. These are
                 heavily resampled (R ~ 740 at best), so all velocity
                 structure is unresolved -- see the caveats below.

  phoenix-hires  Husser et al. (2013) PHOENIX-ACES-AGSS-COND-2011
                 high-resolution library (R ~ 500,000 native), downloaded from
                 Goettingen with download_phoenix.py and processed with
                 Starfish (the astrostarfish package): the spectra are
                 convolved to the requested instrumental resolution with
                 Starfish's FFT-based instrumental_broaden and resampled onto a
                 grid uniform in log(wavelength). Use this source if you care
                 about line profiles, equivalent widths, radial velocities or
                 vsini.

Method (both sources)
---------------------
1. Every (Teff, log g) spectrum ends up on a wavelength grid that is
   uniformly spaced in ln(wavelength). This is REQUIRED by Icarus:
   AtmoGridSpec.Get_flux_doppler applies the per-surface-element Doppler
   shift as a (fractional) pixel shift of val_vel/delta_v, which is only
   correct if every pixel spans the same velocity interval. Neither the
   native CDBS sampling (which varies between ~1.35e-3 and ~6.2e-3 over the
   optical) nor the native Husser sampling is uniform in ln(wav), so the
   resampling is not optional. For --source cdbs it is a plain interpolation
   onto the axis set by --delta-lnwav; for --source phoenix-hires it is done
   by Starfish's HDF5Creator, which band-limits the spectrum with the
   instrumental convolution first and then resamples, so the downsampling
   from R ~ 500,000 does not alias.
2. A mu (cos of emission angle) axis is created by applying the analytic
   Neckel (2005) limb-darkening law from Icarus.Utils.Flux.Limb_darkening,
   evaluated per wavelength. Neither PHOENIX grid provides specific
   intensities, only disk-integrated (surface) fluxes, so limb darkening has
   to be imposed analytically -- exactly as Icarus' own
   Atmo_photo_BTSettl7 loader does for the photometric grids.
3. The result is stored as natural-log flux in an AtmoGridSpec HDF5 file,
   using the same surface-flux convention as the photometric grids so the
   two are mutually consistent (see EXACT_LD_NORMALIZATION).

IMPORTANT CAVEATS -- read before using for science
---------------------------------------------------
- SPECTRAL RESOLUTION (--source cdbs only). The CDBS PHOENIX grid is heavily
  resampled: its native sampling is R ~ 740 (~405 km/s per pixel) from
  4300 A out to 25000 A, and only R ~ 161 below 4300 A. The companion's
  projected velocity semi-amplitude is only ~350 km/s, so ALL velocity
  structure (line profiles, rotational broadening, the orbital Doppler
  shift) is UNRESOLVED. X-shooter itself resolves R ~ 5000-9000, i.e. ~7-10x
  finer than this grid, so the predicted spectrum is a valid phase-dependent
  SED and is fine for predicting broad-band count rates or continuum shape,
  but it must NOT be used to predict line profiles, equivalent widths,
  radial velocities or vsini. For that use --source phoenix-hires.
- LIMB-DARKENING EXTRAPOLATION (both sources). The Neckel (2005) law
  implemented by Utils.Flux.Limb_darkening is only formally valid over
  0.42257-1.100 micron; Limb_darkening silently extrapolates outside that
  range. The X-shooter range used here therefore relies on extrapolation
  below 4226 A and, more substantially, over the whole NIR arm above
  11000 A. The extrapolation is smooth, monotonic and physically plausible
  (the centre-to-limb ratio saturates from 0.51 at 1.1 micron to 0.65 at
  2.48 micron, i.e. weaker limb darkening towards the IR, which is the
  correct qualitative behaviour), but it is not a validated fit there. Note
  also that a grey-in-mu limb-darkening law cannot reproduce the
  centre-to-limb behaviour *within* a line, so mu-resolved line profiles
  from this grid are approximate even at high resolution.
- METALLICITY. Fixed at [M/H] = +0.5 for --source cdbs (the phoenixp05
  grid); set with --Z for --source phoenix-hires (default +0.5, to match).
  The true companion metallicity is unknown and is not fit for.
- AIR vs VACUUM (--source phoenix-hires). The Husser spectra are computed on
  a vacuum wavelength scale and are kept that way by default; pass --air to
  have Starfish convert them to air wavelengths. At R ~ 9000 the two differ
  by ~2-3 resolution elements in the optical, so this MUST be matched to
  whatever scale the observed spectra are on.
- ABSOLUTE CALIBRATION. The grid is in physical units, not relative ones:
  the model spectra are true surface fluxes (verified for CDBS: integral
  F_lam dlam reproduces sigma*T^4; --source phoenix-hires performs and
  reports the same check, and bypasses Starfish's default per-spectrum
  renormalization to 1 L_sun, which would have destroyed the absolute
  scale), and Icarus' area/projection scaling makes star.Flux_doppler return
  the flux density at a distance of 10 parsec. Unlike the photometric grids,
  this grid does NOT use Icarus' hardcoded 4/pi^2 surface-flux ->
  specific-intensity factor by default. That factor is only self-consistent
  with the limb-darkening law where the integral of LD(mu)*mu dmu equals
  pi/8 (near 5000 A), and would recover only 86% of the surface flux at
  3000 A and 119% at 24800 A -- a 33% swing across the X-shooter range that
  would visibly distort the predicted SED. This script instead renormalizes
  per wavelength so the hemispheric integral exactly reproduces the input
  surface flux (EXACT_LD_NORMALIZATION, --icarus-norm to reproduce Icarus'
  behaviour).
- GRID SIZE (--source phoenix-hires). A high-resolution grid is large: the
  wavelength axis alone grows like the resolution, and the mu axis
  multiplies it by N_MU. The full X-shooter range at R = 9000 with 73
  temperatures, 4 log g values and 16 mu points is ~2 GB in float64. The
  script estimates this up front and refuses to proceed past --max-mem-gb
  without --yes; --dtype float32 halves the file size (Icarus casts back to
  float64 on read).

Examples
--------
    ## The original band-integrated-resolution grid, unchanged:
    python build_atmo_grid_spec_J2241-5236.py

    ## High resolution: download the Husser library (~3 GB for the default
    ## selection) and build an R = 9000 grid for X-shooter's VIS arm:
    python build_atmo_grid_spec_J2241-5236.py --source phoenix-hires \\
        --wav-lims 5595 10240 -R 9000 --yes

    ## Same, but reusing an already-downloaded library and a named Starfish
    ## instrument instead of a bare resolving power:
    python build_atmo_grid_spec_J2241-5236.py --source phoenix-hires \\
        --no-download --instrument ESPaDOnS
"""

import argparse
import importlib.util
import logging
import os
import glob
import sys

import numpy as np
from astropy.io import fits

from Icarus.Atmosphere.Atmo import AtmoGridSpec
from Icarus.Utils.Flux import Limb_darkening


##### Configuration
SOURCE = 'cdbs'                  # 'cdbs' or 'phoenix-hires'
FITS_DIR = 'model_spec/grp/redcat/trds/grid/phoenix/phoenixp05'
OUT_FLN = 'atmo_grid_spec_J2241-5236.h5'
OUT_FLN_HIRES = 'atmo_grid_spec_hires_J2241-5236.h5'
TEMP_MIN = 2000.                 # K   (matches the photometric grid)
TEMP_MAX = 12500.                # K
LOGG_LIMS = [3.0, 4.5]      # phoenixp05 has no data above 4.5 for T > 2900 K (see docstring)
##### Wavelength range: the full VLT/X-shooter range.
WAV_MIN = 3000.                  # Angstrom
WAV_MAX = 24800.                 # Angstrom
DELTA_LNWAV = 1.35e-3            # ~405 km/s per pixel; matches the native CDBS
                                 # sampling, which is uniform at this value from
                                 # 4300 A all the way out to 25000 A, so we do
                                 # not invent resolution the models lack.
                                 # (Below 4300 A the native sampling is coarser,
                                 # 6.21e-3 / R ~ 161, so the blue end of the UVB
                                 # arm is interpolated, not truly resolved.)
                                 # Used by --source cdbs only; for
                                 # --source phoenix-hires the spacing follows
                                 # from -R / --oversampling.
N_MU = 16                        # same mu sampling as Atmo_photo_BTSettl7

##### High-resolution (Husser 2013 / astrostarfish) configuration
HIRES_LIB = 'model_spec_hires'   # where download_phoenix.py puts the raw library
HIRES_HDF5 = None                # Starfish HDF5 grid; None -> derived from the output name
HIRES_Z = 0.5                    # [Fe/H]; +0.5 matches the phoenixp05 CDBS grid
HIRES_R = 9000.                  # target resolving power (X-shooter VIS arm)
HIRES_OVERSAMPLING = 3.0         # pixels per resolution FWHM in the output grid
HIRES_AIR = False                # keep the native vacuum wavelength scale
HIRES_PAD = 200.                 # Angstrom of margin kept around the requested
                                 # range while convolving, trimmed afterwards, so
                                 # the FFT wrap-around never reaches the output.
##### The Husser grid only runs over these; the requested limits are clipped.
HIRES_TEMP_RANGE = (2300., 12000.)

##### Surface flux -> specific intensity normalization.
##### True : renormalize per wavelength so the hemispheric integral
#####        2*pi*Int I(mu)*mu dmu exactly recovers the input surface flux.
##### False: reproduce Icarus' hardcoded 4/pi^2 factor (what
#####        Atmo_photo_BTSettl7 uses for the photometric grids). That factor
#####        is only self-consistent with the limb-darkening law near 5000 A,
#####        and is wrong by -14% at 3000 A and +19% at 24800 A -- a large
#####        wavelength-dependent distortion across the X-shooter range.
EXACT_LD_NORMALIZATION = True

##### X-shooter arm boundaries (Angstrom); used for the coverage report only.
XSHOOTER_ARMS = [('UVB', 3000., 5595.),
                 ('VIS', 5595., 10240.),
                 ('NIR', 10240., 24800.)]

C_KMS = 2.99792458e5
SIGMA_SB = 5.670374419e-5        # erg/s/cm^2/K^4

## np.trapz was removed in numpy 2.0 in favour of np.trapezoid.
trapz = getattr(np, 'trapezoid', None) or np.trapz


##-----------------------------------------------------------------------------
## CDBS (low-resolution) source
##-----------------------------------------------------------------------------

def list_temps(fits_dir, temp_min, temp_max):
    temps = []
    for fln in glob.glob(os.path.join(fits_dir, '*_*.fits')):
        t = float(os.path.basename(fln).rsplit('_', 1)[1].replace('.fits', ''))
        if temp_min <= t <= temp_max:
            temps.append((t, fln))
    temps.sort()
    return temps


def load_cdbs(args):
    """
    Read the CDBS/pysynphot phoenixp05 grid and resample it onto an axis
    uniform in ln(wav).

    Returns (temps, logg, wav, flux, delta_lnwav, meta) where flux has shape
    (n_temp, n_logg, n_wav) and is a surface flux in erg/s/cm^2/A.
    """
    temps = list_temps(args.fits_dir, args.temp_lims[0], args.temp_lims[1])
    if not temps:
        raise SystemExit("No CDBS spectra found in {}.".format(args.fits_dir))
    print("Found {} temperature points between {} and {} K.".format(
        len(temps), args.temp_lims[0], args.temp_lims[1]))

    ## Wavelength axis: uniform in ln(wav), as required by AtmoGridSpec.
    delta_lnwav = args.delta_lnwav
    n_wav = int(np.log(args.wav_lims[1]/args.wav_lims[0])/delta_lnwav) + 1
    wav = args.wav_lims[0] * np.exp(np.arange(n_wav)*delta_lnwav)
    print("Wavelength axis: {} points, {:.1f}-{:.1f} A, delta_v = {:.1f} km/s (R ~ {:.0f}).".format(
        n_wav, wav[0], wav[-1], delta_lnwav*C_KMS, 1./delta_lnwav))

    ## Establish the logg axis from the first file.
    with fits.open(temps[0][1]) as f:
        all_logg_cols = [c for c in f[1].data.columns.names if c != 'WAVELENGTH']
    logg_cols = [c for c in all_logg_cols
                 if args.logg_lims[0] <= float(c[1:])/10. <= args.logg_lims[1]]
    logg = np.array([float(c[1:])/10. for c in logg_cols])
    print("logg axis: {} points, {} to {}.".format(logg.size, logg.min(), logg.max()))

    n_temp = len(temps)
    n_logg = logg.size
    check_memory(n_temp, n_logg, args.n_mu, n_wav, args)

    ## Resample every (Teff, logg) spectrum onto the uniform ln(wav) axis.
    flux = np.empty((n_temp, n_logg, n_wav), dtype=float)
    for i, (temp, fln) in enumerate(temps):
        with fits.open(fln) as f:
            data = f[1].data
            wav_native = data['WAVELENGTH']
            for j, col in enumerate(logg_cols):
                flux[i, j] = np.interp(wav, wav_native, data[col])
        print("  processed {} K".format(temp))

    meta = {'source': 'PHOENIX/BT-Settl phoenixp05 (CDBS), resampled uniform in ln(wav)',
            'resolution_note': 'R ~ {:.0f}; velocity structure is UNRESOLVED'.format(1./delta_lnwav)}
    return np.array([t for t, _ in temps]), logg, wav, flux, delta_lnwav, meta


##-----------------------------------------------------------------------------
## Husser (high-resolution) source, via download_phoenix.py + Starfish
##-----------------------------------------------------------------------------

def import_download_phoenix():
    """
    Import download_phoenix.py, which sits next to this script.

    It is a standalone CLI rather than an installed module, so it is loaded by
    path. It must be registered in sys.modules before execution because its
    dataclasses need to be able to look their own module up.
    """
    fln = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'download_phoenix.py')
    if not os.path.exists(fln):
        raise SystemExit("--source phoenix-hires needs download_phoenix.py next to "
                         "this script; {} does not exist.".format(fln))
    spec = importlib.util.spec_from_file_location('download_phoenix', fln)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['download_phoenix'] = mod
    spec.loader.exec_module(mod)
    return mod


def hires_axes(dp, temp_lims, logg_lims):
    """The Husser grid points falling inside the requested limits."""
    temp_lo = max(temp_lims[0], HIRES_TEMP_RANGE[0])
    temp_hi = min(temp_lims[1], HIRES_TEMP_RANGE[1])
    if (temp_lo, temp_hi) != tuple(temp_lims):
        print("NOTE: the Husser grid only runs over {:.0f}-{:.0f} K; the requested "
              "{:.0f}-{:.0f} K was clipped to {:.0f}-{:.0f} K.".format(
                  HIRES_TEMP_RANGE[0], HIRES_TEMP_RANGE[1],
                  temp_lims[0], temp_lims[1], temp_lo, temp_hi))
    temps = np.array([float(t) for t in dp.NOMINAL_TEFF if temp_lo <= t <= temp_hi])
    logg = np.array([float(g) for g in dp.NOMINAL_LOGG
                     if logg_lims[0] <= g <= logg_lims[1]])
    if not temps.size or not logg.size:
        raise SystemExit("Empty selection: the Husser grid has Teff 2300-7000 K in "
                         "100 K steps, 7000-12000 K in 200 K steps, and logg 0.0-6.0 "
                         "in 0.5 dex.")
    return temps, logg


def hires_download(dp, temps, logg, args):
    """Fetch the raw library with download_phoenix.py's own CLI."""
    argv = ['-o', args.hires_lib,
            '--teff-values'] + ['{:.0f}'.format(t) for t in temps] + \
           ['--logg-values'] + ['{:.2f}'.format(g) for g in logg] + \
           ['--Z', '{:.1f}'.format(args.Z),
            '-j', str(args.workers),
            '--max-gb', str(args.max_gb)]
    if args.yes:
        argv.append('--yes')
    print("Downloading the raw library: download_phoenix.py {}".format(
        ' '.join(argv[:6]) + ' ...'))
    rc = dp.main(argv)
    if rc:
        raise SystemExit("download_phoenix.py reported failures (exit {}); not "
                         "building the grid on an incomplete library. Re-run to "
                         "retry only the failures, or pass --no-download once "
                         "the library is complete.".format(rc))


def hires_report_absolute_flux(dp, iface, temp, logg, args):
    """
    Check that the raw spectra really are absolute surface fluxes.

    Integrating F_lam over the *full* Husser range (500 A - 5.5 micron, i.e.
    the untruncated wl array, not the slice we keep) should recover
    sigma*T^4. This is the same sanity check that validates the CDBS grid's
    absolute scale, and it is the thing that breaks first if Starfish's
    per-spectrum renormalization ever sneaks back in.
    """
    fln = os.path.join(args.hires_lib, dp.phoenix_relpath(temp, logg, args.Z))
    if not os.path.exists(fln):
        return
    flux_full = fits.getdata(fln)*1e-8            # erg/s/cm^2/cm -> /A
    f_bol = trapz(flux_full, iface.wl_full)
    ratio = f_bol/(SIGMA_SB*temp**4)
    print("Absolute scale check at {:.0f} K, logg {:.2f}: integral F_lam dlam = "
          "{:.3f} x sigma*T^4.".format(temp, logg, ratio))
    if not 0.9 < ratio < 1.1:
        print("WARNING: that is more than 10% off unity. The spectra may not be "
              "absolute surface fluxes, which would put the whole grid on the "
              "wrong absolute scale.")


def hires_instrument(args, wl_range):
    """
    The Starfish Instrument to convolve to: either one of its named
    instruments, or a bare resolving power turned into a Gaussian FWHM.

    Called both when building the intermediate grid and when writing the
    metadata for an already-built one, so the two can never disagree.
    """
    try:
        from Starfish.grid_tools import instruments as sf_instruments
    except ImportError as exc:
        raise SystemExit("--source phoenix-hires needs Starfish ({}). "
                         "Try: pip install astrostarfish".format(exc))

    if args.instrument:
        cls = getattr(sf_instruments, args.instrument, None)
        if cls is None:
            available = sorted(n for n, o in vars(sf_instruments).items()
                               if isinstance(o, type)
                               and issubclass(o, sf_instruments.Instrument)
                               and o is not sf_instruments.Instrument)
            raise SystemExit("Unknown instrument {!r}. Available: {}".format(
                args.instrument, ', '.join(available)))
        return cls()

    return sf_instruments.Instrument(name='R={:.0f}'.format(args.resolution),
                                     FWHM=C_KMS/args.resolution,
                                     wl_range=wl_range,
                                     oversampling=args.oversampling)


def hires_iface_range(args):
    """
    Wavelength range to read from the raw library.

    Kept wider than what is asked for: the instrumental convolution is done by
    FFT, so it wraps around, and the resampling needs valid neighbours at the
    edges. Everything outside --wav-lims is trimmed off again on read.
    """
    return (max(500., args.wav_lims[0] - 2*HIRES_PAD), args.wav_lims[1] + 2*HIRES_PAD)


def hires_wav_size(args):
    """
    How many wavelength points the finished grid will have.

    Worth knowing before downloading tens of gigabytes: it follows
    deterministically from Starfish's create_log_lam_grid, which rounds the
    number of pixels up to a power of two, so the sampling can be up to twice
    as fine as FWHM/oversampling and the grid up to twice as large as a
    back-of-the-envelope estimate suggests.
    """
    from Starfish.utils import create_log_lam_grid

    ## Mirror the bounds hires_build_hdf5 and HDF5Creator arrive at, including
    ## HDF5Creator's own 50 A buffer, so this is the count and not a guess.
    wav_lo, wav_hi = args.wav_lims
    iface_range = hires_iface_range(args)
    instrument = hires_instrument(args, iface_range)
    wl = create_log_lam_grid(
        instrument.FWHM/instrument.oversampling,
        max(instrument.wl_range[0], iface_range[0], wav_lo - HIRES_PAD - 50.),
        min(instrument.wl_range[1], iface_range[1], wav_hi + HIRES_PAD + 50.))['wl']
    return int(((wl >= wav_lo) & (wl <= wav_hi)).sum())


def hires_build_hdf5(dp, temps, logg, args):
    """
    Convolve the raw library to the requested resolution and store it as a
    Starfish HDF5 grid, whose wavelength axis is uniform in log(wav).

    Starfish does the heavy lifting (FFT instrumental broadening, then
    resampling), but two of its defaults have to be overridden:

    - PHOENIXGridInterface.load_flux renormalizes every spectrum to a
      bolometric luminosity of 1 L_sun. That is right for fitting a
      normalized spectrum and wrong for us: Icarus needs the true surface
      flux to get the absolute scale (and hence the flux ratio between the
      two stars) right, so the subclass below turns it off.
    - The default HDF5 dataset keys embed unformatted floats, which makes
      lookups depend on exact float repr. We pin the precision instead.
    """
    try:
        from Starfish.grid_tools import HDF5Creator, PHOENIXGridInterfaceNoAlpha
    except ImportError as exc:
        raise SystemExit("--source phoenix-hires needs Starfish ({}). "
                         "Try: pip install astrostarfish".format(exc))

    class SurfaceFluxPHOENIX(PHOENIXGridInterfaceNoAlpha):
        """PHOENIX interface that keeps the absolute surface flux."""
        def load_flux(self, parameters, header=False, norm=True):
            ## norm is accepted for signature compatibility and deliberately
            ## ignored: HDF5Creator calls this with the default norm=True.
            out = super().load_flux(parameters, header=header, norm=False)
            if header:
                flux, hdr = out
                hdr['norm'] = 'surface flux (no L_sun renormalization)'
                return flux*1e-8, hdr        # erg/s/cm^2/cm -> erg/s/cm^2/A
            return out*1e-8

    wav_lo, wav_hi = args.wav_lims
    iface_range = hires_iface_range(args)

    instrument = hires_instrument(args, iface_range)
    print("Convolving to {} (FWHM {:.2f} km/s, R ~ {:.0f}), {:.1f} output pixels "
          "per FWHM requested.".format(instrument.name, instrument.FWHM,
                                       C_KMS/instrument.FWHM, instrument.oversampling))
    if instrument.wl_range[0] > wav_lo or instrument.wl_range[1] < wav_hi:
        print("WARNING: {} only covers {:.0f}-{:.0f} A, so the grid will be "
              "truncated to that; the requested {:.0f}-{:.0f} A will not be fully "
              "covered.".format(instrument.name, instrument.wl_range[0],
                                instrument.wl_range[1], wav_lo, wav_hi))

    print("Opening the raw library at {} ({} wavelengths).".format(
        args.hires_lib, 'air' if args.air else 'vacuum'))
    iface = SurfaceFluxPHOENIX(args.hires_lib, air=args.air, wl_range=iface_range)
    dp._extend_par_dicts(iface)
    dp._guard_missing_spectra(iface)
    hires_report_absolute_flux(dp, iface, temps[0], logg[0], args)

    ## HDF5Creator filters points with >= low / <= high against values that come
    ## out of np.arange, so pad the bounds outward by an epsilon.
    eps = 1e-6
    ranges = [[temps.min() - eps, temps.max() + eps],
              [logg.min() - eps, logg.max() + eps],
              [args.Z - eps, args.Z + eps]]

    print("Building the Starfish grid -> {} (this is the slow step: one FFT and "
          "two spline resamplings per spectrum).".format(args.hires_hdf5))
    creator = HDF5Creator(iface, args.hires_hdf5, instrument=instrument,
                          wl_range=[wav_lo - HIRES_PAD, wav_hi + HIRES_PAD],
                          ranges=ranges,
                          key_name='T{0:.0f}_logg{1:.2f}_Z{2:+.1f}')
    creator.process_grid()
    print("Starfish grid written: {} ({:.1f} MB).".format(
        args.hires_hdf5, os.path.getsize(args.hires_hdf5)/1024.**2))


def load_hires(args):
    """
    Download (unless --no-download), convolve and read the Husser library.

    Returns (temps, logg, wav, flux, delta_lnwav, meta) with the same
    conventions as load_cdbs.
    """
    dp = import_download_phoenix()
    temps, logg = hires_axes(dp, args.temp_lims, args.logg_lims)
    print("Selection: {} temperatures ({:.0f}-{:.0f} K), {} logg values "
          "({:.1f}-{:.1f}), [Fe/H] = {:+.1f}.".format(
              temps.size, temps[0], temps[-1], logg.size, logg[0], logg[-1], args.Z))
    ## Check the output size before the download and the convolution, not after.
    check_memory(temps.size, logg.size, args.n_mu, hires_wav_size(args), args,
                 label='Output grid (predicted)')

    if args.download:
        hires_download(dp, temps, logg, args)

    if args.rebuild_hdf5 and os.path.exists(args.hires_hdf5):
        os.remove(args.hires_hdf5)
    if not os.path.exists(args.hires_hdf5):
        hires_build_hdf5(dp, temps, logg, args)
    else:
        print("Reusing the existing Starfish grid {} (--rebuild-hdf5 to "
              "rebuild it).".format(args.hires_hdf5))

    from Starfish.grid_tools import HDF5Interface
    grid = HDF5Interface(args.hires_hdf5)

    ## The Starfish axis is uniform in log10(wav) by construction, so it is
    ## uniform in ln(wav) too and can be used as-is -- no second interpolation.
    ind = (grid.wl >= args.wav_lims[0]) & (grid.wl <= args.wav_lims[1])
    wav = grid.wl[ind]
    if wav.size < 2:
        raise SystemExit("The Starfish grid does not cover {}-{} A.".format(*args.wav_lims))
    delta_lnwav = np.log(wav[1]/wav[0])
    spread = np.ptp(np.diff(np.log(wav)))
    if spread > 1e-3*delta_lnwav:
        raise SystemExit("The Starfish wavelength axis is not uniform in ln(wav) "
                         "(spread {:.3g} vs spacing {:.3g}); AtmoGridSpec requires "
                         "that it is.".format(spread, delta_lnwav))
    print("Wavelength axis: {} points, {:.1f}-{:.1f} A, delta_v = {:.2f} km/s "
          "(sampling R ~ {:.0f}).".format(wav.size, wav[0], wav[-1],
                                          delta_lnwav*C_KMS, 1./delta_lnwav))

    ## Restrict the axes to what actually made it into the grid, then fill the
    ## remaining holes: AtmoGridSpec needs a rectangular grid.
    available = {(round(p[0], 3), round(p[1], 3)) for p in grid.grid_points}
    temps = np.array([t for t in temps if any((round(t, 3), round(g, 3)) in available
                                              for g in logg)])
    if not temps.size:
        raise SystemExit("None of the requested spectra are in {}.".format(args.hires_hdf5))

    check_memory(temps.size, logg.size, args.n_mu, wav.size, args,
                 label='Output grid (actual)')

    flux = np.empty((temps.size, logg.size, wav.size), dtype=float)
    filled = []
    for i, temp in enumerate(temps):
        have = [g for g in logg if (round(temp, 3), round(g, 3)) in available]
        for j, g in enumerate(logg):
            g_use = g if g in have else min(have, key=lambda x: abs(x - g))
            if g_use != g:
                filled.append((temp, g, g_use))
            flux[i, j] = grid.load_flux(np.array([temp, g_use, args.Z]))[ind]
        print("  read {:.0f} K".format(temp))

    if filled:
        print("WARNING: {} (Teff, logg) combinations are absent from the Husser "
              "library and were filled from the nearest available logg (the grid "
              "must be rectangular). This flattens the logg dependence there:".format(
                  len(filled)))
        for temp, g, g_use in filled[:20]:
            print("    {:.0f} K logg {:.1f} <- logg {:.1f}".format(temp, g, g_use))
        if len(filled) > 20:
            print("    ... and {} more".format(len(filled) - 20))

    ## Report the instrumental FWHM, not the pixel spacing: create_log_lam_grid
    ## rounds the number of pixels up to a power of two, so the actual sampling
    ## is finer than FWHM/oversampling by up to a factor of two and says nothing
    ## about the resolution.
    fwhm = hires_instrument(args, (wav[0], wav[-1])).FWHM
    meta = {'source': ('PHOENIX-ACES-AGSS-COND-2011 (Husser 2013) HiRes, [Fe/H] = '
                       '{:+.1f}, convolved and resampled with Starfish'.format(args.Z)),
            'wavelength_scale': 'air' if args.air else 'vacuum',
            'resolution_note': ('instrumental FWHM {:.2f} km/s (R ~ {:.0f}), sampled at '
                                '{:.2f} km/s/pixel'.format(fwhm, C_KMS/fwhm,
                                                           delta_lnwav*C_KMS))}
    return temps, logg, wav, flux, delta_lnwav, meta


##-----------------------------------------------------------------------------
## Shared assembly
##-----------------------------------------------------------------------------

def check_memory(n_temp, n_logg, n_mu, n_wav, args, label='Output grid'):
    """Refuse to silently allocate a grid larger than the caller expects."""
    nbytes = n_temp*n_logg*n_mu*n_wav*np.dtype(args.dtype).itemsize
    print("{}: {} x {} x {} x {} = {:.2f} GB in {}.".format(
        label, n_temp, n_logg, n_mu, n_wav, nbytes/1024.**3, args.dtype))
    if nbytes > args.max_mem_gb*1024.**3 and not args.yes:
        raise SystemExit(
            "That is over the --max-mem-gb limit of {:g} GB. Narrow --wav-lims, "
            "--temp-lims or --logg-lims, lower -R, --n-mu or --oversampling, use "
            "--dtype float32, raise --max-mem-gb, or pass --yes.".format(args.max_mem_gb))


def assemble(temps, logg, wav, flux, delta_lnwav, meta, args):
    """Impose limb darkening, take the log and write the AtmoGridSpec file."""
    ## Guard against non-positive fluxes, which would make log(flux) invalid.
    n_bad = (flux <= 0).sum()
    if n_bad:
        floor = flux[flux > 0].min() * 1e-10
        print("WARNING: {} non-positive flux values ({:.3g}% of the grid); "
              "flooring them at {:.3e} so log(flux) stays finite.".format(
                  n_bad, 100.*n_bad/flux.size, floor))
        flux = np.maximum(flux, floor)

    ## mu axis + analytic limb darkening, evaluated per wavelength.
    ## NOTE: Limb_darkening expects the wavelength in MICRONS.
    mu = np.linspace(0., 1., args.n_mu)
    mu_factor = Limb_darkening(wav*1e-4, mu.reshape(-1, 1))   # (n_mu, n_wav)
    print("Limb darkening (centre-to-limb ratio): {:.3f} at {:.0f} A, {:.3f} at {:.0f} A.".format(
        mu_factor[0, 0], wav[0], mu_factor[0, -1], wav[-1]))

    ## Surface flux -> specific intensity. For a plane-parallel atmosphere with
    ## emergent intensity I(mu) = I0*LD(mu), the surface flux is
    ##     F = 2*pi * Int_0^1 I(mu)*mu dmu = 2*pi*I0 * Int_0^1 LD(mu)*mu dmu,
    ## so flux conservation requires I0 = F / (2*pi*Int LD(mu)*mu dmu).
    ld_int = trapz(mu_factor*mu[:, None], mu, axis=0)      # (n_wav,)
    icarus_norm = 4/np.pi**2
    if args.exact_ld_normalization:
        norm = 1./(2*np.pi*ld_int)
        print("Using exact per-wavelength flux-conserving normalization.")
    else:
        norm = np.full(wav.size, icarus_norm)
        print("Using Icarus' hardcoded 4/pi^2 normalization (not flux-conserving).")
    ## Report how far Icarus' hardcoded factor would have been off, so the
    ## size of the systematic being corrected (or kept) is on the record.
    recovered = 2*np.pi*icarus_norm*ld_int
    print("Icarus' 4/pi^2 factor would recover {:.1%} of the surface flux at "
          "{:.0f} A and {:.1%} at {:.0f} A.".format(
              recovered[0], wav[0], recovered[-1], wav[-1]))

    ## Combine into the (logtemp, logg, mu, wav) grid, in natural-log flux.
    ## Filled one (temp, logg) at a time: the whole grid can be several GB, and
    ## the naive broadcast expression would need three copies of it at once.
    intensity_factor = mu_factor*norm[None, :]
    grid = np.empty((temps.size, logg.size, args.n_mu, wav.size), dtype=args.dtype)
    for i in range(temps.size):
        for j in range(logg.size):
            grid[i, j] = np.log(flux[i, j][None, :]*intensity_factor)

    atmo = AtmoGridSpec(
        data=grid,
        cols=[('logtemp', np.log(temps)), ('logg', logg), ('mu', mu), ('wav', wav)],
        meta=dict({'zp': 0.0,
                   'delta_v': delta_lnwav,
                   'units': 'log(specific intensity), erg/s/cm^2/A/sr-like',
                   'magsys': 'none',
                   'normalization': ('exact per-wavelength flux-conserving'
                                     if args.exact_ld_normalization else 'Icarus 4/pi^2')},
                  **meta))

    if os.path.exists(args.out):
        os.remove(args.out)
    atmo.WriteHDF5(args.out)
    print("Wrote {} ({:.1f} MB, shape {}).".format(
        args.out, os.path.getsize(args.out)/1024.**2, grid.shape))
    print("X-shooter arm coverage:")
    for name, lo, hi in XSHOOTER_ARMS:
        n_in = int(((wav >= lo) & (wav <= hi)).sum())
        print("  {:3s} {:6.0f}-{:6.0f} A : {:4d} grid points".format(name, lo, hi, n_in))


##-----------------------------------------------------------------------------
## CLI
##-----------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=("Build the Icarus AtmoGridSpec spectroscopic atmosphere grid for "
                     "J2241-5236, from either the low-resolution CDBS PHOENIX grid or "
                     "the high-resolution Husser (2013) PHOENIX library."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--source', choices=['cdbs', 'phoenix-hires'], default=SOURCE,
                   help="Model library to build from.")
    p.add_argument('-o', '--out', default=None,
                   help="Output HDF5 grid. Default: {} for cdbs, {} for "
                        "phoenix-hires.".format(OUT_FLN, OUT_FLN_HIRES))
    p.add_argument('--temp-lims', nargs=2, type=float, default=[TEMP_MIN, TEMP_MAX],
                   metavar=('MIN', 'MAX'), help="Teff range in K.")
    p.add_argument('--logg-lims', nargs=2, type=float, default=list(LOGG_LIMS),
                   metavar=('MIN', 'MAX'), help="log g range.")
    p.add_argument('--wav-lims', nargs=2, type=float, default=[WAV_MIN, WAV_MAX],
                   metavar=('MIN', 'MAX'), help="Wavelength range in Angstrom.")
    p.add_argument('--n-mu', type=int, default=N_MU,
                   help="Number of mu (cos emission angle) points.")
    p.add_argument('--icarus-norm', dest='exact_ld_normalization', action='store_false',
                   default=EXACT_LD_NORMALIZATION,
                   help="Use Icarus' hardcoded 4/pi^2 surface-flux factor. Without this "
                        "flag the exact per-wavelength flux-conserving normalization is "
                        "used. (The default shown below is that of the flag's dest, i.e. "
                        "'exact normalization on'.)")
    p.add_argument('--dtype', choices=['float64', 'float32'], default='float64',
                   help="Storage type of the grid. Icarus casts to float64 on read, "
                        "so float32 only halves the file size.")
    p.add_argument('--max-mem-gb', type=float, default=8.0,
                   help="Refuse to build a grid larger than this without --yes.")
    p.add_argument('-y', '--yes', action='store_true',
                   help="Skip the grid-size and download-size confirmations.")

    low = p.add_argument_group('--source cdbs')
    low.add_argument('--fits-dir', default=FITS_DIR, help="CDBS phoenixp05 directory.")
    low.add_argument('--delta-lnwav', type=float, default=DELTA_LNWAV,
                     help="Pixel spacing in ln(wav), i.e. v/c per pixel.")

    hi = p.add_argument_group('--source phoenix-hires')
    hi.add_argument('--hires-lib', default=HIRES_LIB,
                    help="Root of the raw Husser library (download_phoenix.py layout).")
    hi.add_argument('--hires-hdf5', default=HIRES_HDF5,
                    help="Intermediate Starfish HDF5 grid. Default: the output name "
                         "with a _starfish suffix.")
    hi.add_argument('--Z', type=float, default=HIRES_Z,
                    help="[Fe/H]. +0.5 matches the phoenixp05 photometric grids.")
    hi.add_argument('-R', '--resolution', type=float, default=HIRES_R,
                    help="Target resolving power lambda/dlambda.")
    hi.add_argument('--instrument', default=None,
                    help="Named Starfish instrument (TRES, ESPaDOnS, SPEX, ...) to "
                         "convolve to, instead of -R.")
    hi.add_argument('--oversampling', type=float, default=HIRES_OVERSAMPLING,
                    help="Output pixels per resolution FWHM.")
    hi.add_argument('--air', action='store_true', default=HIRES_AIR,
                    help="Convert to air wavelengths (default: keep vacuum).")
    hi.add_argument('--no-download', dest='download', action='store_false',
                    help="Assume the raw library is already complete on disk.")
    hi.add_argument('--rebuild-hdf5', action='store_true',
                    help="Rebuild the intermediate Starfish grid even if it exists.")
    hi.add_argument('-j', '--workers', type=int, default=4,
                    help="Concurrent downloads.")
    hi.add_argument('--max-gb', type=float, default=50.0,
                    help="Download size ceiling passed to download_phoenix.py.")

    p.add_argument('-v', '--verbose', action='store_true',
                   help="Debug-level logging from Starfish and the downloader.")

    args = p.parse_args(argv)
    if args.out is None:
        args.out = OUT_FLN if args.source == 'cdbs' else OUT_FLN_HIRES
    if args.hires_hdf5 is None:
        args.hires_hdf5 = os.path.splitext(args.out)[0] + '_starfish.h5'
    if args.wav_lims[0] >= args.wav_lims[1]:
        p.error("--wav-lims must be increasing")
    if args.n_mu < 2:
        p.error("--n-mu must be at least 2")
    return args


def build(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(message)s',
                        datefmt='%H:%M:%S')
    loader = load_cdbs if args.source == 'cdbs' else load_hires
    temps, logg, wav, flux, delta_lnwav, meta = loader(args)
    assemble(temps, logg, wav, flux, delta_lnwav, meta, args)


if __name__ == '__main__':
    build()
