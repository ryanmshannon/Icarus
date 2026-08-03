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

Data used
---------
PHOENIX/BT-Settl (Allard et al.) synthetic stellar spectra, [M/H] = +0.5,
in the CDBS/pysynphot FITS grid format, read from FITS_DIR below. Each file
holds one effective temperature and provides F_lambda (FLAM, i.e.
erg/s/cm^2/Angstrom) for a set of log(g) values.

Method
------
1. Every (Teff, log g) spectrum is resampled onto a wavelength grid that is
   uniformly spaced in ln(wavelength). This is REQUIRED by Icarus:
   AtmoGridSpec.Get_flux_doppler applies the per-surface-element Doppler
   shift as a (fractional) pixel shift of val_vel/delta_v, which is only
   correct if every pixel spans the same velocity interval. The native CDBS
   sampling is *not* uniform in ln(wav) (it varies between ~1.35e-3 and
   ~6.2e-3 over the optical), so resampling is not optional.
2. A mu (cos of emission angle) axis is created by applying the analytic
   Neckel (2005) limb-darkening law from Icarus.Utils.Flux.Limb_darkening,
   evaluated per wavelength. The CDBS PHOENIX grid only provides
   disk-integrated fluxes, not specific intensities, so limb darkening has
   to be imposed analytically -- exactly as Icarus' own
   Atmo_photo_BTSettl7 loader does for the photometric grids.
3. The result is stored as natural-log flux in an AtmoGridSpec HDF5 file,
   using the same 4/pi^2 surface-intensity convention as the photometric
   grids so the two are mutually consistent.

IMPORTANT CAVEATS -- read before using for science
---------------------------------------------------
- SPECTRAL RESOLUTION. The CDBS PHOENIX grid is heavily resampled: its
  native sampling is R ~ 740 (~405 km/s per pixel) from 4300 A out to
  25000 A, and only R ~ 161 below 4300 A. The companion's projected
  velocity semi-amplitude is only ~350 km/s, so ALL velocity structure
  (line profiles, rotational broadening, the orbital Doppler shift) is
  UNRESOLVED. X-shooter itself resolves R ~ 5000-9000, i.e. ~7-10x finer
  than this grid, so the predicted spectrum is a valid phase-dependent SED
  and is fine for predicting broad-band count rates or continuum shape,
  but it must NOT be used to predict line profiles, equivalent widths,
  radial velocities or vsini. For that, rebuild from the full-resolution
  BT-Settl spectra.
- LIMB-DARKENING EXTRAPOLATION. The Neckel (2005) law implemented by
  Utils.Flux.Limb_darkening is only formally valid over 0.42257-1.100
  micron; Limb_darkening silently extrapolates outside that range. The
  X-shooter range used here therefore relies on extrapolation below
  4226 A and, more substantially, over the whole NIR arm above 11000 A.
  The extrapolation is smooth, monotonic and physically plausible (the
  centre-to-limb ratio saturates from 0.51 at 1.1 micron to 0.65 at 2.48
  micron, i.e. weaker limb darkening towards the IR, which is the correct
  qualitative behaviour), but it is not a validated fit there.
- Metallicity is fixed at [M/H] = +0.5 (the phoenixp05 grid); the true
  companion metallicity is unknown and is not fit for.
- ABSOLUTE CALIBRATION. The grid is in physical units, not relative ones:
  the CDBS spectra are true surface fluxes (verified: integral F_lam dlam
  reproduces sigma*T^4), and Icarus' area/projection scaling makes
  star.Flux_doppler return the flux density at a distance of 10 parsec.
  Unlike the photometric grids, this grid does NOT use Icarus' hardcoded
  4/pi^2 surface-flux -> specific-intensity factor by default. That factor
  is only self-consistent with the limb-darkening law where the integral
  of LD(mu)*mu dmu equals pi/8 (near 5000 A), and would recover only 86%
  of the surface flux at 3000 A and 119% at 24800 A -- a 33% swing across
  the X-shooter range that would visibly distort the predicted SED. This
  script instead renormalizes per wavelength so the hemispheric integral
  exactly reproduces the input surface flux (EXACT_LD_NORMALIZATION, set
  to False to reproduce Icarus' behaviour). Consequently the absolute
  scale of this grid differs from that of the photometric grids by that
  same wavelength-dependent factor.

>>> python build_atmo_grid_spec_J2241-5236.py
"""

import os
import glob

import numpy as np
from astropy.io import fits

from Icarus.Atmosphere.Atmo import AtmoGridSpec
from Icarus.Utils.Flux import Limb_darkening


##### Configuration
FITS_DIR = 'model_spec/grp/redcat/trds/grid/phoenix/phoenixp05'
OUT_FLN = 'atmo_grid_spec_J2241-5236.h5'
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
N_MU = 16                        # same mu sampling as Atmo_photo_BTSettl7

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


def list_temps(fits_dir, temp_min, temp_max):
    temps = []
    for fln in glob.glob(os.path.join(fits_dir, '*_*.fits')):
        t = float(os.path.basename(fln).rsplit('_', 1)[1].replace('.fits', ''))
        if temp_min <= t <= temp_max:
            temps.append((t, fln))
    temps.sort()
    return temps


def build():
    temps = list_temps(FITS_DIR, TEMP_MIN, TEMP_MAX)
    print("Found {} temperature points between {} and {} K.".format(len(temps), TEMP_MIN, TEMP_MAX))

    ## Wavelength axis: uniform in ln(wav), as required by AtmoGridSpec.
    n_wav = int(np.log(WAV_MAX/WAV_MIN)/DELTA_LNWAV) + 1
    wav = WAV_MIN * np.exp(np.arange(n_wav)*DELTA_LNWAV)
    print("Wavelength axis: {} points, {:.1f}-{:.1f} A, delta_v = {:.1f} km/s (R ~ {:.0f}).".format(
        n_wav, wav[0], wav[-1], DELTA_LNWAV*2.99792458e5, 1./DELTA_LNWAV))

    ## Establish the logg axis from the first file.
    with fits.open(temps[0][1]) as f:
        all_logg_cols = [c for c in f[1].data.columns.names if c != 'WAVELENGTH']
    logg_cols = [c for c in all_logg_cols if LOGG_LIMS[0] <= float(c[1:])/10. <= LOGG_LIMS[1]]
    logg = np.array([float(c[1:])/10. for c in logg_cols])
    print("logg axis: {} points, {} to {}.".format(logg.size, logg.min(), logg.max()))

    n_temp = len(temps)
    n_logg = logg.size

    ## Resample every (Teff, logg) spectrum onto the uniform ln(wav) axis.
    flux = np.empty((n_temp, n_logg, n_wav), dtype=float)
    for i, (temp, fln) in enumerate(temps):
        with fits.open(fln) as f:
            data = f[1].data
            wav_native = data['WAVELENGTH']
            for j, col in enumerate(logg_cols):
                flux[i, j] = np.interp(wav, wav_native, data[col])
        print("  processed {} K".format(temp))

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
    mu = np.linspace(0., 1., N_MU)
    mu_factor = Limb_darkening(wav*1e-4, mu.reshape(-1, 1))   # (N_MU, n_wav)
    print("Limb darkening (centre-to-limb ratio): {:.3f} at {:.0f} A, {:.3f} at {:.0f} A.".format(
        mu_factor[0, 0], wav[0], mu_factor[0, -1], wav[-1]))

    ## Surface flux -> specific intensity. For a plane-parallel atmosphere with
    ## emergent intensity I(mu) = I0*LD(mu), the surface flux is
    ##     F = 2*pi * Int_0^1 I(mu)*mu dmu = 2*pi*I0 * Int_0^1 LD(mu)*mu dmu,
    ## so flux conservation requires I0 = F / (2*pi*Int LD(mu)*mu dmu).
    ld_int = np.trapz(mu_factor*mu[:, None], mu, axis=0)      # (n_wav,)
    icarus_norm = 4/np.pi**2
    if EXACT_LD_NORMALIZATION:
        norm = 1./(2*np.pi*ld_int)
        print("Using exact per-wavelength flux-conserving normalization.")
    else:
        norm = np.full(n_wav, icarus_norm)
        print("Using Icarus' hardcoded 4/pi^2 normalization (not flux-conserving).")
    ## Report how far Icarus' hardcoded factor would have been off, so the
    ## size of the systematic being corrected (or kept) is on the record.
    recovered = 2*np.pi*icarus_norm*ld_int
    print("Icarus' 4/pi^2 factor would recover {:.1%} of the surface flux at "
          "{:.0f} A and {:.1%} at {:.0f} A.".format(
              recovered[0], wav[0], recovered[-1], wav[-1]))

    ## Combine into the (logtemp, logg, mu, wav) grid, in natural-log flux.
    grid = np.log(flux[:, :, None, :] * (mu_factor*norm[None, :])[None, None, :, :])

    logtemp = np.log(np.array([t for t, _ in temps]))
    atmo = AtmoGridSpec(
        data=grid,
        cols=[('logtemp', logtemp), ('logg', logg), ('mu', mu), ('wav', wav)],
        meta={'zp': 0.0,
              'delta_v': DELTA_LNWAV,
              'units': 'log(specific intensity), erg/s/cm^2/A/sr-like',
              'magsys': 'none',
              'source': 'PHOENIX/BT-Settl phoenixp05 (CDBS), resampled uniform in ln(wav)',
              'normalization': ('exact per-wavelength flux-conserving'
                                if EXACT_LD_NORMALIZATION else 'Icarus 4/pi^2'),
              'resolution_note': 'R ~ {:.0f}; velocity structure is UNRESOLVED'.format(1./DELTA_LNWAV)})

    if os.path.exists(OUT_FLN):
        os.remove(OUT_FLN)
    atmo.WriteHDF5(OUT_FLN)
    print("Wrote {} ({:.1f} MB, shape {}).".format(
        OUT_FLN, os.path.getsize(OUT_FLN)/1024.**2, grid.shape))
    print("X-shooter arm coverage:")
    for name, lo, hi in XSHOOTER_ARMS:
        n_in = int(((wav >= lo) & (wav <= hi)).sum())
        print("  {:3s} {:6.0f}-{:6.0f} A : {:4d} grid points".format(name, lo, hi, n_in))


if __name__ == '__main__':
    build()
