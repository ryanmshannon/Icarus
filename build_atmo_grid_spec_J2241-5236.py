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
- SPECTRAL RESOLUTION. The CDBS PHOENIX grid is heavily resampled: over
  4300-9000 A its native sampling corresponds to R ~ 600-740, i.e. ~400-500
  km/s per pixel. The companion's projected velocity semi-amplitude is only
  ~350 km/s, so ALL velocity structure (line profiles, rotational
  broadening, the orbital Doppler shift) is UNRESOLVED in this grid. The
  computed spectrum is a valid phase-dependent SED, but it must NOT be used
  for radial-velocity, line-profile or vsini work. For that, rebuild this
  grid from the full-resolution BT-Settl spectra.
- The wavelength range is restricted to WAV_MIN..WAV_MAX below, chosen to
  stay inside the validity range of the Neckel (2005) limb-darkening law
  (0.42257-1.100 micron), which Limb_darkening silently extrapolates
  outside of.
- Metallicity is fixed at [M/H] = +0.5 (the phoenixp05 grid); the true
  companion metallicity is unknown and is not fit for.
- The absolute flux calibration is not independently validated (same
  caveat as the photometric grids).

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
WAV_MIN = 4300.                  # Angstrom -- inside the Neckel (2005) LD
WAV_MAX = 9000.                  # Angstrom    validity range (4226-11000 A)
DELTA_LNWAV = 1.35e-3            # ~405 km/s per pixel; matches the finest
                                 # native CDBS sampling, so we do not invent
                                 # resolution the underlying models lack.
N_MU = 16                        # same mu sampling as Atmo_photo_BTSettl7


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

    ## Combine into the (logtemp, logg, mu, wav) grid, in natural-log flux.
    ## The 4/pi^2 factor matches the convention used by Atmo_photo_BTSettl7
    ## for the photometric grids, so the two sets are mutually consistent.
    grid = np.log(flux[:, :, None, :] * mu_factor[None, None, :, :] * 4/np.pi**2)

    logtemp = np.log(np.array([t for t, _ in temps]))
    atmo = AtmoGridSpec(
        data=grid,
        cols=[('logtemp', logtemp), ('logg', logg), ('mu', mu), ('wav', wav)],
        meta={'zp': 0.0,
              'delta_v': DELTA_LNWAV,
              'units': 'log(F_lambda), erg/s/cm^2/A (times 4/pi^2)',
              'magsys': 'none',
              'source': 'PHOENIX/BT-Settl phoenixp05 (CDBS), resampled uniform in ln(wav)',
              'resolution_note': 'R ~ {:.0f}; velocity structure is UNRESOLVED'.format(1./DELTA_LNWAV)})

    if os.path.exists(OUT_FLN):
        os.remove(OUT_FLN)
    atmo.WriteHDF5(OUT_FLN)
    print("Wrote {} ({:.1f} MB, shape {}).".format(
        OUT_FLN, os.path.getsize(OUT_FLN)/1024.**2, grid.shape))


if __name__ == '__main__':
    build()
