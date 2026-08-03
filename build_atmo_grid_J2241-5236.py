# Licensed under a 3-clause BSD style license - see LICENSE

"""
build_atmo_grid_J2241-5236.py

Builds the photometric atmosphere grids needed by model_J2241-5236.py:
    atmo_models_J2241-5236.txt              (index file)
    atmo_grid_J2241-5236_{g,r,i}.h5         (per-band grids, in Icarus'
                                              AtmoGridPhot HDF5 format)

Data used
---------
- PHOENIX/BT-Settl (Allard et al.) synthetic stellar spectra, solar-ish
  metallicity ([M/H] = +0.5), in the CDBS/pysynphot FITS grid format,
  read from FITS_DIR below (a local copy of the "phoenix/phoenixp05" grid).
  Per the grid's own AA_README, these are the BT-Settl models (original
  filenames of the form "lte067-1.5-0.0.BT-Settl.7.bz2") repackaged by
  STScI/CDBS -- i.e. exactly the model family Icarus.Atmosphere's
  Atmo_photo_BTSettl7 loader expects.
- SDSS g/r/i total-system (photon-counting) response curves from the SVO
  Filter Profile Service (http://svo2.cab.inta-csic.es/theory/fps/), saved
  under filters/SDSS_{g,r,i}.dat. This is a standard, public, non-
  proprietary photometric system -- a reasonable stand-in since the
  discovery/companion papers for PSR J2241-5236 do not specify an
  instrument-specific filter response.

Method
------
For every (Teff, log g) point in the FITS grid, the emergent surface flux
spectrum (F_lambda, erg/s/cm^2/Angstrom) is folded through each SDSS
filter's photon-counting response to produce one band-averaged F_nu
(erg/s/cm^2/Hz) value:

    <F_nu> = Integral[ F_nu(lambda) * T(lambda)/lambda dlambda ]
             / Integral[ T(lambda)/lambda dlambda ]

Those (temp, logg, flux) tables are written out and handed to Icarus's own
Atmo_photo_BTSettl7 loader, which applies an analytic (Neckel 2005-based)
limb-darkening law to build the full (logtemp, logg, mu) grid that Icarus'
Photometry class needs. The result is repackaged as an
Icarus.Atmosphere.Atmo.AtmoGridPhot object and written to HDF5.

The band's zero-point is fixed at the standard AB value (zp = -48.60, i.e.
flux in F_nu space directly gives m_AB = -2.5*log10(F_nu) - 48.60), and the
per-band extinction ratio (A_band/A_V) is computed with Icarus' own
Utils.Flux.Extinction (O'Donnell 1994, Rv=3.1) at each band's pivot
wavelength, so it is self-consistent with how Icarus.Photometry applies the
"av" fit parameter elsewhere.

IMPORTANT CAVEATS -- read before using for science
---------------------------------------------------
- ABSOLUTE CALIBRATION. The grid is in physical units, not relative ones.
  The CDBS spectra are true surface fluxes (verified: integral F_lam dlam
  reproduces sigma*T^4), and because Icarus expresses surface areas in
  units of orbital separation^2 while Star._Proj supplies (a/10pc)^2, the
  resulting magnitudes are ABSOLUTE magnitudes -- which is exactly why
  Photometry.Calc_chi2 can fit a distance modulus on top of them.
  The known systematic is the surface-flux -> specific-intensity
  conversion: Atmo_photo_BTSettl7 uses a hardcoded 4/pi^2 factor that is
  only exactly consistent with the limb-darkening law where the integral
  of LD(mu)*mu dmu equals pi/8, i.e. near 5000 A. The reconstructed
  surface flux is off by about -5% at 4300 A (g), +5% at 6200 A (r) and
  +9% at 7500 A (i) -- a smooth, band-dependent systematic of order
  0.05-0.1 mag in colour. For the mock/test use of this script that is
  harmless, since Calc_chi2(..., do_offset=True) fits a free per-band
  offset which absorbs it; for real absolute photometry it should be
  checked against a spectrophotometric standard such as Vega.
- Metallicity is fixed at [M/H]=+0.5 (the phoenixp05 grid); the true
  companion metallicity is unknown and not fit for.
- Limb darkening is Atmo_photo_BTSettl7's built-in analytic approximation,
  not derived from PHOENIX intensity spectra (this FITS grid only provides
  disk-integrated flux, not per-mu intensities).
- Only Teff in [TEMP_MIN, TEMP_MAX] is included (see below), chosen to
  bracket the Tnight/Tday values used in model_J2241-5236.py with margin.

>>> python build_atmo_grid_J2241-5236.py
"""

import os
import glob

import numpy as np
from astropy.io import fits

from Icarus.Atmosphere.Atmo_photo_BTSettl7 import Atmo_phot_BTSettl7
from Icarus.Atmosphere.Atmo import AtmoGridPhot
from Icarus.Utils.Flux import Extinction


##### Configuration
FITS_DIR = 'model_spec/grp/redcat/trds/grid/phoenix/phoenixp05'
FILTER_DIR = 'filters'
RAW_DIR = 'atmo_grid_raw_J2241-5236'          # intermediate (temp,logg,flux) tables
BANDS = ['g', 'r', 'i']
TEMP_MIN = 2000.                               # K
TEMP_MAX = 12500.                              # K -- brackets Tnight/Tday with margin
## The phoenixp05 CDBS grid only has real data for log g = 3.0-4.5 across the
## full temperature range; the g50/g55 columns are identically zero for every
## T > 2900 K. Including them would put log(0) = -inf planes in the grid, and
## since the modelled surface spans log g ~ 4.3-4.6 (straddling 4.5), roughly
## half the surface elements would interpolate against -inf and contribute
## exactly zero flux. Capping at 4.5 makes Icarus mildly extrapolate instead,
## which is far better behaved.
LOGG_LIMS = [3.0, 4.5]
C_ANGSTROM_PER_S = 2.99792458e18               # speed of light, Angstrom/s


def list_temps(fits_dir, temp_min, temp_max):
    fitfiles = glob.glob(os.path.join(fits_dir, '*_*.fits'))
    temps = []
    for fln in fitfiles:
        base = os.path.basename(fln)
        t = float(base.rsplit('_', 1)[1].replace('.fits', ''))
        if temp_min <= t <= temp_max:
            temps.append((t, fln))
    temps.sort()
    return temps


def load_filter(band):
    fln = os.path.join(FILTER_DIR, 'SDSS_{}.dat'.format(band))
    wav, resp = np.loadtxt(fln, unpack=True)
    return wav, resp


def pivot_wavelength(wav, resp):
    """ Photon-counting pivot wavelength (Tokunaga & Vacca 2005). """
    return np.sqrt(np.trapz(resp*wav, wav) / np.trapz(resp/wav, wav))


def effective_width(wav, resp):
    return np.trapz(resp, wav) / resp.max()


def band_flux_nu(wav_spec, flam_spec, wav_filt, resp_filt):
    """
    Photon-weighted band-averaged F_nu (erg/s/cm^2/Hz) given a F_lambda
    spectrum (erg/s/cm^2/Angstrom) on wav_spec (Angstrom) and a
    photon-counting filter response resp_filt on wav_filt (Angstrom).
    """
    wmin, wmax = wav_filt.min(), wav_filt.max()
    inband = (wav_spec >= wmin) & (wav_spec <= wmax)
    w = wav_spec[inband]
    resp = np.interp(w, wav_filt, resp_filt, left=0., right=0.)
    fnu = flam_spec[inband] * w**2 / C_ANGSTROM_PER_S
    num = np.trapz(fnu * resp / w, w)
    den = np.trapz(resp / w, w)
    return num / den


def build_raw_tables():
    """
    Reads the PHOENIX FITS grid and writes, for each band, a (temp, logg,
    flux) text table under RAW_DIR, suitable for Atmo_photo_BTSettl7.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    temps = list_temps(FITS_DIR, TEMP_MIN, TEMP_MAX)
    print("Found {} temperature points between {} and {} K.".format(len(temps), TEMP_MIN, TEMP_MAX))

    filters = {band: load_filter(band) for band in BANDS}
    for band, (wav_filt, resp_filt) in filters.items():
        print("{} band: pivot = {:.1f} A, effective width = {:.1f} A".format(
            band, pivot_wavelength(wav_filt, resp_filt), effective_width(wav_filt, resp_filt)))

    raw_rows = {band: [] for band in BANDS}
    for temp, fln in temps:
        with fits.open(fln) as f:
            data = f[1].data
            wav_spec = data['WAVELENGTH']
            logg_cols = [name for name in data.columns.names if name != 'WAVELENGTH']
            for col in logg_cols:
                logg = float(col[1:]) / 10.
                flam_spec = data[col]
                for band in BANDS:
                    wav_filt, resp_filt = filters[band]
                    flux = band_flux_nu(wav_spec, flam_spec, wav_filt, resp_filt)
                    raw_rows[band].append((temp, logg, flux))
        print("  processed {} K".format(temp))

    raw_flns = {}
    for band in BANDS:
        fln = os.path.join(RAW_DIR, 'raw_{}.txt'.format(band))
        rows = np.array(raw_rows[band])
        np.savetxt(fln, rows, fmt='%.6e', header='temp logg flux(F_nu, erg/s/cm^2/Hz)')
        raw_flns[band] = fln
        print("Wrote {} ({} rows).".format(fln, len(rows)))

    return raw_flns, filters


def build_hdf5_grids(raw_flns, filters):
    zp = -48.60  # standard AB zero-point (flux already in F_nu, erg/s/cm^2/Hz)
    for band in BANDS:
        wav_filt, resp_filt = filters[band]
        pivot_angstrom = pivot_wavelength(wav_filt, resp_filt)
        width_angstrom = effective_width(wav_filt, resp_filt)
        pivot_micron = pivot_angstrom * 1e-4
        ## NOTE: Atmo_phot_BTSettl7 expects `wav`/`dwav` in CENTIMETRES -- it
        ## internally does self.wav*1e4 to get microns for the limb-darkening
        ## law (Utils.Flux.Limb_darkening) and self.wav*1e8 to get Angstroms.
        ## Passing microns here instead would evaluate the Neckel (2005) law
        ## at ~4700 microns rather than ~0.47 micron, which silently returns a
        ## nearly wavelength-independent and far too weak limb darkening
        ## (centre-to-limb ratio ~0.75 instead of ~0.19 in g).
        pivot_cm = pivot_angstrom * 1e-8
        width_cm = width_angstrom * 1e-8
        ext = float(Extinction(np.array([pivot_micron]), Rv=3.1))

        atmo = Atmo_phot_BTSettl7(raw_flns[band], wav=pivot_cm, dwav=width_cm,
                                   zp=zp, ext=ext, logg_lims=LOGG_LIMS, AB=True)

        if not np.isfinite(atmo.grid).all():
            raise RuntimeError(
                "Band {}: {} non-finite values in the grid. This means the "
                "underlying model spectra were zero/missing for some "
                "(Teff, logg) combination -- check LOGG_LIMS against the "
                "coverage of the model grid.".format(
                    band, int((~np.isfinite(atmo.grid)).sum())))

        grid_phot = AtmoGridPhot(data=atmo.grid,
                                  cols=[('logtemp', atmo.logtemp), ('logg', atmo.logg), ('mu', atmo.mu)],
                                  meta={'zp': zp, 'ext': ext, 'filter': 'SDSS.{}'.format(band),
                                        'pivot': pivot_wavelength(wav_filt, resp_filt),
                                        'delta_w': effective_width(wav_filt, resp_filt),
                                        'units': 'log(F_nu), erg/s/cm^2/Hz', 'magsys': 'AB',
                                        'source': 'PHOENIX/BT-Settl phoenixp05 (CDBS) + SVO SLOAN/SDSS.{}'.format(band)})

        h5_fln = 'atmo_grid_J2241-5236_{}.h5'.format(band)
        grid_phot.WriteHDF5(h5_fln)
        print("Wrote {} (ext={:.3f}, pivot={:.1f} A).".format(h5_fln, ext, pivot_wavelength(wav_filt, resp_filt)))


def write_index_file():
    fln = 'atmo_models_J2241-5236.txt'
    with open(fln, 'w') as f:
        f.write("# Index file for Icarus.Photometry.Photometry -- lists one line per band.\n")
        f.write("# Format: band  filename\n")
        f.write("#\n")
        f.write("# Built by build_atmo_grid_J2241-5236.py from the PHOENIX/BT-Settl\n")
        f.write("# phoenixp05 grid and SVO SDSS g/r/i filter curves. See that script's\n")
        f.write("# docstring for the method used and important caveats (metallicity fixed\n")
        f.write("# at [M/H]=+0.5; grid is in absolute-magnitude units, with a\n")
        f.write("# wavelength-dependent ~5-10% intensity-normalization systematic).\n")
        for band in BANDS:
            f.write("{}  atmo_grid_J2241-5236_{}.h5\n".format(band, band))
    print("Wrote {}.".format(fln))


if __name__ == '__main__':
    raw_flns, filters = build_raw_tables()
    build_hdf5_grids(raw_flns, filters)
    write_index_file()
