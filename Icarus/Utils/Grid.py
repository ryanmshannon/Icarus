# Licensed under a 3-clause BSD style license - see LICENSE

import os

try:
    from scipy import weave 
except:
    try:
        import weave
    except:
        print('weave cannot be import from scipy nor on its own.')

from .import_modules import *

logger = logging.getLogger(__name__)


##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##
## Grid utilities
## Contain functions that pertain to the "atmosphere grid-
## related" purposes such as various kinds of interpolation
## in order to extract fluxes.
##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##


def _Trilinear(grid, jx, jy, jz, wx, wy, wz):
    """
    Shared vectorized trilinear interpolation helper used by several
    Interp_* functions below, replacing what used to be repeated
    weave.inline C blocks (scipy.weave is no longer available).

    grid: ndarray with the interpolated axes as its first three dimensions
        (may have further dimensions, e.g. wavelength, which are broadcast
        over).
    jx, jy, jz: ndarray of lower-bound integer indices along each axis.
    wx, wy, wz: ndarray of fractional weights along each axis (weight of
        the upper-bound point).
    """
    w1x, w0x = wx, 1.-wx
    j0x, j1x = jx, jx+1
    w1y, w0y = wy, 1.-wy
    j0y, j1y = jy, jy+1
    w1z, w0z = wz, 1.-wz
    j0z, j1z = jz, jz+1
    return w1z*(w0y*(w0x*grid[j0x,j0y,j1z] + w1x*grid[j1x,j0y,j1z])
               + w1y*(w0x*grid[j0x,j1y,j1z] + w1x*grid[j1x,j1y,j1z])) \
         + w0z*(w0y*(w0x*grid[j0x,j0y,j0z] + w1x*grid[j1x,j0y,j0z])
               + w1y*(w0x*grid[j0x,j1y,j0z] + w1x*grid[j1x,j1y,j0z]))

def _Trilinear_at_wav(grid, jx, jy, jz, wx, wy, wz, jw):
    """
    Same as _Trilinear, but also selects along a 4th (wavelength) axis at
    integer index array jw (exact, not interpolated -- callers blend two
    calls at neighbouring wavelength indices themselves). jw may carry
    extra broadcast dimensions beyond jx/jy/jz (e.g. jw of shape
    (nsurf,nwav) vs jx of shape (nsurf,1), broadcasting to (nsurf,nwav)).
    """
    w1x, w0x = wx, 1.-wx
    j0x, j1x = jx, jx+1
    w1y, w0y = wy, 1.-wy
    j0y, j1y = jy, jy+1
    w1z, w0z = wz, 1.-wz
    j0z, j1z = jz, jz+1
    return w1z*(w0y*(w0x*grid[j0x,j0y,j1z,jw] + w1x*grid[j1x,j0y,j1z,jw])
               + w1y*(w0x*grid[j0x,j1y,j1z,jw] + w1x*grid[j1x,j1y,j1z,jw])) \
         + w0z*(w0y*(w0x*grid[j0x,j0y,j0z,jw] + w1x*grid[j1x,j0y,j0z,jw])
               + w1y*(w0x*grid[j0x,j1y,j0z,jw] + w1x*grid[j1x,j1y,j0z,jw]))

def Interp_3Dgrid(grid, wx, wy, wz, jx, jy, jz):
    """
    """
    grid = np.ascontiguousarray(grid, dtype=float)
    wx = np.ascontiguousarray(wx, dtype=float)
    wy = np.ascontiguousarray(wy, dtype=float)
    wz = np.ascontiguousarray(wz, dtype=float)
    jx = np.ascontiguousarray(jx, dtype=int)
    jy = np.ascontiguousarray(jy, dtype=int)
    jz = np.ascontiguousarray(jz, dtype=int)
    ## Pure-NumPy replacement for the weave.inline block above (scipy.weave
    ## is no longer available).
    return _Trilinear(grid, jx, jy, jz, wx, wy, wz)

def Interp_photometry(grid, wteff, wlogg, wmu, jteff, jlogg, jmu, area, val_mu):
    """
    Simple interpolation of an atmosphere grid having axes (logtemp, logg, mu).

    The interpolation takes a set of points to be interpolated and summed
    together.

    Parameters
    ----------
    grid : ndarray
        Atmosphere grid, with dimensions (logtemp, logg, mu, wav).
    wteff, wlogg, wmu : ndarray
        Weights of the temperature, logg, mu.
    jteff, jlogg, jmu : ndarray
        Fractional position of the temperature, logg, mu.
    area : ndarray
        Area (i.e. weight) of each surface element for the summation.
    val_mu : ndarray
        Value of the cross-section visible to us.

    Returns
    -------
    flux : scalar
        Flux integrated over the surface.
    """
    code = """
    double fl = 0.;
    #pragma omp parallel shared(grid,wteff,wlogg,wmu,jteff,jlogg,jmu,area,val_mu,nsurf,fl) default(none)
    {
    double w1teff, w0teff, w1logg, w0logg, w1mu, w0mu, tmp_fl;
    int j0teff, j1teff, j0logg, j1logg, j0mu, j1mu;
    #pragma omp for reduction(+:fl)
    for (int i=0; i<nsurf; i++) {
        w1teff = wteff(i);
        w0teff = 1.-w1teff;
        j0teff = jteff(i);
        j1teff = 1.+j0teff;
        w1logg = wlogg(i);
        w0logg = 1.-w1logg;
        j0logg = jlogg(i);
        j1logg = 1.+j0logg;
        w1mu = wmu(i);
        w0mu = 1.-w1mu;
        j0mu = jmu(i);
        j1mu = 1.+j0mu;
        tmp_fl = w1mu*(w0logg*(w0teff*grid(j0teff,j0logg,j1mu) + w1teff*grid(j1teff,j0logg,j1mu)) \
                      + w1logg*(w0teff*grid(j0teff,j1logg,j1mu) + w1teff*grid(j1teff,j1logg,j1mu))) \
                + w0mu*(w0logg*(w0teff*grid(j0teff,j0logg,j0mu) + w1teff*grid(j1teff,j0logg,j0mu)) \
                      + w1logg*(w0teff*grid(j0teff,j1logg,j0mu) + w1teff*grid(j1teff,j1logg,j0mu)));
        fl = fl + exp(tmp_fl) * area(i) * val_mu(i);
    }
    }
    return_val = fl;
    """
    grid = np.ascontiguousarray(grid, dtype=float)
    wteff = np.ascontiguousarray(wteff, dtype=float)
    wlogg = np.ascontiguousarray(wlogg, dtype=float)
    wmu = np.ascontiguousarray(wmu, dtype=float)
    jteff = np.ascontiguousarray(jteff, dtype=int)
    jlogg = np.ascontiguousarray(jlogg, dtype=int)
    jmu = np.ascontiguousarray(jmu, dtype=int)
    area = np.ascontiguousarray(area, dtype=float)
    val_mu = np.ascontiguousarray(val_mu, dtype=float)
    ## Pure-NumPy replacement for the weave.inline block above (scipy.weave
    ## is no longer available).
    tmp_fl = _Trilinear(grid, jteff, jlogg, jmu, wteff, wlogg, wmu)
    fl = float(np.sum(np.exp(tmp_fl) * area * val_mu))
    return fl

def Interp_photometry_doppler(grid, wteff, wlogg, wmu, jteff, jlogg, jmu, area, val_mu, val_vel, grid_doppler):
    """
    Simple interpolation of an atmosphere grid having axes (logtemp, logg, mu),
    which also takes into account Doppler boosting using coefficients stored in
    a dedicated grid.

    Parameters
    ----------
    The interpolation takes a set of points to be interpolated and summed together.
    grid : ndarray
        Atmosphere grid, with dimensions (logtemp, logg, mu, wav).
    wteff, wlogg, wmu : ndarray
        Weights of the temperature, logg, mu.
    jteff, jlogg, jmu : ndarray
        Fractional position of the temperature, logg, mu.
    area : ndarray
        Area (i.e. weight) of each surface element for the summation.
    val_mu : ndarray
        Value of the cross-section visible to us.
    val_vel : ndarray
        Value of the velocity, in v/c units.
    grid_doppler : ndarray
        Doppler boosting coefficients, with dimensions similar to grid.

    Returns
    -------
    flux : scalar
        Flux integrated over the surface, with Doppler boosting.
    """
    grid = np.ascontiguousarray(grid, dtype=float)
    wteff = np.ascontiguousarray(wteff, dtype=float)
    wlogg = np.ascontiguousarray(wlogg, dtype=float)
    wmu = np.ascontiguousarray(wmu, dtype=float)
    jteff = np.ascontiguousarray(jteff, dtype=int)
    jlogg = np.ascontiguousarray(jlogg, dtype=int)
    jmu = np.ascontiguousarray(jmu, dtype=int)
    area = np.ascontiguousarray(area, dtype=float)
    val_mu = np.ascontiguousarray(val_mu, dtype=float)
    grid_doppler = np.ascontiguousarray(grid_doppler, dtype=float)
    val_vel = np.ascontiguousarray(val_vel, dtype=float)
    ## Pure-NumPy replacement for the weave.inline block above (scipy.weave
    ## is no longer available).
    tmp_fl = _Trilinear(grid, jteff, jlogg, jmu, wteff, wlogg, wmu)
    tmp_doppler = _Trilinear(grid_doppler, jteff, jlogg, jmu, wteff, wlogg, wmu)
    fl = float(np.sum(np.exp(tmp_fl) * area * val_mu * (1 + tmp_doppler * val_vel)))
    return fl

def Interp_photometry_doppler_nosum(grid, wteff, wlogg, wmu, jteff, jlogg, jmu, area, val_mu, val_vel, grid_doppler):
    """
    Simple interpolation of an atmosphere grid having axes (logtemp, logg, mu),
    which also takes into account Doppler boosting using coefficients stored in
    a dedicated grid.

    Note: As opposed to Interp_photometry_doppler, this function does not sum
    the surface elements.

    Parameters
    ----------
    The interpolation takes a set of points to be interpolated.
    grid : ndarray
        Atmosphere grid, with dimensions (logtemp, logg, mu, wav).
    wteff, wlogg, wmu : ndarray
        Weights of the temperature, logg, mu.
    jteff, jlogg, jmu : ndarray
        Fractional position of the temperature, logg, mu.
    area : ndarray
        Area (i.e. weight) of each surface element for the summation.
    val_mu : ndarray
        Value of the cross-section visible to us.
    val_vel : ndarray
        Value of the velocity, in v/c units.
    grid_doppler : ndarray
        Doppler boosting coefficients, with dimensions similar to grid.

    Returns
    -------
    flux : ndarray
        Flux _not_ integrated over the surface.
    """
    grid = np.ascontiguousarray(grid, dtype=float)
    wteff = np.ascontiguousarray(wteff, dtype=float)
    wlogg = np.ascontiguousarray(wlogg, dtype=float)
    wmu = np.ascontiguousarray(wmu, dtype=float)
    jteff = np.ascontiguousarray(jteff, dtype=int)
    jlogg = np.ascontiguousarray(jlogg, dtype=int)
    jmu = np.ascontiguousarray(jmu, dtype=int)
    area = np.ascontiguousarray(area, dtype=float)
    val_mu = np.ascontiguousarray(val_mu, dtype=float)
    grid_doppler = np.ascontiguousarray(grid_doppler, dtype=float)
    val_vel = np.ascontiguousarray(val_vel, dtype=float)
    ## Pure-NumPy replacement for the weave.inline block above (scipy.weave
    ## is no longer available).
    tmp_fl = _Trilinear(grid, jteff, jlogg, jmu, wteff, wlogg, wmu)
    tmp_doppler = _Trilinear(grid_doppler, jteff, jlogg, jmu, wteff, wlogg, wmu)
    fl = np.exp(tmp_fl) * area * val_mu * (1 + tmp_doppler * val_vel)
    return fl

def Interp_photometry_details(grid, wteff, wlogg, wmu, jteff, jlogg, jmu, area, val_mu, v, val_teff):
    """
    Simple interpolation of an atmosphere grid having axes (logtemp, logg, mu),
    which also takes into account Doppler boosting using coefficients stored in
    a dedicated grid.

    Note: Some additional quantities are calculated, such as the flux-weighted
    velocity, temperature and vsini.

    Parameters
    ----------
    The interpolation takes a set of points to be interpolated and summed together.
    grid : ndarray
        Atmosphere grid, with dimensions (logtemp, logg, mu, wav).
    wteff, wlogg, wmu : ndarray
        Weights of the temperature, logg, mu.
    jteff, jlogg, jmu : ndarray
        Fractional position of the temperature, logg, mu.
    area : ndarray
        Area (i.e. weight) of each surface element for the summation.
    val_mu : ndarray
        Value of the cross-section visible to us.
    val_vel : ndarray
        Value of the velocity, in v/c units.
    val_teff : ndarray
        Value of the temperatures.

    Returns
    -------
    flux : scalar
        Flux integrated over the surface.
    Keff : scalar
        Flux-weighted radial velocity.
    vsini : scalar
        Estimated vsini.
    Teff : scalar
        Flux-weighted temperature.
    """
    grid = np.ascontiguousarray(grid, dtype=float)
    wteff = np.ascontiguousarray(wteff, dtype=float)
    wlogg = np.ascontiguousarray(wlogg, dtype=float)
    wmu = np.ascontiguousarray(wmu, dtype=float)
    jteff = np.ascontiguousarray(jteff, dtype=int)
    jlogg = np.ascontiguousarray(jlogg, dtype=int)
    jmu = np.ascontiguousarray(jmu, dtype=int)
    area = np.ascontiguousarray(area, dtype=float)
    val_mu = np.ascontiguousarray(val_mu, dtype=float)
    v = np.ascontiguousarray(v, dtype=float)
    val_teff = np.ascontiguousarray(val_teff, dtype=float)
    ## Pure-NumPy replacement for the weave.inline block above (scipy.weave
    ## is no longer available).
    tmp_fl = np.exp(_Trilinear(grid, jteff, jlogg, jmu, wteff, wlogg, wmu)) * area * val_mu
    fl = float(np.sum(tmp_fl))
    Keff = float(np.sum(v * tmp_fl)) / fl
    KeffSquare = float(np.sum(v*v * tmp_fl))
    Teff = float(np.sum(np.exp(val_teff) * tmp_fl)) / fl
    vsini = float(np.sqrt((KeffSquare/fl) - Keff*Keff))
    return fl, Keff, vsini, Teff

def Interp_photometry_Keff(grid, wteff, wlogg, wmu, jteff, jlogg, jmu, area, val_mu, v):
    """
    Simple interpolation of an atmosphere grid having axes (logtemp, logg, mu),
    which also takes into account Doppler boosting using coefficients stored in
    a dedicated grid.

    Note: The flux-weighted velocity is also returned.

    Parameters
    ----------
    The interpolation takes a set of points to be interpolated and summed together.
    grid : ndarray
        Atmosphere grid, with dimensions (logtemp, logg, mu, wav).
    wteff, wlogg, wmu : ndarray
        Weights of the temperature, logg, mu.
    jteff, jlogg, jmu : ndarray
        Fractional position of the temperature, logg, mu.
    area : ndarray
        Area (i.e. weight) of each surface element for the summation.
    val_mu : ndarray
        Value of the cross-section visible to us.
    val_vel : ndarray
        Value of the velocity, in v/c units.

    Returns
    -------
    flux : scalar
        Flux integrated over the surface.
    Keff : scalar
        Flux-weighted radial velocity.
    """
    grid = np.ascontiguousarray(grid, dtype=float)
    wteff = np.ascontiguousarray(wteff, dtype=float)
    wlogg = np.ascontiguousarray(wlogg, dtype=float)
    wmu = np.ascontiguousarray(wmu, dtype=float)
    jteff = np.ascontiguousarray(jteff, dtype=int)
    jlogg = np.ascontiguousarray(jlogg, dtype=int)
    jmu = np.ascontiguousarray(jmu, dtype=int)
    area = np.ascontiguousarray(area, dtype=float)
    val_mu = np.ascontiguousarray(val_mu, dtype=float)
    v = np.ascontiguousarray(v, dtype=float)
    ## Pure-NumPy replacement for the weave.inline block above (scipy.weave
    ## is no longer available).
    tmp_fl = np.exp(_Trilinear(grid, jteff, jlogg, jmu, wteff, wlogg, wmu)) * area * val_mu
    fl = float(np.sum(tmp_fl))
    Keff = float(np.sum(v * tmp_fl)) / fl
    return fl, Keff

def Interp_photometry_nosum(grid, wteff, wlogg, wmu, jteff, jlogg, jmu, area, val_mu):
    """
    Simple interpolation of an atmosphere grid having axes (logtemp, logg, mu).

    Note: As opposed to Interp_photometry, this function does not sum
    the surface elements.

    Parameters
    ----------
    The interpolation takes a set of points to be interpolated.
    grid : ndarray
        Atmosphere grid, with dimensions (logtemp, logg, mu, wav).
    wteff, wlogg, wmu : ndarray
        Weights of the temperature, logg, mu.
    jteff, jlogg, jmu : ndarray
        Fractional position of the temperature, logg, mu.
    area : ndarray
        Area (i.e. weight) of each surface element for the summation.
    val_mu : ndarray
        Value of the cross-section visible to us.

    Returns
    -------
    flux : ndarray
        Flux _not_ integrated over the surface.
    """
    grid = np.ascontiguousarray(grid, dtype=float)
    wteff = np.ascontiguousarray(wteff, dtype=float)
    wlogg = np.ascontiguousarray(wlogg, dtype=float)
    wmu = np.ascontiguousarray(wmu, dtype=float)
    jteff = np.ascontiguousarray(jteff, dtype=int)
    jlogg = np.ascontiguousarray(jlogg, dtype=int)
    jmu = np.ascontiguousarray(jmu, dtype=int)
    area = np.ascontiguousarray(area, dtype=float)
    val_mu = np.ascontiguousarray(val_mu, dtype=float)
    ## Pure-NumPy replacement for the weave.inline block above (scipy.weave
    ## is no longer available).
    fl = np.exp(_Trilinear(grid, jteff, jlogg, jmu, wteff, wlogg, wmu)) * area * val_mu
    return fl

def Interp_doppler(grid, wteff, wlogg, wmu, wwav, jteff, jlogg, jmu, jwav, area, val_mu):
    """
    Simple interpolation of an atmosphere grid having axes (logtemp, logg, mu, wav).

    This grid interpolation is made for a grid which is linear in the velocity
    or redshift space, e.g. log lambda.

    Note: Because of the Doppler shift, the interpolation on the wavelength
        will necessarily go out of bound, on the lower or upper range. We
        assume that the atmosphere grid has a broader spectral coverage than
        the data.

    Parameters
    ----------
    The interpolation takes a set of points to be interpolated.
    grid : ndarray
        Atmosphere grid, with dimensions (logtemp, logg, mu, wav).
    wteff, wlogg, wmu, wwav : ndarray
        Weights of the temperature, logg, mu, wav.
    jteff, jlogg, jmu, jwav : ndarray
        Fractional position of the temperature, logg, mu, wav.
    area : ndarray
        Area (i.e. weight) of each surface element for the summation.
    val_mu : ndarray
        Value of the cross-section visible to us.

    Returns
    -------
    spectrum : ndarray
        Spectrum integrated over the surface.
    """
    logger.log(9, "start")
    grid = np.ascontiguousarray(grid, dtype=float)
    wteff = np.ascontiguousarray(wteff, dtype=float)
    wlogg = np.ascontiguousarray(wlogg, dtype=float)
    wmu = np.ascontiguousarray(wmu, dtype=float)
    jteff = np.ascontiguousarray(jteff, dtype=int)
    jlogg = np.ascontiguousarray(jlogg, dtype=int)
    jmu = np.ascontiguousarray(jmu, dtype=int)
    area = np.ascontiguousarray(area, dtype=float)
    val_mu = np.ascontiguousarray(val_mu, dtype=float)
    wwav = np.ascontiguousarray(wwav, dtype=float)
    jwav = np.ascontiguousarray(jwav, dtype=int)
    nsurf = jteff.size
    nwav = grid.shape[-1]
    fl = np.zeros(nwav, dtype=float)
    ## Pure-NumPy replacement for the weave.inline block above (scipy.weave
    ## is no longer available). For each surface element, the local
    ## spectrum is quadrilinearly interpolated (via _Trilinear_at_wav, at
    ## the two neighbouring wavelength indices j0wavk/j1wavk that its
    ## Doppler shift maps each output pixel k to) and blended, then
    ## accumulated onto the output wavelength grid. Processed in chunks of
    ## surface elements to bound the (nsurf_chunk x nwav) intermediate
    ## arrays' memory footprint.
    kk = np.arange(nwav)
    chunk = max(1, 20_000_000 // max(nwav,1))
    for start in range(0, nsurf, chunk):
        sl = slice(start, start+chunk)
        jteff_c = jteff[sl,None]; wteff_c = wteff[sl,None]
        jlogg_c = jlogg[sl,None]; wlogg_c = wlogg[sl,None]
        jmu_c = jmu[sl,None]; wmu_c = wmu[sl,None]
        w1wav_c = wwav[sl,None]; w0wav_c = 1.-w1wav_c
        j0wavk = jwav[sl,None] + kk[None,:]
        j1wavk = j0wavk+1
        mask_low = j0wavk < 0
        mask_high = j1wavk >= nwav
        j0wavk = np.where(mask_low, 0, np.where(mask_high, nwav-1, j0wavk))
        j1wavk = np.where(mask_low, 0, np.where(mask_high, nwav-1, j1wavk))
        val0 = _Trilinear_at_wav(grid, jteff_c, jlogg_c, jmu_c, wteff_c, wlogg_c, wmu_c, j0wavk)
        val1 = _Trilinear_at_wav(grid, jteff_c, jlogg_c, jmu_c, wteff_c, wlogg_c, wmu_c, j1wavk)
        tmp_fl = w0wav_c*val0 + w1wav_c*val1
        contrib = np.exp(tmp_fl) * (area[sl]*val_mu[sl])[:,None]
        fl += contrib.sum(axis=0)
    logger.log(9, "end")
    return fl

def Interp_doppler_savememory(grid, wteff, wlogg, wmu, wwav, jteff, jlogg, jmu, jwav, mu_grid, area, val_mu):
    """
    Simple interpolation of an atmosphere grid having axes (logtemp, logg, wav).

    This grid interpolation is made for a grid which is linear in the velocity
    or redshift space, e.g. log lambda.

    The limb darkening is implement by sourcing values from an external grid
    containing limb darkening coefficients.

    Note: Because of the Doppler shift, the interpolation on the wavelength
        will necessarily go out of bound, on the lower or upper range. We
        assume that the atmosphere grid has a broader spectral coverage than
        the data.

    Parameters
    ----------
    The interpolation takes a set of points to be interpolated.
    grid : ndarray
        Atmosphere grid, with dimensions (logtemp, logg, mu, wav).
    wteff, wlogg, wmu, wwav : ndarray
        Weights of the temperature, logg, mu, wav.
    jteff, jlogg, jmu, jwav : ndarray
        Fractional position of the temperature, logg, mu, wav.
    area : ndarray
        Area (i.e. weight) of each surface element for the summation.
    val_mu : ndarray
        Value of the cross-section visible to us.
    mu_grid : ndarray
        Grid of limb darkening having axes (mu, wav).

    Returns
    -------
    spectrum : ndarray
        Spectrum integrated over the surface.

    NOTE: This is becoming obsolete.
    """
    logger.log(9, "start")
    grid = np.ascontiguousarray(grid, dtype=float)
    wteff = np.ascontiguousarray(wteff, dtype=float)
    wlogg = np.ascontiguousarray(wlogg, dtype=float)
    wmu = np.ascontiguousarray(wmu, dtype=float)
    jteff = np.ascontiguousarray(jteff, dtype=int)
    jlogg = np.ascontiguousarray(jlogg, dtype=int)
    jmu = np.ascontiguousarray(jmu, dtype=int)
    area = np.ascontiguousarray(area, dtype=float)
    val_mu = np.ascontiguousarray(val_mu, dtype=float)
    wwav = np.ascontiguousarray(wwav, dtype=float)
    jwav = np.ascontiguousarray(jwav, dtype=int)
    mu_grid = np.ascontiguousarray(mu_grid, dtype=float)
    nsurf = jteff.size
    nwav = grid.shape[-1]
    fl = np.zeros(nwav, dtype=float)
    ## Pure-NumPy replacement for the weave.inline block above (scipy.weave
    ## is no longer available). Same wavelength-shift-and-blend scheme as
    ## Interp_doppler, but here `grid` only has (logtemp, logg, wav) axes
    ## (bilinear, not trilinear) and the limb darkening is a separate
    ## (mu, wav) table applied multiplicatively outside the exp(). Chunked
    ## over surface elements to bound memory.
    kk = np.arange(nwav)
    chunk = max(1, 20_000_000 // max(nwav,1))
    for start in range(0, nsurf, chunk):
        sl = slice(start, start+chunk)
        j0teff = jteff[sl,None]; j1teff = j0teff+1
        w1teff = wteff[sl,None]; w0teff = 1.-w1teff
        j0logg = jlogg[sl,None]; j1logg = j0logg+1
        w1logg = wlogg[sl,None]; w0logg = 1.-w1logg
        j0mu = jmu[sl,None]; j1mu = j0mu+1
        w1mu = wmu[sl,None]; w0mu = 1.-w1mu
        w1wav = wwav[sl,None]; w0wav = 1.-w1wav
        j0wavk = jwav[sl,None] + kk[None,:]
        j1wavk = j0wavk+1
        mask_low = j0wavk < 0
        mask_high = j1wavk >= nwav
        j0wavk = np.where(mask_low, 0, np.where(mask_high, nwav-1, j0wavk))
        j1wavk = np.where(mask_low, 0, np.where(mask_high, nwav-1, j1wavk))
        val0 = np.exp(w0logg*(w0teff*grid[j0teff,j0logg,j0wavk] + w1teff*grid[j1teff,j0logg,j0wavk])
                     + w1logg*(w0teff*grid[j0teff,j1logg,j0wavk] + w1teff*grid[j1teff,j1logg,j0wavk]))
        val1 = np.exp(w0logg*(w0teff*grid[j0teff,j0logg,j1wavk] + w1teff*grid[j1teff,j0logg,j1wavk])
                     + w1logg*(w0teff*grid[j0teff,j1logg,j1wavk] + w1teff*grid[j1teff,j1logg,j1wavk]))
        tmp_fl = w0wav*(w0mu*mu_grid[j0mu,j0wavk] + w1mu*mu_grid[j1mu,j0wavk])*val0 \
               + w1wav*(w0mu*mu_grid[j0mu,j1wavk] + w1mu*mu_grid[j1mu,j1wavk])*val1
        contrib = tmp_fl * (area[sl]*val_mu[sl])[:,None]
        fl += contrib.sum(axis=0)
    logger.log(9, "end")
    return fl

def Interp_doppler_savememory_linear(grid, wteff, wlogg, wmu, jteff, jlogg, jmu, mu_grid, area, val_mu, val_vel, z0):
    """
    Simple interpolation of an atmosphere grid having axes (logtemp, logg, wav).

    This grid interpolation is made for a grid which is linear in lambda space.

    The limb darkening is implement by sourcing values from an external grid
    containing limb darkening coefficients.

    Note: Because of the Doppler shift, the interpolation on the wavelength
        will necessarily go out of bound, on the lower or upper range. We
        assume that the atmosphere grid has a broader spectral coverage than
        the data.

    Parameters
    ----------
    The interpolation takes a set of points to be interpolated.
    grid : ndarray
        Atmosphere grid, with dimensions (logtemp, logg, mu, wav).
    wteff, wlogg, wmu, wwav : ndarray
        Weights of the temperature, logg, mu, wav.
    jteff, jlogg, jmu, jwav : ndarray
        Fractional position of the temperature, logg, mu, wav.
    area : ndarray
        Area (i.e. weight) of each surface element for the summation.
    val_mu : ndarray
        Value of the cross-section visible to us.
    mu_grid : ndarray
        Grid of limb darkening having axes (mu, wav).
    z0: delta_lambda / lambda0 of the grid.
        Interpolated lambda bin is: k' = (z+1)*k + z/z0
        Derivation: z+1 = lambda'/lambda
                        = (k'*delta_lambda+lambda0) / (k*delta_lambda+lambda0)
            ... which is solved for n'.

    Returns
    -------
    spectrum : ndarray
        Spectrum integrated over the surface.

    NOTE: This is becoming obsolete.
    """
    grid = np.ascontiguousarray(grid, dtype=float)
    wteff = np.ascontiguousarray(wteff, dtype=float)
    wlogg = np.ascontiguousarray(wlogg, dtype=float)
    wmu = np.ascontiguousarray(wmu, dtype=float)
    jteff = np.ascontiguousarray(jteff, dtype=int)
    jlogg = np.ascontiguousarray(jlogg, dtype=int)
    jmu = np.ascontiguousarray(jmu, dtype=int)
    area = np.ascontiguousarray(area, dtype=float)
    val_mu = np.ascontiguousarray(val_mu, dtype=float)
    val_vel = np.ascontiguousarray(val_vel, dtype=float)
    mu_grid = np.ascontiguousarray(mu_grid, dtype=float)
    nsurf = jteff.size
    nwav = grid.shape[-1]
    fl = np.zeros(nwav, dtype=float)
    ## Pure-NumPy replacement for the weave.inline block above (scipy.weave
    ## is no longer available). Same structure as Interp_doppler_savememory,
    ## but the Doppler shift is computed continuously per (surface element,
    ## output pixel) via relativistic velocity/z0 instead of a precomputed
    ## per-surface-element integer shift. Chunked over surface elements to
    ## bound memory. (Note: the original weave code also referenced
    ## `wwav`/`jwav`, which are not among this function's parameters --
    ## dead/broken leftovers from another variant, dropped here.)
    kk = np.arange(nwav)
    zplusone = np.sqrt((1.+val_vel)/(1.-val_vel))
    chunk = max(1, 20_000_000 // max(nwav,1))
    for start in range(0, nsurf, chunk):
        sl = slice(start, start+chunk)
        j0teff = jteff[sl,None]; j1teff = j0teff+1
        w1teff = wteff[sl,None]; w0teff = 1.-w1teff
        j0logg = jlogg[sl,None]; j1logg = j0logg+1
        w1logg = wlogg[sl,None]; w0logg = 1.-w1logg
        j0mu = jmu[sl,None]; j1mu = j0mu+1
        w1mu = wmu[sl,None]; w0mu = 1.-w1mu

        kprime = zplusone[sl,None]*kk[None,:] + (zplusone[sl,None]-1.)/z0
        mask_high = kprime >= nwav
        mask_low = kprime < 0
        j0k = np.where(mask_high, nwav-1, np.where(mask_low, 0, kprime.astype(int)))
        ## Clip defensively: kprime in [nwav-1, nwav) lands in the "else"
        ## branch with j0k = nwav-1, so j1k = j0k+1 would be nwav (an
        ## out-of-bounds index) -- an edge case present in the original
        ## C code too (relying on the fact that its weight w0k there is
        ## near 1, making the out-of-bounds w1k contribution negligible;
        ## clipped here since NumPy indexing does not tolerate it).
        j1k = np.clip(np.where(mask_high, nwav-1, np.where(mask_low, 0, j0k+1)), 0, nwav-1)
        w1k = np.where(mask_high | mask_low, 1., kprime - j0k)
        w0k = 1. - w1k

        val0 = w0logg*(w0teff*grid[j0teff,j0logg,j0k] + w1teff*grid[j1teff,j0logg,j0k]) \
             + w1logg*(w0teff*grid[j0teff,j1logg,j0k] + w1teff*grid[j1teff,j1logg,j0k])
        val1 = w0logg*(w0teff*grid[j0teff,j0logg,j1k] + w1teff*grid[j1teff,j0logg,j1k]) \
             + w1logg*(w0teff*grid[j0teff,j1logg,j1k] + w1teff*grid[j1teff,j1logg,j1k])
        tmp_fl = (w1mu*mu_grid[j1mu,j0k] + w0mu*mu_grid[j0mu,j0k]) * (w0k*val0) \
               + (w1mu*mu_grid[j1mu,j1k] + w0mu*mu_grid[j0mu,j1k]) * (w1k*val1)
        contrib = tmp_fl * (area[sl]*val_mu[sl])[:,None]
        fl += contrib.sum(axis=0)
    return fl

def Interp_doppler_nomu(grid, wteff, wlogg, wwav, jteff, jlogg, jwav, area, val_mu):
    """
    Simple interpolation of an atmosphere grid having axes (logtemp, logg, wav).

    This grid interpolation is made for a grid which is linear in the velocity
    or redshift space, e.g. log lambda.

    Note: Because of the Doppler shift, the interpolation on the wavelength
        will necessarily go out of bound, on the lower or upper range. We
        assume that the atmosphere grid has a broader spectral coverage than
        the data.

    Parameters
    ----------
    The interpolation takes a set of points to be interpolated.
    grid : ndarray
        Atmosphere grid, with dimensions (logtemp, logg, mu, wav).
    wteff, wlogg, wmu, wwav : ndarray
        Weights of the temperature, logg, mu, wav.
    jteff, jlogg, jmu, jwav : ndarray
        Fractional position of the temperature, logg, mu, wav.
    area : ndarray
        Area (i.e. weight) of each surface element for the summation.
    val_mu : ndarray
        Value of the cross-section visible to us.
    mu_grid : ndarray
        Grid of limb darkening having axes (mu, wav).
    z0: delta_lambda / lambda0 of the grid.
        Interpolated lambda bin is: k' = (z+1)*k + z/z0
        Derivation: z+1 = lambda'/lambda
                        = (k'*delta_lambda+lambda0) / (k*delta_lambda+lambda0)
            ... which is solved for n'.

    Returns
    -------
    spectrum : ndarray
        Spectrum integrated over the surface.
    """
    grid = np.ascontiguousarray(grid, dtype=float)
    wteff = np.ascontiguousarray(wteff, dtype=float)
    wlogg = np.ascontiguousarray(wlogg, dtype=float)
    jteff = np.ascontiguousarray(jteff, dtype=int)
    jlogg = np.ascontiguousarray(jlogg, dtype=int)
    area = np.ascontiguousarray(area, dtype=float)
    val_mu = np.ascontiguousarray(val_mu, dtype=float)
    wwav = np.ascontiguousarray(wwav, dtype=float)
    jwav = np.ascontiguousarray(jwav, dtype=int)
    nsurf = jteff.size
    nwav = grid.shape[-1]
    fl = np.zeros(nwav, dtype=float)
    ## Pure-NumPy replacement for the weave.inline block above (scipy.weave
    ## is no longer available). Same wavelength-shift-and-blend scheme as
    ## Interp_doppler, but bilinear over (logtemp, logg) only, with no mu
    ## dependence and no exp() (linear, not log-flux). Chunked over surface
    ## elements to bound memory. (Note: the original weave code also
    ## referenced `wmu`/`jmu`, which are not among this function's
    ## parameters -- dead/broken leftovers from another variant, dropped
    ## here.)
    kk = np.arange(nwav)
    chunk = max(1, 20_000_000 // max(nwav,1))
    for start in range(0, nsurf, chunk):
        sl = slice(start, start+chunk)
        j0teff = jteff[sl,None]; j1teff = j0teff+1
        w1teff = wteff[sl,None]; w0teff = 1.-w1teff
        j0logg = jlogg[sl,None]; j1logg = j0logg+1
        w1logg = wlogg[sl,None]; w0logg = 1.-w1logg
        w1wav = wwav[sl,None]; w0wav = 1.-w1wav
        j0wavk = jwav[sl,None] + kk[None,:]
        j1wavk = j0wavk+1
        mask_low = j0wavk < 0
        mask_high = j1wavk >= nwav
        j0wavk = np.where(mask_low, 0, np.where(mask_high, nwav-1, j0wavk))
        j1wavk = np.where(mask_low, 0, np.where(mask_high, nwav-1, j1wavk))
        val0 = w0logg*(w0teff*grid[j0teff,j0logg,j0wavk] + w1teff*grid[j1teff,j0logg,j0wavk]) \
             + w1logg*(w0teff*grid[j0teff,j1logg,j0wavk] + w1teff*grid[j1teff,j1logg,j0wavk])
        val1 = w0logg*(w0teff*grid[j0teff,j0logg,j1wavk] + w1teff*grid[j1teff,j0logg,j1wavk]) \
             + w1logg*(w0teff*grid[j0teff,j1logg,j1wavk] + w1teff*grid[j1teff,j1logg,j1wavk])
        tmp_fl = w0wav*val0 + w1wav*val1
        contrib = tmp_fl * (area[sl]*val_mu[sl])[:,None]
        fl += contrib.sum(axis=0)
    return fl

def Interp_spectroscopy(grid, wteff, wlogg, wmu, jteff, jlogg, jmu, area, val_mu, wav, wav0, dwav):
    """
    Simple interpolation of an atmosphere grid having axes (logtemp, logg, mu, wav).

    This grid interpolation is made for a grid which is regular and linear in
    the wavelength space.

    Note: Because of the Doppler shift, the interpolation on the wavelength
        will necessarily go out of bound, on the lower or upper range. We
        assume that the atmosphere grid has a broader spectral coverage than
        the data.

    Parameters
    ----------
    The interpolation takes a set of points to be interpolated.
    grid : ndarray
        Atmosphere grid, with dimensions (logtemp, logg, mu, wav).
    wteff, wlogg, wmu : ndarray
        Weights of the temperature, logg, mu to interpolate at.
    jteff, jlogg, jmu : ndarray
        Lower index of the position of the temperature, logg, mu to interpolate at.
    area : ndarray
        Area (i.e. weight) of each surface element for the summation.
    val_mu : ndarray
        Value of the cross-section visible to us.
    wav : ndarray
        Values of the wavelength to interpolate at.
    wav0 : float
        Lower wavelength value of the grid.
    dwav : float
        Spacing of the grid.

    Returns
    -------
    spectrum : ndarray
        Spectrum integrated over the surface.
    """
    logger.log(9, "start")
    grid = np.ascontiguousarray(grid, dtype=float)
    wteff = np.ascontiguousarray(wteff, dtype=float)
    wlogg = np.ascontiguousarray(wlogg, dtype=float)
    wmu = np.ascontiguousarray(wmu, dtype=float)
    jteff = np.ascontiguousarray(jteff, dtype=int)
    jlogg = np.ascontiguousarray(jlogg, dtype=int)
    jmu = np.ascontiguousarray(jmu, dtype=int)
    area = np.ascontiguousarray(area, dtype=float)
    val_mu = np.ascontiguousarray(val_mu, dtype=float)
    wav = np.ascontiguousarray(wav, dtype=float)
    wav0 = float(wav0)
    dwav = float(dwav)
    nsurf = jteff.size
    nwav = wav.size
    nwav_arr = grid.shape[-1]
    fl = np.zeros(nwav, dtype=float)
    ## Pure-NumPy replacement for the weave.inline block above (scipy.weave
    ## is no longer available). Unlike Interp_doppler, the wavelength index
    ## here depends only on the fixed target wavelength grid `wav` (not per
    ## surface element), so it is computed once up front; each surface
    ## element then contributes its (teff,logg,mu) quadrilinear interpolation
    ## at those two wavelength indices, blended and summed. Chunked over
    ## surface elements to bound memory.
    tmp_wav = (wav-wav0)/dwav
    j0wav = np.trunc(tmp_wav).astype(int)
    w1wav = tmp_wav - np.trunc(tmp_wav)  # matches C's fmod(tmp_wav, 1.)
    w0wav = 1.-w1wav
    j1wav = j0wav+1
    mask_low = j0wav < 0
    mask_high = j1wav >= nwav_arr
    j0wav = np.where(mask_low, 0, np.where(mask_high, nwav_arr-1, j0wav))
    j1wav = np.where(mask_low, 0, np.where(mask_high, nwav_arr-1, j1wav))

    chunk = max(1, 20_000_000 // max(nwav,1))
    for start in range(0, nsurf, chunk):
        sl = slice(start, start+chunk)
        jteff_c = jteff[sl,None]; wteff_c = wteff[sl,None]
        jlogg_c = jlogg[sl,None]; wlogg_c = wlogg[sl,None]
        jmu_c = jmu[sl,None]; wmu_c = wmu[sl,None]
        val0 = _Trilinear_at_wav(grid, jteff_c, jlogg_c, jmu_c, wteff_c, wlogg_c, wmu_c, j0wav[None,:])
        val1 = _Trilinear_at_wav(grid, jteff_c, jlogg_c, jmu_c, wteff_c, wlogg_c, wmu_c, j1wav[None,:])
        tmp_fl = w0wav[None,:]*val0 + w1wav[None,:]*val1
        contrib = np.exp(tmp_fl) * (area[sl]*val_mu[sl])[:,None]
        fl += contrib.sum(axis=0)
    logger.log(9, "end")
    return fl

def Interp_spectroscopy_doppler(grid, wteff, wlogg, wmu, jteff, jlogg, jmu, area, val_mu, wav, wav0, dwav, val_vel):
    """
    Simple interpolation of an atmosphere grid having axes
    (logtemp, logg, mu, wav).

    This grid interpolation is made for a grid which is regular and linear in
    the wavelength space.

    This grid interpolation takes into account the Doppler shift due to each
    surface element.

    This grid interpolation also takes into account the Doppler boosting
    component. I_nu/nu^3 is a Lorentz invariant (and hence I_lambda/nu^5).
    Therefore,
        I(nu) = (nu/nu')^3 I'(nu')
        or
        I(lambda) = (nu/nu')^5 I'(lambda')
    where, in the non-relativistic case (v<<c)
        nu/nu' = 1 + v/c
    and
        (nu/nu')^n ~ 1 + n*v/c
    In this case, we have F_lambda and so the boosting is
        F(lambda) = F(lambda') * (1+5v/c)

    Note: Because of the Doppler shift, the interpolation on the wavelength
        will necessarily go out of bound, on the lower or upper range. We
        assume that the atmosphere grid has a broader spectral coverage than
        the data.

    Parameters
    ----------
    grid : ndarray
        Atmosphere grid, with dimensions (logtemp, logg, mu, wav).
        The flux values are natural logarithm in energy per unit time per unit
        solid angle per unit wavelength.
    wteff, wlogg, wmu : ndarray
        Weights of the temperature, logg, mu to interpolate at.
    jteff, jlogg, jmu : ndarray
        Lower index of the position of the temperature, logg, mu to interpolate
        at.
    area : ndarray
        Area (i.e. weight) of each surface element for the summation.
    val_mu : ndarray
        Value of the cross-section visible to us.
    wav : ndarray
        Values of the wavelength to interpolate at.
    wav0 : float
        Lower wavelength value of the grid.
    dwav : float
        Spacing of the grid.
    val_vel : ndarray
        Velocity value (in v/c unit) of each surface element.

    Returns
    -------
    spectrum : ndarray
        Spectrum integrated over the surface.
    """
    logger.log(9, "start")
    grid = np.ascontiguousarray(grid, dtype=float)
    wteff = np.ascontiguousarray(wteff, dtype=float)
    wlogg = np.ascontiguousarray(wlogg, dtype=float)
    wmu = np.ascontiguousarray(wmu, dtype=float)
    jteff = np.ascontiguousarray(jteff, dtype=int)
    jlogg = np.ascontiguousarray(jlogg, dtype=int)
    jmu = np.ascontiguousarray(jmu, dtype=int)
    area = np.ascontiguousarray(area, dtype=float)
    val_mu = np.ascontiguousarray(val_mu, dtype=float)
    wav = np.ascontiguousarray(wav, dtype=float)
    wav0 = float(wav0)
    dwav = float(dwav)
    val_vel = np.ascontiguousarray(val_vel, dtype=float)
    nsurf = jteff.size
    nwav = wav.size
    nwav_arr = grid.shape[-1]
    fl = np.zeros(nwav, dtype=float)
    ## Pure-NumPy replacement for the weave.inline block above (scipy.weave
    ## is no longer available). Same scheme as Interp_spectroscopy, but the
    ## wavelength index now depends on both surface element (via its
    ## Doppler velocity) and output pixel, so it is computed per
    ## (surface-element, pixel) pair; a (1+5v) boosting factor is applied
    ## per surface element. Chunked over surface elements to bound memory.
    chunk = max(1, 20_000_000 // max(nwav,1))
    for start in range(0, nsurf, chunk):
        sl = slice(start, start+chunk)
        jteff_c = jteff[sl,None]; wteff_c = wteff[sl,None]
        jlogg_c = jlogg[sl,None]; wlogg_c = wlogg[sl,None]
        jmu_c = jmu[sl,None]; wmu_c = wmu[sl,None]
        val_vel_c = val_vel[sl,None]

        tmp_wav = (wav[None,:]*(1+val_vel_c) - wav0)/dwav
        j0wav = np.trunc(tmp_wav).astype(int)
        w1wav = tmp_wav - np.trunc(tmp_wav)  # matches C's fmod(tmp_wav, 1.)
        w0wav = 1.-w1wav
        j1wav = j0wav+1
        mask_low = j0wav < 0
        mask_high = j1wav >= nwav_arr
        j0wav = np.where(mask_low, 0, np.where(mask_high, nwav_arr-1, j0wav))
        j1wav = np.where(mask_low, 0, np.where(mask_high, nwav_arr-1, j1wav))

        val0 = _Trilinear_at_wav(grid, jteff_c, jlogg_c, jmu_c, wteff_c, wlogg_c, wmu_c, j0wav)
        val1 = _Trilinear_at_wav(grid, jteff_c, jlogg_c, jmu_c, wteff_c, wlogg_c, wmu_c, j1wav)
        tmp_fl = w0wav*val0 + w1wav*val1
        contrib = np.exp(tmp_fl) * (area[sl]*val_mu[sl]*(1+5*val_vel[sl]))[:,None]
        fl += contrib.sum(axis=0)
    logger.log(9, "end")
    return fl

