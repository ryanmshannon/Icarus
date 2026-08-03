# Licensed under a 3-clause BSD style license - see LICENSE

import sys
import os

try:
    from scipy import weave 
except:
    try:
        import weave
    except:
        print('weave cannot be import from scipy nor on its own.')

try:
    from numba import autojit
except:
    print("Cannot load the numba module.")

from .import_modules import *

logger = logging.getLogger(__name__)


##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##
## Time series utilities
## Contain functions that pertain to "time series-related"
## purposes such as convolution, interpolation, rebinning, etc.
##----- ----- ----- ----- ----- ----- ----- ----- ----- -----##


def Convolve_gaussian_tophat(arr, sigma=1., top=1):
    """
    Convolve an array with a Gaussian and a tophat
    function along the last dimension.

    arr (array): Array of values to be convolved.
    sigma (float): The width (sigma) of the Gaussian.
    top (int): The width of the tophat.

    Note: This function works on a multi-dimensional array
        but will only apply the convolution on the last
        axis (i.e. wavelength if it is a spectrum array).
    """
    ## We define the gaussian kernel
    m_gauss = int(4*sigma+0.5)
    w_gauss = 2*m_gauss+1
    k_gauss = np.exp(-0.5*(np.arange(w_gauss)-m_gauss)**2/sigma**2)
    ## We define the tophat kernel
    w_top = int(top)
    ## If the tophat's width is even, we need to center it so the width is odd in order to preserve the phase in the convolution
    if w_top%2 == 0:
        w_top += 1
        k_top = np.ones(w_top)
        k_top[0] = 0.5
        k_top[-1] = 0.5
    else:
        k_top = np.ones(w_top)
    ## Calculating the full kernel
    if w_gauss > w_top:
        kernel = scipy.ndimage.convolve1d(k_gauss, k_top, mode='constant', cval=0.0)
    else:
        kernel = scipy.ndimage.convolve1d(k_top, k_gauss, mode='constant', cval=0.0)
    ## Normalizing the kernel so the sum is unity
    kernel /= kernel.sum()
    ## Applying the kernel to the array of values
    newarr = scipy.ndimage.convolve1d(arr, kernel, axis=-1)
    return newarr

def Doppler_shift_spectrum(fref, wref, wobs, v):
    """
    Simple Doppler shifting of a spectrum using a linear interpolation.

    This Doppler shifting takes into account the Doppler boosting
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
    fref : ndarray
        Rest flux in energy per unit time per unit solid angle per unit
        wavelength.
    wref : ndarray
        Rest wavelengths
    wobs : ndarray
        Wavelengths to be interpolated at
    v : float
        Velocity in v/c unit
            Positive velocity: blueshift
            Negative velocity: redshift

    Returns
    -------
    fobs : ndarray
        Doppler shifted and boosted spectrum.
    """
    logger.log(5, "start")
    wref = np.ascontiguousarray(wref, dtype=float)
    fref = np.ascontiguousarray(fref, dtype=float)
    wobs = np.ascontiguousarray(wobs, dtype=float)
    v = float(v)
    nref = wref.size
    nobs = wobs.size
    ## Pure-NumPy replacement for the weave.inline block above (scipy.weave
    ## is no longer available). Same bisection-based linear interpolation as
    ## Getaxispos_vector, applied to the Doppler-shifted wavelengths, with
    ## the (1+5v) boosting factor applied to the interpolated flux.
    wav = wobs*(1+v)
    ascending = wref[-1] > wref[0]
    if ascending:
        j = np.searchsorted(wref, wav, side='right') - 1
    else:
        j = (nref - 1) - np.searchsorted(wref[::-1], wav, side='left')
    j = np.clip(j, 0, nref-2)
    w = (wav - wref[j]) / (wref[j+1] - wref[j])
    fobs = (fref[j]*(1-w) + fref[j+1]*w) * (1+5*v)
    logger.log(5, "end")
    return fobs

def Doppler_shift_spectrum_integrate(fref, wobs, v, refstart, refstep):
    """
    Takes a reference spectrum, Doppler shifts it, and calculate
    the new spectral flux values at the provided observed wavelengths.

    - Assumes constant bin size and separation for the reference spectrum.
    - Assumes that the observed spectrum bin size is larger than
        the reference spectrum bin size and performs the integration.
        If it was smaller, a simple interpolation would be enough.
    - Takes into account the Doppler boosting component. I_nu/nu^3 is a
        Lorentz invariant (and hence I_lambda/nu^5).
        In this case, we have F_lambda and so the boosting is
            F(lambda) = F(lambda') * (1+5v/c)

            (see Doppler_shift_spectrum for a full explanation)

    fref: reference flux values
    wobs: observed wavelengths
    v: Doppler velocity shift (in m/s)
    refstart: wavelength of the first reference spectrum data point
    refstep: wavelength step size of the reference spectrum

    N.B. Could be optimized for the case of constant binning for the
    observed spectrum.
    """
    nobs = wobs.size
    nref = fref.size
    fref = np.ascontiguousarray(fref, dtype=float)
    wobs = np.ascontiguousarray(wobs, dtype=float)
    v = float(v)
    refstart = float(refstart)
    refstep = float(refstep)
    ## Pure-NumPy replacement for the weave.inline block above (scipy.weave
    ## is no longer available). For each observed bin [wl,wu] (Doppler
    ## shifted by `scale`), integrates (with edge fractions) the reference
    ## spectrum bins it covers, normalizes by bin width, and applies the
    ## (1+5v) boosting factor -- identical arithmetic to the C code it
    ## replaces, vectorized over all bins at once. Bins entirely below/above
    ## the reference spectrum's range are assigned the first/last reference
    ## flux value directly (no normalization/boosting), as in the original.
    scale = 1.+v
    wl = np.empty(nobs, dtype=float)
    wu = np.empty(nobs, dtype=float)
    wl[0] = wobs[0] - (wobs[1]-wobs[0])*0.5
    wu[0] = (wobs[0]+wobs[1])*0.5
    wl[1:-1] = (wobs[1:-1]+wobs[:-2])*0.5
    wu[1:-1] = (wobs[1:-1]+wobs[2:])*0.5
    wl[-1] = (wobs[-1]+wobs[-2])*0.5
    wu[-1] = wobs[-1] + (wobs[-1]-wobs[-2])*0.5
    wl *= scale
    wu *= scale

    refposl = (wl - refstart) / refstep
    refposu = (wu - refstart) / refstep
    ## C's (int) cast truncates toward zero, unlike np.floor.
    irefl = np.trunc(refposl+0.5).astype(np.int64)
    irefu = np.trunc(refposu+0.5).astype(np.int64)

    fbin = np.empty(nobs, dtype=float)
    mask_below = irefl < 0
    mask_above = (~mask_below) & (irefu > nref-1)
    mask_valid = (~mask_below) & (~mask_above)

    fbin[mask_below] = fref[0]
    fbin[mask_above] = fref[-1]

    il = irefl[mask_valid]
    iu = irefu[mask_valid]
    rl = refposl[mask_valid]
    ru = refposu[mask_valid]
    mask_eq = il == iu
    fval = np.where(mask_eq,
                     (ru-rl) * fref[il],
                     (0.5-(rl-il))*fref[il] + (0.5+(ru-np.where(mask_eq, il, iu)))*fref[np.where(mask_eq, il, iu)])
    ## Add the whole reference bins strictly between il and iu (empty range
    ## when mask_eq or when iu == il+1), via a prefix sum.
    cumfref = np.concatenate(([0.], np.cumsum(fref)))
    whole = np.where(mask_eq, 0., cumfref[iu] - cumfref[il+1])
    fval = fval + whole
    fval *= refstep/(wu[mask_valid]-wl[mask_valid])
    fval *= (1+5*v)
    fbin[mask_valid] = fval
    return fbin

def FFTConvolve1D(in1, in2, axis=-1):
    """
    Convolve a N-dimensional array with a one dimensional kernel using FFT
    along a specified axis.

    Parameters
    ----------
    in1 : ndarray
        Input array to operate the convolution on. Can be any dimension.
    in2 : ndarray
        Input convolution kernel. Must be 1-dimensional.
        The dimension of in2 must much the axis dimension of in1 over which the
        convolution is performed.
    axis : int
        Axis over which the convolution is performed.

    Returns
    -------
    convarr : ndarray
        Convolved array having the same dimensions as in1. Note that the
        convolution implicitely uses the "same" method, applied to in1.
    """
    ## Making sure that the dimensions match
    #if in1.shape[axis] != in2.size:
    #    raise ValueError("The 'axis' dimension of in1 must match the size of in2")

    ## Formatting the kernel to match the input array
    in2 = in2.copy()
    s2 = np.ones(in1.ndim, dtype=int)
    s2[axis] = in2.size
    in2.shape = s2

    ## Working out the size of the convolution array and the slice to extract
    size = in1.shape[axis] + in2.size - 1
    fftslice = [slice(l) for l in in1.shape]
    fftslice[axis] = slice(0, int(size))
    fftslice = tuple(fftslice)
    
    ## Using 2**n FFT for speed
    fftsize = 2**int(np.ceil(np.log2(size)))

    ## Applying the convolution theorem in the Fourier space
    fftarr = scipy.fftpack.fft(in1, fftsize, axis=axis)
    fftarr *= scipy.fftpack.fft(in2, fftsize, axis=axis)
    convarr = scipy.fftpack.ifft(fftarr, axis=axis)[fftslice].copy()

    return scipy.signal.signaltools._centered(convarr, in1.shape)

def Getaxispos_scalar(xold, xnew):
    """
    Given a scalar xnew, returns the index and fractional weight
    that corresponds to the nearest linear interpolation from
    the vector xold.

    xold: vector of values to be interpolated from.
    xnew: scalar value to be interpolated.

    weight,index = Getaxispos_scalar(xold, xnew)
    """
    ## Pure-NumPy replacement for the weave.inline block above (scipy.weave
    ## is no longer available). Scalar special-case of Getaxispos_vector's
    ## bisection search.
    xold = np.ascontiguousarray(xold, dtype=float)
    xnew = float(xnew)
    n = xold.shape[0]
    ascending = xold[-1] > xold[0]
    if ascending:
        j = np.searchsorted(xold, xnew, side='right') - 1
    else:
        j = (n - 1) - np.searchsorted(xold[::-1], xnew, side='left')
    j = int(np.clip(j, 0, n-2))
    w = (xnew - xold[j]) / (xold[j+1] - xold[j])
    return w,j

def Getaxispos_vector(xold, xnew):
    """
    Given a vector xnew, returns the indices and fractional weights
    that corresponds to their nearest linear interpolation from
    the vector xold.

    xold: vector of values to be interpolated from.
    xnew: vector of values to be interpolated.

    weights,indices = Getaxispos_scalar(xold, xnew)
    """
    logger.log(5, "start")
    xold = np.ascontiguousarray(xold, dtype=float)
    xnew = np.ascontiguousarray(xnew, dtype=float)
    n = xold.shape[0]
    ## Pure-NumPy replacement for the weave.inline block above (scipy.weave
    ## is no longer available). Equivalent to the bisection search it
    ## replaces: for each xnew, finds j (clipped to [0, n-2]) such that
    ## xold[j] and xold[j+1] bracket xnew (xold can be ascending or
    ## descending), plus the fractional weight w for linear interpolation.
    ascending = xold[-1] > xold[0]
    if ascending:
        j = np.searchsorted(xold, xnew, side='right') - 1
    else:
        j = (n - 1) - np.searchsorted(xold[::-1], xnew, side='left')
    j = np.clip(j, 0, n-2)
    w = (xnew - xold[j]) / (xold[j+1] - xold[j])
    logger.log(5, "end")
    return w,j

def General_polynomial_fit(y, x=None, err=None, coeff=1, Xfnct=None, Xfnct_offset=False, chi2=True):
    """
    Best-fit generalized polynomial to a function minimizing:
    chi2 = sum_i( [y(x_i) - sum_k( a_k * X_k(x_i) )]**2 / err_i**2 )
    X_k(x_i) = O_k(x_i)
        if Xfnct=None, i.e. O_k is a simple polynomial of order k
    X_k(x_i) = O_k(x_i)*f(x_i)
        if Xfnct=f(x_i) and Xfnct_offset=False
    X_k(x_i) = O_k(x_i)*f(x_i) + offset
        if Xfnct=f(x_i) and Xfnct_offset=True

    y: the y values, shape (n)
    x (None): the x values, shape (n)
    err (None): the error values, shape (1) or (n)
    coeff (1): the number of coefficients to the generalized polynomial
            function to be fitted (>= 1)
    Xfnct (None): a function to generalize the polynomial, shape (n)
    Xfnct_offset (False): whether the polynomial includes a constant offset or not
    chi2 (bool): If true, will also return the chi-square.

    Returns generalized polynomial coefficients
        shape (coeff)
        i.e. (a_n, a_(n-1), ..., a_1, a_0)
    """
    y = np.ascontiguousarray(y, dtype=float)
    n = y.size
    if x is None:
        x = np.arange(n, dtype=float)
    else:
        x = np.ascontiguousarray(x, dtype=float)
    if err is None:
        err = np.ones(n, dtype=float)
    elif np.size(err) == 1:
        err = np.ones(n, dtype=float)*err
    else:
        err = np.ascontiguousarray(err, dtype=float)
    if Xfnct is None:
        Xfnct = np.ones(n, dtype=float)
    else:
        Xfnct = np.ascontiguousarray(Xfnct, dtype=float)
    if Xfnct_offset:
        Xfnct_offset = 1
    else:
        Xfnct_offset = 0
    ## Pure-NumPy replacement for the weave.inline block above (scipy.weave
    ## is no longer available). The C code's recurrence a(i,coeff-1-k) =
    ## a(i,coeff-k)*x(i) (for k>=1, or k>=2 in the offset case) just builds
    ## successive powers of x(i), so each column of the design matrix is a
    ## power of x times a base value -- computed directly here rather than
    ## recursively.
    base = Xfnct/err
    a = np.empty((n,coeff), dtype=float)
    if Xfnct_offset == 1:
        if coeff > 1:
            powers = np.arange(coeff-2, -1, -1)
            a[:,:coeff-1] = base[:,None] * (x[:,None] ** powers[None,:])
        a[:,coeff-1] = 1./err
    else:
        powers = np.arange(coeff-1, -1, -1)
        a[:,:] = base[:,None] * (x[:,None] ** powers[None,:])
    b = y/err
    tmp = np.linalg.lstsq(a, b, rcond=None)
    if chi2:
        return tmp[0], tmp[1][0]
    return tmp[0]

def Interp_linear(y, x, xnew):
    """
    Given a vector xnew, returns the indices and fractional weights
    that corresponds to their nearest linear interpolation from
    the vector xold.

    y: y variables to be interpolated from.
    x: x variables to be interpolated from.
    xnew: x variables to be interpolated at.

    weights,indices = Getaxispos_scalar(xold, xnew)
    """
    logger.log(5, "start")
    x = np.ascontiguousarray(x, dtype=float)
    y = np.ascontiguousarray(y, dtype=float)
    xnew = np.ascontiguousarray(xnew, dtype=float)
    n_old = x.size
    n_new = xnew.size
    ## Pure-NumPy replacement for the weave.inline block above (scipy.weave
    ## is no longer available). Same bisection search as Getaxispos_vector,
    ## directly evaluating the linearly-interpolated y values.
    ascending = x[-1] > x[0]
    if ascending:
        j = np.searchsorted(x, xnew, side='right') - 1
    else:
        j = (n_old - 1) - np.searchsorted(x[::-1], xnew, side='left')
    j = np.clip(j, 0, n_old-2)
    w = (xnew - x[j]) / (x[j+1] - x[j])
    ynew = y[j]*(1-w) + y[j+1]*w
    logger.log(5, "end")
    return ynew

def Interp_linear2(y, weights, inds):
    """
    Given some weights and indices (from Getaxispos), evaluate the linear
    interpolation of the original time series.

    >>> x = np.arange(100.)
    >>> y = np.sin(x/10)
    >>> xnew = np.arange(20.)*5+0.3
    >>> weights,indices = Getaxispos_scalar(x, xnew)
    >>> ynew = Utils.Interp_integrate(y, weights, indices)
    """
    ## Pure-NumPy replacement for the weave.inline block above (scipy.weave
    ## is no longer available): a simple gather + weighted sum.
    y = np.ascontiguousarray(y, dtype=float)
    weights = np.ascontiguousarray(weights, dtype=float)
    inds = np.ascontiguousarray(inds, dtype=int)
    if y.ndim == 1:
        ynew = y[inds]*(1.-weights) + y[inds+1]*weights
    elif y.ndim == 2:
        ynew = y[:,inds]*(1.-weights)[None,:] + y[:,inds+1]*weights[None,:]
    else:
        print("Number of dimensions > 2 not supported!")
        return
    return ynew

def Interp_linear_integrate(y, x, xnew):
    """
    Resample a time series (x,y) at the values xnew by performing an
    integration within each new bin of the old time series using the Euler
    method. Here we assume that the new time series is undersampling the old
    one, otherwise it is just equivalent to linearly interpolating.

    Parameters
    ----------
    y : (N,...) ndarray
        y values to interpolate from. The array can be multi-dimensional. The
        interpolation will be carried along the first axis.
    x : (N,) ndarray
        x values to interpolate from. y = f(x)
    xnew : (M,) ndarray
        x values to interpolate at.

    Return
    ------
    ynew : (M,...) ndarray
        y values interpolated at. The first dimension is the same as xnew,
        while the other dimensions, if any, will match the other dimensions of
        x.

    >>> x = np.arange(100.)
    >>> y = np.sin(x/10)
    >>> xnew = np.arange(20.)*5+0.3
    >>> ynew = Interp_linear_integrate(y, x, xnew)
    """
    shape = list(y.shape)
    shape[0] = xnew.size
    ynew = np.zeros(shape, dtype=float)
    i = 0
    ii = 0
    while ii < xnew.size:
        weight = 0.
        val = 0.
        if ii == 0:
            xnewl = xnew[ii]-(xnew[ii+1]-xnew[ii])*0.5
        else:
            xnewl = (xnew[ii]+xnew[ii-1])*0.5
        if ii == xnew.size-1:
            xnewr = xnew[ii]+(xnew[ii]-xnew[ii-1])*0.5
        else:
            xnewr = (xnew[ii+1]+xnew[ii])*0.5
        while i < x.size:
            if i == 0:
                xl = x[i]-(x[i+1]-x[i])*0.5
            else:
                xl = (x[i]+x[i-1])*0.5
            if i == x.size-1:
                xr = x[i]+(x[i]-x[i-1])*0.5
            else:
                xr = (x[i+1]+x[i])*0.5
            ## Bin completely inside
            if xl >= xnewl and xr <= xnewr:
                weight += xr-xl
                val += y[i]*(xr-xl)
            ## Bin overlapping the right side
            elif xl < xnewr and xr > xnewr:
                weight += xnewr-xl
                val += y[i]*(xnewr-xl)
                ## Means we have to move to next xnew bin
                break
            ## Bin overlapping the left side
            elif xl < xnewl and xr > xnewl:
                weight += xr-xnewl
                val += y[i]*(xr-xnewl)
            ## Bin to the right
            elif xl >= xnewr:
                ## Means we are done
                break
            ## Bin to the left
            elif xr <= xnewl:
                pass
            ## This condition should not happen
            else:
                pass
            i += 1
        ## Add the sum to ynew
        if weight != 0:
            ynew[ii] = val/weight
        ii += 1
    return ynew

if 'numba' in sys.modules:
    Interp_linear_integrate = autojit(Interp_linear_integrate)

def Resample_linlog(xold):
    """
    Resample a linear wavelength vector to a log space and
    returns the new vector and the Doppler shift z.

    The resampling is done such that the largest wavelength interval
    is conserved in order to preserve the spectral resolution.

    The Doppler shift is:
        1+z = lambda_1 / lambda_0

    In the non-relativistic limit:
        z = v/c

    >>> xnew, z = Resample_linlog(xold)
    """
    z = xold[-2] / xold[-1] - 1
    ## The number of data points to cover the spectal range is
    n = np.ceil( np.log(xold[0]/xold[-1]) / np.log(1+z) ) + 1
    xnew = xold[-1] * (1+z)**np.arange(n)[::-1]
    return xnew, np.abs(z)

def Resample_loglin(xold):
    """
    Resample a log wavelength vector to a linear space.

    The resampling is done such that the smallest wavelength interval
    is conserved in order to preserve the spectral resolution.

    >>> xnew = Resample_loglin(xold)
    """
    step = xold[1] - xold[0]
    xnew = np.arange(xold[0], xold[-1]+step, step)
    return xnew



