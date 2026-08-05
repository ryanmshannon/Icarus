# Licensed under a 3-clause BSD style license - see LICENSE

"""
model_J2241-5236.py

Models the phase-resolved optical light curve of the irradiated companion
star of the "black widow" pulsar PSR J2241-5236, using Icarus.

Data and atmosphere grid
------------------------
Icarus does not distribute observational data or atmosphere grids. The
files pointed to by `data_fln` and `atmo_fln` below must be created by
the user and follow the format documented in the docstring of
Icarus.Photometry.Photometry:
    - `data_fln` is an index file listing, for each photometric band, the
      band name, the phase/magnitude/magnitude-error column ids, the
      phase shift, the band calibration error, and the per-band data
      file (which itself holds the phase-resolved photometry).
    - `atmo_fln` points to the atmosphere grid (e.g. a BT-Settl grid)
      appropriate for a strongly irradiated, very-low-mass companion.
    - the spectroscopic grid used for the model spectrum is built by
      build_atmo_grid_spec_J2241-5236.py and selected with --spec-atmo
      (or --hires for the high-resolution one). The script reads its pixel
      scale from the grid's own metadata, so which library the grid came
      from is not hardcoded anywhere below.

Usage
-----
    ## Low-resolution (R ~ 740) SED over the full X-shooter range:
    python model_J2241-5236.py

    ## High-resolution: Halpha at quadrature, where the companion's radial
    ## velocity is largest (requires the grid to cover that wavelength):
    python model_J2241-5236.py --hires --spec-phase 0.75 \\
        --plot-range 6540 6590 --save-spec spec_halpha_p075.txt

    ## As it would actually be observed: placed at the source's distance and
    ## reddened, rather than left at Icarus' native 10 pc:
    python model_J2241-5236.py --hires --distance 1.0 --av 0.1

System parameters
------------------
The pulsar spin/orbital timing solution is from the discovery paper:
    Keith, M. J., et al. 2011, MNRAS, 414, 1292
        Porb ~ 3.5 hr, projected semi-major axis (asini) ~ 9.2e-4 lt-s

The companion heating/irradiation model and the resulting neutron star
mass are from:
    Kandel, D., Romani, R. W., & An, H. 2019, ApJ, arXiv:1908.00992

Note: the values below are the best available approximations to the
published parameters. Users should double-check them against the paper
before using this script for scientific analysis.
"""

import argparse

import Icarus
from Icarus.Utils.import_modules import *


##### Data and atmosphere grid description files (see module docstring).
data_fln = 'data_J2241-5236.txt'
atmo_fln = 'atmo_models_J2241-5236.txt'
##### Spectroscopic atmosphere grid, used for the model spectrum at the end of
##### this script. Built by build_atmo_grid_spec_J2241-5236.py, either from the
##### low-resolution CDBS grid (the default, R ~ 740) or from the Husser
##### high-resolution library (--source phoenix-hires). Override with
##### --spec-atmo; everything downstream adapts to the grid's own resolution,
##### which is read from its metadata rather than assumed.
SPEC_ATMO_FLN = 'atmo_grid_spec_J2241-5236.h5'
SPEC_ATMO_FLN_HIRES = 'atmo_grid_spec_hires_J2241-5236.h5'
##### VLT/X-shooter arm boundaries (Angstrom). The spectroscopic grid need not
##### span all three (a high-resolution grid is usually built for one arm at a
##### time); arms outside its range are simply skipped. Used for the coverage
##### report and the spectrum plot below.
XSHOOTER_ARMS = [('UVB', 3000., 5595.),
                 ('VIS', 5595., 10240.),
                 ('NIR', 10240., 24800.)]
ndiv = 5

##### Distance and extinction to PSR J2241-5236, used only to place the model
##### spectrum on an observable flux scale (--distance/--dm/--av). Icarus itself
##### works at 10 pc: Star._Proj hardcodes that, so every flux the model returns
##### is an absolute one and the distance is applied here, afterwards.
#####
##### IMPORTANT: set these to the values you intend to use and cite. They are
##### NOT constrained by anything in this repository, and the light-curve fit
##### does not measure them either -- Photometry.Plot fits a free offset per
##### band, which absorbs distance, extinction and any grid zero-point error
##### into one number per band (see the note near the Get_flux call below).
DISTANCE_KPC = 1.1                    # None -> must be given with --distance/--dm
AV = 0.05                               # V-band extinction, magnitudes
RV = 3.1                               # A_V/E(B-V); 3.1 is the standard diffuse-ISM value


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Model the phase-resolved light curve and spectrum of the "
                    "PSR J2241-5236 companion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--spec-atmo', default=SPEC_ATMO_FLN,
                   help="Spectroscopic atmosphere grid built by "
                        "build_atmo_grid_spec_J2241-5236.py.")
    p.add_argument('--hires', dest='spec_atmo', action='store_const',
                   const=SPEC_ATMO_FLN_HIRES,
                   help="Shorthand for --spec-atmo {}.".format(SPEC_ATMO_FLN_HIRES))
    p.add_argument('--spec-phase', type=float, default=0.5,
                   help="Orbital phase of the model spectrum, in Icarus' convention "
                        "(0 = night side / faintest, 0.5 = day side / brightest).")
    p.add_argument('--plot-range', nargs=2, type=float, default=None,
                   metavar=('MIN', 'MAX'),
                   help="Wavelength range of the spectrum plot, in Angstrom. Default: "
                        "the whole grid.")
    d = p.add_argument_group('distance and extinction of the model spectrum')
    d.add_argument('--distance', type=float, default=DISTANCE_KPC, metavar='KPC',
                   help="Distance in kpc. Without this (and without --dm) the "
                        "spectrum is left at Icarus' native 10 pc, i.e. absolute.")
    d.add_argument('--dm', type=float, default=None, metavar='MAG',
                   help="Distance modulus, as an alternative to --distance.")
    d.add_argument('--av', type=float, default=AV, metavar='MAG',
                   help="V-band extinction applied to the spectrum, using the "
                        "O'Donnell (1994) law from Icarus.Utils.Flux.Extinction.")
    d.add_argument('--rv', type=float, default=RV, metavar='RATIO',
                   help="A_V/E(B-V) for the extinction law.")
    d.add_argument('--cardelli', action='store_true',
                   help="Use the Cardelli et al. (1989) extinction law instead of "
                        "O'Donnell (1994).")

    p.add_argument('--save-spec', default=None, metavar='PATH',
                   help="Also write the model spectrum as two columns, wavelength "
                        "(Angstrom) and flux (erg/s/cm^2/A) at whatever distance "
                        "the spectrum was placed at.")
    args = p.parse_args(argv)
    if args.dm is not None and args.distance is not None:
        p.error("--distance and --dm are mutually exclusive")
    if args.dm is None and args.distance is not None:
        if args.distance <= 0:
            p.error("--distance must be positive")
        ##### Distance modulus WITHOUT the extinction term: the extinction is
        ##### applied per wavelength below, so folding it in here as well would
        ##### double-count it.
        args.dm = Icarus.Utils.Flux.Distance_to_distance_modulus(args.distance)
    elif args.dm is not None:
        args.distance = Icarus.Utils.Flux.Distance_modulus_to_distance(args.dm)
    return args


args = parse_args()
spec_atmo_fln = args.spec_atmo


##### Pulsar timing solution (Reardon et al. 2021).
porb = 0.145672*86400                    # orbital period, in seconds
asini = 25.7e-3                        # pulsar projected semi-major axis, in light-seconds


##### Companion/system parameters (Kandel, Romani & An 2019).
incl = 49.7 * cts.degree                # orbital inclination
M_ns = 1.5                             # neutron star (pulsar) mass, in Msun
corotation = 1.                        # tidally-locked companion
filling = 0.66                         # nearly Roche-lobe-filling companion
gravdark = 0.08                        # gravity darkening coefficient (convective envelope)
Tnight = 3000.                         # unheated (night-side) base temperature, in K
#Tday = 10000.                          # approximate irradiated (day-side) temperature, in K
Tday=5000.

##### Deriving the mass ratio and the companion's velocity semi-amplitude
##### from the pulsar timing solution and the assumed neutron star mass,
##### using Icarus' binary utility functions.
mass_function = Icarus.Utils.Binary.Mass_function1(asini, porb)
q = Icarus.Utils.Binary.Mass_ratio1(mass_function, M_ns, incl)    # q = M_ns/M_companion
k2 = cts.TWOPI * asini * cts.c / porb                             # pulsar velocity semi-amplitude, in m/s
k1 = k2 * q                                                       # companion velocity semi-amplitude, in m/s

print( "Mass function: {:.3e} Msun".format(mass_function) )
print( "Derived mass ratio q = M_ns/M_companion = {:.1f}".format(q) )
print( "Companion mass ~ {:.4f} Msun".format(M_ns/q) )
print( "Companion velocity semi-amplitude K1 ~ {:.1f} km/s".format(k1/1e3) )


##### Deriving the irradiation temperature such that, at the sub-stellar
##### point, Tday**4 = Tnight**4 + Tirr**4.
tirr = (Tday**4 - Tnight**4)**0.25


##### Full parameter vector for Icarus.Photometry.Photometry.Get_flux:
##### [q, porb, incl, k1, corotation, filling, gravdark, Tnight, tirr]
par = np.r_[q, porb, incl, k1, corotation, filling, gravdark, Tnight, tirr]


##### Loading the data into an Icarus.Photometry object (failure to do so
##### is likely due to missing data/atmosphere model files -- see the
##### module docstring above).
print( "Loading the data into an Icarus.Photometry object.\n" )
fit = Icarus.Photometry.Photometry(atmo_fln, data_fln, ndiv)


##### Calculating the model light curves at the observed orbital phases.
print( "Calculating the model light curves of the companion star.\n" )
flux_model = fit.Get_flux(par, verbose=True)


##### Calculating the model spectrum at a given orbital phase.
#####
##### NOTE ON PHASE CONVENTION: this is Icarus' convention, NOT the paper's.
##### Icarus defines phase 0 as the modelled star (the companion) at inferior
##### conjunction, i.e. its unheated night side faces us and it is at its
##### faintest; phase 0.5 is the heated day side facing us, at its brightest.
##### The paper's convention is offset by +0.25 (its max is at 0.75), which is
##### what the phase_shift column of data_J2241-5236.txt corrects for.
spec_phase = args.spec_phase           # 0.5 = day side / maximum brightness

print( "Calculating the model spectrum at orbital phase {} (Icarus convention;"
       " = phase {} in the paper's convention).\n".format(spec_phase, (spec_phase+0.25) % 1) )

print( "Reading the spectroscopic grid {}.".format(spec_atmo_fln) )
spec_atmo = Icarus.Atmosphere.AtmoGridSpec.ReadHDF5(spec_atmo_fln)

##### Everything below adapts to the grid actually loaded. delta_v is the pixel
##### spacing in v/c units, which is what Flux_doppler shifts by, so it -- not
##### any assumption about which library the grid came from -- sets what the
##### model spectrum can and cannot be used for.
spec_delta_v = spec_atmo.meta['delta_v'] * cts.c / 1e3      # km/s per pixel
spec_pix_per_k1 = (k1/1e3) / spec_delta_v
##### Velocity structure is resolved once the orbital shift spans more than a
##### few pixels. Below that the line diagnostics are meaningless.
spec_resolved = spec_pix_per_k1 > 3.
for key in ('source', 'normalization', 'resolution_note', 'wavelength_scale'):
    if key in spec_atmo.meta:
        print( "  {}: {}".format(key, spec_atmo.meta[key]) )
print( "  sampling: {:.2f} km/s per pixel, i.e. K1 = {:.0f} km/s spans {:.1f}"
       " pixels.\n".format(spec_delta_v, k1/1e3, spec_pix_per_k1) )

##### fit.Get_flux above already called Make_surface with these parameters,
##### but we call it explicitly so this block does not depend on that.
fit.Make_surface(par)

##### Flux_doppler sums the (Doppler-shifted) contribution of every visible
##### surface element. atmo_doppler is only needed for photometry grids, so it
##### is left as None here.
spec_wav = np.asarray(spec_atmo.cols['wav'])          # Angstrom
spec_flux = fit.star.Flux_doppler(spec_phase, atmo_grid=spec_atmo)

##### UNITS: this is an absolute, physically calibrated flux -- specifically
##### the flux density the companion would have at a distance of 10 parsec, in
##### erg/s/cm^2/Angstrom. (Star._Proj returns (a/10pc)^2 and the surface areas
##### are in units of orbital separation^2, so area*proj = A_phys/(10 pc)^2.)
##### --distance/--dm below places it at the real distance by scaling by
##### (10 pc / d)^2, and --av reddens it; without them the spectrum stays
##### absolute. Note that the light-curve fit does NOT independently measure
##### either quantity: Photometry.Plot fits one free offset per band, which
##### absorbs distance, extinction and grid zero-point together, so the
##### distance used here is an input you must justify, not a fit result.
##### Accuracy: the underlying PHOENIX spectra are true surface fluxes
##### (integral F_lam dlam = sigma T^4). How faithfully the grid turns those
##### into the specific intensities Icarus integrates depends on how it was
##### built: with the default flux-conserving normalization the hemispheric
##### integral reproduces the surface flux exactly at every wavelength, whereas
##### Icarus' hardcoded 4/pi^2 factor (build ... --icarus-norm) is only
##### consistent with the limb-darkening law near 5000 A and is off by -14% at
##### 3000 A to +19% at 24800 A. The grid's 'normalization' metadata, printed
##### above, says which one is in force.
print( "Model spectrum: {} points from {:.1f} to {:.1f} A".format(
        spec_wav.size, spec_wav[0], spec_wav[-1]) )
print( "  flux at 10 pc, min/max: {:.4e} / {:.4e} erg/s/cm^2/A".format(
        spec_flux.min(), spec_flux.max()) )

##### Place the spectrum at the real distance and redden it. Both are pure
##### multiplicative factors applied after the surface integration, so nothing
##### about the model itself changes -- only the scale it is reported on.
spec_where = "10 pc"
if args.dm is not None:
    spec_flux = spec_flux * 10**(-0.4*args.dm)
    spec_where = "{:.3g} kpc".format(args.distance)
    print( "  scaled to d = {:.4g} kpc (DM = {:.3f} mag): min/max {:.4e} /"
           " {:.4e} erg/s/cm^2/A".format(args.distance, args.dm,
                                         spec_flux.min(), spec_flux.max()) )
if args.av:
    ##### Extinction expects microns and returns A_lambda/A_V.
    a_lambda = args.av * Icarus.Utils.Flux.Extinction(spec_wav*1e-4, Rv=args.rv,
                                                      cardelli=args.cardelli)
    if not np.isfinite(a_lambda).all():
        print( "  WARNING: the extinction law is undefined over part of this"
               " wavelength range ({} points); those points are left"
               " unreddened.".format(int((~np.isfinite(a_lambda)).sum())) )
        a_lambda = np.nan_to_num(a_lambda)
    spec_flux = spec_flux * 10**(-0.4*a_lambda)
    spec_where += ", A_V = {:.2f}".format(args.av)
    print( "  reddened with A_V = {:.3f} mag, Rv = {:.2f} ({}): A_lambda ="
           " {:.3f} at {:.0f} A to {:.3f} at {:.0f} A; min/max {:.4e} /"
           " {:.4e} erg/s/cm^2/A".format(
                args.av, args.rv, 'Cardelli 1989' if args.cardelli else "O'Donnell 1994",
                a_lambda[0], spec_wav[0], a_lambda[-1], spec_wav[-1],
                spec_flux.min(), spec_flux.max()) )
elif args.dm is None:
    print( "  NOTE: left at 10 pc (absolute). Pass --distance KPC (or --dm) and"
           " --av to put the spectrum on an observable flux scale." )

##### CROSS-CHECK OF THE ABSOLUTE SCALE.
##### The spectrum is only as absolute as the model's flux zero-point, and the
##### photometry can test that: the offsets Photometry fits are what the DATA
##### require on top of the 10 pc model, whereas the distance and extinction
##### assumed here PREDICT dm + ext*A_V. The two should agree. Any residual is
##### how wrong the absolute flux scale is -- a mixture of the grid zero-point,
##### the assumed Tnight/tirr/filling and the distance itself -- and it applies
##### to the spectrum just as much as to the photometry, since both come from
##### the same PHOENIX library and the same surface integration. Multiply any
##### predicted count rate (and divide any predicted RV precision) by the flux
##### factor printed below to carry the systematic through.
if args.dm is not None:
    chi2_off, extras_off = fit.Calc_chi2(par, do_offset=True, full_output=True)
    offset_fitted = np.asarray(extras_off['offset'])
    offset_predicted = args.dm + np.asarray(fit.data['ext'])*args.av
    resid = offset_fitted - offset_predicted
    print( "  absolute-scale cross-check against the photometry"
           " (fitted offset vs dm + ext*A_V):" )
    for band, o_fit, o_pred, r in zip(fit.data['id'], offset_fitted,
                                      offset_predicted, resid):
        print( "    {:3s}: fitted {:+7.3f}, predicted {:+7.3f}, residual {:+6.3f} mag"
               " (model too bright by x{:.2f})".format(
                    band, o_fit, o_pred, r, 10**(0.4*r)) )
    print( "    mean residual {:+.3f} +/- {:.3f} mag -> the model's absolute flux is"
           " off by a factor of {:.2f}. This is NOT corrected in the spectrum"
           " above.".format(resid.mean(), resid.std(), 10**(0.4*resid.mean())) )

##### Per-arm summary for VLT/X-shooter.
print( "  VLT/X-shooter arm coverage (mean flux at {}):".format(spec_where) )
for arm_name, arm_lo, arm_hi in XSHOOTER_ARMS:
    sel = (spec_wav >= arm_lo) & (spec_wav <= arm_hi)
    if sel.any():
        print( "    {:3s} {:6.0f}-{:6.0f} A : {:4d} pts, mean {:.3e} erg/s/cm^2/A".format(
                arm_name, arm_lo, arm_hi, int(sel.sum()), spec_flux[sel].mean()) )
if spec_resolved:
    print( "  Velocity structure is resolved: the orbital Doppler shift spans"
           " {:.0f} pixels at quadrature, and because Flux_doppler shifts each"
           " surface element by its own velocity, the rotational broadening of"
           " the tidally-locked companion and the phase-dependent line"
           " asymmetry come out of the surface integration. Line profiles and"
           " equivalent widths are meaningful predictions here; try"
           " --spec-phase 0.25 or 0.75 (quadrature, maximum radial velocity)"
           " and --plot-range to zoom in on a line.\n".format(spec_pix_per_k1) )
else:
    print( "  NOTE: the grid samples only {:.0f} km/s per pixel (R ~ {:.0f}),"
           " while X-shooter resolves R ~ 5000-9000. Line profiles, equivalent"
           " widths and the {:.0f} km/s orbital Doppler shift are therefore"
           " UNRESOLVED here: use this to predict continuum shape and"
           " broad-band count rates, not line diagnostics. Rebuild with"
           " 'build_atmo_grid_spec_J2241-5236.py --source phoenix-hires' and"
           " re-run with --hires for that.\n".format(
                spec_delta_v, cts.c/1e3/spec_delta_v, k1/1e3) )

if args.save_spec:
    np.savetxt(args.save_spec, np.c_[spec_wav, spec_flux],
               header="wavelength (A)  flux at {} (erg/s/cm^2/A)\n"
                      "phase {} (Icarus convention), grid {}".format(
                          spec_where, spec_phase, spec_atmo_fln))
    print( "  wrote the spectrum to {}\n".format(args.save_spec) )


##### Plotting the observed and modelled light curves, phase by phase, and
##### the model spectrum at the chosen orbital phase.
if pylab:
    fit.Plot(par)

    plot_lo, plot_hi = args.plot_range if args.plot_range else (spec_wav[0], spec_wav[-1])

    pylab.figure(figsize=(10, 5))
    ##### Shade the X-shooter arms behind the spectrum.
    for (arm_name, arm_lo, arm_hi), arm_col in zip(XSHOOTER_ARMS,
                                                    ['#4c72b0', '#55a868', '#c44e52']):
        if arm_hi < plot_lo or arm_lo > plot_hi:
            continue
        pylab.axvspan(arm_lo, arm_hi, color=arm_col, alpha=0.10)
        pylab.text(np.sqrt(max(arm_lo, plot_lo)*min(arm_hi, plot_hi)), 0.96, arm_name,
                   color=arm_col, ha='center', va='top',
                   transform=pylab.gca().get_xaxis_transform())
    pylab.plot(spec_wav, spec_flux, 'k-', lw=0.8)
    ##### Over a wide range the SED spans orders of magnitude and log-log is the
    ##### only readable choice; zoomed in on lines it flattens out and linear
    ##### axes show the line depths and profiles properly.
    if plot_hi/plot_lo > 1.5:
        pylab.xscale('log')
        pylab.yscale('log')
    pylab.xlim(plot_lo, plot_hi)
    sel = (spec_wav >= plot_lo) & (spec_wav <= plot_hi)
    if sel.any():
        pylab.ylim(0.95*spec_flux[sel].min(), 1.05*spec_flux[sel].max())
    pylab.xlabel("Wavelength (Angstrom)")
    pylab.ylabel("Flux (erg/s/cm$^2$/$\\AA$)")
    pylab.title("PSR J2241-5236 companion, predicted VLT/X-shooter spectrum at "
                "orbital phase {} ({:.1f} km/s/pixel, {})".format(
                    spec_phase, spec_delta_v, spec_where))
    pylab.tight_layout()

    pylab.show()
