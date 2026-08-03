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

import Icarus
from Icarus.Utils.import_modules import *


##### Data and atmosphere grid description files (see module docstring).
data_fln = 'data_J2241-5236.txt'
atmo_fln = 'atmo_models_J2241-5236.txt'
ndiv = 5


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
Tday = 10000.                          # approximate irradiated (day-side) temperature, in K


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


##### Plotting the observed and modelled light curves, phase by phase.
if pylab:
    fit.Plot(par)
    pylab.show()
