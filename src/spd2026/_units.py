"""
The units shared by the observations and the figures which draw them.

Written down here rather than taken from whatever each observation happens
to arrive in, since two figures on consecutive slides which label the same
quantity differently read as two different measurements.
"""

import astropy.units as u

__all__ = [
    "unit_radiance",
]

#: The spectral radiance every observation is converted to.
#:
#: Per angstrom rather than per nanometer, which is what IRIS gives back,
#: because the lines in question are a fraction of an angstrom wide and
#: because the synthetic ESIS scene is already in these units.
unit_radiance = u.erg / (u.AA * u.s * u.sr * u.cm**2)
