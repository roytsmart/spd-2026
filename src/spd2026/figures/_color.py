"""
The color scale shared by the figures which show a spectral cube as a
false-color image.

Keeping these here rather than in each figure means the figures can be read
against each other: a color in one is the same measurement as that color in
another.
"""

import astropy.units as u

__all__ = [
    "velocity_color_default",
    "percentile_default",
]

#: The Doppler velocity mapped to each end of the visible spectrum, so that
#: green is light at rest, red is moving away, and blue is moving toward us.
velocity_color_default = 150 * u.km / u.s

#: The percentile of the signal placed at the top of the brightness scale.
#: A percentile rather than a maximum, so that a few bright pixels cannot
#: darken everything else.
#:
#: The brightness is scaled separately at each wavelength, so that the faint
#: light in the wings of a line is not lost against the bright light at its
#: center.
percentile_default = 99.5
