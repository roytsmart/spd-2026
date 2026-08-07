"""
The explosive event which the figures are about.

Written down here rather than in each figure, since a figure of the raster
and a figure of the inversion which quietly came to be about different
events would still look right, and the whole point of putting them on
consecutive slides is that they are about the same one.

Given as a place on the sky rather than as a cell, because the raster and
the recovered cube are on different grids: each figure takes the cell of its
own grid nearest to this, which is as close as the two can be brought.
"""

import astropy.units as u

__all__ = [
    "x_event_default",
    "y_event_default",
]

#: The horizontal position of the explosive event, measured from the center
#: of the field.
x_event_default = -43 * u.arcsec

#: The vertical position of the explosive event.
y_event_default = -30.4 * u.arcsec
