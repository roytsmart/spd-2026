"""
The geometry shared by the figures which show the same patch of sky, so that
the sky does not move when one slide follows another.

The positions are given rather than worked out, because a layout which
arranges itself does so from what each figure happens to contain, and two
figures containing different things would not agree. These were originally
measured from a layout which arranged itself, so they are close to what one
would choose, but they are now the definition.
"""

import matplotlib.axes
import named_arrays as na

__all__ = [
    "figsize_default",
    "bottom_default",
    "height_default",
    "rect_image_default",
    "rect_image_right_default",
    "rect_colorbar_default",
    "rect_spectrum_default",
    "rect_profile_default",
    "set_limits_sky",
]

#: The width and height of the figure in inches, the shape of a slide.
figsize_default = (13.33, 7.5)

#: The bottom of every panel, as a fraction of the figure.
#:
#: Shared by all of them, so that a panel which appears on one slide lines up
#: with the panels which were on the slide before it.
bottom_default = 0.1017

#: The height of every panel, as a fraction of the figure.
height_default = 0.8596

#: The position of the image of the sky, as a fraction of the figure.
rect_image_default = (0.2093, bottom_default, 0.3770, height_default)

#: The position of the color key, as a fraction of the figure.
rect_colorbar_default = (0.0647, bottom_default, 0.0242, height_default)

#: The position of a second image of the sky, beside the first, for the
#: figures which compare two pictures of the same patch.
#:
#: The same size as the first, so that neither picture is given more room
#: than the other, and far enough to the right to leave the first its
#: vertical axis.
rect_image_right_default = (0.6000, bottom_default, 0.3770, height_default)

#: The position of the spectrum along the slit, as a fraction of the figure.
rect_spectrum_default = (0.6135, bottom_default, 0.1548, height_default)

#: The position of the spectral line profile, as a fraction of the figure.
rect_profile_default = (0.7750, bottom_default, 0.1548, height_default)


def set_limits_sky(
    ax: "matplotlib.axes.Axes",
    position: "na.AbstractCartesian2dVectorArray",
) -> None:
    """
    Give the axes exactly the extent of the sky.

    Left to itself an axes takes its extent from whatever was drawn on it,
    which is not the same thing: an axes sharing its vertical extent with a
    plot of a single slit ends up with the extent of that slit rather than of
    the whole raster. Since the image is drawn to scale, an extent which is
    off by a percent moves and resizes the image by a percent, which is
    visible when one figure follows another.

    Parameters
    ----------
    ax
        The axes to set the extent of.
    position
        The positions on the sky which the axes should cover.
    """
    ax.set_xlim(position.x.min().ndarray, position.x.max().ndarray)
    ax.set_ylim(position.y.min().ndarray, position.y.max().ndarray)
