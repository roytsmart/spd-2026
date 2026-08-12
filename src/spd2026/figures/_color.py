"""
The color scale shared by the figures which show a spectral cube as a
false-color image.

Keeping these here rather than in each figure means the figures can be read
against each other: a color in one is the same measurement as that color in
another.
"""

import matplotlib.pyplot as plt
import astropy.units as u

__all__ = [
    "velocity_color_default",
    "percentile_default",
    "cmap_line_default",
    "span_line_default",
    "colors_line",
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


#: The colormap the spectral lines are drawn from.
#:
#: Sampled in order of formation temperature, so that the color of a curve
#: says how hot the plasma it came from is rather than only telling one curve
#: from another.
#:
#: Chosen for how far apart five samples of it look, which is what a panel of
#: ten curves needs: the perceptually uniform maps take even steps, and even
#: steps across five samples are small ones, so their middle lines are hard to
#: tell apart where the curves run together. The price is that this map is not
#: perceptually uniform, and that a step in it is not a step in temperature.
cmap_line_default = "turbo"

#: The part of :obj:`cmap_line_default` to use.
#:
#: Short of both ends, which are dark enough to be hard to tell from each
#: other and from black.
span_line_default = (0.05, 0.95)


def colors_line(
    num: int,
    cmap: None | str = None,
    span: None | tuple[float, float] = None,
) -> list[tuple[float, float, float, float]]:
    """
    A color for each spectral line, in order of formation temperature.

    Parameters
    ----------
    num
        The number of lines to find colors for.
    cmap
        The colormap to take them from.
        If :obj:`None`, :obj:`cmap_line_default`.
    span
        The part of the colormap to use.
        If :obj:`None`, :obj:`span_line_default`.
    """
    if cmap is None:
        cmap = cmap_line_default
    if span is None:
        span = span_line_default

    colormap = plt.get_cmap(cmap)

    # One line would otherwise divide by zero, and belongs at the cool end.
    denominator = max(num - 1, 1)

    return [
        colormap(span[0] + (span[1] - span[0]) * i / denominator) for i in range(num)
    ]
