"""
The instrument models used by the figures in this presentation.
"""

import named_arrays as na
import esis

__all__ = [
    "instrument",
]


def instrument(
    num_wavelength: int = 5,
    axis_wavelength: str = "fit-wavelength",
    num_distribution: int = 0,
) -> esis.optics.Instrument:
    """
    A model of the ESIS instrument as it flew in 2019.

    This is :func:`esis.flights.f1.optics.distortion_fit`, the optical design
    with the distortion parameters measured from the flight images, with one
    change: the grid of wavelengths traced through the system is replaced by
    an evenly-spaced grid spanning the passband.

    By default that grid is the three spectral lines the instrument was built
    to observe, which is what is wanted when studying those lines, but not
    when imaging a synthetic scene which is continuous in wavelength.
    The replacement grid also has its own axis name, so that it does not
    collide with the wavelength axis of the scene being imaged.

    The passband is different for each channel, so the ends of the grid are
    averaged over the channels, giving every channel the same grid.

    Parameters
    ----------
    num_wavelength
        The number of wavelengths in the grid.
    axis_wavelength
        The name of the logical axis corresponding to changing wavelength.
    num_distribution
        The number of Monte Carlo samples to draw when computing the
        uncertainty of the model.
        Zero, the default, gives a model without uncertainties.
        The measured parameters do have uncertainties, but propagating them
        costs this many times as much memory and time as not, so it is left
        to the caller to ask for.
    """
    result = esis.flights.f1.optics.distortion_fit(
        num_distribution=num_distribution,
    )

    result.wavelength = na.linspace(
        start=result.wavelength_min.mean(),
        stop=result.wavelength_max.mean(),
        axis=axis_wavelength,
        num=num_wavelength,
    )

    return result
