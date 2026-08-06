"""
The scene as seen at the resolution the instrument can actually distinguish.
"""

import astropy.units as u
import named_arrays as na
import optika
import ctis
from ._caching import memory
from ._observations import scene_esis
from ._instruments import instrument

__all__ = [
    "coordinates_degraded",
    "scene_degraded",
]


def coordinates_degraded(
    velocity_max: u.Quantity = 250 * u.km / u.s,
    system: "None | optika.systems.AbstractSequentialSystem" = None,
) -> na.TemporalDopplerPositionalVectorArray:
    """
    The grid the instrument can distinguish, as the vertices of its cells.

    A cell of this grid is one plate scale across and one dispersion wide,
    which is as fine as ESIS can resolve and no finer. It is the grid an
    inversion recovers the scene onto, and therefore the grid the scene has
    to be put on before the two can be compared.

    The plate scale and the dispersion are read off the linearized model of
    the instrument rather than given, so that they follow the model.

    Parameters
    ----------
    velocity_max
        The Doppler velocity at each end of the spectrum.
    system
        The linearized model of the instrument to read the plate scale and
        the dispersion from.
        If :obj:`None`, the model of :func:`spd2026.instrument` is
        linearized, which takes a while, so a caller which has already
        linearized one should hand it over.

    Examples
    --------

    The size of a cell of the grid.

    .. jupyter-execute::

        import numpy as np
        import spd2026

        coordinates = spd2026.coordinates_degraded()

        velocity = coordinates.velocity
        position = coordinates.position

        print(np.diff(velocity.ndarray).mean())
        print(np.diff(position.x.ndarray).mean())
    """
    scene = scene_esis()

    if system is None:
        system = instrument().system.linearize()

    axis_wavelength = scene.axis_wavelength
    axis_x = scene.axis_detector_x
    axis_y = scene.axis_detector_y

    # How far apart two wavelengths, and two places, have to be for the
    # instrument to put them in different pixels.
    coefficients = system.distortion.fit.coefficients
    wavelength_rest = scene.inputs.wavelength_rest

    dispersion = 1 / coefficients.components["wavelength"].length.ndarray.mean()
    dispersion = (wavelength_rest + dispersion * u.pix).to(
        u.km / u.s,
        equivalencies=u.doppler_optical(wavelength_rest),
    )
    dispersion = dispersion / u.pix

    plate_scale = 1 / coefficients.components["position.x"].length.ndarray.mean()
    plate_scale = plate_scale.to(u.arcsec / u.pix)

    return na.TemporalDopplerPositionalVectorArray.from_velocity(
        # Each step of the raster was taken at a different time, and the scene
        # they make together is treated as though it were taken at once.
        time=scene.inputs.time.mean(axis_x),
        velocity=ctis.arange(
            start=-velocity_max,
            stop=+velocity_max,
            step=dispersion * u.pix,
            axis=axis_wavelength,
        ),
        wavelength_rest=wavelength_rest,
        position=ctis.arange(
            start=scene.inputs.position.min(),
            stop=scene.inputs.position.max(),
            step=plate_scale * u.pix,
            axis=na.Cartesian2dVectorArray(x=axis_x, y=axis_y),
        ),
    )


@memory.cache
def scene_degraded(
    velocity_max: u.Quantity = 250 * u.km / u.s,
) -> na.FunctionArray[
    na.TemporalDopplerPositionalVectorArray,
    na.ScalarArray,
]:
    """
    The synthetic ESIS scene on the grid the instrument can distinguish.

    This is what an inversion could recover if it were perfect: the scene of
    :func:`spd2026.scene_esis` with nothing taken away but the resolution.
    Comparing a recovered scene against this one rather than against the
    original separates what the inversion failed to recover from what no
    instrument with this plate scale and this dispersion could have recorded
    in the first place.

    The resampling is conservative and treats the scene as the density it is,
    so the light in the scene is neither created nor destroyed by moving it
    onto a coarser grid.

    Parameters
    ----------
    velocity_max
        The Doppler velocity at each end of the spectrum.
    """
    scene = scene_esis()
    coordinates = coordinates_degraded(velocity_max=velocity_max)

    outputs = ctis.regrid(
        # The grids carry a time as well as a wavelength and a position, and
        # the resampling is only over the last two, so each is asked for the
        # spectral and positional part of itself.
        coordinates_input=scene.inputs.spectral_positional,
        coordinates_output=coordinates.spectral_positional,
        values_input=scene.outputs,
        axis_wavelength=scene.axis_wavelength,
        axis_position=(scene.axis_detector_x, scene.axis_detector_y),
    )

    return na.FunctionArray(
        inputs=coordinates,
        outputs=outputs,
    )
