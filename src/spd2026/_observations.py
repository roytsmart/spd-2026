"""
The observations used by the figures in this presentation.
"""

import dataclasses
import numpy.typing as npt
import astropy.units as u
import named_arrays as na
import iris
import esis
from ._caching import memory
from ._units import unit_radiance

__all__ = [
    "time_default",
    "window_default",
    "observation_iris",
    "scene_esis",
]

#: The time of the IRIS observation containing the explosive event.
time_default = "2013-10-22 11:30"

#: The IRIS spectral window containing the explosive event.
window_default = "Si IV 1394"


@memory.cache
def _outputs_iris_despiked(
    time: str,
    window: str,
) -> npt.NDArray:
    """
    The despiked signal of the IRIS raster, without units.

    Despiking the whole spectral window takes several minutes,
    so the result is cached.

    Only the signal is cached, and only its values, since the coordinates are
    cheap to load and a plain array is the one thing which can be both stored
    and memory-mapped reliably.

    Parameters
    ----------
    time
        The time of the observation to download.
    window
        The name of the spectral window to load.
    """
    result = iris.sg.open(
        time=time,
        window=window,
    )

    result = na.despike(
        array=result,
        axis=(result.axis_wavelength, result.axis_detector_y),
    )

    return result.outputs.ndarray.value


def observation_iris(
    time: str = time_default,
    window: str = window_default,
    slice_wavelength: slice = slice(750, 1250),
) -> iris.sg.SpectrographObservation:
    """
    The IRIS spectrograph raster containing the explosive event.

    The cosmic ray spikes have been removed from the signal, and the signal
    has been converted from instrument units into a spectral radiance, so
    that it can be compared with anything else measured in physical units.

    Parameters
    ----------
    time
        The time of the observation to download.
    window
        The name of the spectral window to load.
    slice_wavelength
        The range of wavelength pixels to keep,
        chosen to isolate the spectral line from the rest of the window.
    """
    result = iris.sg.open(
        time=time,
        window=window,
    )

    outputs = _outputs_iris_despiked(
        time=time,
        window=window,
    )

    result = dataclasses.replace(
        result,
        outputs=na.ScalarArray(
            ndarray=outputs << na.unit(result.outputs),
            axes=result.outputs.axes,
        ),
    )

    result = result[{result.axis_wavelength: slice_wavelength}]

    # The raster is a patch of sky far from the center of the disk, but the
    # instrument which observes the scene made from it is modeled as pointing
    # along its own boresight, so a scene left where it was found falls off
    # the edge of the detector. Placing the raster on the boresight is what
    # the ``mart-iris`` notebook in :mod:`esis` does, and doing it here means
    # the raster and everything derived from it share one set of coordinates.
    result.inputs.position = result.inputs.position - result.inputs.position.mean()

    result = result.radiance

    # IRIS gives a radiance per nanometer and the synthetic scene is per
    # angstrom, which made two figures showing the same patch of sky label
    # their color keys differently. Converted here so that everything
    # downstream is on one scale, see :mod:`._units`.
    return dataclasses.replace(
        result,
        outputs=result.outputs.to(unit_radiance),
    )


@memory.cache
def scene_esis(
    time: str = time_default,
    window: str = window_default,
    **kwargs,
) -> na.FunctionArray[
    na.TemporalDopplerPositionalVectorArray,
    na.ScalarArray,
]:
    r"""
    The same raster as :func:`observation_iris`, as a synthetic ESIS scene.

    The IRIS raster is shifted and scaled onto the
    :math:`\text{O\,V}\;630\,\AA` line, so that it represents what ESIS would
    have seen looking at the same piece of the Sun.
    See :func:`esis.flights.f1.data.synth.scene_iris` for the details.

    Despiking happens inside that function, so the result is cached here
    rather than reusing the despiked raster from :func:`observation_iris`.

    Parameters
    ----------
    time
        The time of the observation to download.
    window
        The name of the spectral window to load.
    kwargs
        Additional keyword arguments passed to
        :func:`esis.flights.f1.data.synth.scene_iris`.
    """
    scene = esis.flights.f1.data.synth.scene_iris(
        time_start=time,
        window=window,
        velocity_max=250 * u.km / u.s,
        **kwargs,
    )

    # Placed on the boresight for the same reason as in
    # :func:`observation_iris`, and repeated here rather than taken from it
    # because `scene_iris` opens its own copy of the raster. The two start
    # from the same coordinates, so subtracting each mean leaves them on the
    # same ones.
    scene.inputs.position = scene.inputs.position - scene.inputs.position.mean()

    # `scene_iris` hands back a radiance per nanometer, which is what
    # :mod:`iris` reports and what it now passes through, while the raster
    # this is compared against is per angstrom. Converted here for the same
    # reason as in :func:`observation_iris`, see :mod:`._units`.
    scene.outputs = scene.outputs.to(unit_radiance)

    scene = scene[{scene.axis_time: 0}]
    scene.timedelta = scene.timedelta[{scene.axis_time: 0}]
    return scene
