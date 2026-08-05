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

    The cosmic ray spikes have been removed from the signal.

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

    return result[{result.axis_wavelength: slice_wavelength}]


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
    scene = scene[{scene.axis_time: 0}]
    scene.timedelta = scene.timedelta[{scene.axis_time: 0}]
    return scene
