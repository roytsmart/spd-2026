"""
The observations used by the figures in this presentation.
"""

import dataclasses
import numpy.typing as npt
import named_arrays as na
import iris
from ._caching import memory

__all__ = [
    "observation_iris",
]


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
    time: str = "2013-10-22 11:30",
    window: str = "Si IV 1394",
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
