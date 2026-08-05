"""
Backprojections of the synthetic ESIS images onto the sky.
"""

import named_arrays as na
from ._caching import memory
from ._observations import scene_esis
from ._instruments import instrument
from ._images import images_simulated

__all__ = [
    "backprojections_simulated",
]


@memory.cache
def backprojections_simulated() -> na.FunctionArray[
    na.SpectralPositionalVectorArray,
    na.ScalarArray,
]:
    """
    Where ESIS disperses each wavelength to on the sky.

    The scene is imaged one wavelength at a time rather than summed over
    wavelength, and each of those images is then projected back onto the sky
    as though its light had all been at the same wavelength.
    Every image therefore lands where the instrument disperses that
    wavelength to, rather than where it came from, and the spread of the
    images across the sky is the dispersion of the instrument.

    The wavelength of each image is restored afterwards, so that the result
    still knows which wavelength each one belongs to and can be colored by it.

    The result is a spectral radiance in the same units as the scene it came
    from, so the two can be compared directly.
    """
    scene = scene_esis()

    system = instrument().system.linearize()

    images = images_simulated()

    # One image per wavelength, instead of one image summed over wavelength.
    image = system.image(scene, integrate=False)

    # Give every wavelength the same coordinate, so that the backprojection
    # cannot undo the dispersion it is meant to show.
    wavelength = image.inputs.wavelength
    image.inputs.wavelength = images.inputs.wavelength

    # The sky is likewise a single wavelength wide.
    coordinates = scene.inputs.replace(wavelength=images.inputs.wavelength)

    # Left to itself the backprojection comes back in photon units, which are
    # the natural units of the sensor. Asking for the units of the scene makes
    # the two directly comparable.
    result = system.backproject(
        image,
        coordinates.spectral_positional,
        unit=scene.outputs.unit,
    )

    result.inputs.wavelength = wavelength

    return result
