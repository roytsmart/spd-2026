"""
Synthetic ESIS images of the observations in this presentation.
"""

import named_arrays as na
from ._caching import memory
from ._observations import scene_esis
from ._instruments import instrument

__all__ = [
    "images_simulated",
]


@memory.cache
def images_simulated() -> na.FunctionArray[
    na.SpectralPositionalVectorArray,
    na.ScalarArray,
]:
    """
    The synthetic ESIS images of :func:`spd2026.scene_esis`.

    This is what ESIS would have recorded if it had been pointed at the piece
    of the Sun that IRIS observed.
    Each channel disperses the scene along a different direction, so the four
    images together are the projections that an inversion has to work from.

    The scene is imaged using a linearized version of
    :func:`spd2026.instrument`, which images scenes by conservative
    regridding rather than by raytracing each one. Building that linear model
    is the expensive part, which is why the result is cached.

    The spectral axis is integrated over, so the result is one image per
    channel, in electrons.
    It includes noise, which means the cached result is one particular
    realization of it, and that pixels which saw nothing are as often
    negative as positive.
    """
    scene = scene_esis()

    system = instrument().system.linearize()

    return system.image(scene)
