from ._observations import (
    time_default,
    window_default,
    observation_iris,
    scene_esis,
)
from ._instruments import instrument
from ._images import images_simulated
from . import figures

__all__ = [
    "time_default",
    "window_default",
    "observation_iris",
    "scene_esis",
    "instrument",
    "images_simulated",
    "figures",
]
