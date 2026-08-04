import pathlib

__all__ = [
    "default_path",
]

default_path = pathlib.Path(__file__).parent.parent.parent.parent / "figures"
