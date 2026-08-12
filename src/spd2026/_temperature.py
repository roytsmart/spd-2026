"""
The temperature at which each of the lines ESIS recovered is formed.

Kept here rather than in each figure so that the figures order their lines
the same way as each other, and so that the numbers can be argued with in one
place.

These are not in :mod:`esis`, which records a rest wavelength, a radiance and
a width for each line but not a temperature, so they are written down here
until they are.
"""

from collections.abc import Sequence
import astropy.units as u

__all__ = [
    "temperature_line",
    "order_temperature",
]

#: The temperature at which each line is formed, keyed by the label the
#: Level-4 product gives it.
#:
#: The temperature of the peak of the ionization fraction in equilibrium, as
#: tabulated by the Chianti Atomic Database, which is the same source the
#: rest wavelengths in :mod:`esis` are calculated from. The values for
#: O III and O IV are the ones already asserted in the docstrings of
#: :mod:`esis.flights.f1.spectrum`, so at least those two agree with the
#: package by construction.
#:
#: He I is the weakest of them. Helium in the chromosphere is not in
#: ionization equilibrium, being ionized by coronal radiation rather than by
#: the local temperature, so a single formation temperature for it is a
#: convention for placing it below the transition-region lines rather than a
#: statement about where the line is formed.
temperature_line = {
    "He I 584": 10**4.3 * u.K,
    "O III 600": 10**4.95 * u.K,
    "O IV 608+610": 10**5.18 * u.K,
    "O V 630": 10**5.4 * u.K,
    "Mg X 610+625": 10**6.05 * u.K,
}


def order_temperature(label: Sequence[str]) -> list[int]:
    """
    The order which puts a set of lines from coolest to hottest.

    Given as the indices to take rather than as the labels themselves, since
    what has to be reordered alongside them is everything else the figure
    holds one of per line.

    Parameters
    ----------
    label
        The labels of the lines to order, as the Level-4 product gives them.
    """
    missing = [x for x in label if x not in temperature_line]
    if missing:  # pragma: nocover
        raise ValueError(
            f"no formation temperature is recorded for {missing}, "
            f"add it to `spd2026._temperature.temperature_line`"
        )

    return sorted(range(len(label)), key=lambda i: temperature_line[label[i]])
