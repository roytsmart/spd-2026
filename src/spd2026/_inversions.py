"""
The scenes recovered from the synthetic ESIS images.
"""

import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import ctis
import esis
from ._caching import memory
from ._observations import scene_esis
from ._instruments import instrument
from ._images import images_simulated
from ._degraded import coordinates_degraded

__all__ = [
    "InversionSim",
    "inversion_sim",
]


@dataclasses.dataclass(eq=False, repr=False)
class InversionSim:
    """
    What is worth keeping of an inversion.

    The result which :mod:`ctis` gives back holds the inverter which produced
    it, and through it the model of the instrument, which cannot be stored and
    read back again: https://github.com/sun-data/named-arrays/issues/199.
    This holds the parts which can.
    """

    solutions: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray]
    """The recovered scene at every iteration."""

    mean_chi_squared: na.ScalarArray
    """How far the images predicted by each iteration were from the real ones."""

    correlation_residual: na.ScalarArray
    """How much of what each iteration could not explain looks like signal."""

    success: bool
    """Whether the inversion converged."""

    message: str
    """What the inversion had to say for itself."""

    axis_iteration: str = "iteration"
    """The name of the logical axis along which the iterations lie."""

    @property
    def solution(self) -> na.FunctionArray:
        """The recovered scene, which is the last iteration."""
        return self.solutions[{self.axis_iteration: ~0}]

    @property
    def iteration(self) -> na.ScalarArray:
        """The number of each iteration."""
        return na.arange(
            start=0,
            stop=self.solutions.shape[self.axis_iteration],
            axis=self.axis_iteration,
        )


@memory.cache
def inversion_sim(
    velocity_max: u.Quantity = 250 * u.km / u.s,
    num_iteration: int = 50,
    threshold_convergence: float = 1e-6,
    width_guess: u.Quantity = esis.flights.f1.spectrum.O_V.width_doppler,
    intermediate: bool = True,
) -> InversionSim:
    """
    The scene recovered from the synthetic ESIS images using MART.

    The four images are inverted together into one spectral cube, the same
    way the ``mart-iris`` notebook in :mod:`esis` inverts a scene made from
    an IRIS raster.

    The scene is recovered onto a grid of the instrument's own making rather
    than onto the grid of the scene which produced the images. A cell of that
    grid is one plate scale across and one dispersion wide, so the grid is as
    fine as the instrument can distinguish and no finer, which is both what
    can honestly be asked of an inversion and far cheaper than the grid of
    the raster.

    The guess the algorithm starts from says as little as it can: the same
    Gaussian line, at rest and as wide as the line is measured to be, at
    every place on the sky. Only its total is taken from the images, being
    the light the four of them hold between them. This is the guess the
    flight data has been inverted with.

    A guess built from the images, such as the faintest of the four projected
    back, starts nearer the answer and converges sooner, but then every
    structure in the recovered scene has to be argued not to have been put
    there by the guess. A flat field cannot be accused of that: whatever
    structure comes back was found by the inversion.

    Parameters
    ----------
    velocity_max
        The Doppler velocity at each end of the recovered spectrum.
    num_iteration
        The greatest number of iterations to take, if the inversion has not
        stopped improving before then.
    threshold_convergence
        How much better an iteration has to be than the one before it for
        another to be worth taking.

        This is a threshold on the *improvement* in the reduced chi squared
        from one iteration to the next, not on the chi squared itself, so a
        smaller number means the inversion is allowed to keep going while it
        is still gaining a little, rather than a better fit being demanded of
        it.
    width_guess
        The standard deviation of the Gaussian the guess starts from.
        The measured Doppler width of the line itself by default, which is
        the width the flight data has been inverted with.

        A shape with tails, rather than one which reaches zero, because MART
        multiplies: a cell where the guess is zero stays zero however many
        iterations are taken, and the fast wings of an explosive event are
        the thing being looked for.
    intermediate
        Whether to keep every iteration rather than only the last, which is
        what makes it possible to see whether the inversion converged.
    """
    if width_guess is None:
        width_guess = velocity_max

    scene = scene_esis()
    images = images_simulated()

    optics = instrument()
    system = optics.system.linearize()

    axis_wavelength = scene.axis_wavelength
    axis_x = scene.axis_detector_x
    axis_y = scene.axis_detector_y
    axis_channel = optics.axis_channel

    # The grid the instrument can distinguish, which is also the grid
    # :func:`spd2026.scene_degraded` puts the scene on, so that what is
    # recovered here can be compared with what was there to recover.
    coordinates = coordinates_degraded(
        velocity_max=velocity_max,
        system=system,
    )

    instrument_ctis = ctis.instruments.OptikaInstrument(
        system=system,
        coordinates_scene=coordinates,
        channel="Channel " + optics.camera.channel.astype(str),
        axis_channel=axis_channel,
        axis_wavelength=axis_wavelength,
        axis_scene_xy=(axis_x, axis_y),
    )

    unit = na.unit(scene.outputs)
    backprojection = instrument_ctis.backproject(images, unit=unit)

    # The shape of the grid the scene is recovered on, taken from a
    # backprojection, which is already on it.
    shape = {
        axis: num
        for axis, num in backprojection.outputs.shape.items()
        if axis != axis_channel
    }

    # Where the guess puts the light in wavelength: a line at rest, as wide
    # as the line is measured to be.
    velocity = coordinates.velocity.cell_centers(axis_wavelength)
    guess_spectral = np.exp(-np.square(velocity / width_guess) / 2)
    guess_spectral = guess_spectral / guess_spectral.sum()

    # How much light there is to put, which each image holds a copy of.
    guess_total = (
        backprojection.outputs.sum().to(unit) / images.outputs.shape[axis_channel]
    )

    # Spread evenly across the sky, so that the guess says nothing about
    # where the light is and every structure in the recovered scene has come
    # from the images rather than from where the inversion started.
    guess = guess_spectral * guess_total / (shape[axis_x] * shape[axis_y])
    guess = na.broadcast_to(guess, shape)
    guess = np.maximum(guess, 0 * unit)

    inverter = ctis.inverters.MartInverter(
        instrument=instrument_ctis,
        num_iteration=num_iteration,
        threshold_convergence=threshold_convergence,
        intermediate=intermediate,
    )

    result = inverter(images, guess=guess)

    return InversionSim(
        solutions=result.solutions,
        mean_chi_squared=result.mean_chi_squared,
        correlation_residual=result.correlation_residual,
        success=result.success,
        message=result.message,
        axis_iteration=inverter.axis_iteration,
    )
