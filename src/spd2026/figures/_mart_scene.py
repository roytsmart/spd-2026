"""
What MART recovers, beside what there was to recover.
"""

import pathlib
import numpy as np
import matplotlib.animation
import matplotlib.artist
import matplotlib.axes
import matplotlib.ticker
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization
import named_arrays as na
from .._observations import scene_esis
from .._degraded import scene_degraded
from .._inversions import InversionSim, inversion_sim
from ._color import velocity_color_default, percentile_default
from ._layout import (
    figsize_default,
    rect_image_default,
    rect_image_right_default,
    rect_colorbar_default,
    set_limits_sky,
)
from ._path import default_path

__all__ = [
    "mart_scene",
]


def mart_scene(
    inversion: "None | InversionSim" = None,
    velocity_color: u.Quantity = velocity_color_default,
    percentile: float = percentile_default,
    figsize: tuple[float, float] = figsize_default,
    dpi: float = 150,
    fps: int = 24,
    fps_video: int = 24,
    suffix: str = ".mp4",
    path: None | pathlib.Path = None,
) -> pathlib.Path:
    """
    Watch MART recover the scene, beside the scene it is recovering.

    The left panel is the synthetic scene on the grid ESIS can distinguish,
    which is the best any inversion of these images could do. The right panel
    is what MART has recovered, one frame per iteration. What is still
    different between them when the movie stops is what the inversion did not
    get, as opposed to what the instrument was never able to record.

    Both panels are the same patch of sky on the same grid, colored the same
    way and scaled the same way, so the only thing which can differ between
    them is the scene itself.

    The left panel sits where :func:`spd2026.figures.iris_ee` and
    :func:`spd2026.figures.blink` put the sky, so this figure can follow them
    on a slide without the sky moving.

    Parameters
    ----------
    inversion
        The inversion to show.
        If :obj:`None`, :func:`spd2026.inversion_sim`, the one which was not
        told the answer.
    velocity_color
        The Doppler velocity mapped to each end of the visible spectrum,
        which is the range used by the other figures, so that a color means
        the same thing in each of them.
    percentile
        The percentile of the scene placed at the top of the brightness
        scale, separately at each wavelength.
    figsize
        The width and height of the figure in inches.
    dpi
        The resolution of the saved animation in dots per inch.
    fps
        The number of iterations shown per second.
    fps_video
        The frame rate of the file itself, for the movie formats.
        Players have trouble with a movie of only a few frames, so each frame
        is repeated as many times as it takes to reach this rate.
        Ignored for a GIF, which records how long to hold each frame.
    suffix
        The file type of the animation, either ``".mp4"`` or ``".gif"``.
        Ignored if `path` is given.
    path
        The location to save the animation.
        If :obj:`None`, it is saved alongside the other figures.
    """
    degraded = scene_degraded()
    inv = inversion_sim() if inversion is None else inversion

    solutions = inv.solutions
    axis_iteration = inv.axis_iteration
    axis_velocity = "velocity"
    axis_xy = ("detector_x", "detector_y")

    coordinates = degraded.inputs
    velocity = coordinates.velocity
    position = coordinates.position

    # Both panels are held to the extent of the scene rather than of what is
    # drawn on them, which is what keeps this figure registered against the
    # others, see :mod:`._layout`.
    position_limit = scene_esis().inputs.position

    # The scene is the reference, so its scale is the one both panels use.
    # Fixed for every frame, since a scale worked out per iteration would
    # stretch each one to fill the same range and hide the convergence.
    vmax = np.nanpercentile(degraded.outputs, percentile, axis=axis_xy)

    chi_squared = inv.mean_chi_squared.mean("channel")

    if path is None:
        path = default_path / f"mart-scene{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)

    with astropy.visualization.quantity_support():

        fig = plt.figure(figsize=figsize)
        ax_scene = fig.add_axes(rect_image_default)
        ax_mart = fig.add_axes(rect_image_right_default)
        cax = fig.add_axes(rect_colorbar_default)
        cax_twin = cax.twinx()

        def draw(ax, outputs):
            """Draw one picture of the sky, and say what its colors mean."""
            return na.plt.rgbmesh(
                velocity,
                position.x,
                position.y,
                C=outputs,
                axis_wavelength=axis_velocity,
                ax=ax,
                vmin=0,
                vmax=vmax,
                wavelength_min=-velocity_color,
                wavelength_max=+velocity_color,
            )

        # The scene does not change from one iteration to the next, so it and
        # the key are drawn once rather than for every frame.
        colorbar = draw(ax_scene, degraded.outputs)

        unit = na.unit(colorbar.inputs.x)
        equivalency = u.doppler_optical(coordinates.wavelength_rest)
        unit_wavelength = na.unit(coordinates.wavelength_rest)

        na.plt.pcolormesh(
            colorbar.inputs.x / unit,
            colorbar.inputs.y.to(unit_wavelength, equivalencies=equivalency),
            C=colorbar.outputs,
            axis_rgb=axis_velocity,
            ax=cax,
        )
        na.plt.pcolormesh(
            colorbar.inputs.x / unit,
            colorbar.inputs.y,
            C=colorbar.outputs,
            axis_rgb=axis_velocity,
            ax=cax_twin,
        )

        cax.set_ylabel(f"wavelength ({unit_wavelength:latex_inline})")
        cax.set_ylim(
            (-velocity_color).to(unit_wavelength, equivalencies=equivalency),
            (+velocity_color).to(unit_wavelength, equivalencies=equivalency),
        )
        cax.set_xlim(0, (colorbar.inputs.x.max() / unit).ndarray)
        cax.set_xlabel(f"{unit:latex_inline}")
        cax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=2))
        cax.tick_params(axis="x", labelrotation=45, labelsize="small")

        cax_twin.set_ylabel(f"velocity ({velocity_color.unit:latex_inline})")
        cax_twin.set_ylim(-velocity_color, +velocity_color)

        for ax in (ax_scene, ax_mart):
            set_limits_sky(ax, position_limit)
            ax.set_aspect("equal")
            ax.set_xlabel(f"helioprojective $x$ ({position.x.unit:latex_inline})")

        ax_scene.set_ylabel(f"helioprojective $y$ ({position.y.unit:latex_inline})")
        ax_scene.set_title("scene, at the resolution ESIS can distinguish")

        # The right panel is beside the left one and reads against it, so it
        # does not repeat its vertical axis.
        ax_mart.tick_params(axis="y", labelleft=False)

        if path.suffix == ".gif":
            repeat = 1
            writer = matplotlib.animation.PillowWriter(fps=fps)
        else:
            repeat = max(round(fps_video / fps), 1)
            writer = matplotlib.animation.FFMpegWriter(
                fps=fps * repeat,
                codec="h264",
                # `yuv420p` rather than the full color resolution of
                # `yuv444p`, which needs a profile PowerPoint cannot open.
                extra_args=["-pix_fmt", "yuv420p", "-crf", "14"],
            )

        transient = []
        drawn = [-1]

        def func(index: int) -> list[matplotlib.artist.Artist]:

            iteration = index // repeat
            if iteration == drawn[0]:
                return transient
            drawn[0] = iteration

            for artist in transient:
                artist.remove()
            transient.clear()

            draw(ax_mart, solutions.outputs[{axis_iteration: iteration}])
            transient.extend(ax_mart.collections)

            for artist in transient:
                artist.set_rasterized(True)

            # Reset every frame, since drawing on the axes widens them and
            # the two panels have to keep exactly the same extent.
            set_limits_sky(ax_mart, position_limit)

            chi = chi_squared[{axis_iteration: iteration}].ndarray
            ax_mart.set_title(
                f"MART iteration {iteration + 1}, "
                rf"$\langle \chi^2 \rangle = {chi:0.3f}$"
            )

            return transient

        for artist in [*ax_scene.collections, *cax.collections, *cax_twin.collections]:
            artist.set_rasterized(True)

        ani = matplotlib.animation.FuncAnimation(
            fig=fig,
            func=func,
            frames=solutions.outputs.shape[axis_iteration] * repeat,
        )

        ani.save(
            filename=path,
            writer=writer,
            dpi=dpi,
        )

    plt.close(fig)

    return path


if __name__ == "__main__":

    print(mart_scene())
