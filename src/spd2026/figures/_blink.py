"""
A blink comparison of the observed scene and the scene ESIS recovers from it.
"""

import pathlib
import dataclasses
import numpy as np
import matplotlib.animation
import matplotlib.artist
import matplotlib.colorbar
import matplotlib.ticker
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization
import named_arrays as na
from .._observations import observation_iris, scene_esis
from .._backprojections import backprojections_simulated
from ._color import velocity_color_default, percentile_default
from ._layout import (
    figsize_default,
    rect_image_default,
    rect_colorbar_default,
    set_limits_sky,
)
from ._path import default_path

__all__ = [
    "blink",
]


def _crop(
    a: na.FunctionArray,
    axis: str,
    velocity: na.AbstractScalar,
    velocity_limit: u.Quantity,
) -> na.FunctionArray:
    """Restrict `a` to the cells whose center is within `velocity_limit`."""
    center = velocity.cell_centers(axis)

    where_upper = +velocity_limit < center
    where_lower = -velocity_limit < center

    index_lower = np.nanargmax(where_lower)[axis].ndarray

    if where_upper.any():
        index_upper = np.nanargmax(where_upper)[axis].ndarray
    else:  # pragma: nocover
        index_upper = None

    return a[{axis: slice(index_lower, index_upper)}]


def blink(
    velocity_limit: u.Quantity = 250 * u.km / u.s,
    velocity_color: u.Quantity = velocity_color_default,
    percentile: float = percentile_default,
    figsize: tuple[float, float] = figsize_default,
    dpi: float = 150,
    fps: int = 2,
    fps_video: int = 24,
    suffix: str = ".mp4",
    path: None | pathlib.Path = None,
) -> pathlib.Path:
    """
    Blink between the observed scene and the scene ESIS recovers from it.

    The frames are, in order, the IRIS raster the scene was built from, the
    synthetic O V scene, and the scene backprojected from each of the four
    ESIS channels.
    Every frame is on the same patch of sky, so blinking between them shows
    what the instrument keeps and what it loses.

    The frames are colored by Doppler velocity rather than by wavelength,
    since the raster is a different spectral line from the rest, and are all
    converted to a spectral radiance at the same spectral resolution. The
    color scale is the one used by :func:`spd2026.figures.iris_ee`, so a
    color means the same thing in either figure.

    The brightness of each frame is scaled separately at each wavelength, so
    the blink compares structure rather than brightness.

    Parameters
    ----------
    velocity_limit
        The Doppler velocity range kept in every frame.
    velocity_color
        The Doppler velocity mapped to each end of the visible spectrum,
        which is the same as the range used by
        :func:`spd2026.figures.iris_ee`, so that the two figures can be read
        against each other.
    percentile
        The percentile of each frame placed at the top of its brightness
        scale, separately at each wavelength.
    figsize
        The width and height of the figure in inches.
    dpi
        The resolution of the saved animation in dots per inch.
    fps
        The number of times per second the frame changes, which is the rate
        of the blink.
    fps_video
        The frame rate of the file itself, for the movie formats.
        Players have trouble with a movie of only a few frames, so each frame
        is repeated as many times as it takes to reach this rate, which leaves
        the blink looking the same but gives the file an ordinary frame rate.
        Ignored for a GIF, which records how long to hold each frame.
    suffix
        The file type of the animation, either ``".mp4"`` or ``".gif"``.
        Ignored if `path` is given.
        The MP4 is smaller and does not quantize the color, which matters
        here since the color is the measurement, but it only loops if the
        program playing it is told to.
    path
        The location to save the animation.
        If :obj:`None`, it is saved alongside the other figures.
    """
    obs = observation_iris()
    scene = scene_esis()
    back = backprojections_simulated()

    axis_channel = "channel"
    axis_xy = ("detector_x", "detector_y")

    # The raster has a length-one time axis which the other frames do not.
    obs = obs[{obs.axis_time: 0}]

    # Each backprojection was made onto a sky one wavelength bin wide, so its
    # spectral radiance is the light of one wavelength spread over the whole
    # range. Scaling by the number of bins in the scene puts it back onto the
    # spectral resolution of the scene, which is what makes the radiance of
    # the two comparable.
    wavelength = back.inputs.wavelength.ndarray
    bins = float((wavelength.max() - wavelength.min()) / np.diff(wavelength).mean())
    back = dataclasses.replace(back, outputs=back.outputs * bins)

    # The backprojections are the only frames without a Doppler coordinate,
    # so it is computed from the rest wavelength of the line they represent.
    def velocity_of(a: na.FunctionArray) -> na.AbstractScalar:
        return a.inputs.wavelength.to(
            u.km / u.s,
            equivalencies=u.doppler_optical(scene.inputs.wavelength_rest),
        )

    obs = _crop(obs, obs.axis_wavelength, obs.inputs.velocity, velocity_limit)
    scene = _crop(scene, scene.axis_wavelength, scene.inputs.velocity, velocity_limit)
    back = _crop(back, "velocity", velocity_of(back), velocity_limit)

    velocity_back = velocity_of(back)

    # The rest wavelength travels with each frame, since the key gives the
    # wavelength as well as the velocity and the raster is a different line
    # from the rest.
    frames = [
        (
            "IRIS Si IV 1394",
            obs.axis_wavelength,
            obs.inputs.velocity,
            obs.inputs.wavelength_rest,
            obs.outputs,
        ),
        (
            "scene O V 630",
            scene.axis_wavelength,
            scene.inputs.velocity,
            scene.inputs.wavelength_rest,
            scene.outputs,
        ),
    ]
    for i in range(back.outputs.shape[axis_channel]):
        frames += [
            (
                f"channel {i + 1} backprojected",
                "velocity",
                velocity_back,
                scene.inputs.wavelength_rest,
                back.outputs[{axis_channel: i}],
            )
        ]

    position = scene.inputs.position

    with astropy.visualization.quantity_support():

        # Placed where the figure of the explosive event puts the same image
        # and the same key, so that the sky does not move between the two.
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(rect_image_default)
        cax = fig.add_axes(rect_colorbar_default)
        cax_twin = cax.twinx()

        drawn = [-1]

        def func(index: int) -> list[matplotlib.artist.Artist]:

            # Each frame is held for several frames of the file, and redrawing
            # it every time would cost as much again for no difference.
            index = index // repeat
            if index == drawn[0]:
                return [*ax.collections, *cax.collections, *cax_twin.collections]
            drawn[0] = index

            label, axis, velocity, wavelength_rest, outputs = frames[index]

            for artist in ax.collections:
                artist.remove()
            for artist in [*cax.collections, *cax_twin.collections]:
                artist.remove()

            colorbar = na.plt.rgbmesh(
                velocity,
                position.x,
                position.y,
                C=outputs,
                axis_wavelength=axis,
                ax=ax,
                vmin=0,
                vmax=np.nanpercentile(outputs, percentile, axis=axis_xy),
                wavelength_min=-velocity_color,
                wavelength_max=+velocity_color,
            )
            # Matplotlib remembers the units given to an axis, and the frames
            # arrive with equivalent but differently spelled units, so the unit
            # is divided out and named in the label instead.
            unit_radiance = na.unit(colorbar.inputs.x)

            # The key is drawn twice, once against the wavelength of the line
            # this frame belongs to and once against the Doppler velocity, so
            # that it carries both scales as the key of the explosive event
            # figure does.
            equivalency = u.doppler_optical(wavelength_rest)
            unit_wavelength = na.unit(wavelength_rest)

            na.plt.pcolormesh(
                colorbar.inputs.x / unit_radiance,
                colorbar.inputs.y.to(unit_wavelength, equivalencies=equivalency),
                C=colorbar.outputs,
                axis_rgb=axis,
                ax=cax,
            )
            na.plt.pcolormesh(
                colorbar.inputs.x / unit_radiance,
                colorbar.inputs.y,
                C=colorbar.outputs,
                axis_rgb=axis,
                ax=cax_twin,
            )

            cax.set_ylabel(f"wavelength ({unit_wavelength:latex_inline})")
            cax_twin.set_ylabel(f"velocity ({velocity_color.unit:latex_inline})")

            # The frames reach beyond the range which is given a color, so the
            # key is held to the range it describes, as `show` does.
            cax.set_ylim(
                (-velocity_color).to(unit_wavelength, equivalencies=equivalency),
                (+velocity_color).to(unit_wavelength, equivalencies=equivalency),
            )
            cax_twin.set_ylim(-velocity_color, +velocity_color)

            # Removing the last frame does not reset the limits, and autoscale
            # only ever widens them, so a frame fainter than the one before it
            # would be squeezed into the left edge unless they are set here.
            cax.set_xlim(0, (colorbar.inputs.x.max() / unit_radiance).ndarray)

            # The key is narrow, so it can only carry a couple of ticks, and
            # they have to be turned to keep them from running into each other.
            cax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=2))
            cax.tick_params(axis="x", labelrotation=45, labelsize="small")
            cax.set_xlabel(f"{unit_radiance:latex_inline}")

            for artist in [*ax.collections, *cax.collections, *cax_twin.collections]:
                artist.set_rasterized(True)

            set_limits_sky(ax, position)
            ax.set_aspect("equal")
            ax.set_xlabel(f"helioprojective $x$ ({position.x.unit:latex_inline})")
            ax.set_ylabel(f"helioprojective $y$ ({position.y.unit:latex_inline})")
            ax.set_title(label)

            return [*ax.collections, *cax.collections, *cax_twin.collections]

        if path is None:
            path = default_path / f"blink{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".gif":
            # A GIF stores how long to hold each frame, so one frame each is
            # all it needs.
            repeat = 1
            writer = matplotlib.animation.PillowWriter(fps=fps)
        else:
            # A movie of six frames at two per second is unusual enough that
            # players stumble over it, so each frame is held for as many
            # frames of the file as it takes to reach an ordinary frame rate.
            repeat = max(round(fps_video / fps), 1)
            writer = matplotlib.animation.FFMpegWriter(
                fps=fps * repeat,
                codec="h264",
                # `yuv444p` would keep the full resolution of the color, which
                # here is the measurement, but it needs a profile that
                # PowerPoint cannot open. `yuv420p` is the one everything
                # plays, so the quality is bought back with a low `crf`
                # instead.
                extra_args=["-pix_fmt", "yuv420p", "-crf", "14"],
            )

        ani = matplotlib.animation.FuncAnimation(
            fig=fig,
            func=func,
            frames=len(frames) * repeat,
        )

        ani.save(
            filename=path,
            writer=writer,
            dpi=dpi,
        )

    plt.close(fig)

    return path
