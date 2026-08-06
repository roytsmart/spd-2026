"""
A blink between the scene and what each ESIS channel makes of it.
"""

import pathlib
import dataclasses
import numpy as np
import matplotlib.animation
import matplotlib.artist
import matplotlib.ticker
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization
import named_arrays as na
from .._observations import scene_esis
from .._backprojections import backprojections_simulated
from ._blink import _crop
from ._color import velocity_color_default, percentile_default
from ._layout import (
    figsize_default,
    rect_image_default,
    rect_colorbar_default,
    set_limits_sky,
)
from ._path import default_path

__all__ = [
    "blink_channels",
]


def blink_channels(
    velocity_limit: u.Quantity = 250 * u.km / u.s,
    velocity_color: u.Quantity = velocity_color_default,
    percentile: float = percentile_default,
    cmap: str = "gray",
    figsize: tuple[float, float] = figsize_default,
    dpi: float = 150,
    fps: int = 2,
    fps_video: int = 24,
    suffix: str = ".mp4",
    path: None | pathlib.Path = None,
) -> pathlib.Path:
    """
    Blink between the scene and what each ESIS channel makes of it.

    The first frame is the synthetic scene, colored by Doppler velocity.
    The rest are the scene projected back from each of the four channels,
    in gray.

    A channel records one image, and cannot tell light which arrived from
    one place at one wavelength from light which arrived from another place
    at another. What one of them gives back therefore has no color to it,
    and the blink is the difference between what is there and what any one
    channel can say about it.

    The figure is laid out like :func:`spd2026.figures.iris_ee` and
    :func:`spd2026.figures.blink`, so that the sky does not move between
    them.

    Parameters
    ----------
    velocity_limit
        The Doppler velocity range kept in every frame.
    velocity_color
        The Doppler velocity mapped to each end of the visible spectrum in
        the first frame.
    percentile
        The percentile of each frame placed at the top of its brightness
        scale.
    cmap
        The colormap of the frames which have no color of their own.
    figsize
        The width and height of the figure in inches.
    dpi
        The resolution of the saved animation in dots per inch.
    fps
        The number of times per second the frame changes, which is the rate
        of the blink.
    fps_video
        The frame rate of the file itself, for the movie formats.
    suffix
        The file type of the animation, either ``".mp4"`` or ``".gif"``.
        Ignored if `path` is given.
    path
        The location to save the animation.
        If :obj:`None`, it is saved alongside the other figures.
    """
    scene = scene_esis()
    back = backprojections_simulated()

    axis_channel = "channel"
    axis_velocity = "velocity"
    axis_xy = ("detector_x", "detector_y")

    # As in the blink: each backprojection was made onto a sky one wavelength
    # bin wide, so it is scaled back onto the spectral resolution of the scene.
    wavelength = back.inputs.wavelength.ndarray
    bins = float((wavelength.max() - wavelength.min()) / np.diff(wavelength).mean())
    back = dataclasses.replace(back, outputs=back.outputs * bins)

    velocity_back = back.inputs.wavelength.to(
        u.km / u.s,
        equivalencies=u.doppler_optical(scene.inputs.wavelength_rest),
    )

    scene = _crop(scene, scene.axis_wavelength, scene.inputs.velocity, velocity_limit)
    back = _crop(back, axis_velocity, velocity_back, velocity_limit)

    # What a channel records is one number per pixel, so the wavelengths are
    # added up rather than kept apart.
    recorded = back.integrate(component="wavelength", axis=axis_velocity)

    frames = [("scene O V 630", None)]
    for i in range(recorded.outputs.shape[axis_channel]):
        frames += [(f"channel {i + 1}", i)]

    position = scene.inputs.position

    # Kept as plain numbers, since the shades of gray are worked out by
    # matplotlib rather than by `named_arrays`, and it has no use for a unit.
    unit_radiance = na.unit(recorded.outputs)
    vmax_recorded = float(
        np.nanpercentile(recorded.outputs.ndarray.to_value(unit_radiance), percentile)
    )

    with astropy.visualization.quantity_support():

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(rect_image_default)
        cax = fig.add_axes(rect_colorbar_default)
        cax_twin = cax.twinx()

        if path is None:
            path = default_path / f"blink-channels{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".gif":
            repeat = 1
            writer = matplotlib.animation.PillowWriter(fps=fps)
        else:
            repeat = max(round(fps_video / fps), 1)
            writer = matplotlib.animation.FFMpegWriter(
                fps=fps * repeat,
                codec="h264",
                extra_args=["-pix_fmt", "yuv420p", "-crf", "14"],
            )

        drawn = [-1]

        def func(index: int) -> list[matplotlib.artist.Artist]:

            index = index // repeat
            if index == drawn[0]:
                return [*ax.collections, *cax.collections, *cax_twin.collections]
            drawn[0] = index

            label, channel = frames[index]

            # Emptied rather than cleared. Clearing an axes would also
            # forget that the twin belongs on the right, and everything drawn
            # below is in plain numbers anyway, so there are no units left on
            # an axis for the next frame to disagree with.
            for a in (ax, cax, cax_twin):
                for artist in [*a.collections, *a.images]:
                    artist.remove()

            cax_twin.set_visible(channel is None)

            if channel is None:
                colorbar = na.plt.rgbmesh(
                    scene.inputs.velocity,
                    position.x,
                    position.y,
                    C=scene.outputs,
                    axis_wavelength=scene.axis_wavelength,
                    ax=ax,
                    vmin=0,
                    vmax=np.nanpercentile(scene.outputs, percentile, axis=axis_xy),
                    wavelength_min=-velocity_color,
                    wavelength_max=+velocity_color,
                )
                unit = na.unit(colorbar.inputs.x)
                equivalency = u.doppler_optical(scene.inputs.wavelength_rest)
                unit_wavelength = na.unit(scene.inputs.wavelength_rest)

                wavelength = colorbar.inputs.y.to(
                    unit_wavelength,
                    equivalencies=equivalency,
                )
                na.plt.pcolormesh(
                    colorbar.inputs.x / unit,
                    wavelength / unit_wavelength,
                    C=colorbar.outputs,
                    axis_rgb=scene.axis_wavelength,
                    ax=cax,
                )
                na.plt.pcolormesh(
                    colorbar.inputs.x / unit,
                    colorbar.inputs.y / velocity_color.unit,
                    C=colorbar.outputs,
                    axis_rgb=scene.axis_wavelength,
                    ax=cax_twin,
                )

                cax.set_ylabel(f"wavelength ({unit_wavelength:latex_inline})")
                cax.set_ylim(
                    (-velocity_color).to_value(unit_wavelength, equivalency),
                    (+velocity_color).to_value(unit_wavelength, equivalency),
                )
                cax.set_xlim(0, (colorbar.inputs.x.max() / unit).ndarray)
                cax.set_xlabel(f"{unit:latex_inline}")
                cax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=2))
                cax.tick_params(axis="x", labelrotation=45, labelsize="small")

                cax_twin.set_ylabel(f"velocity ({velocity_color.unit:latex_inline})")
                cax_twin.set_ylim(
                    -velocity_color.value,
                    +velocity_color.value,
                )

            else:
                na.plt.pcolormesh(
                    position.x,
                    position.y,
                    C=recorded.outputs[{axis_channel: channel}] / unit_radiance,
                    ax=ax,
                    cmap=cmap,
                    vmin=0,
                    vmax=vmax_recorded,
                )

                # A ramp of the same gray, so that the key says what the
                # shades of the image are worth.
                ramp = np.linspace(0, 1, 256).reshape(-1, 1)
                cax.imshow(
                    ramp,
                    aspect="auto",
                    cmap=cmap,
                    origin="lower",
                    extent=(0, 1, 0, vmax_recorded),
                )
                # Every limit is set here as well as above, since nothing is
                # cleared between frames and the two kinds of frame put
                # different things on the same axes.
                cax.set_xlim(0, 1)
                cax.set_xticks([])
                cax.set_xlabel("")
                cax.set_ylim(0, vmax_recorded)
                cax.set_ylabel(f"radiance ({unit_radiance:latex_inline})")

            for artist in [*ax.collections, *cax.collections, *cax_twin.collections]:
                artist.set_rasterized(True)

            set_limits_sky(ax, position)
            ax.set_aspect("equal")
            ax.set_xlabel(f"helioprojective $x$ ({position.x.unit:latex_inline})")
            ax.set_ylabel(f"helioprojective $y$ ({position.y.unit:latex_inline})")
            ax.set_title(label)

            return [*ax.collections, *cax.collections, *cax_twin.collections]

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
