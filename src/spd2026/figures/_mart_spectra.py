"""
The spectra MART recovers, beside the spectra there were to recover.
"""

import pathlib
import numpy as np
import matplotlib.animation
import matplotlib.artist
import matplotlib.axes
import matplotlib.cm
import matplotlib.colors
import matplotlib.ticker
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization
import named_arrays as na
from .._degraded import scene_degraded
from .._inversions import InversionSim, inversion_sim
from ._color import percentile_default
from ._event import x_event_default, y_event_default
from ._layout import (
    figsize_default,
    bottom_default,
    height_default,
)
from ._path import default_path

__all__ = [
    "mart_spectra",
]

#: The positions of the three spectra and the two keys, as fractions of the
#: figure.
#:
#: Packed rather than placed on the grid the other figures share, since this
#: figure holds five panels where they hold two or three, and the spectra
#: are tall and narrow. Each panel is given only the room its own labels
#: need: the three spectra share a vertical axis, so only the first pays for
#: one, and the profile carries its scale on the outside, so the gap to its
#: left is only what separates it from the key.
_rect_key_scene = (0.062, bottom_default, 0.016, height_default)
_rect_degraded = (0.150, bottom_default, 0.150, height_default)
_rect_mart = (0.310, bottom_default, 0.150, height_default)
_rect_residual = (0.470, bottom_default, 0.150, height_default)
_rect_key_residual = (0.628, bottom_default, 0.016, height_default)
_rect_profile = (0.730, bottom_default, 0.205, height_default)


def mart_spectra(
    inversion: "None | InversionSim" = None,
    x_event: u.Quantity = x_event_default,
    y_event: u.Quantity = y_event_default,
    velocity_limit: u.Quantity = 250 * u.km / u.s,
    velocity_gap: u.Quantity = 200 * u.km / u.s,
    percentile: float = percentile_default,
    speed_sound: u.Quantity = 76 * u.km / u.s,
    cmap: str = "viridis",
    cmap_residual: str = "gray",
    linewidth: float = 2,
    headroom: float = 1.5,
    figsize: tuple[float, float] = figsize_default,
    dpi: float = 150,
    fps: int = 24,
    fps_video: int = 24,
    suffix: str = ".mp4",
    path: None | pathlib.Path = None,
) -> pathlib.Path:
    """
    Watch MART recover the spectrum of an explosive event.

    The first three panels are the spectrum along the slit through the
    event: the scene at the resolution ESIS can distinguish, what MART has
    recovered at this iteration, and the difference between them. The last
    panel is the line profile at the event itself, with the scene and the
    reconstruction drawn on top of one another.

    Only those two, since what is left over is the gap between them and can
    be read straight off. A third curve for it would be a second thing to
    follow which says nothing the first two do not.

    The residual is the scene minus the reconstruction, so light is light the
    inversion failed to recover and dark is light it invented. It is drawn on
    the same scale as the two panels beside it, running from minus to plus
    rather than from zero, so that the size of what is left over can be read
    against the size of what was there, and the middle of its colormap is
    where the two agree.

    The event is given as a place on the sky and the nearest cell of the
    recovered grid is shown. :func:`spd2026.figures.iris_ee` is given the
    same place, so the two figures are about the same event even though
    they are on different grids.

    Parameters
    ----------
    inversion
        The inversion to show.
        If :obj:`None`, :func:`spd2026.inversion_sim`, the one which was not
        told the answer.
    x_event
        The horizontal position of the explosive event, measured from the
        center of the field.
    y_event
        The vertical position of the explosive event.
    velocity_limit
        The Doppler velocity range to display.
    velocity_gap
        The half width of the gap in the marker which points at the row the
        profile was taken from, wide enough that the marker does not lie
        across the event it is pointing at.
    percentile
        The percentile of the scene placed at the top of the brightness
        scale, which the reconstruction and the residual share.
    speed_sound
        The sound speed to mark in the profile.
        O V forms near :math:`\\log T = 5.4`, where the sound speed is
        roughly 76 km/s for a fully ionized plasma with :math:`\\mu = 0.6`.
    cmap
        The colormap of the scene and the reconstruction.
    cmap_residual
        The colormap of the residual.
        Its range runs from minus to plus, so the middle of the colormap is
        where the inversion got it right, the dark end is light it invented
        and the light end is light it failed to recover.
    linewidth
        The width of the line profiles.
    headroom
        The height of the profile panel as a multiple of the height of the
        profile of the event, which leaves room for the legend.
    figsize
        The width and height of the figure in inches.
    dpi
        The resolution of the saved animation in dots per inch.
    fps
        The number of iterations shown per second.
    fps_video
        The frame rate of the file itself, for the movie formats.
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
    axis_x = "detector_x"
    axis_y = "detector_y"

    coordinates = degraded.inputs
    velocity = coordinates.velocity
    position = coordinates.position

    unit_position = na.unit(position.x)
    unit_radiance = na.unit(degraded.outputs)

    # The event is given as a place on the sky rather than as a cell, and the
    # nearest cell of the recovered grid is the one shown.
    centers_x = position.x.cell_centers(axis_x).ndarray.squeeze()
    centers_y = position.y.cell_centers(axis_y).ndarray.squeeze()
    index_slit = {axis_x: int(np.abs(centers_x - x_event).argmin())}
    index = index_slit | {axis_y: int(np.abs(centers_y - y_event).argmin())}

    x_event = centers_x[index_slit[axis_x]].to_value(unit_position)
    y_event = centers_y[index[axis_y]].to_value(unit_position)

    # Kept as plain numbers, since the colors are worked out by matplotlib,
    # which has no use for a unit, and since the three panels would otherwise
    # each teach their axes a unit of their own.
    spectrum_degraded = degraded.outputs[index_slit].to_value(unit_radiance)
    profile_degraded = degraded.outputs[index].to_value(unit_radiance)

    # The scene is the reference, so its scale is the one all three panels
    # use, and the residual runs from minus that to plus it.
    vmax = float(np.nanpercentile(spectrum_degraded.ndarray, percentile))

    # The profile panel has to hold every iteration, since its limits cannot
    # move from one frame to the next without the curves appearing to change
    # size. Worked out over all of them, and over the scene as well, since
    # either may be the taller.
    profiles_mart = solutions.outputs[index].to_value(unit_radiance)

    ylim_profile = (
        0,
        headroom
        * max(
            float(np.nanmax(profiles_mart.ndarray)),
            float(np.nanmax(profile_degraded.ndarray)),
        ),
    )

    chi_squared = inv.mean_chi_squared.mean("channel")

    label_unit = f"({unit_radiance:latex_inline})"

    if path is None:
        path = default_path / f"mart-spectra{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)

    with astropy.visualization.quantity_support():

        fig = plt.figure(figsize=figsize)
        ax_degraded = fig.add_axes(_rect_degraded)
        ax_mart = fig.add_axes(_rect_mart)
        ax_residual = fig.add_axes(_rect_residual)
        cax_scene = fig.add_axes(_rect_key_scene)
        cax_residual = fig.add_axes(_rect_key_residual)
        ax_profile = fig.add_axes(_rect_profile)

        axs_spectra = (ax_degraded, ax_mart, ax_residual)

        for ax in axs_spectra[1:]:
            ax.sharey(ax_degraded)
        ax_profile.sharex(ax_degraded)

        def spectrum(ax, values, norm, colormap):
            """Draw one spectrum along the slit."""
            return na.plt.pcolormesh(
                velocity,
                position.y,
                C=values,
                ax=ax,
                cmap=colormap,
                norm=norm,
            )

        norm_scene = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
        norm_residual = matplotlib.colors.Normalize(vmin=-vmax, vmax=+vmax)

        # The scene does not change from one iteration to the next, so it and
        # the two keys are drawn once rather than for every frame.
        spectrum(ax_degraded, spectrum_degraded, norm_scene, cmap)

        fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm_scene, cmap=cmap),
            cax=cax_scene,
            label=f"radiance {label_unit}",
        )
        # Read from the outside in, as the keys of the other figures are,
        # which also keeps it clear of the panel it stands beside.
        cax_scene.yaxis.set_ticks_position("left")
        cax_scene.yaxis.set_label_position("left")
        fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm_residual, cmap=cmap_residual),
            cax=cax_residual,
            label=f"scene $-$ MART {label_unit}",
        )

        ax_degraded.set_title("scene")
        ax_residual.set_title("residual")
        ax_degraded.set_ylabel(f"helioprojective $y$ ({unit_position:latex_inline})")

        for ax in axs_spectra:
            ax.set_xlabel(f"velocity ({velocity_limit.unit:latex_inline})")
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3))
        for ax in axs_spectra[1:]:
            ax.tick_params(axis="y", labelleft=False)

        # Which row of the slit the profile beside these panels was taken
        # from. Drawn as a pair of segments with a gap in the middle, so that
        # it points at the event from either side without lying across it.
        # The ends are fractions of the panel, which the velocity can be
        # turned into directly, since the panel spans `velocity_limit` either
        # way.
        gap = (velocity_gap / velocity_limit).to_value(u.dimensionless_unscaled)
        for ax in axs_spectra:
            for xmin, xmax in [
                (0, (1 - gap) / 2),
                ((1 + gap) / 2, 1),
            ]:
                ax.axhline(
                    y=y_event * unit_position,
                    xmin=xmin,
                    xmax=xmax,
                    color="red",
                )

        # Drawn underneath the profiles, so that they read as a backdrop
        # against which the profiles are compared.
        for sign in (-1, +1):
            ax_profile.axvline(
                sign * speed_sound.value,
                color="red",
                linestyle="--",
                label="$c_s$" if sign > 0 else None,
                zorder=-1,
            )

        ax_profile.set_xlim(-velocity_limit.value, +velocity_limit.value)
        ax_profile.set_xlabel(f"velocity ({velocity_limit.unit:latex_inline})")

        # On the outside, away from the residual and its key, which is also
        # where :func:`spd2026.figures.iris_ee` puts the scale of its profile.
        ax_profile.yaxis.tick_right()
        ax_profile.yaxis.set_label_position("right")
        ax_profile.set_ylabel(f"radiance {label_unit}")
        ax_profile.set_title(
            f"$x = {x_event:0.1f}$, $y = {y_event:0.1f}$ "
            f"{unit_position:latex_inline}"
        )

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

        def artists(ax):
            """Everything on the axes which one of the plots below could add."""
            return {*ax.collections, *ax.patches, *ax.lines, *ax.images}

        transient = []
        drawn = [-1]

        def func(index_frame: int) -> list[matplotlib.artist.Artist]:

            iteration = index_frame // repeat
            if iteration == drawn[0]:
                return transient
            drawn[0] = iteration

            for artist in transient:
                artist.remove()
            transient.clear()

            outputs = solutions.outputs[{axis_iteration: iteration}]
            spectrum_mart = outputs[index_slit].to_value(unit_radiance)
            profile_mart = outputs[index].to_value(unit_radiance)

            before = {ax: artists(ax) for ax in (*axs_spectra, ax_profile)}

            spectrum(ax_mart, spectrum_mart, norm_scene, cmap)
            spectrum(
                ax_residual,
                spectrum_degraded - spectrum_mart,
                norm_residual,
                cmap_residual,
            )

            # Given a color rather than taking the next one, since the
            # profiles are drawn again for every iteration and the color
            # cycle would otherwise have moved on by the next frame.
            na.plt.stairs(
                velocity,
                profile_degraded,
                label="scene",
                color="C0",
                linewidth=linewidth,
                ax=ax_profile,
            )
            na.plt.stairs(
                velocity,
                profile_mart,
                label="MART",
                color="C1",
                linewidth=linewidth,
                ax=ax_profile,
            )

            for ax in (*axs_spectra, ax_profile):
                transient.extend(artists(ax) - before[ax])

            for artist in transient:
                artist.set_rasterized(True)

            # Set every frame, since drawing on an axes only ever widens it,
            # and to the range which holds every iteration, so that the
            # curves do not appear to change size from one frame to the next.
            ax_profile.set_ylim(*ylim_profile)

            handles, labels = ax_profile.get_legend_handles_labels()
            order = np.argsort([label == "$c_s$" for label in labels], kind="stable")
            ax_profile.legend(
                [handles[i] for i in order],
                [labels[i] for i in order],
                loc="upper right",
            )

            chi = chi_squared[{axis_iteration: iteration}].ndarray
            ax_mart.set_title(
                f"MART {iteration + 1}, " rf"$\chi^2$={chi:0.3f}",
                fontsize="small",
            )

            return transient

        for artist in ax_degraded.collections:
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

    print(mart_spectra())
