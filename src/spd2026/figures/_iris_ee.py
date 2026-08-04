"""
An IRIS observation of a transition region explosive event,
built up one panel at a time.
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.colorbar
import matplotlib.ticker
import astropy.units as u
import astropy.visualization
import colorsynth
import named_arrays as na
from .._observations import observation_iris
from ._path import default_path

__all__ = [
    "iris_ee",
]


def iris_ee(
    time: str = "2013-10-22 11:30",
    window: str = "Si IV 1394",
    index_time: int = 0,
    index_x: int = 230,
    index_y: int = 483,
    velocity_limit: u.Quantity = 250 * u.km / u.s,
    velocity_color: u.Quantity = 100 * u.km / u.s,
    velocity_limit_key: u.Quantity = 150 * u.km / u.s,
    velocity_gap: u.Quantity = 150 * u.km / u.s,
    position_gap: u.Quantity = 5 * u.arcsec,
    speed_sound: u.Quantity = 44 * u.km / u.s,
    linewidth: float = 2,
    height_key: float = 0.5,
    figsize: tuple[float, float] = (13.33, 7.5),
    dpi: float = 200,
    path: None | pathlib.Path = None,
) -> list[pathlib.Path]:
    """
    Save a sequence of three figures, each adding one more panel.

    The panels are, in the order they are revealed:

    1. The raster displayed as a false-color image, where the hue is the
       Doppler velocity.
    2. The spectrum along the slit, together with the red line marking the
       slit in the first panel.
    3. The spectral line profile at the explosive event, compared to the
       median profile of the whole raster, together with the red marker
       straddling the explosive event in the second panel.

    Each red marker therefore appears at the same time as the panel it points
    at, and never before it.

    Every figure has exactly the same layout,
    so that the panels appear one at a time without the earlier panels moving.

    Parameters
    ----------
    time
        The time of the observation to download.
    window
        The name of the spectral window to load.
    index_time
        The index along the time axis to display.
    index_x
        The index of the slit position containing the explosive event.
    index_y
        The index along the slit of the explosive event.
    velocity_limit
        The Doppler velocity range to display in the second and third panels.
    velocity_color
        The Doppler velocity mapped to each end of the visible spectrum in
        the first panel.
        The same value is used for the raster and for the color key,
        so that the key always describes the image.
    velocity_limit_key
        The Doppler velocity range to display in the color key.
    velocity_gap
        The half-width of the gap in the marker straddling the explosive
        event in the second panel.
    position_gap
        The half-height of the gap in the marker straddling the explosive
        event in the first panel.
    speed_sound
        The sound speed to mark in the third panel.
        Si IV forms near :math:`\\log T = 4.9`, where the sound speed is
        roughly 43 km/s for a fully ionized plasma with :math:`\\mu = 0.6`.
    linewidth
        The width of the spectral line profiles in the third panel.
    height_key
        The height of the color key in the first figure,
        as a fraction of the height of the panels it stands in for.
    figsize
        The width and height of the figures in inches.
    dpi
        The resolution used for the parts of the figures which are too
        detailed to store as vectors.
    path
        The directory in which to save the figures.
        If :obj:`None`, they are saved alongside the other figures.
    """
    obs = observation_iris(
        time=time,
        window=window,
    )

    axis_time = obs.axis_time
    axis_x = obs.axis_detector_x
    axis_y = obs.axis_detector_y

    index_slit = {axis_time: index_time, axis_x: index_x}
    index = index_slit | {axis_y: index_y}

    if path is None:
        path = default_path
    path.mkdir(parents=True, exist_ok=True)

    velocity = obs.inputs.velocity
    position = obs.inputs.position

    unit_position = na.unit(position.x)
    x_slit = position.x[index].ndarray.value
    y_event = position.y[index].ndarray.value

    median = np.nanmedian(obs.outputs, axis=(axis_time, axis_x, axis_y))

    with astropy.visualization.quantity_support():

        fig, axs = plt.subplots(
            ncols=3,
            figsize=figsize,
            constrained_layout=True,
            gridspec_kw=dict(
                width_ratios=[0.56, 0.22, 0.22],
            ),
        )
        cax, _ = matplotlib.colorbar.make_axes(
            axs[0],
            location="left",
            pad=0.15,
        )

        # The raster as a false-color image.
        obs.show(
            index_time=index_time,
            ax=axs[0],
            cax=cax,
            velocity_min=-velocity_color,
            velocity_max=+velocity_color,
        )

        # The spectrum along the slit.
        axs[1].sharey(axs[0])
        na.plt.pcolormesh(
            velocity[index_slit],
            position.y[index_slit],
            C=obs.outputs[index_slit].value,
            vmin=0,
            vmax=np.nanpercentile(obs.outputs[index_slit].value, 99.9),
            ax=axs[1],
        )
        axs[1].set_ylabel("")
        axs[1].tick_params(axis="y", labelleft=False)
        axs[1].set_title(f"$x = {x_slit:0.1f}$ {unit_position:latex_inline}")

        # The line profile at the explosive event.
        axs[2].sharex(axs[1])
        na.plt.stairs(
            velocity[index],
            obs.outputs[index],
            label=r"EE $1394\,\AA$",
            linewidth=linewidth,
            ax=axs[2],
        )
        na.plt.stairs(
            velocity,
            median,
            label=r"median $1394\,\AA$",
            linewidth=linewidth,
            ax=axs[2],
        )
        # Drawn underneath the profiles so that they read as a backdrop
        # against which the profiles are compared.
        axs[2].axvline(
            +speed_sound.value,
            color="red",
            linestyle="--",
            label="$c_s$",
            zorder=-1,
        )
        axs[2].axvline(
            -speed_sound.value,
            color="red",
            linestyle="--",
            zorder=-1,
        )
        axs[2].yaxis.tick_right()
        axs[2].yaxis.set_label_position("right")
        axs[2].set_ylim(None, 300)
        axs[2].set_title(f"$y = {y_event:0.1f}$ {unit_position:latex_inline}")
        axs[2].legend()

        # The second and third panels share this axis, so setting it once
        # sets it for both.
        axs[1].set_xlim(-velocity_limit.value, +velocity_limit.value)

    # Each color mesh has hundreds of thousands of cells, which would make an
    # enormous SVG if they were stored as vectors, so draw them as embedded
    # images instead. The axes, text, and lines stay vectors.
    for ax in fig.axes:
        for artist in ax.collections:
            artist.set_rasterized(True)

    # Freeze the layout so that the hidden panels still take up space and the
    # visible panels do not move between figures.
    fig.canvas.draw()
    fig.set_layout_engine("none")

    # Both markers are drawn as a pair of segments with a gap in the middle,
    # so that they point at the feature without covering it. Their endpoints
    # are in axes coordinates, so they can only be computed once the limits
    # are final and the figure has been drawn.
    def _fraction(
        ax: matplotlib.axes.Axes,
        point: tuple[float, float],
        component: int,
    ) -> float:
        """The position of a point in the coordinates of the given axes."""
        return ax.transAxes.inverted().transform(ax.transData.transform(point))[component]

    gap_position = position_gap.to_value(na.unit(position.y))
    lines_slit = [
        axs[0].axvline(
            x=x_slit,
            ymin=ymin,
            ymax=ymax,
            color="red",
        )
        for ymin, ymax in [
            (0, _fraction(axs[0], (x_slit, y_event - gap_position), 1)),
            (_fraction(axs[0], (x_slit, y_event + gap_position), 1), 1),
        ]
    ]

    gap_velocity = velocity_gap.to_value(na.unit(velocity))
    lines_event = [
        axs[1].axhline(
            y=y_event,
            xmin=xmin,
            xmax=xmax,
            color="red",
        )
        for xmin, xmax in [
            (0, _fraction(axs[1], (-gap_velocity, y_event), 0)),
            (_fraction(axs[1], (+gap_velocity, y_event), 0), 1),
        ]
    ]

    # A key showing which Doppler velocity maps to which color in the raster,
    # which is only useful while the raster is the only panel on screen.
    # It is placed in the space that the later panels will occupy, using their
    # frozen positions, so that adding it cannot move anything else.
    bbox = axs[1].get_position().union([axs[1].get_position(), axs[2].get_position()])
    height = height_key * bbox.height

    # Half an inch of room on the left so that the vertical axis label does
    # not run into the raster, and the same on the right so that the labels
    # of the second vertical axis are not clipped by the edge of the figure.
    pad = 0.5 / figsize[0]

    # An inch of room underneath for the wavelength scale, which hangs below
    # the axes. The key sits at the bottom of the space left by the panels
    # which have not appeared yet, leaving the top of that space free.
    pad_lower = 1.0 / figsize[1]

    ax_key = fig.add_axes(
        (
            bbox.x0 + pad,
            bbox.y0 + pad_lower,
            bbox.width - 2 * pad,
            height,
        )
    )

    with astropy.visualization.quantity_support():

        velocity_center = velocity.cell_centers(obs.axis_wavelength).ndarray.squeeze()

        # The private function which `colorsynth` uses to map the spectral
        # axis onto the visible spectrum. It is given the same velocity range
        # as the raster, so that the key describes the colors in the image
        # rather than some other mapping.
        transform_wavelength = colorsynth._colorsynth._transform_wavelength(
            velocity_center,
            -1,
            -velocity_color,
            +velocity_color,
            None,
        )
        wavelength_visible = transform_wavelength(velocity_center)

        # The response of each color channel to a spectrum which is nonzero
        # at only one point along the spectral axis.
        spd = np.diagflat(np.ones(wavelength_visible.shape))
        XYZ = colorsynth.XYZcie1931_from_spd(
            spd,
            wavelength_visible[..., np.newaxis],
            axis=0,
        )
        XYZ = XYZ / XYZ.max(axis=1, keepdims=True)
        XYZ = XYZ * np.array([0.9505, 1.0000, 1.0890])[..., np.newaxis]
        red, green, blue = colorsynth.sRGB(XYZ, axis=0)

        for channel, color in zip([red, green, blue], ["red", "green", "blue"]):
            ax_key.plot(velocity_center, channel, color=color)

        ax_key_twin = ax_key.twinx()
        na.plt.stairs(
            velocity,
            median,
            label=r"median $1394\,\AA$",
            color="black",
            linewidth=linewidth,
            ax=ax_key_twin,
        )
        ax_key_twin.legend()

        ax_key.set_xlim(-velocity_limit_key.value, +velocity_limit_key.value)
        ax_key.set_ylim(-0.2, None)
        ax_key.set_ylabel("sRGB response")
        ax_key_twin.set_ylim(0, None)

        # The same curves against the visible wavelength they are mapped onto.
        # The mapping is linear, so setting the limits is enough to line this
        # axis up with the one below it.
        unit_visible = wavelength_visible.unit
        ax_key_top = ax_key.twiny()
        ax_key_top.set_xlim(
            transform_wavelength(-velocity_limit_key).to_value(unit_visible),
            transform_wavelength(+velocity_limit_key).to_value(unit_visible),
        )
        ax_key_top.set_xlabel(f"visible wavelength ({unit_visible:latex_inline})")

        # A third scale, below the velocity, giving the wavelength of the
        # spectral line itself. The velocity and the wavelength are two
        # parameterizations of the same grid of pixels, so one is interpolated
        # onto the other rather than assuming a Doppler convention.
        unit_velocity = na.unit(velocity)
        unit_line = na.unit(obs.inputs.wavelength)
        ax_key_line = ax_key.twiny()
        ax_key_line.set_xlim(
            *np.interp(
                x=[-velocity_limit_key.to_value(unit_velocity)]
                + [+velocity_limit_key.to_value(unit_velocity)],
                xp=velocity[{axis_time: index_time}].ndarray.to_value(unit_velocity),
                fp=obs.inputs.wavelength[{axis_time: index_time}].ndarray.to_value(
                    unit_line
                ),
            )
        )
        ax_key_line.xaxis.set_ticks_position("bottom")
        ax_key_line.xaxis.set_label_position("bottom")
        ax_key_line.spines["bottom"].set_position(("outward", 36))
        for spine in ax_key_line.spines.values():
            spine.set_visible(False)
        ax_key_line.spines["bottom"].set_visible(True)
        ax_key_line.patch.set_visible(False)
        ax_key_line.set_xlabel(f"wavelength ({unit_line:latex_inline})")

        # These tick labels are long enough to run into each other at the
        # width of this axes, so use fewer of them than usual.
        ax_key_line.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))

    # Each panel appears along with the marker in the panel to its left which
    # points at it, so that a marker is never on screen before the panel
    # explaining it. The color key occupies the space of the panels which have
    # not appeared yet, so it goes away as soon as the first of them arrives.
    shown = [
        [ax_key, ax_key_twin, ax_key_top, ax_key_line],
        [axs[1], *lines_slit],
        [axs[2], *lines_event],
    ]
    hidden = [
        [],
        [ax_key, ax_key_twin, ax_key_top, ax_key_line],
        [],
    ]

    for group in shown[1:]:
        for artist in group:
            artist.set_visible(False)

    result = []
    for i, (group_shown, group_hidden) in enumerate(zip(shown, hidden)):
        for artist in group_shown:
            artist.set_visible(True)
        for artist in group_hidden:
            artist.set_visible(False)
        path_i = path / f"iris-ee-{i + 1}.svg"
        fig.savefig(path_i, dpi=dpi)
        result.append(path_i)

    plt.close(fig)

    return result


if __name__ == "__main__":

    for p in iris_ee():
        print(p)
