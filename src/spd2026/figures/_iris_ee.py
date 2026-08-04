"""
An IRIS observation of a transition region explosive event,
built up one panel at a time.
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.colorbar
import astropy.units as u
import astropy.visualization
import named_arrays as na
import iris
from ._path import default_path

__all__ = [
    "observation",
    "iris_ee",
]


def observation(
    time: str = "2013-10-22 11:30",
    window: str = "Si IV 1394",
    slice_wavelength: slice = slice(750, 1250),
) -> iris.sg.SpectrographObservation:
    """
    The IRIS spectrograph raster containing the explosive event.

    Parameters
    ----------
    time
        The time of the observation to download.
    window
        The name of the spectral window to load.
    slice_wavelength
        The range of wavelength pixels to keep,
        chosen to isolate the spectral line from the rest of the window.
    """
    result = iris.sg.open(
        time=time,
        window=window,
    )

    return result[{result.axis_wavelength: slice_wavelength}]


def iris_ee(
    time: str = "2013-10-22 11:30",
    window: str = "Si IV 1394",
    index_time: int = 0,
    index_x: int = 230,
    index_y: int = 483,
    velocity_limit: u.Quantity = 250 * u.km / u.s,
    velocity_gap: u.Quantity = 150 * u.km / u.s,
    position_gap: u.Quantity = 5 * u.arcsec,
    speed_sound: u.Quantity = 44 * u.km / u.s,
    linewidth: float = 2,
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
    figsize
        The width and height of the figures in inches.
    dpi
        The resolution used for the parts of the figures which are too
        detailed to store as vectors.
    path
        The directory in which to save the figures.
        If :obj:`None`, they are saved alongside the other figures.
    """
    obs = observation(
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
            np.nanmedian(obs.outputs, axis=(axis_time, axis_x, axis_y)),
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
            zorder=0,
        )
        axs[2].axvline(
            -speed_sound.value,
            color="red",
            linestyle="--",
            zorder=0,
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

    # Each panel appears along with the marker in the panel to its left which
    # points at it, so that a marker is never on screen before the panel
    # explaining it.
    revealed = [
        [],
        [axs[1], *lines_slit],
        [axs[2], *lines_event],
    ]

    for group in revealed[1:]:
        for artist in group:
            artist.set_visible(False)

    result = []
    for i, group in enumerate(revealed):
        for artist in group:
            artist.set_visible(True)
        path_i = path / f"iris-ee-{i + 1}.svg"
        fig.savefig(path_i, dpi=dpi)
        result.append(path_i)

    plt.close(fig)

    return result


if __name__ == "__main__":

    for p in iris_ee():
        print(p)
