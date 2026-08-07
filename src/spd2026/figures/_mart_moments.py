"""
How well the moments of the recovered line profiles match the true ones.
"""

import pathlib
import numpy as np
import matplotlib.animation
import matplotlib.artist
import matplotlib.axes
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization
import named_arrays as na
from .._degraded import scene_degraded
from .._inversions import InversionSim, inversion_sim
from ._layout import figsize_default
from ._path import default_path

__all__ = [
    "mart_moments",
]

#: The positions of the three histograms and of the key above each, as
#: fractions of the figure.
#:
#: Each histogram compares a quantity with itself, so it is drawn square and
#: the line of perfect recovery runs at forty five degrees. Three squares
#: side by side on a slide can only be so large, so the room left over goes
#: above them, where the iteration is named.
_width = 0.265
_height = 0.471
_bottom = 0.125
_bottom_key = 0.665
_height_key = 0.028
_lefts = (0.075, 0.3925, 0.710)


def mart_moments(
    inversion: "None | InversionSim" = None,
    axis: str = "velocity",
    num_bins: int = 50,
    range_radiance: tuple[u.Quantity, u.Quantity] = (
        0 * u.erg / (u.s * u.sr * u.cm**2),
        3e3 * u.erg / (u.s * u.sr * u.cm**2),
    ),
    range_median: tuple[u.Quantity, u.Quantity] = (
        -80 * u.km / u.s,
        +80 * u.km / u.s,
    ),
    range_iqr: tuple[u.Quantity, u.Quantity] = (
        0 * u.km / u.s,
        100 * u.km / u.s,
    ),
    percentile_radiance: float = 75,
    percentile_color: float = 97,
    figsize: tuple[float, float] = figsize_default,
    dpi: float = 150,
    fps: int = 24,
    fps_video: int = 24,
    suffix: str = ".mp4",
    path: None | pathlib.Path = None,
) -> pathlib.Path:
    """
    Watch the moments of the recovered profiles against the true ones.

    Three column-normalized two-dimensional histograms, one frame per
    iteration: the radiance of each line profile, the median of each profile,
    which is where the plasma is moving, and the interquartile range, which
    is how fast it is moving apart. The true value runs along the horizontal
    axis and the recovered one along the vertical, so a perfect inversion
    would put every count on the dashed diagonal.

    Column-normalized, so each column of a panel is the distribution of what
    the inversion recovered given what was there, which means a faint part of
    the scene is judged as harshly as a bright one.

    This is the figure the ``mart-iris`` notebook in :mod:`esis` draws with
    ``plot_moments``, made once per iteration instead of once at the end, so
    that the moments can be watched turning away from the truth as the
    inversion begins to fit the noise.

    Parameters
    ----------
    inversion
        The inversion to show.
        If :obj:`None`, :func:`spd2026.inversion_sim`, the one which was not
        told the answer.
    axis
        The logical axis along which to compute the moments.
    num_bins
        The number of bins along each side of each histogram.
    range_radiance
        The range of the radiance histogram.
    range_median
        The range of the median histogram.
    range_iqr
        The range of the interquartile range histogram.
    percentile_radiance
        Places on the sky fainter than this percentile of the true radiance
        are left out, since the moments of a profile with no light in it are
        meaningless.
    percentile_color
        The percentile of the counts placed at the top of the color scale.
        Worked out over every iteration and then held fixed, so that a frame
        can be compared with the one before it.

        Kept below the ninety ninth percentile, above which the scale is set
        by columns holding only a handful of places on the sky. Normalizing
        such a column hands one of its bins the whole probability, and a
        scale reaching all the way to one leaves the cloud which the panel
        exists to show almost black.
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
    truth = scene_degraded()
    inv = inversion_sim() if inversion is None else inversion

    solutions = inv.solutions
    axis_iteration = inv.axis_iteration
    num_iteration = solutions.outputs.shape[axis_iteration]

    bins = dict(true=num_bins, reconstructed=num_bins)

    def moments(a: na.FunctionArray) -> tuple:
        """The radiance, the median velocity, and the width of each profile."""
        width_cell = a.inputs.wavelength.volume_cell(axis)
        radiance = (a.outputs * width_cell).sum(axis)
        median = na.pdf.median(x=a.inputs.velocity, f=a.outputs, axis=axis)
        iqr = na.pdf.iqr(x=a.inputs.velocity, f=a.outputs, axis=axis)
        return radiance, median, iqr

    radiance_truth, median_truth, iqr_truth = moments(truth)

    # The moments of a profile with no light in it say nothing, so the faint
    # part of the scene is left out of every panel.
    threshold = np.nanpercentile(radiance_truth, percentile_radiance)
    where = radiance_truth > threshold

    ranges = (range_radiance, range_median, range_iqr)
    truths = (radiance_truth, median_truth, iqr_truth)

    # Every frame is worked out before any is drawn, so that one color scale
    # can be chosen for the whole movie.
    histograms = []
    correlations = []
    for i in range(num_iteration):
        recon = na.FunctionArray(
            inputs=solutions.inputs,
            outputs=solutions.outputs[{axis_iteration: i}],
        )
        row = []
        row_r = []
        for value_truth, value_recon, (low, high) in zip(
            truths,
            moments(recon),
            ranges,
        ):
            histogram = na.histogram2d(
                value_truth,
                value_recon,
                bins=bins,
                min=low,
                max=high,
                weights=where,
            )
            # Normalized down each column, so that each is the distribution
            # of what came back given what was there.
            histogram = histogram / histogram.sum("reconstructed")
            histogram.outputs = np.nan_to_num(
                x=histogram.outputs,
                posinf=0,
                neginf=0,
            )
            row.append(histogram)
            row_r.append(
                na.stats.pearsonr(
                    x=value_truth,
                    y=value_recon,
                    where=where & np.isfinite(value_recon),
                ).ndarray
            )
        histograms.append(row)
        correlations.append(row_r)

    vmax = [
        max(np.nanpercentile(h[j].outputs, percentile_color) for h in histograms)
        for j in range(3)
    ]

    labels = ("radiance", "median", "IQR")

    if path is None:
        path = default_path / f"mart-moments{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)

    with astropy.visualization.quantity_support():

        fig = plt.figure(figsize=figsize)
        axs = [fig.add_axes((left, _bottom, _width, _height)) for left in _lefts]
        caxs = [
            fig.add_axes((left, _bottom_key, _width, _height_key)) for left in _lefts
        ]

        title = fig.text(
            x=0.5,
            y=0.925,
            s="",
            ha="center",
            va="center",
            fontsize="x-large",
        )

        texts = [
            ax.text(
                x=0.05,
                y=0.95,
                s="",
                transform=ax.transAxes,
                ha="left",
                va="top",
                color="white",
            )
            for ax in axs
        ]

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

        transient = []
        drawn = [-1]

        def func(index: int) -> list[matplotlib.artist.Artist]:

            iteration = index // repeat
            if iteration == drawn[0]:
                return transient
            first = drawn[0] < 0
            drawn[0] = iteration

            for artist in transient:
                artist.remove()
            transient.clear()

            for j, (ax, cax) in enumerate(zip(axs, caxs)):

                image = na.plt.pcolormesh(
                    C=histograms[iteration][j],
                    ax=ax,
                    vmin=0,
                    vmax=vmax[j],
                )
                transient.extend(ax.collections)

                # The scale is the same in every frame, so the key is only
                # worth drawing once.
                if first:
                    plt.colorbar(
                        image.ndarray.item(),
                        cax=cax,
                        orientation="horizontal",
                        label="probability",
                    )
                    cax.xaxis.set_ticks_position("top")
                    cax.xaxis.set_label_position("top")

                    # The line an inversion which recovered everything would
                    # lie along, drawn through the middle of the truth.
                    point = np.nanmean(truths[j]).ndarray.value
                    ax.axline(
                        (point, point),
                        slope=1,
                        color="tab:red",
                        linestyle="dashed",
                    )
                    ax.set_aspect("equal")
                    unit = na.unit(truths[j])
                    ax.set_xlabel(f"true {labels[j]} ({unit:latex_inline})")
                    ax.set_ylabel(f"recovered {labels[j]} ({unit:latex_inline})")

                texts[j].set_text(f"Pearson's $r = {correlations[iteration][j]:.03f}$")

            for artist in transient:
                artist.set_rasterized(True)

            title.set_text(f"MART iteration {iteration + 1}")

            return [*transient, *texts, title]

        ani = matplotlib.animation.FuncAnimation(
            fig=fig,
            func=func,
            frames=num_iteration * repeat,
        )

        ani.save(
            filename=path,
            writer=writer,
            dpi=dpi,
        )

    plt.close(fig)

    return path


if __name__ == "__main__":

    print(mart_moments())
