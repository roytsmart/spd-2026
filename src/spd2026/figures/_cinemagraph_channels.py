"""
An animated loop of the ESIS Level-1 images captured by all four channels.
"""

import pathlib
import matplotlib.animation
import matplotlib.artist
import matplotlib.pyplot as plt
import named_arrays as na
from ._cinemagraph import frames
from ._layout import figsize_default
from ._path import default_path

__all__ = [
    "cinemagraph_channels",
]


def cinemagraph_channels(
    threshold: float = 0,
    normalize: bool = False,
    cmap: str = "gray",
    percentile_vmin: float = 1,
    percentile_vmax: float = 99.9,
    figsize: tuple[float, float] = figsize_default,
    dpi: float = 150,
    fps: int = 5,
    timestamp: bool = True,
    suffix: str = ".mp4",
    path: None | pathlib.Path = None,
) -> pathlib.Path:
    """
    Loop over the ESIS Level-1 images of all four channels at once.

    The four channels are shown two by two, as they are arranged around the
    optical axis, and they share their axes and their color scale: the same
    place on a sensor is the same place in every panel, and the same shade is
    the same number of electrons.

    All four look at the same Sun at the same moment through the same
    atmosphere, so what differs between the panels is the instrument. Each
    disperses the scene in a different direction, which is what makes the
    inversion possible and what these images are the evidence for.

    Every frame of the flight is shown, and none is rescaled, so the images
    brighten and fade as the rocket climbs out of the absorbing atmosphere
    and falls back into it. That is most of what happens: the observation is
    five minutes long and only the middle minute of it is at full signal.

    Parameters
    ----------
    threshold
        The minimum median signal in a frame, expressed as a fraction of the
        largest median signal in the observation, for that frame to be kept.
        Measured over the four channels together, so that they keep the same
        frames.
    normalize
        If :obj:`True`, scale each frame so that every frame of a channel has
        the same median signal, which stops the loop pulsing as the rocket
        rises and falls.
        Each channel is scaled to its own median, so the channels keep their
        brightnesses relative to one another.
    cmap
        The colormap used to map the signal to colors.
    percentile_vmin
        The percentile of the signal to place at the bottom of the color
        scale.
    percentile_vmax
        The percentile of the signal to place at the top of the color scale.
        The brightest pixels are cosmic ray residuals which are orders of
        magnitude brighter than the Sun, so this should stay well below 100.
    figsize
        The width and height of the figure in inches.
    dpi
        The resolution of the saved animation in dots per inch.
    fps
        The number of frames per second in the saved animation.
    timestamp
        If :obj:`True`, write the time each frame was taken along the top of
        the figure.
    suffix
        The file type of the animation, either ``".mp4"`` or ``".gif"``.
        Ignored if `path` is given.
    path
        The location to save the animation.
        If :obj:`None`, it is saved alongside the other figures.
    """
    obs = frames(
        channel=None,
        threshold=threshold,
        normalize=normalize,
    )

    axis_channel = obs.axis_channel
    axis_time = obs.axis_time

    # Ordered like an image, so that a frame of a channel can be handed
    # straight to `imshow`.
    outputs = obs.outputs.transpose(
        (axis_channel, axis_time, obs.axis_y, obs.axis_x),
    )

    # One color scale for every channel and every frame, so that a shade
    # means the same thing wherever it appears.
    vmin = outputs.percentile(percentile_vmin).ndarray
    vmax = outputs.percentile(percentile_vmax).ndarray

    images = outputs.ndarray
    unit = na.unit(outputs)
    if unit is not None:
        images = images.to_value(unit)
        vmin = vmin.to_value(unit)
        vmax = vmax.to_value(unit)

    channel = obs.channel.ndarray
    time = obs.inputs.time[{axis_channel: 0}]

    num_channel = outputs.shape[axis_channel]
    num_time = outputs.shape[axis_time]

    if path is None:
        path = default_path / f"cinemagraph-channels{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=figsize,
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    axs = axs.ravel()

    # A sensor is twice as wide as it is tall, so four of them stacked two by
    # two are twice as wide as they are tall, while a slide is not quite
    # twice. There is no room to spare, so every part of the figure which is
    # not an image is made as small as it can be read at.
    # `w_pad` and `h_pad` are what keeps the labels off the edge of the
    # figure, so they are small rather than nothing, while the space between
    # the panels is left at nothing.
    fig.get_layout_engine().set(
        w_pad=0.05,
        h_pad=0.05,
        wspace=0.01,
        hspace=0.01,
    )

    artists = []
    for i in range(num_channel):
        ax = axs[i]
        artists.append(
            ax.imshow(
                X=images[i, 0],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                origin="lower",
                # Equal, since a pixel of the sensor is square and stretching
                # the images to fill the slide would misrepresent the shape
                # of everything on them.
                aspect="equal",
            )
        )
        # Written on the image rather than above it, since a title would cost
        # a row of its own between the panels, and the corner of a sensor is
        # empty in any case.
        ax.text(
            x=0.01,
            y=0.97,
            s=channel[i],
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="white",
        )

    for ax in axs[2:]:
        ax.set_xlabel("detector $x$ (pix)")
    for ax in axs[::2]:
        ax.set_ylabel("detector $y$ (pix)")

    fig.colorbar(
        artists[0],
        ax=axs.tolist(),
        label=f"signal ({unit:latex_inline})",
        # Held close and kept thin, since the width it takes is width the
        # images do not get.
        pad=0.01,
        fraction=0.025,
    )

    if timestamp:
        # Placed at a fixed spot rather than made a title, for two reasons:
        # the band along the top is empty anyway, being the difference
        # between the shape of four sensors and the shape of a slide, so it
        # costs the images nothing; and a title is laid out with the rest of
        # the figure, which would move everything a little whenever the width
        # of the text changed from one frame to the next.
        text = fig.text(
            x=0.5,
            y=0.978,
            s="",
            ha="center",
            va="top",
        )
    else:
        text = None

    def func(frame: int) -> list[matplotlib.artist.Artist]:
        for i, image in enumerate(artists):
            image.set_data(images[i, frame])
        if text is None:
            return artists
        t = time[{axis_time: frame}].ndarray
        text.set_text(t.strftime("%Y-%m-%d %H:%M:%S UTC"))
        return [*artists, text]

    ani = matplotlib.animation.FuncAnimation(
        fig=fig,
        func=func,
        frames=num_time,
    )

    if path.suffix == ".gif":
        writer = matplotlib.animation.PillowWriter(fps=fps)
    else:
        writer = matplotlib.animation.FFMpegWriter(
            fps=fps,
            codec="h264",
            extra_args=["-pix_fmt", "yuv420p", "-crf", "18"],
        )

    ani.save(
        filename=path,
        writer=writer,
        dpi=dpi,
    )

    plt.close(fig)

    return path


if __name__ == "__main__":

    print(cinemagraph_channels())
