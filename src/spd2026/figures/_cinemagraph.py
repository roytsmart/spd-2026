"""
An animated loop of the ESIS Level-1 images captured by a single channel.
"""

import pathlib
import dataclasses
import matplotlib.animation
import matplotlib.artist
import matplotlib.pyplot as plt
import named_arrays as na
import esis
from ._path import default_path

__all__ = [
    "frames",
    "cinemagraph",
]


def frames(
    channel: None | str = "Channel 3",
    threshold: float = 0.8,
    normalize: bool = True,
) -> esis.data.Level_1:
    """
    The subset of the ESIS Level-1 images which are suitable for animating.

    The signal recorded during the flight ramps up as the rocket ascends out of
    the absorbing upper atmosphere and ramps back down as it descends,
    so the frames at the start and the end of the observation are much fainter
    than the rest.
    This function isolates a single channel, discards those faint frames,
    and optionally rescales the remaining frames to a common brightness
    so that the animation loops without pulsing.

    Parameters
    ----------
    channel
        The human-readable name of the channel to select,
        see :attr:`esis.data.abc.AbstractChannelData.channel`.
        If :obj:`None`, every channel is kept.
    threshold
        The minimum median signal in a frame, expressed as a fraction of the
        largest median signal in the observation, for that frame to be kept.
    normalize
        If :obj:`True`, scale each frame so that every frame has the same
        median signal.
    """
    obs = esis.flights.f1.data.level_1()

    axis_channel = obs.axis_channel
    axis_time = obs.axis_time
    axis_xy = obs.axis_xy

    if channel is not None:
        obs = obs[{axis_channel: obs.channel == channel}]

    # The median signal in each frame, which falls off at the start and the
    # end of the flight due to absorption by the upper atmosphere.
    #
    # Which frames to keep is decided by the channels together, since they
    # look through the same atmosphere and every channel has to keep the same
    # frames for them to be shown side by side.
    signal = obs.outputs.median(axis=axis_xy)
    signal_mean = signal.mean(axis=axis_channel)

    obs = obs[{axis_time: signal_mean > threshold * signal_mean.max()}]

    if normalize:
        signal = obs.outputs.median(axis=axis_xy)
        obs = dataclasses.replace(
            obs,
            outputs=obs.outputs * signal.mean(axis=axis_time) / signal,
        )

    return obs


def cinemagraph(
    channel: str = "Channel 3",
    threshold: float = 0.8,
    normalize: bool = True,
    cmap: str = "gray",
    percentile_vmin: float = 1,
    percentile_vmax: float = 99.9,
    fps: int = 5,
    timestamp: bool = False,
    suffix: str = ".gif",
    path: None | pathlib.Path = None,
) -> pathlib.Path:
    """
    Save an animation looping over the ESIS Level-1 images of one channel.

    The images fill the entire frame: there are no axes, ticks, labels,
    colorbar, or margins,
    and each pixel of the animation is one pixel of the sensor.

    Parameters
    ----------
    channel
        The human-readable name of the channel to animate.
    threshold
        The minimum median signal in a frame, expressed as a fraction of the
        largest median signal in the observation, for that frame to be kept.
    normalize
        If :obj:`True`, scale each frame so that every frame has the same
        median signal.
    cmap
        The colormap used to map the signal to colors.
    percentile_vmin
        The percentile of the signal to place at the bottom of the color scale.
    percentile_vmax
        The percentile of the signal to place at the top of the color scale.
        The brightest pixels are cosmic ray residuals which are orders of
        magnitude brighter than the Sun, so this should stay well below 100.
    fps
        The number of frames per second in the saved animation.
    timestamp
        If :obj:`True`, overlay the time of each frame in the bottom right
        corner of the image.
    suffix
        The file type of the animation, either ``".gif"`` or ``".mp4"``.
        Ignored if `path` is given.
        Note that an MP4 is much smaller than a GIF, but only loops if the
        program playing it is configured to loop.
    path
        The location to save the animation.
        If :obj:`None`, the animation is saved alongside the other figures.
    """
    obs = frames(
        channel=channel,
        threshold=threshold,
        normalize=normalize,
    )

    axis_time = obs.axis_time
    axis_channel = obs.axis_channel

    if path is None:
        stem = channel.lower().replace(" ", "-")
        path = default_path / f"cinemagraph-{stem}{suffix}"

    path.parent.mkdir(parents=True, exist_ok=True)

    time = obs.inputs.time[{axis_channel: 0}]

    # Gather the images into a single array ordered like an image so that each
    # frame can be handed straight to `imshow`.
    outputs = obs.outputs[{axis_channel: 0}]
    outputs = outputs.transpose((axis_time, obs.axis_y, obs.axis_x))

    # A single color scale shared by every frame.
    vmin = outputs.percentile(percentile_vmin).ndarray
    vmax = outputs.percentile(percentile_vmax).ndarray

    images = outputs.ndarray
    unit = na.unit(outputs)
    if unit is not None:
        images = images.to_value(unit)
        vmin = vmin.to_value(unit)
        vmax = vmax.to_value(unit)

    # A single set of axes which fills the entire figure, sized so that the
    # animation has exactly the same number of pixels as the sensor.
    dpi = 100
    fig = plt.figure(
        figsize=(obs.num_x / dpi, obs.num_y / dpi),
        dpi=dpi,
    )
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()

    image = ax.imshow(
        X=images[0],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        aspect="auto",
    )

    if timestamp:
        text = ax.text(
            x=0.99,
            y=0.01,
            s="",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color="white",
        )
    else:
        text = None

    def func(frame: int) -> list[matplotlib.artist.Artist]:
        image.set_data(images[frame])
        if text is None:
            return [image]
        t = time[{axis_time: frame}].ndarray
        text.set_text(t.strftime("%H:%M:%S"))
        return [image, text]

    ani = matplotlib.animation.FuncAnimation(
        fig=fig,
        func=func,
        frames=obs.shape[axis_time],
    )

    if path.suffix == ".gif":
        writer = matplotlib.animation.PillowWriter(fps=fps)
    else:
        writer = matplotlib.animation.FFMpegWriter(
            fps=fps,
            codec="h264",
            # `yuv420p` is the pixel format understood by the widest range of
            # players, and `crf` is the quality, where smaller is better.
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

    print(cinemagraph(suffix=".gif"))
    print(cinemagraph(suffix=".mp4"))
