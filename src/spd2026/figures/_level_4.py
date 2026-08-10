"""
The O V scene ESIS recovered from its own flight images.
"""

import pathlib
import matplotlib.animation
import matplotlib.artist
import matplotlib.cm
import matplotlib.colors
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import astropy.time
import astropy.visualization
import named_arrays as na
import sdo
import esis
from ._color import velocity_color_default, percentile_default
from ._layout import figsize_default, bottom_default, height_default
from ._path import default_path

__all__ = [
    "path_level_4_default",
    "level_4",
    "level_4_velocity",
    "level_4_lines",
]

#: Where the Level-4 product of the O V line is kept.
#:
#: The published copy lives on a share, and reading it from there takes an
#: hour where reading it from a local disk takes five seconds, so a local
#: copy is used if one has been made and the share is the fallback.
path_level_4_default = pathlib.Path.home() / ".spd2026/data/esis_level_4_o_v_630.fits"

#: Where the published copy lives, if there is no local one.
path_level_4_remote = pathlib.Path(
    "Z:/esis_level4_gpu/fits/esis_level_4_o_v_630.fits",
)

#: The position of the image of the sky, as a fraction of the figure.
#:
#: Wider than the rect the other figures use, since this is the whole field
#: rather than the patch the raster covers, and it is nearly square: at the
#: height of a slide a square image is as wide as it is tall, and this rect
#: is only there to leave it room and to keep it clear of the key.
#: The gap between the key and the image has to hold four things: the ticks
#: and label of the velocity scale on the right of the key, and the ticks and
#: label of the vertical axis of the image.
_rect_image = (0.225, bottom_default, 0.52, height_default)
_rect_key = (0.065, bottom_default, 0.024, height_default)

#: The power of ten taken out of the brightness scale of the key, so that its
#: two tick labels are short numbers rather than a pair of digits with an
#: exponent hung underneath them, on top of the unit.
_scale_key = 1e11

#: Where event E is, the compact bidirectional event which has been studied
#: so far, as recorded in the README distributed with the Level-4 product.
position_event_e = na.Cartesian2dVectorArray(
    x=47.8 * u.arcsec,
    y=-87.8 * u.arcsec,
)


def _open(path_data: None | pathlib.Path) -> "esis.data.Level_4":
    """Read the Level-4 product, preferring a local copy to the share."""
    if path_data is None:
        path_data = path_level_4_default
        if not path_data.exists():  # pragma: nocover
            path_data = path_level_4_remote
    return esis.data.Level_4.from_fits(path_data)


def level_4(
    path_data: None | pathlib.Path = None,
    center: None | na.Cartesian2dVectorArray = None,
    radius: u.Quantity = 60 * u.arcsec,
    index_time_reference: int = 15,
    velocity_color: u.Quantity = velocity_color_default,
    percentile: float = percentile_default,
    figsize: tuple[float, float] = figsize_default,
    dpi: float = 150,
    fps: int = 5,
    timestamp: bool = True,
    suffix: str = ".mp4",
    path: None | pathlib.Path = None,
) -> pathlib.Path:
    r"""
    Loop over the :math:`\text{O\,V}\;630\,\AA` scene ESIS recovered.

    This is the flight, not a simulation of it: the Level-4 product is the
    time-dependent MART inversion of the thirty Level-1 exposures, on a grid
    of three quarters of an arcsecond and seventeen and a half kilometers per
    second, registered to the sky so that a feature fixed on the Sun stays
    put from one frame to the next.

    The scene is drawn as a false-color image, where the hue is the Doppler
    velocity, on the same color scale as the figures of the synthetic scene.
    A color therefore means the same thing here as it does there, and what
    ESIS actually recovered can be held against what it was expected to.

    The whole field is shown, which is some fourteen arcminutes across,
    where the figures built from the IRIS raster cover a couple of
    arcminutes of it.

    Parameters
    ----------
    path_data
        The Level-4 file to read.
        If :obj:`None`, a local copy if one exists and the published copy on
        the share otherwise, see :obj:`path_level_4_default`.
    center
        The place on the sky to look at.
        If :obj:`None`, the whole field. :obj:`position_event_e` is where
        the event studied so far sits.
    radius
        The half width of the region shown when `center` is given.
    index_time_reference
        The frame whose brightness sets the color scale of every frame.
        The default is the frame the distortion fit was optimized against,
        which is also near the middle of the flight, where the rocket is
        highest and the signal is strongest. The frames at either end are
        therefore dimmer, as they were.
    velocity_color
        The Doppler velocity mapped to each end of the visible spectrum,
        which is the range the other figures use.
    percentile
        The percentile of the reference frame placed at the top of the
        brightness scale, separately at each wavelength.
    figsize
        The width and height of the figure in inches.
    dpi
        The resolution of the saved animation in dots per inch.
    fps
        The number of frames per second in the saved animation.
    timestamp
        Whether to write the time of each exposure above the image.
    suffix
        The file type of the animation, either ``".mp4"`` or ``".gif"``.
        Ignored if `path` is given.
    path
        The location to save the animation.
        If :obj:`None`, it is saved alongside the other figures.

    Notes
    -----
    The radiometry of the four channels is relative rather than absolute,
    so the structure and the velocities of this scene are to be trusted
    further than its brightness.
    """
    a = _open(path_data)

    axis_time = a.axis_time
    axis_wavelength = a.axis_wavelength
    axis_xy = (a.axis_x, a.axis_y)

    position = a.inputs.position
    wavelength = a.inputs.wavelength
    wavelength_rest = a.wavelength_center.ndarray[0]

    # The product is given in wavelength, while the color scale of the other
    # figures is given in Doppler velocity, so the axis is expressed the way
    # the color scale expects to be told about it.
    equivalency = u.doppler_optical(wavelength_rest)
    velocity = wavelength.to(u.km / u.s, equivalencies=equivalency)

    unit_wavelength = na.unit(wavelength)
    unit_position = na.unit(position.x)

    # One scale for the whole flight, taken from the brightest part of it,
    # so that the rise and fall of the signal as the rocket climbs out of
    # the atmosphere and falls back into it is left in rather than divided
    # out.
    reference = a.outputs[{axis_time: index_time_reference}]
    vmax = np.nanpercentile(reference, percentile, axis=axis_xy)

    time = a.inputs.time

    if path is None:
        stem = "level-4-o-v"
        if center is not None:
            stem = f"{stem}-event"
        path = default_path / f"{stem}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)

    with astropy.visualization.quantity_support():

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(_rect_image)
        cax = fig.add_axes(_rect_key)
        cax_twin = cax.twinx()

        if timestamp:
            text = fig.text(
                x=0.5,
                y=0.985,
                s="",
                ha="center",
                va="top",
            )
        else:
            text = None

        if path.suffix == ".gif":
            writer = matplotlib.animation.PillowWriter(fps=fps)
        else:
            writer = matplotlib.animation.FFMpegWriter(
                fps=fps,
                codec="h264",
                extra_args=["-pix_fmt", "yuv420p", "-crf", "14"],
            )

        drawn = [-1]

        def func(index: int) -> list[matplotlib.artist.Artist]:

            if index == drawn[0]:
                return [*ax.collections]
            first = drawn[0] < 0
            drawn[0] = index

            for artist in ax.collections:
                artist.remove()

            colorbar = na.plt.rgbmesh(
                velocity,
                position.x,
                position.y,
                C=a.outputs[{axis_time: index}],
                axis_wavelength=axis_wavelength,
                ax=ax,
                vmin=0,
                vmax=vmax,
                wavelength_min=-velocity_color,
                wavelength_max=+velocity_color,
            )

            # The scale never changes, so the key is only drawn once.
            if first:
                unit = na.unit(colorbar.inputs.x)
                brightness = colorbar.inputs.x / unit / _scale_key

                na.plt.pcolormesh(
                    brightness,
                    colorbar.inputs.y.to(
                        unit_wavelength,
                        equivalencies=equivalency,
                    ),
                    C=colorbar.outputs,
                    axis_rgb=axis_wavelength,
                    ax=cax,
                )
                na.plt.pcolormesh(
                    brightness,
                    colorbar.inputs.y,
                    C=colorbar.outputs,
                    axis_rgb=axis_wavelength,
                    ax=cax_twin,
                )

                cax.set_ylabel(f"wavelength ({unit_wavelength:latex_inline})")
                cax.set_ylim(
                    (-velocity_color).to(unit_wavelength, equivalencies=equivalency),
                    (+velocity_color).to(unit_wavelength, equivalencies=equivalency),
                )
                cax.set_xlim(0, brightness.max().ndarray)
                cax.set_xlabel(
                    f"$10^{{{int(np.log10(_scale_key))}}}$ {unit:latex_inline}"
                )
                cax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=2))
                cax.tick_params(axis="x", labelrotation=45, labelsize="small")

                cax_twin.set_ylabel(f"velocity ({velocity_color.unit:latex_inline})")
                cax_twin.set_ylim(-velocity_color, +velocity_color)

                for artist in [*cax.collections, *cax_twin.collections]:
                    artist.set_rasterized(True)

            for artist in ax.collections:
                artist.set_rasterized(True)

            # A crop is a change of limits rather than of the data: every
            # cell is drawn and the ones outside are clipped, so the region
            # shown is drawn at the resolution of the axes.
            if center is None:
                ax.set_xlim(position.x.min().ndarray, position.x.max().ndarray)
                ax.set_ylim(position.y.min().ndarray, position.y.max().ndarray)
            else:
                ax.set_xlim((center.x - radius).value, (center.x + radius).value)
                ax.set_ylim((center.y - radius).value, (center.y + radius).value)
            ax.set_aspect("equal")
            ax.set_xlabel(f"helioprojective $x$ ({unit_position:latex_inline})")
            ax.set_ylabel(f"helioprojective $y$ ({unit_position:latex_inline})")

            if text is not None:
                t = time[{axis_time: index}].ndarray
                text.set_text(t.strftime("%Y-%m-%d %H:%M:%S UTC"))

            return [*ax.collections]

        ani = matplotlib.animation.FuncAnimation(
            fig=fig,
            func=func,
            frames=a.outputs.shape[axis_time],
        )

        ani.save(
            filename=path,
            writer=writer,
            dpi=dpi,
        )

    plt.close(fig)

    return path


def level_4_velocity(
    path_data: None | pathlib.Path = None,
    center: None | na.Cartesian2dVectorArray = None,
    radius: u.Quantity = 60 * u.arcsec,
    velocity_limit: u.Quantity = 80 * u.km / u.s,
    cmap: str = "RdBu_r",
    color_bad: str = "black",
    figsize: tuple[float, float] = figsize_default,
    dpi: float = 150,
    fps: int = 5,
    timestamp: bool = True,
    suffix: str = ".mp4",
    path: None | pathlib.Path = None,
) -> pathlib.Path:
    r"""
    Loop over the Doppler shift ESIS recovered.

    The same flight as :func:`level_4`, reduced from a spectrum at every
    place on the sky to a single number: the median of the line profile,
    which is the velocity that half the light of that pixel is moving faster
    than and half slower.

    Drawn the way a Doppler shift is conventionally drawn: blue where the
    plasma is coming toward us, red where it is going away, and white where
    it is at rest. The two signs then separate at a glance rather than
    having to be read off a ramp, and, since the quiet Sun is near rest, the
    background settles into the pale middle of the colormap instead of into
    a busy mid-tone.

    The range is a great deal narrower than the range of the spectrum: the
    line moves by a few kilometers per second almost everywhere, so a scale
    reaching the ends of the recovered spectrum would leave the whole field
    the same shade of gray.

    Parameters
    ----------
    path_data
        The Level-4 file to read.
        If :obj:`None`, a local copy if one exists and the published copy on
        the share otherwise, see :obj:`path_level_4_default`.
    center
        The place on the sky to look at.
        If :obj:`None`, the whole field. :obj:`position_event_e` is where
        the event studied so far sits.
    radius
        The half width of the region shown when `center` is given.
    velocity_limit
        The Doppler velocity at each end of the colormap.
    cmap
        The colormap.
        One which runs through white at its middle, since the quantity is
        signed and its middle is where the plasma is at rest.
    color_bad
        The color of the places with no median to report.
        Outside the field stop no light was recorded, so the median of
        nothing is nothing. It has to be a color the colormap itself never
        takes, or those places would read as a velocity: the middle of a
        diverging colormap is white, so white would say the plasma was at
        rest rather than absent.
    figsize
        The width and height of the figure in inches.
    dpi
        The resolution of the saved animation in dots per inch.
    fps
        The number of frames per second in the saved animation.
    timestamp
        Whether to write the time of each exposure above the image.
    suffix
        The file type of the animation, either ``".mp4"`` or ``".gif"``.
        Ignored if `path` is given.
    path
        The location to save the animation.
        If :obj:`None`, it is saved alongside the other figures.

    Notes
    -----
    A median is only as meaningful as the profile it is taken of. Where the
    line is faint the profile is mostly noise and so is its median, which is
    why the faintest places look like static rather than like plasma.
    """
    a = _open(path_data)

    axis_time = a.axis_time
    axis_wavelength = a.axis_wavelength

    position = a.inputs.position
    wavelength_rest = a.wavelength_center.ndarray[0]
    equivalency = u.doppler_optical(wavelength_rest)
    velocity = a.inputs.wavelength.to(u.km / u.s, equivalencies=equivalency)

    unit_position = na.unit(position.x)
    unit_velocity = velocity_limit.unit

    num_time = a.outputs.shape[axis_time]

    # Worked out for every frame before any is drawn, which costs a few
    # hundred megabytes and takes a few seconds, and saves recomputing a
    # frame each time the animation revisits it.
    median = [
        na.pdf.median(
            x=velocity,
            f=a.outputs[{axis_time: i}],
            axis=axis_wavelength,
        ).to_value(unit_velocity)
        for i in range(num_time)
    ]

    time = a.inputs.time

    colormap = plt.get_cmap(cmap).with_extremes(bad=color_bad)

    if path is None:
        stem = "level-4-o-v-velocity"
        if center is not None:
            stem = f"{stem}-event"
        path = default_path / f"{stem}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)

    with astropy.visualization.quantity_support():

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(_rect_image)
        cax = fig.add_axes(_rect_key)

        norm = matplotlib.colors.Normalize(
            vmin=-velocity_limit.to_value(unit_velocity),
            vmax=+velocity_limit.to_value(unit_velocity),
        )
        fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap),
            cax=cax,
            label=f"median Doppler velocity ({unit_velocity:latex_inline})",
        )
        cax.yaxis.set_ticks_position("left")
        cax.yaxis.set_label_position("left")

        if timestamp:
            text = fig.text(x=0.5, y=0.985, s="", ha="center", va="top")
        else:
            text = None

        if path.suffix == ".gif":
            writer = matplotlib.animation.PillowWriter(fps=fps)
        else:
            writer = matplotlib.animation.FFMpegWriter(
                fps=fps,
                codec="h264",
                extra_args=["-pix_fmt", "yuv420p", "-crf", "14"],
            )

        drawn = [-1]

        def func(index: int) -> list[matplotlib.artist.Artist]:

            if index == drawn[0]:
                return [*ax.collections]
            drawn[0] = index

            for artist in ax.collections:
                artist.remove()

            na.plt.pcolormesh(
                position.x,
                position.y,
                C=median[index],
                ax=ax,
                cmap=colormap,
                norm=norm,
            )

            for artist in ax.collections:
                artist.set_rasterized(True)

            # Set every frame, since removing the last one does not put the
            # limits back and drawing the next only ever widens them.
            #
            # A crop is a change of limits rather than of the data: every
            # cell is drawn and the ones outside are clipped, which costs a
            # little time and means the region shown is drawn at the full
            # resolution of the axes rather than of the whole field.
            if center is None:
                ax.set_xlim(position.x.min().ndarray, position.x.max().ndarray)
                ax.set_ylim(position.y.min().ndarray, position.y.max().ndarray)
            else:
                ax.set_xlim((center.x - radius).value, (center.x + radius).value)
                ax.set_ylim((center.y - radius).value, (center.y + radius).value)
            ax.set_aspect("equal")
            ax.set_xlabel(f"helioprojective $x$ ({unit_position:latex_inline})")
            ax.set_ylabel(f"helioprojective $y$ ({unit_position:latex_inline})")

            if text is not None:
                t = time[{axis_time: index}].ndarray
                text.set_text(t.strftime("%Y-%m-%d %H:%M:%S UTC"))

            return [*ax.collections]

        ani = matplotlib.animation.FuncAnimation(
            fig=fig,
            func=func,
            frames=num_time,
        )

        ani.save(filename=path, writer=writer, dpi=dpi)

    plt.close(fig)

    return path


def _intensity(a: "esis.data.Level_4") -> na.AbstractScalarArray:
    r"""
    The light of each line window, integrated over its Doppler range.

    :attr:`esis.data.Level_4.intensity` adds the cells of a window together
    without multiplying by the width of a cell, so what it returns is still a
    spectral radiance: its unit keeps the :math:`\text{\AA}^{-1}` that an
    integral over wavelength would have cancelled.

    Each window is a grid uniform in Doppler velocity about its own rest
    wavelength, so its cells are all one width, and that width is the only
    thing the sum is missing. The windows do not share it, though, running
    from :math:`0.0341\,\text{\AA}` at He I to :math:`0.0368\,\text{\AA}` at
    O V, so leaving it out is not one constant factor across the lines but
    five different ones, and the lines cannot be held against each other
    until it is put back.

    Parameters
    ----------
    a
        The Level-4 product to integrate.
    """
    axis = a.axis_wavelength

    # One more vertex than there are cells, and the extra one is why the
    # window itself cannot be used: it stops at the last cell.
    num = a.num_velocity + 1

    result = []
    for i in range(a.num_line):
        vertices = a.inputs.wavelength[{axis: slice(i * num, (i + 1) * num)}]
        width = (vertices[{axis: -1}] - vertices[{axis: 0}]) / a.num_velocity
        result.append(a.outputs[a.window(i)].sum(axis) * width)

    return na.stack(result, axis=a.axis_line)


def _aia(
    a: "esis.data.Level_4",
    wavelength: u.Quantity = 304 * u.AA,
    tolerance: u.Quantity = 8 * u.s,
) -> tuple[np.ndarray, list[float], u.UnitBase]:
    r"""
    An AIA image of the same field at the same moment as each ESIS frame.

    AIA takes a full disk every twelve seconds and ESIS took a frame every
    ten, so there is an AIA image within a few seconds of every ESIS one, and
    the nearest is the one taken. The two instruments are not synchronized,
    so the offset wanders through the flight rather than being fixed.

    The images are registered, which turns them solar north up on a common
    plate scale of six tenths of an arcsecond and puts disk center at the
    reference pixel. That leaves a grid which is separable and identical from
    one frame to the next, so the crop to the ESIS field is worked out once
    and the rest is slicing.

    Parameters
    ----------
    a
        The Level-4 product whose frames are to be matched.
    wavelength
        The AIA channel.
    tolerance
        How far from an ESIS frame to look for an AIA one.
        Half the AIA cadence would do; a little more, so that a frame is
        found even where the two instruments drift furthest apart.

    Notes
    -----
    Every frame is a separate query, which is slower than asking for the
    whole flight at once but holds only one full disk in memory at a time:
    thirty of them would be four gigabytes, where thirty crops are a quarter
    of that.
    """
    axis_time = a.axis_time
    time = a.inputs.time
    num_time = time.shape[axis_time]

    position = a.inputs.position
    unit_position = na.unit(position.x)

    images = []
    extent = None
    unit = None
    index_x = index_y = None

    for i in range(num_time):

        t = astropy.time.Time(time[{axis_time: i}].ndarray)

        b = sdo.aia.open(
            time_start=t - tolerance,
            time_stop=t + tolerance,
            wavelength=wavelength,
            register=True,
        )

        # The axis a filtergram keeps its channels on, which it names itself
        # and is whichever of its axes is not time and not a detector one.
        (axis_wavelength,) = set(b.outputs.shape) - {
            b.axis_time,
            b.axis_detector_x,
            b.axis_detector_y,
        }

        # The one AIA frame nearest this ESIS frame, of the two or three the
        # window catches.
        time_b = astropy.time.Time(np.ravel(b.inputs.time.ndarray))
        index = int(np.argmin(np.abs((time_b - t).to_value(u.s))))

        index_frame = {b.axis_time: index, axis_wavelength: 0}

        if extent is None:

            # The vertices of the grid, which number one more than the cells
            # and are what a crop has to be expressed in.
            vertex = b.inputs[index_frame].position
            vertex_x = vertex.x[{b.axis_detector_y: 0}].ndarray.to(unit_position)
            vertex_y = vertex.y[{b.axis_detector_x: 0}].ndarray.to(unit_position)

            index_x = _crop(vertex_x, position.x)
            index_y = _crop(vertex_y, position.y)

            extent = [
                vertex_x[index_x.start].value,
                vertex_x[index_x.stop].value,
                vertex_y[index_y.start].value,
                vertex_y[index_y.stop].value,
            ]

        outputs = b.outputs[
            index_frame
            | {
                b.axis_detector_x: index_x,
                b.axis_detector_y: index_y,
            }
        ]

        unit = na.unit(outputs)

        # Named rather than assumed, since `imshow` reads the first axis as
        # the one running up the screen. Cast down on the way out: a count is
        # not worth eight bytes, and thirty frames of them would be worth two
        # gigabytes.
        outputs = outputs.transpose(
            axes=(b.axis_detector_y, b.axis_detector_x),
        )
        images.append(outputs.ndarray.value.astype(np.float32))

    return np.stack(images), extent, unit


def _crop(vertex: u.Quantity, coordinate: na.AbstractScalarArray) -> slice:
    """
    The cells of a grid which cover the span of a coordinate.

    Parameters
    ----------
    vertex
        The vertices of the grid to crop, in increasing order.
    coordinate
        The coordinate whose span is to be covered.
    """
    start = int(np.searchsorted(vertex, coordinate.min().ndarray, side="right")) - 1
    stop = int(np.searchsorted(vertex, coordinate.max().ndarray, side="left"))
    start = max(start, 0)
    stop = min(stop, vertex.size - 1)
    return slice(start, stop)


def level_4_lines(
    path_data: None | pathlib.Path = None,
    wavelength_aia: None | u.Quantity = 304 * u.AA,
    percentile: float = 99.5,
    cmap: str = "gray",
    ncols: int = 3,
    figsize: tuple[float, float] = figsize_default,
    dpi: float = 150,
    fps: int = 5,
    timestamp: bool = True,
    suffix: str = ".mp4",
    path: None | pathlib.Path = None,
) -> pathlib.Path:
    """
    Loop over the intensity of every line ESIS recovered.

    The Level-4 product is inverted one spectral window at a time and
    distributed one file per window, and a whole directory of them reads back
    as a single product of several lines. This shows the intensity of each,
    the light of that window integrated over its Doppler range, so that the
    same moment of the flight can be seen in every line at once.

    The lines are formed at very different temperatures, from the
    chromosphere to the corona, so what changes between the panels is which
    part of the atmosphere is being looked at rather than which part of the
    Sun.

    Each panel is scaled to itself. The lines differ in brightness by far
    more than the structure within any one of them, so a single scale would
    leave the faint windows black; the price is that the panels say which
    structures appear in which line rather than which line is brighter.

    The last panel is AIA rather than ESIS, cropped to the same field and
    matched frame for frame. It is there to be recognized: AIA 304 is a
    channel everybody has looked at, so it says whether what ESIS recovered
    is the Sun that was there, and its own line is He II 304, which is formed
    near where He I 584 is and can be read against it.

    Parameters
    ----------
    path_data
        The directory of Level-4 files to read.
        If :obj:`None`, the directory the local copies are kept in.
    wavelength_aia
        The AIA channel drawn in the panel after the lines.
        If :obj:`None`, that panel is left empty and no AIA data is fetched.
    percentile
        The percentile of each line placed at the top of its own brightness
        scale.
    cmap
        The colormap.
    ncols
        The number of panels across.
    figsize
        The width and height of the figure in inches.
    dpi
        The resolution of the saved animation in dots per inch.
    fps
        The number of frames per second in the saved animation.
    timestamp
        Whether to write the time of each exposure above the panels.
    suffix
        The file type of the animation, either ``".mp4"`` or ``".gif"``.
        Ignored if `path` is given.
    path
        The location to save the animation.
        If :obj:`None`, it is saved alongside the other figures.
    """
    if path_data is None:
        path_data = path_level_4_default.parent

    a = esis.data.Level_4.from_fits(path_data)

    axis_line = a.axis_line
    axis_time = a.axis_time

    intensity = _intensity(a)
    num_line = intensity.shape[axis_line]
    num_time = intensity.shape[axis_time]

    position = a.inputs.position
    unit_position = na.unit(position.x)

    # A uniform grid, so the panels can be drawn as images, which is both
    # quicker than a mesh and cheap to update: only the array behind each
    # image changes from one frame to the next.
    extent = [
        position.x.min().ndarray.value,
        position.x.max().ndarray.value,
        position.y.min().ndarray.value,
        position.y.max().ndarray.value,
    ]

    unit_intensity = na.unit(intensity)

    # Frame first and then the axis running up the screen, which is the order
    # `imshow` reads, so that drawing a frame is a plain index.
    images = [
        intensity[{axis_line: i}]
        .transpose(axes=(axis_time, a.axis_y, a.axis_x))
        .to_value(unit_intensity)
        .ndarray
        for i in range(num_line)
    ]
    extents = [extent] * num_line
    labels = list(a.label_line)
    units = [unit_intensity] * num_line

    # AIA is another instrument on another grid, so it is fetched and cropped
    # rather than sliced out of the product, but from here on it is one more
    # panel like the rest.
    if wavelength_aia is not None:
        images_aia, extent_aia, unit_aia = _aia(a, wavelength_aia)
        images.append(images_aia)
        extents.append(extent_aia)
        labels.append(f"AIA {wavelength_aia.to_value(u.AA):.0f}")
        units.append(unit_aia)

    num_panel = len(images)

    # Each panel against its own brightest places, over the whole flight, so
    # that it neither fades with the signal nor is set by one frame.
    vmax = [float(np.nanpercentile(im, percentile)) for im in images]

    time = a.inputs.time

    nrows = -(-num_panel // ncols)

    if path is None:
        path = default_path / f"level-4-lines{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    axs = np.asarray(axs).ravel()

    artists = []
    for i in range(num_panel):
        ax = axs[i]
        artists.append(
            ax.imshow(
                images[i][0],
                origin="lower",
                extent=extents[i],
                cmap=cmap,
                vmin=0,
                vmax=vmax[i],
                aspect="equal",
            )
        )
        ax.set_title(labels[i])
        fig.colorbar(
            artists[i],
            ax=ax,
            fraction=0.045,
            pad=0.02,
            label=f"intensity ({units[i]:latex_inline})",
        )

    # The panels nothing landed in have nothing to say.
    for ax in axs[num_panel:]:
        ax.set_axis_off()

    # Every panel is cropped to the field ESIS saw, including the AIA one,
    # which arrives as a piece of a full disk and would otherwise set the
    # limits of all of them: the axes are shared.
    axs[0].set_xlim(extent[0], extent[1])
    axs[0].set_ylim(extent[2], extent[3])

    # The lowest panel of each column, which is not the bottom row where a
    # column runs out early: `sharex` hides the tick labels of every panel
    # with another below it, so a panel with an empty one below would
    # otherwise be left with no scale at all.
    for column in range(ncols):
        index = max(i for i in range(num_panel) if i % ncols == column)
        axs[index].set_xlabel(
            f"helioprojective $x$ ({unit_position:latex_inline})",
        )
        axs[index].tick_params(labelbottom=True)
    for ax in axs[::ncols]:
        ax.set_ylabel(f"helioprojective $y$ ({unit_position:latex_inline})")

    # A suptitle rather than free text at the top of the figure, so that the
    # layout leaves room for it instead of laying it over the titles.
    if timestamp:
        text = fig.suptitle("")
    else:
        text = None

    def func(index: int) -> list[matplotlib.artist.Artist]:
        for i, image in enumerate(artists):
            image.set_data(images[i][index])
        if text is None:
            return artists
        t = time[{axis_time: index}].ndarray
        text.set_text(t.strftime("%Y-%m-%d %H:%M:%S UTC"))
        return [*artists, text]

    if path.suffix == ".gif":
        writer = matplotlib.animation.PillowWriter(fps=fps)
    else:
        writer = matplotlib.animation.FFMpegWriter(
            fps=fps,
            codec="h264",
            extra_args=["-pix_fmt", "yuv420p", "-crf", "18"],
        )

    ani = matplotlib.animation.FuncAnimation(fig=fig, func=func, frames=num_time)
    ani.save(filename=path, writer=writer, dpi=dpi)

    plt.close(fig)

    return path
