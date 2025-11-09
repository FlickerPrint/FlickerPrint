#!/usr/bin/env python
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple
from functools import reduce

import flickerprint as gc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def create_axes(
    n_axes, col_wrap=4, axes_height=4, fig_width=None, aspect=1.0, **subplot_kw
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Create a figure and sub-axes, while specifing the size of the sub figures.

    This emulated the behaviour of the seaborn facet grid, where we give a size
    for the individual figures rather than the overall plot.

    The remaining axes are blanked using plt.axis('off')

    Only one of ``axes_height`` or ``fig_width`` should be not None as these
    options conflict. To maintain backwards compatibility, ``axes_height`` will
    override ``fig_width``.

    Parameters
    ----------
    n_axes: int
        Number of sub-axes to create
    col_wrap: int
        How many columns in the figure, next axes will go into new rows.
    axes_height: float or None
        Height of the axes, in inches by default
    fig_width: float or None
        Total width of the figure, conflicts with axes height
    aspect: float
        Width/Height ratio of the sub-axes

    Returns
    -------
    fig, [axes]
        The axes are returned in a flat array, [0,...,n_axes-1]

    """

    n_cols = min(n_axes, col_wrap)
    n_rows = np.ceil(n_axes / n_cols).astype(int)
    n_axes_blank = n_cols * n_rows - n_axes

    # print(f'n_axes = {n_axes}, nBlank = {n_axes_blank}, n_rows = {n_rows}, n_cols = {n_cols}')
    # raise SystemExit

    if fig_width is None:
        axes_width = axes_height * aspect
        fig_width = n_cols * axes_width
        fig_height = n_rows * axes_height

    else:
        axes_width = fig_width / n_cols
        axes_height = axes_width / aspect
        fig_height = axes_height * n_rows

    fig, axs = plt.subplots(
        ncols=n_cols, nrows=n_rows, figsize=(fig_width, fig_height), **subplot_kw,
    )

    # Keep the axes returned in a 1d array
    if n_rows > 1:
        axs = axs.flatten()

    # Blank the axis at the end of the list if we create more than specified
    for blank_axes in range(n_axes_blank):
        axs[-(blank_axes + 1)].axis("off")

    return fig, axs


def hide_axis_lables(ax):
    """ Given an axis, remove the text and spacing used in the tick marks. """
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis("off")


def annotate_axis(ax, label, pos=None, **fontKwargs):
    """Place a label in the figure.

    This defaults to the top left, useful for labelling sub-figures.
    """
    if pos is None:
        pos = (0.05, 0.9)

    ax.annotate(
        label,
        xy=pos,
        xycoords="axes fraction",
        horizontalalignment="left",
        verticalalignment="baseline",
        **fontKwargs,
    )


def set_labels(axs, ylabels=None, xlabels=None, **fontkwargs):
    """Given a set of axes, label those that are in the outermost positions.

    We use the in-built methods on the axes to create y-labels on the left most
    axes and add x-labels to the bottom row.
    """
    if not isinstance(axs, Iterable):
        axs = [axs]
    for ax in axs:
        if ylabels is not None and ax.get_subplotspec().is_first_col():
            ax.set_ylabel(ylabels, **fontkwargs)
        if xlabels is not None and ax.get_subplotspec().is_first_col():
            ax.set_xlabel(xlabels, **fontkwargs)


def force_integer_ticks(ax, x=False, y=False):
    """ Force the tick marks on either of the axes to be in integers. """
    if x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if y:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def format_si(number, precision=3, s="", latex=False):
    """Add SI metric prefix to the number and print to a given specification.

    If the magnitude of the number is not within the range 10^-24 to 10^24 then
    we simply return the number in scientific notation.
    """
    if number == 0:
        return f"{0:{s}.{precision}g} "
    # Round the number first, this tends catches the case when 999 rounds to 1000
    number = float(f"{number:.{precision}e}")
    base_thousand = np.floor(np.log(abs(number)) / np.log(1e3))

    # Ensure that we have a single digits starting with a sign, so that we can
    # look them up in a dictionary.
    prefix_key = f"{int(base_thousand):+1d}"

    # Shift by multiples of 1000 so the number is in the range (1, 1000]
    shortened_digits = number / (1e3 ** base_thousand)

    prefix_dict = {
        # '-8':'y', '-7':'z', '-6':'a', '-5':'f',
        "-4": "p",
        "-3": "n",
        "-2": r"$\mu$" if latex else "μ",
        "-1": "m",
        "+0": "",
        "+1": "k",
        "+2": "M",
        "+3": "G",
        "+4": "T",
        # '+5':'P', '+6':'E', '+7':'Z', '+8':'Y',
    }

    if prefix_key in prefix_dict:
        return f"{shortened_digits:{s}.{precision}g} {prefix_dict[prefix_key]}"
    else:
        return f"{number:{s}.{precision}g} "


def image_comp(
    images: List[np.ndarray],
    save_name: Path,
    titles: List[str] = None,
    axes_height=4,
    fig_width=None,
    col_wrap=4,
    cmap="inferno",
    norm=False,
):
    """Plot several images for comparison.

    Intended to be a one-line convince to compare plots.
    TODO: Add colour-bar and equalisation options.
    """
    n_images = len(images)
    fig, axs = create_axes(
        n_images, col_wrap=col_wrap, axes_height=axes_height, fig_width=fig_width,
    )

    if norm:
        max_vals = [a.max() for a in images]
        max_val = max(max_vals)
        min_vals = [a.min() for a in images]
        min_val = min(min_vals)

    for image, ax, num in zip(images, axs, range(n_images)):
        if norm:
            ax.imshow(image, cmap=cmap, vmax=max_val, vmin=min_val)
        else:
            ax.imshow(image, cmap=cmap)
        if titles is not None:
            ax.set_title(titles[num], size=10)
        hide_axis_lables(ax)

    save_figure_and_trim(save_name, padding=0.05, tl_padding=0.3)


def save_figure_and_trim(
    save_name: Path,
    args=None,
    additional_metadata=None,
    padding=0.15,
    tl_padding=1.08,
    fig=None,
    despine=True,
    dpi=330,
):
    """ Save the figure with metadata and crop the boundaries. """

    # Use the most recent figure if None is provided
    if fig is None:
        fig = plt.gcf()

    # Remove the right and top axis
    axs = fig.get_axes()
    if despine:
        [despine_axis(ax) for ax in axs]

    metadata = {
        "Exp:Creating Script": f"{Path(sys.argv[0]).resolve()}",
        "Exp:Working Dir": f"{os.getcwd()}",
    }

    # Provide the command line arguments
    if args is not None:
        for key, value in vars(args).items():
            metadata[f"Exp:arg-{key}"] = f"{str(value)}"

    # Add user provided keywords
    if additional_metadata is not None:
        metadata.update(additional_metadata)

    plotKwargs = {}
    if padding:
        plotKwargs = dict(bbox_inches="tight", pad_inches=padding)

    fig.tight_layout(pad=tl_padding)
    fig.savefig(save_name, dpi=dpi, **plotKwargs, metadata=metadata)
    plt.close(fig=fig)


def despine_axis(ax):
    """Remove the top and right axis.

    This emulates seaborn.despine, but doesn't require the modules.
    """
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def polar2cart(theta, radii):
    """ Convert polar angles to (x, y). """
    x = radii * np.cos(theta)
    y = radii * np.sin(theta)
    return (x, y)


def cart2polar(point):
    radius = np.sqrt(point[:, 0] ** 2 + point[:, 1] ** 2 + point[:, 2] ** 2)
    theta = np.arccos(point[:, 2] / radius)
    phi = np.arctan2(point[:, 1], point[:, 0])
    return np.array([radius, theta, phi])


def create_figure_directory(name) -> Path:
    """ Create a figure directory within the project structure. """
    print(gec.__file__)


save = save_figure_and_trim
