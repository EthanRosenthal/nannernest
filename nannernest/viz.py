from typing import List, Optional, Tuple

import matplotlib as mpl
from matplotlib.axes._axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from nannernest.nesting import BoundingBox
from nannernest.segmentation import Segment
from nannernest.slicing import BananaCircle
from nannernest.slices import Slice

plt.rcParams["image.origin"] = "lower"


def mask_to_color(
    mask: np.ndarray,
    color: List[float],
    thresh: Optional[float] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    rows, cols = mask.shape
    out = np.ones((rows, cols, 4), dtype=np.float32)
    out[..., :3] = color
    if thresh:
        threshed = mask > thresh
        out[threshed, 3] *= alpha

        out[~threshed, 3] *= 0
    else:
        out[..., 3] *= mask * alpha
    return out


def plot_segment(
    ax: Axes, segment: Segment, mask_color: List[float], fig_scaler: float
) -> Axes:
    ax.add_patch(
        plt.Rectangle(
            (segment.box[0], segment.box[1]),
            segment.box[2] - segment.box[0],
            segment.box[3] - segment.box[1],
            fill=False,
            edgecolor="red",
            linewidth=3.5 * fig_scaler,
        )
    )
    ax.text(
        segment.box[0],
        segment.box[1] - 2,
        "{:s} {:.3f}".format(segment.name, segment.score),
        bbox=dict(facecolor="blue", alpha=0.5),
        fontsize=14 * fig_scaler,
        color="white",
    )
    ax.imshow(mask_to_color(segment.mask, mask_color))
    return ax


def plot_slices(ax: Axes, slices: List[Slice], fig_scaler: float) -> Axes:
    cmap = sns.color_palette("Paired")

    nested_slices = [s for s in slices if s.nested_coords is not None]
    if nested_slices:
        # Copy nested slices over and only plot these slices.
        slices = list(nested_slices)
    # else: Do nothing. We will plot all possible slices on the banana.

    for slice_ in slices:
        point_array = np.array(
            list(
                zip(
                    slice_.upper_right,
                    slice_.lower_right,
                    slice_.lower_left,
                    slice_.upper_left,
                )
            )
        ).T
        ax.add_patch(
            mpl.patches.Polygon(
                point_array,
                closed=True,
                alpha=0.5,
                edgecolor="black",
                facecolor=cmap[slice_.index % len(cmap)],
            )
        )
        ax.annotate(
            f"{slice_.index}",
            slice_.centroid,
            ha="center",
            va="center",
            fontsize=12 * fig_scaler,
        )

        if slice_.nested_coords is not None:
            ax.add_patch(
                mpl.patches.Polygon(
                    slice_.nested_coords,
                    closed=True,
                    alpha=0.5,
                    edgecolor="black",
                    facecolor=cmap[slice_.index % len(cmap)],
                )
            )
            centroid = slice_.nested_coords.mean(axis=0)
            ax.annotate(
                f"{slice_.index}",
                centroid,
                ha="center",
                va="center",
                fontsize=12 * fig_scaler,
            )

    return ax


def plot(
    image: np.ndarray,
    banana: Optional[Segment] = None,
    bread: Optional[Segment] = None,
    slices: Optional[List[Slice]] = None,
    banana_skeleton: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    banana_circle: Optional[BananaCircle] = None,
    banana_centroid: Optional[Tuple[float, float]] = None,
    bread_box: Optional[BoundingBox] = None,
    mask_color: Optional[List[float]] = None,
    output: Optional[str] = "perfect_sandwich.jpg",
    show: bool = False,
):

    # Default color is green   [  R,   G,   B]
    mask_color = mask_color or [0.0, 1.0, 0.0]

    # The figure scaling stuff is from https://stackoverflow.com/a/34769840
    dpi = 80
    height, width, _ = np.array(image).shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)
    # Use fig_scaler to scale all fonts relative to the figure size.
    fig_scaler = max(figsize) / 10

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.imshow(image, aspect="equal")

    for segment in (banana, bread):
        if segment is not None:
            ax = plot_segment(ax, segment, mask_color, fig_scaler)

    if banana_skeleton is not None:
        ax.plot(
            *banana_skeleton, ".", linewidth=1 * fig_scaler, markersize=10 * fig_scaler
        )

    if banana_circle is not None:
        ax.plot(
            *banana_circle.draw(), "orange", linestyle="--", linewidth=2 * fig_scaler,
        )
        # Center of the circle
        ax.plot(
            *banana_circle.center,
            "o",
            markerfacecolor="orange",
            markeredgecolor="None",
            markersize=15 * fig_scaler,
        )

    if banana_centroid is not None:
        # Mask centroid
        ax.plot(
            banana_centroid[1], banana_centroid[0], "or", markersize=15 * fig_scaler,
        )

    if bread_box is not None:
        num_points = len(bread_box.vertices)
        for i in range(num_points):
            ax.plot(
                [bread_box.vertices[i, 0], bread_box.vertices[(i + 1) % num_points, 0]],
                [bread_box.vertices[i, 1], bread_box.vertices[(i + 1) % num_points, 1]],
                color="black",
                linestyle="dashed",
                linewidth=2 * fig_scaler,
            )

    if slices is not None:
        ax = plot_slices(ax, slices, fig_scaler)

    ax.set(xlim=[-0.5, width - 0.5], ylim=[-0.5, height - 0.5], aspect=1)

    if output is not None:
        fig.savefig(output, dpi=dpi, transparent=True)

    if show:
        plt.show()
    else:
        return ax
