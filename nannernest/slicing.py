from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from skimage import measure
from skimage.morphology import skeletonize

from nannernest.slices import Slice

plt.rcParams["image.origin"] = "lower"

MASK_THRESH = 0.8


class BananaCircle:
    def __init__(
        self, center: Tuple[float, float], radius: float, num_points: int = 201
    ):
        self.theta = np.linspace(0, 2 * np.pi, num_points)
        self.center = center
        self.radius = radius

    @property
    def xc(self) -> float:
        return self.center[0]

    @property
    def yc(self) -> float:
        return self.center[1]

    def draw(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self.xc + self.radius * np.cos(self.theta),
            self.yc + self.radius * np.sin(self.theta),
        )


def calc_banana_skeleton(
    mask: np.ndarray, threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    skeleton = skeletonize(mask > threshold)
    # coordinates of the barycenter
    skeleton_coords = np.where(skeleton)
    # Remember that the columns are the "x" direction
    # and the rows are the "y" direction
    # So switch them here
    skeleton_coords = (skeleton_coords[1], skeleton_coords[0])
    return skeleton_coords


def fit_banana_arc(
    banana_skeleton: Tuple[np.ndarray, np.ndarray]
) -> Tuple[Tuple[float, float], float]:
    """
    Fit a circle to the banana skeleton. The below is straight-up grabbed from
    https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html#Using-scipy.optimize.leastsq

    Parameters
    ----------
    banana_skeleton : tuple(np.ndarray, np.ndarray)
        The coordinates of the banana skeleton as a tuple of (x, y) where x and y are
        arrays.

    Returns
    -------
    center_est
        The estimated center of the circle
    r_est
        The estimated radius of the circle

    """

    def radial_distance(
        xc: float, yc: float, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """calculate the distance of each 2D points from the center (xc, yc)"""
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def radial_distance_differences(
        circle: Tuple[float, float], x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        xc, yc = circle
        """ 
        calculate the algebraic distance between the data points and the mean circle 
        centered at c=(xc, yc) 
        """
        r = radial_distance(xc, yc, x, y)
        return r - r.mean()

    skeleton_center = np.mean(banana_skeleton[0]), np.mean(banana_skeleton[1])

    center_est, ier = scipy.optimize.leastsq(
        radial_distance_differences, skeleton_center, args=banana_skeleton
    )

    r_dist = radial_distance(*center_est, *banana_skeleton)
    r_est = r_dist.mean()

    return center_est, r_est


def calc_banana_centroid(mask: np.ndarray, mask_thresh: float) -> Tuple[float, float]:
    """Given a segmentation banana mask, find the centroid of the masked region"""
    return measure.regionprops((mask > mask_thresh).astype(int), mask)[0].centroid


def circle(
    x: float, y: float, r: float, theta: Union[np.ndarray, float]
) -> Tuple[Union[float, np.ndarray], Union[np.ndarray, float]]:
    return (x + r * np.cos(theta), y + r * np.sin(theta))


def create_phi_space(
    banana_centroid: Tuple[float, float],
    banana_circle: BananaCircle,
    num_points: int = 201,
) -> np.ndarray:
    """
    Create an array of angles centered at the angle that the banana centroid makes with
    respect to the banana circle center (phi_0). The angles span a 3 pi range. For the
    life of me, I can't remember why the angles span a 3 pi range, but the results
    seem to be worse if I reduce the range.
    """

    # Get phi_0, the angle of the centroid wrt the circle center.
    # Specifically, treat the horizontal axis of the image as zero degrees.
    # We will measure all angles relative to the angle that the
    # line between circle center and centroid makes.
    dy = banana_centroid[0] - banana_circle.yc
    dx = banana_centroid[1] - banana_circle.xc
    phi_0 = np.arctan2(dy, dx)

    # For the life of me, I can't remember why the below spans a
    phi_space = np.linspace(phi_0 - 3 * np.pi / 4, phi_0 + 3 * np.pi / 4, num_points)
    return phi_space


def assemble_profiles(phi_space, mask, banana_circle, linewidth=5):

    profiles = []
    for phi in phi_space:
        profiles.append(
            measure.profile_line(
                mask.T,  # For some reason gotta take the transpose
                banana_circle.center,
                circle(
                    banana_circle.xc, banana_circle.yc, 3 * banana_circle.radius, phi
                ),
                linewidth=linewidth,
                mode="constant",
            )[: 2 * int(banana_circle.radius)]
        )

    profiles = np.vstack(profiles)
    return profiles


def find_edges(mat: np.ndarray, axis: int = 1) -> np.ndarray:
    return np.sort(np.argsort(-np.abs(np.diff(np.max(mat, axis=axis), append=0)))[:2])


def find_banana_termini(
    profiles: np.ndarray, mask_threshold: float, frac: float = 0.30
) -> Tuple[int, int]:
    """Find the banana start and end indices of `profiles`"""
    edges = find_edges(profiles > mask_threshold, axis=1)
    length = edges[1] - edges[0]
    delta = int(np.round(frac * length))

    edge1_area = profiles[edges[0] : edges[0] + delta, :].mean()
    edge2_area = profiles[edges[1] - delta : edges[1], :].mean()

    start, end = edges[np.argsort([edge1_area, edge2_area])]

    end_area = max([edge1_area, edge2_area])

    # Now, remove the stem
    # The sign handles the case where start is > end.
    sign = np.sign(end - start)
    sliding_start_areas = np.array(
        [
            profiles[start + sign * i : start + sign * (i + delta) : sign, :].mean()
            for i in np.arange(delta)
        ]
    )
    stem = np.argmin(np.abs(sliding_start_areas - end_area))
    return start + sign * stem, end


def slice_bottom_top(
    profile_slices: np.ndarray, threshold: float
) -> Tuple[float, float]:
    bottom, top = np.sort(
        np.argsort(np.diff((profile_slices > threshold).max(axis=0)))[-2:]
    )
    return bottom, top


def assemble_slices(
    num_slices: int,
    start: float,
    end: float,
    phi_space: np.ndarray,
    banana_circle: BananaCircle,
    profiles: np.ndarray,
    mask_threshold: float,
    ratio: float,
) -> List[Slice]:
    slice_points = np.round(np.linspace(start, end, num_slices + 1)).astype(np.int32)
    slices = []
    for index, (slice_start, slice_end) in enumerate(
        zip(slice_points[:-1], slice_points[1:])
    ):
        slice_ = slice(slice_start, slice_end, np.sign(slice_end - slice_start))
        slice_bottom, slice_top = slice_bottom_top(profiles[slice_, :], mask_threshold)

        # TODO: Convert the slice class into one that either stores both the
        # bounding box coordinates _and_ the polar coordinates. We will need the
        # radius (slice_top - slice_bottom) in order to create the elliptical
        # cross-sectional view of the slices.
        upper_right = circle(
            banana_circle.xc, banana_circle.yc, slice_top, phi_space[slice_end]
        )
        lower_right = circle(
            banana_circle.xc, banana_circle.yc, slice_bottom, phi_space[slice_end]
        )
        lower_left = circle(
            banana_circle.xc, banana_circle.yc, slice_bottom, phi_space[slice_start]
        )
        upper_left = circle(
            banana_circle.xc, banana_circle.yc, slice_top, phi_space[slice_start]
        )

        slices.append(
            Slice(index, upper_right, lower_right, lower_left, upper_left, ratio=ratio)
        )
    return slices


def angular_slice(
    mask: np.ndarray,
    banana_circle: BananaCircle,
    banana_centroid: Tuple[float, float],
    mask_threshold: float,
    ratio: float,
    num_slices: int,
) -> List[Slice]:
    phi_space = create_phi_space(banana_centroid, banana_circle, num_points=201)
    profiles = assemble_profiles(phi_space, mask, banana_circle, linewidth=2)
    start, end = find_banana_termini(profiles, mask_threshold)

    slices = assemble_slices(
        # We'll throw out the first  and last slice, so let's cut 2 extra.
        num_slices + 2,
        start,
        end,
        phi_space,
        banana_circle,
        profiles,
        mask_threshold,
        ratio,
    )
    # Throw out the first and last slices.
    return slices[1:-1]


def run(
    mask: np.ndarray,
    num_slices: int = 22,
    peel_scaler: float = 0.8,
    mask_threshold: float = 0.6,
    ellipse_ratio: float = 0.85,
) -> Tuple[
    List[Slice], BananaCircle, Tuple[float, float], Tuple[np.ndarray, np.ndarray]
]:
    # The first step of slicing involves turning the banana into a parametrized circle.

    # Reduce the banana to a single curved line, 1 pixel wide.
    # See https://scikit-image.org/docs/dev/auto_examples/edges/plot_skeleton.html
    banana_skeleton = calc_banana_skeleton(mask, mask_threshold)

    # Fit a circle to the skeleton
    banana_center, banana_radius = fit_banana_arc(banana_skeleton)
    banana_circle = BananaCircle(banana_center, banana_radius)

    banana_centroid = calc_banana_centroid(mask, mask_threshold)

    slices = angular_slice(
        mask,
        banana_circle,
        banana_centroid,
        mask_threshold,
        ellipse_ratio,
        num_slices,
    )

    for s in slices:
        s.major_axis *= peel_scaler
    return slices, banana_circle, banana_centroid, banana_skeleton
