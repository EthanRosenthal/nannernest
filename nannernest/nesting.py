from dataclasses import dataclass, field
from glob import glob
import os
from typing import List, Tuple
import xml.etree.ElementTree as etree

import nest2D
import numpy as np
from scipy.spatial import ConvexHull
from skimage import measure
from svgpath2mpl import parse_path
import sympy
from sympy.solvers import solve
from sympy import Symbol


from nannernest.segmentation import Segment
from nannernest.slices import Slice


_MM_IN_COORD_UNITS = 1000000
# L is a magic number from nest2D and is related to the fixed size with which the
# SVG output is written.
_NEST_L = 150000000


@dataclass
class BoundingBox:
    vertices: np.ndarray
    area: float
    height: float
    width: float
    angle: float
    reference: np.ndarray = field(init=False)

    def __post_init__(self):
        self.reference = self.vertices[1, :]


def calc_elliptical_polygon_scalers(num_points: int) -> Tuple[float, float]:
    theta = 2 * np.pi / (num_points - 1)
    a, b = Symbol("a", positive=True), Symbol("b", positive=True)
    da, db = Symbol("da", positive=True), Symbol("db", positive=True)
    # fmt: off
    f1 = (
        (
            (-b * sympy.sin(theta/2))
            / ((a + da) - a * sympy.cos(theta/2))
        )
        - (
            ((b * sympy.sin(theta/2) - (b + db) * sympy.sin(theta))
             / (a * sympy.cos(theta/2) - (a + da) * sympy.cos(theta)))
        )
    )

    f2 = (
        (
            ((b + db) * sympy.sin(theta) - b * sympy.sin(3 * theta / 2))
            / ((a + da) * sympy.cos(theta) - a * sympy.cos(3 * theta / 2))
        )
        - (
            ((b * sympy.sin(3 * theta / 2) - (b + db) * sympy.sin(2 * theta))
             / ((a * sympy.cos(3 * theta / 2) - (a + da) * sympy.cos(2 * theta))))
        )
    )
    # fmt: on
    res = solve([f1, f2], da, db, dict=True)

    # Below tells us what to scale a and b by in our
    # ellipse equation.
    # The original derivation was (a + da)
    # The solutions are in the form of da = coeff * a
    # So, we return in the form of (1 + coeff) * a
    major_scaler = 1 + res[0][da].coeff(a)
    minor_scaler = 1 + res[0][db].coeff(b)
    return major_scaler, minor_scaler


def ellipse_to_polygon(
    slice_: Slice, num_points: int, major_scaler: float, minor_scaler: float
) -> List[Tuple[float, float]]:
    theta = np.linspace(0, 2 * np.pi, num_points + 1)
    # Divide major axis by 2 so that it turns into a radius
    x = (major_scaler * slice_.major_axis / 2) * np.cos(theta)
    y = (minor_scaler * slice_.minor_axis / 2) * np.sin(theta)
    # Reverse the order so that it goes clockwise because nest2D assigns positive
    # (negative) area to clockwise (counterclockwise) polygons
    return list(zip(x, y))[::-1]


def polygons_to_nest_items(polygons: List[List[Tuple[float, float]]], scaler: float):
    items = []
    for polygon in polygons:
        items.append(
            nest2D.Item(
                [nest2D.Point(int(x * scaler), int(y * scaler)) for (x, y) in polygon]
            )
        )
    return items


def max_slices(items, box):
    best = None
    for idx in range(1, len(items)):
        pgrp = nest2D.nest(items[:idx], box)
        if len(pgrp) > 1:
            break
        else:
            best = idx

    if best is None:
        raise RuntimeError("Cannot find good nesting")
    return best


def nest(polygons, bread_box):
    L_x = int(bread_box.width)
    L_y = int(bread_box.height)
    svg_scaler = _NEST_L / L_x
    items = polygons_to_nest_items(polygons, svg_scaler)
    box = nest2D.Box(_NEST_L, int(_NEST_L * L_y / L_x))
    max_slice_idx = max_slices(items, box)
    pgrp = nest2D.nest(items[:max_slice_idx], box)
    writer = nest2D.SVGWriter()
    writer.write_packgroup(pgrp)
    writer.save()


def assign_nested_coords(
    slices: List[Slice], svg_file: str, bread_box: BoundingBox
) -> List[Slice]:
    Y_MAX = 210
    L_x = int(bread_box.width)
    L_y = int(bread_box.height)
    L_y_mm = _NEST_L * L_y / L_x / _MM_IN_COORD_UNITS
    reverse_scaler = _MM_IN_COORD_UNITS * L_x / _NEST_L

    tree = etree.parse(svg_file)
    root = tree.getroot()
    path_elems = root.findall(".//{http://www.w3.org/2000/svg}path")
    paths = [parse_path(elem.attrib["d"]) for elem in path_elems]

    # Rotation matrix to rotate to bread box angle
    rotation_mat = np.array(
        [
            [np.cos(bread_box.angle), -np.sin(bread_box.angle)],
            [np.sin(bread_box.angle), np.cos(bread_box.angle)],
        ]
    )

    for idx, (slice_, path) in enumerate(zip(slices, paths)):

        # Translate to origin
        vertices = np.array(path.vertices)
        vertices[:, 1] -= Y_MAX - L_y_mm
        # Scale to original image size
        scaled = vertices * reverse_scaler

        # Rotate polygons so that they're back at the bread box angle
        # dot product is 2 x 2 * 2 x num_points -> 2 x num_points
        # Transpose that, and we're back to num_points x 2
        scaled = np.dot(rotation_mat, scaled.T).T

        # Translate to bread
        # scaled[:, 0] += bread.box[0]
        # scaled[:, 1] += bread.box[1]
        scaled[:, 0] += bread_box.reference[0]
        scaled[:, 1] += bread_box.reference[1]

        slice_.nested_coords = scaled

    return slices


def minimum_bounding_box(hull_points_2d: np.ndarray) -> BoundingBox:
    """
    Adapted from this blog post
    https://chadrick-kwag.net/python-implementation-of-rotating-caliper-algorithm/

    hull_points_2d: array of hull points. each element should have [x,y] format
    """

    # Compute edges (x2-x1,y2-y1)
    edges = np.zeros((len(hull_points_2d) - 1, 2))  # empty 2 column array
    for i in range(len(edges)):
        edge_x = hull_points_2d[i + 1, 0] - hull_points_2d[i, 0]
        edge_y = hull_points_2d[i + 1, 1] - hull_points_2d[i, 1]
        edges[i] = [edge_x, edge_y]

    # Calculate edge angles   atan2(y/x)
    edge_angles = np.zeros((len(edges)))  # empty 1 column array
    for i in range(len(edge_angles)):
        edge_angles[i] = np.arctan2(edges[i, 1], edges[i, 0])

    # Check for angles in 1st quadrant
    for i in range(len(edge_angles)):
        # want strictly positive answers
        edge_angles[i] = np.abs(edge_angles[i] % (np.pi / 2))

    # Remove duplicate angles
    edge_angles = np.unique(edge_angles)

    min_bbox = None
    min_area = np.finfo("float").max

    for i in range(len(edge_angles)):

        # Create rotation matrix to shift points to baseline
        # R = [ cos(theta)      , cos(theta-PI/2)
        #       cos(theta+PI/2) , cos(theta)     ]
        R = np.array(
            [
                [np.cos(edge_angles[i]), np.cos(edge_angles[i] - (np.pi / 2))],
                [np.cos(edge_angles[i] + (np.pi / 2)), np.cos(edge_angles[i])],
            ]
        )

        # Apply this rotation to convex hull points
        rot_points = np.dot(R, np.transpose(hull_points_2d))  # 2x2 * 2xn

        # Find min/max x,y points
        min_x = np.nanmin(rot_points[0], axis=0)
        max_x = np.nanmax(rot_points[0], axis=0)
        min_y = np.nanmin(rot_points[1], axis=0)
        max_y = np.nanmax(rot_points[1], axis=0)

        # Calculate height/width/area of this bounding rectangle
        width = max_x - min_x
        height = max_y - min_y
        area = width * height

        if area > min_area:
            continue
        else:
            min_area = area

        # Calculate center point and restore to original coordinate system
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_point = np.dot([center_x, center_y], R)

        # Calculate corner points and restore to original coordinate system
        corner_points = np.zeros((4, 2))  # empty 2 column array
        corner_points[0] = np.dot([max_x, min_y], R)
        corner_points[1] = np.dot([min_x, min_y], R)
        corner_points[2] = np.dot([min_x, max_y], R)
        corner_points[3] = np.dot([max_x, max_y], R)

        min_bbox = BoundingBox(
            vertices=corner_points,
            area=area,
            height=height,
            width=width,
            angle=edge_angles[i],
        )
    if min_bbox is None:
        raise RuntimeError("Unable to find a minimum bounding box")
    return min_bbox


def make_bread_convex_hull(bread: Segment, mask_threshold: float) -> np.ndarray:
    props = measure.regionprops((bread.mask > mask_threshold).astype(int), bread.mask)[
        0
    ]
    x_offset, y_offset, *_ = props.bbox
    convex_x, convex_y = np.where(props.convex_image)
    convex_x += x_offset
    convex_y += y_offset
    points = np.vstack((convex_x, convex_y)).T
    convex_hull = ConvexHull(points)
    convex_hull = points[convex_hull.vertices, ::-1]  # Flip it around
    return convex_hull


def make_bread_box(bread: Segment, mask_threshold: float) -> BoundingBox:
    bread_chull = make_bread_convex_hull(bread, mask_threshold)
    return minimum_bounding_box(bread_chull)


def run(
    slices: List[Slice],
    bread: Segment,
    mask_threshold: float = 0.6,
    num_points: int = 30,
) -> Tuple[List[Slice], BoundingBox]:
    major_scaler, minor_scaler = calc_elliptical_polygon_scalers(num_points)
    polygons = [
        ellipse_to_polygon(slice_, num_points, major_scaler, minor_scaler)
        for slice_ in slices
    ]
    bread_box = make_bread_box(bread, mask_threshold)
    nest(polygons, bread_box)

    SVG_FILE = "out.svg"
    slices = assign_nested_coords(slices, SVG_FILE, bread_box)

    # Delete any svg files that were made in the nesting process. If multiple bins are
    # required for nesting, then multiple svg files numbered out1.svg, out2.svg, etc...
    # will be created.
    for f in glob("out*.svg"):
        os.remove(f)
    return slices, bread_box
