from typing import Tuple, Optional

import numpy as np


class Slice:
    def __init__(
        self,
        index: int,
        upper_right: Tuple[float, float],
        lower_right: Tuple[float, float],
        lower_left: Tuple[float, float],
        upper_left: Tuple[float, float],
        nested_coords: Optional[np.ndarray] = None,
        ratio: float = 0.85,
    ):
        self.index = index
        self.upper_right = upper_right
        self.lower_right = lower_right
        self.lower_left = lower_left
        self.upper_left = upper_left
        self.nested_coords = nested_coords
        self.ratio = ratio
        self.major_axis = self.calc_major_axis()

    @property
    def minor_axis(self) -> float:
        return self.major_axis * self.ratio

    @property
    def centroid(self) -> Tuple[float, float]:
        return tuple(
            np.array(
                [self.upper_right, self.lower_right, self.lower_left, self.upper_left]
            ).mean(axis=0)
        )

    def calc_major_axis(self) -> float:
        first = euclidean_distance(self.upper_right, self.lower_right)
        second = euclidean_distance(self.upper_left, self.lower_left)
        return (first + second) / 2


def euclidean_distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
