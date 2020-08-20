import numpy as np
import pytest

from nannernest import slicing


@pytest.fixture(scope="module")
def profiles():
    return np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )


def test_calc_banana_centroid_single_region():
    mask = np.array([[0.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 0.0]])
    centroid = slicing.calc_banana_centroid(mask, 0.2)
    assert centroid == (1, 1)


def test_find_edges(profiles):
    edges = slicing.find_edges(profiles, axis=1)
    assert np.allclose(edges, np.array([1, 4]))


def test_slice_bottom_top(profiles):
    bottom, top = slicing.slice_bottom_top(profiles[2:5, :], 0.5)
    assert bottom == 0
    assert top == 5
