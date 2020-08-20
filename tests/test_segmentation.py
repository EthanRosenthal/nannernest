import numpy as np
import pytest
import torch

from nannernest import segmentation
from nannernest.exceptions import SegmentNotFound


@pytest.fixture(scope="module")
def segmentation_output():
    output = {
        "boxes": torch.tensor([[0, 0, 2, 2], [3, 3, 5, 5]], dtype=torch.float),
        "labels": torch.tensor([52, 54], dtype=torch.int64),  # banana, sandwich
        "scores": torch.tensor([0.9, 0.7]),
        "masks": torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.9, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.9, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ],
            dtype=torch.float,
        ).unsqueeze(1),
    }
    return output


@pytest.fixture(scope="function")
def segments():
    return [
        segmentation.Segment(
            label=np.int64(52),
            score=np.float32(0.9),
            mask=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.9, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
            box=np.array([0, 0, 2, 2]),
        ),
        segmentation.Segment(
            label=np.int64(54),
            score=np.float32(0.7),
            # fmt: of
            mask=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.9, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
            box=np.array([3, 3, 5, 5]),
        ),
    ]


def test_run_integration_test(image_path):
    segmentation.run(image_path)


def test_to_segments(segmentation_output, segments):
    results = segmentation.to_segments(segmentation_output)
    for result, segment in zip(results, segments):
        assert result == segment


class Test_find_best_segment:
    def test_banana_is_found(self, segments):
        banana = segmentation.find_best_segment(segments, "banana")
        assert banana == segments[0]

    def test_banana_is_not_found(self, segments):
        with pytest.raises(SegmentNotFound):
            # Slice out the banana
            segmentation.find_best_segment(segments[1:], "banana")

    def test_best_segment_of_multiple_segments_is_found(self, segments):
        best_segment = segmentation.find_best_segment(
            segments, ["banana", "sandwich", "cake"]
        )
        assert best_segment == segments[0]
