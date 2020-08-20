from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Union
import warnings

import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision import transforms

from nannernest.exceptions import SegmentNotFound


SegmentationOutput = Dict[str, torch.Tensor]


@dataclass
class Segment:
    label: np.int64
    name: str = field(init=False)
    score: np.float32
    mask: np.ndarray
    box: np.ndarray

    def __post_init__(self):
        self.name = coco_category_name(self.label)
        self.mask = self.mask.squeeze()

    def __eq__(self, other):
        return (
            self.label == other.label
            and self.name == other.name
            and self.score == other.score
            and np.allclose(self.mask, other.mask)
            and np.allclose(self.box, other.box)
        )


def load_model():
    return maskrcnn_resnet50_fpn(pretrained=True)


def segment_image(
    image_path: Path, model: torch.nn.Module
) -> Tuple[np.ndarray, SegmentationOutput]:
    model = model.eval()

    preprocess = transforms.Compose([transforms.ToTensor()])

    input_image = Image.open(image_path)

    # Flip the image here so that rendering the
    # image with the origin of the image array
    # at the lower left (like a regular plot)
    # makes the image look rightside up.
    input_image = ImageOps.flip(input_image)

    input_tensor = preprocess(input_image)
    # Convert to a mini-batch
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model = model.to("cuda")

    with torch.no_grad():
        with warnings.catch_warnings():
            # There's a pytorch deprecation warning triggered inside of torchvision.
            # I'd like to not have to filter this warning at some point.
            warnings.filterwarnings("ignore", category=UserWarning)
            output = model(input_batch)

    return input_image, output[0]


def coco_category_name(label: int) -> str:
    COCO_INSTANCE_CATEGORY_NAMES = [
        "__background__",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "N/A",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "N/A",
        "backpack",
        "umbrella",
        "N/A",
        "N/A",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "N/A",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "N/A",
        "dining table",
        "N/A",
        "N/A",
        "toilet",
        "N/A",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "N/A",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]
    try:
        return COCO_INSTANCE_CATEGORY_NAMES[label]
    except IndexError:
        raise ValueError(f"Label {label} not found")


def to_segments(segmentation_output: SegmentationOutput) -> List[Segment]:
    boxes = segmentation_output["boxes"].cpu().numpy()
    labels = segmentation_output["labels"].cpu().numpy()
    scores = segmentation_output["scores"].cpu().numpy()
    masks = segmentation_output["masks"].cpu().numpy()
    segments = []
    for box, label, score, mask in zip(boxes, labels, scores, masks):
        segments.append(Segment(label=label, score=score, mask=mask, box=box))
    return segments


def find_best_segment(
    segments: List[Segment], labels: Union[str, List[str]]
) -> Segment:
    if isinstance(labels, str):
        labels = [labels]

    best_segment = None
    for segment in segments:
        if segment.name in labels:
            if best_segment is None or segment.score > best_segment.score:
                best_segment = segment
    if best_segment is None:
        raise SegmentNotFound(f"None of these labels {labels} detected in image")
    return best_segment


def run(image_path: Path) -> Tuple[np.ndarray, Segment, Segment]:
    model = load_model()
    image, segmentation_output = segment_image(image_path, model)
    segments = to_segments(segmentation_output)
    banana = find_best_segment(segments, "banana")
    bread = find_best_segment(segments, ["sandwich", "cake"])

    return image, banana, bread
