from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def image_path():
    return Path(__file__).parent / "assets" / "test_image.jpg"
