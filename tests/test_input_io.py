"""Input pipeline tests (Pillow image path; video depends on OpenCV)."""

import io

import numpy as np
import pytest
from PIL import Image

from vision.input_io import InputError, load_image_from_bytes, rgb_to_bgr
from vision.model_loader import opencv_available
from vision.preprocessing import decode_image_bytes


def _png_bytes(size=(32, 24), mode="RGB") -> bytes:
    img = Image.new(mode, size, color=(40, 80, 120) if mode == "RGB" else 128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_load_png():
    data = _png_bytes()
    p = load_image_from_bytes(data, "x.png")
    assert p.width == 32 and p.height == 24
    assert p.rgb.shape == (24, 32, 3)


def test_load_rgba_converts_rgb():
    data = _png_bytes(mode="RGBA")
    p = load_image_from_bytes(data, "x.png")
    assert p.rgb.shape[2] == 3


def test_decode_image_bytes_bgr():
    data = _png_bytes()
    bgr = decode_image_bytes(data)
    assert bgr.ndim == 3
    assert bgr.shape[2] == 3


def test_rgb_to_bgr_roundtrip_shape():
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255
    bgr = rgb_to_bgr(rgb)
    assert bgr[0, 0, 2] == 255


def test_empty_image_raises():
    with pytest.raises(InputError):
        load_image_from_bytes(b"", "x.png")


def test_corrupt_image_raises():
    with pytest.raises(InputError):
        load_image_from_bytes(b"not-an-image", "x.png")


def test_opencv_flag_is_bool():
    assert isinstance(opencv_available(), bool)
