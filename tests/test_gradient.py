import sys

import numpy as np
import pytest
from PIL import Image

import branded_qr
from branded_qr import _gradient_color, _hex_to_rgb, make_branded_qr

URL = "https://www.schmidtsciences.org/schmidt-observatory-system/"


def _corner_means(img: Image.Image):
    """Mean RGB of the non-white pixels in the bottom-left and top-right thirds of the image."""
    a = np.asarray(img.convert("RGB")).astype(float)
    h, w, _ = a.shape
    ink = a.sum(axis=2) < 600  # drop white background and anti-aliased edges

    def mean(region_mask):
        m = ink & region_mask
        assert m.sum() > 100
        return a[m].mean(axis=0)

    yy, xx = np.mgrid[0:h, 0:w]
    bottom_left = mean((xx < w / 3) & (yy > 2 * h / 3))
    top_right = mean((xx > 2 * w / 3) & (yy < h / 3))
    return bottom_left, top_right


def test_gradient_color_orientation():
    n = 21
    start, end = (255, 0, 0), (0, 0, 255)
    assert _gradient_color(start, end, 0, n - 1, n) == "#ff0000"  # bottom-left
    assert _gradient_color(start, end, n - 1, 0, n) == "#0000ff"  # top-right


def test_gradient_applied_bottom_left_to_top_right():
    img = make_branded_qr(url=URL, logo_path="data/sos_logo.png", gradient_colors=("#ff0000", "#0000ff"))
    bl, tr = _corner_means(img)
    assert bl[0] > bl[2] + 60  # red dominates bottom-left
    assert tr[2] > tr[0] + 60  # blue dominates top-right


def test_schmidt_preset_uses_logo_colours():
    bl, tr = _corner_means(make_branded_qr(url=URL, university="schmidt"))
    teal, purple = np.array(_hex_to_rgb("#0FB5B5")), np.array(_hex_to_rgb("#774D9F"))
    assert np.linalg.norm(bl - teal) < np.linalg.norm(bl - purple)
    assert np.linalg.norm(tr - purple) < np.linalg.norm(tr - teal)


def test_explicit_finder_colour_overrides_preset_gradient():
    def black_pixels(img):
        a = np.asarray(img.convert("RGB"))
        return int((a.max(axis=2) < 20).sum())

    assert black_pixels(make_branded_qr(url=URL, university="schmidt")) == 0
    assert black_pixels(make_branded_qr(url=URL, university="schmidt", finder_dark_color="#000000")) > 1000


@pytest.mark.parametrize("colour,expected", [
    ("#0FB5B5", (15, 181, 181)),
    ("#fff", (255, 255, 255)),
    ("teal", (0, 128, 128)),
    ("rgb(1, 2, 3)", (1, 2, 3)),
    ("#11223344", (17, 34, 51)),
])
def test_hex_to_rgb_accepts_pillow_colours(colour, expected):
    assert _hex_to_rgb(colour) == expected


@pytest.mark.parametrize("colour", ["#12345", "notacolour", "#0FB5B5ZZ"])
def test_hex_to_rgb_rejects_invalid(colour):
    with pytest.raises(ValueError):
        _hex_to_rgb(colour)


def test_cli_gradient(tmp_path, monkeypatch):
    out = tmp_path / "cli.png"
    monkeypatch.setattr(sys, "argv", [
        "branded-qr", URL, "data/sos_logo.png", "--gradient", "#ff0000", "#0000ff", "-o", str(out),
    ])
    branded_qr.main()
    bl, tr = _corner_means(Image.open(out))
    assert bl[0] > bl[2] + 60
    assert tr[2] > tr[0] + 60
