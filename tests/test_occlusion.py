import inspect
from typing import Tuple

import segno
import numpy as np
from PIL import Image

import branded_qr as bqr


def _occlusion_ratio(mat: Tuple[Tuple[bool, ...], ...], img_w: int, logo_wh: Tuple[int, int], pad_px: int) -> float:
    n = len(mat)
    circle_radius = (max(logo_wh[0], logo_wh[1]) + 2 * pad_px) / 2.0
    circle_cx = img_w / 2.0
    circle_cy = img_w / 2.0
    qr_scale = 16  # keep in sync with default
    border_modules = 8  # keep in sync with default
    total_dark = 0
    occluded = 0
    module_radius = qr_scale / 2.0
    for yy in range(n):
        for xx in range(n):
            if mat[yy][xx]:
                total_dark += 1
                px = (xx + border_modules) * qr_scale
                py = (yy + border_modules) * qr_scale
                mx = px + module_radius
                my = py + module_radius
                d = ((mx - circle_cx) ** 2 + (my - circle_cy) ** 2) ** 0.5
                if d < circle_radius:
                    occluded += 1
    return occluded / max(1, total_dark)


def test_default_occlusion_below_threshold():
    # Read defaults from function signature to stay in sync
    sig = inspect.signature(bqr.make_branded_qr)
    target_frac = sig.parameters["target_frac"].default
    pad_frac = sig.parameters["pad_frac"].default
    qr_scale = sig.parameters["qr_scale"].default
    border_modules = sig.parameters["border_modules"].default
    occl_thresh = sig.parameters["occlusion_threshold"].default

    url = "https://example.com"
    sg_qr = segno.make(url, error="h", boost_error=True)
    try:
        mat = sg_qr.matrix
    except AttributeError:
        mat = sg_qr.to_matrix()
    n = len(mat)

    img_w = (n + 2 * border_modules) * qr_scale

    # Use MQ logo for aspect; any preset logo works
    logo_path = bqr.UNIVERSITY_PRESETS["mq"]["logo_path"]
    logo = Image.open(logo_path).convert("RGBA")
    w, h = logo.size

    target_w = int(img_w * target_frac)
    scale = target_w / max(1, w)
    new_w = max(1, target_w)
    new_h = max(1, int(h * scale))

    pad_px = max(8, int(new_w * pad_frac))

    ratio = _occlusion_ratio(mat, img_w, (new_w, new_h), pad_px)

    # Require that occluded fraction stays conservatively below threshold + margin
    assert ratio <= occl_thresh + 0.05
