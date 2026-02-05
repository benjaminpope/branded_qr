from __future__ import annotations

import io
import argparse
from typing import Optional, Tuple

import segno
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

__all__ = ["make_branded_qr"]
__version__ = "0.1.1"

# Preset logo and finder color configurations for common universities
UNIVERSITY_PRESETS = {
    "mq": {
        "logo_path": "data/mq_colour.png",
        "finder_dark_color": None,
    },
    "unisq": {
        "logo_path": "data/unisq_shield_plain.png",
        "finder_dark_color": "#3c2d4d",
    },
    "sydney": {
        "logo_path": "data/sydlogo.png",
        "finder_dark_color": None,
    },
}


def make_branded_qr(
    url: str,
    logo_path: Optional[str] = None,
    *,
    target_frac: float = 0.18,
    pad_frac: float = 0.28,
    smooth_sigma: float = 1.2,
    ring_thickness: int = 0,
    ring_color: Tuple[int, int, int, int] = (200, 200, 200, 255),
    qr_scale: int = 10,
    border_modules: int = 4,
    error: str = "h",
    data_dark: str = "black",
    finder_from_logo: bool = True,
    finder_dark_color: Optional[str] = None,
    module_shape: str = "circle",  # "circle" or "square"
    edge_clearance: float = 1.0,
    save_path: Optional[str] = None,
    university: Optional[str] = None,
) -> Image.Image:
    """Generate a branded QR code with a circular inset and logo.

    Version: 0.1.1

    Minimal Usage:
    - img = make_branded_qr(url="https://example.com", logo_path="logo.png")
    - UniSQ branding prefers finder_dark_color="#3c2d4d".

    Parameters
    - url: target URL/text encoded into the QR.
    - logo_path: path to the logo image file.
    - target_frac: fraction of final QR width used for logo.
    - pad_frac: fraction of logo width used for white circular padding.
    - smooth_sigma: Gaussian blur for circle edge softness.
    - ring_thickness: optional outline width for the circle (0 disables).
    - ring_color: outline RGBA color.
    - qr_scale: pixel size per module.
    - border_modules: quiet-zone size in modules.
    - error: segno error-correction level (e.g., 'l','m','q','h').
    - data_dark: color for data modules (e.g., 'black' or '#RRGGBB').
    - finder_from_logo: when True, sample darkest color from logo for finder marks.
    - finder_dark_color: explicit finder dark color (e.g., '#802020'); overrides sampling.
    - module_shape: 'circle' or 'square' for data modules.
    - edge_clearance: multiplier for skipping dots overlapping the inset edge.
    - save_path: optional output filename; if provided, image is saved.

    Notes
    - For colleagues at UniSQ, set finder_dark_color to "#3c2d4d" for brand consistency.

    Returns: PIL.Image.Image of the composed QR.
    """
    # Resolve presets based on university, if provided
    if university:
        uni_key = university.strip().lower()
        preset = UNIVERSITY_PRESETS.get(uni_key)
        if preset is None:
            raise ValueError(f"Unknown university '{university}'. Choose one of: {', '.join(sorted(UNIVERSITY_PRESETS.keys()))}.")
        if logo_path is None:
            logo_path = preset.get("logo_path")
        if finder_dark_color is None and preset.get("finder_dark_color") is not None:
            finder_dark_color = preset.get("finder_dark_color")
        # If a preset doesn't set finder_dark_color, we keep sampling from logo (finder_from_logo=True)

    if logo_path is None:
        raise ValueError("logo_path must be provided, or specify a supported university preset via 'university'.")

    sg_qr = segno.make(url, error=error)
    try:
        mat = sg_qr.matrix
    except AttributeError:
        mat = sg_qr.to_matrix()
    n = len(mat)

    # Output image width in pixels
    img_w = (n + 2 * border_modules) * qr_scale

    # Load and resize logo
    logo_rgba = Image.open(logo_path).convert("RGBA")
    w, h = logo_rgba.size
    target_w = int(img_w * target_frac)
    scale = target_w / max(1, w)
    new_w = max(1, target_w)
    new_h = max(1, int(h * scale))
    new_logo = logo_rgba.resize((new_w, new_h), Image.LANCZOS)

    # Determine finder dark color
    if finder_dark_color is not None:
        finder_dark_hex = finder_dark_color
    else:
        def luminance(c: Tuple[int, int, int]) -> float:
            r, g, b = c
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        if finder_from_logo:
            logo_small = new_logo.convert("RGB").resize((64, 64), Image.LANCZOS)
            colors = logo_small.getcolors(64 * 64) or []
            rgb_colors = [col for cnt, col in colors] if colors else [tuple(p) for p in logo_small.getdata()]
            filtered = [c for c in rgb_colors if luminance(c) < 245] or rgb_colors
            darkest = sorted(set(filtered), key=luminance)[0]
            finder_dark_hex = "#{:02x}{:02x}{:02x}".format(*darkest)
        else:
            finder_dark_hex = "black"

    # Build circular background
    pad = max(8, int(new_w * pad_frac))
    diameter = max(new_w, new_h) + 2 * pad
    bg_size = (diameter, diameter)
    circle_radius = diameter // 2
    bg_center = (img_w // 2, img_w // 2)

    bg = Image.new("RGBA", bg_size, (255, 255, 255, 0))
    circle_mask = Image.new("L", bg_size, 0)
    draw_mask = ImageDraw.Draw(circle_mask)
    draw_mask.ellipse((0, 0, diameter, diameter), fill=255)
    circle_mask = circle_mask.filter(ImageFilter.GaussianBlur(smooth_sigma))
    white_circle = Image.new("RGBA", bg_size, (255, 255, 255, 255))
    bg = Image.composite(white_circle, bg, circle_mask)

    if ring_thickness > 0:
        ring_layer = Image.new("RGBA", bg_size, (0, 0, 0, 0))
        ring_draw = ImageDraw.Draw(ring_layer)
        inset = ring_thickness / 2
        ring_draw.ellipse((inset, inset, diameter - inset, diameter - inset),
                          outline=ring_color, width=ring_thickness)
        bg = Image.alpha_composite(bg, ring_layer)

    # Centroid placement for logo visual mass
    arr = np.array(new_logo)
    rgb = arr[..., :3].astype(np.float32)
    alpha = arr[..., 3]
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    fg = (alpha > 10) & (lum < 245)
    if fg.any():
        ys, xs = np.nonzero(fg)
        cx_logo = xs.mean()
        cy_logo = ys.mean()
    else:
        cx_logo = new_w / 2
        cy_logo = new_h / 2
    base_x = (diameter - new_w) // 2
    base_y = (diameter - new_h) // 2
    offset_x = int(round(new_w / 2 - cx_logo))
    offset_y = int(round(new_h / 2 - cy_logo))
    paste_x = max(0, min(diameter - new_w, base_x + offset_x))
    paste_y = max(0, min(diameter - new_h, base_y + offset_y))
    bg.paste(new_logo, (paste_x, paste_y), new_logo)

    # Render QR modules
    QRimg = Image.new("RGB", (img_w, img_w), "white")
    draw = ImageDraw.Draw(QRimg)
    module_radius = qr_scale / 2.0
    circle_cx, circle_cy = bg_center

    def in_finder(x: int, y: int, size: int) -> bool:
        return ((x < 7 and y < 7) or (x >= size - 7 and y < 7) or (x < 7 and y >= size - 7))

    for y in range(n):
        for x in range(n):
            if mat[y][x]:
                px = (x + border_modules) * qr_scale
                py = (y + border_modules) * qr_scale
                mx = px + module_radius
                my = py + module_radius
                d = ((mx - circle_cx) ** 2 + (my - circle_cy) ** 2) ** 0.5
                is_finder = in_finder(x, y, n)
                color = finder_dark_hex if is_finder else data_dark
                if not is_finder and abs(d - circle_radius) < edge_clearance * module_radius:
                    continue
                if is_finder:
                    draw.rectangle([px, py, px + qr_scale - 1, py + qr_scale - 1], fill=color)
                else:
                    if module_shape == "circle":
                        draw.ellipse([px, py, px + qr_scale - 1, py + qr_scale - 1], fill=color)
                    else:
                        draw.rectangle([px, py, px + qr_scale - 1, py + qr_scale - 1], fill=color)

    # Paste circular background centered
    pos = (bg_center[0] - circle_radius, bg_center[1] - circle_radius)
    QRimg.paste(bg, (int(pos[0]), int(pos[1])), bg.split()[-1])

    if save_path:
        QRimg.save(save_path)
    return QRimg


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a branded QR code with a circular logo inset.")
    parser.add_argument("url", help="URL or text to encode")
    parser.add_argument("logo_path", nargs="?", default=None, help="Path to logo image (optional if --university is supplied)")
    parser.add_argument("--university", type=str, choices=["mq", "unisq", "sydney"], help="Preset branding: mq | unisq | sydney")
    parser.add_argument("-o", "--output", dest="save_path", default="branded_qr.png", help="Output image path")
    parser.add_argument("--target-frac", type=float, default=0.18)
    parser.add_argument("--pad-frac", type=float, default=0.28)
    parser.add_argument("--smooth-sigma", type=float, default=1.2)
    parser.add_argument("--ring-thickness", type=int, default=0)
    parser.add_argument("--ring-color", type=str, default="#c8c8c8")
    parser.add_argument("--qr-scale", type=int, default=10)
    parser.add_argument("--border-modules", type=int, default=4)
    parser.add_argument("--error", type=str, default="h", choices=["l", "m", "q", "h"]) 
    parser.add_argument("--data-dark", type=str, default="black")
    parser.add_argument("--finder-from-logo", action="store_true", default=True)
    parser.add_argument("--finder-dark-color", type=str, default=None)
    parser.add_argument("--module-shape", type=str, default="circle", choices=["circle", "square"]) 
    parser.add_argument("--edge-clearance", type=float, default=1.0)

    args = parser.parse_args()

    # Convert ring_color hex to RGBA if provided as hex
    ring_color = args.ring_color
    if isinstance(ring_color, str) and ring_color.startswith("#") and len(ring_color) in (4, 7):
        rc = ring_color.lstrip("#")
        if len(rc) == 3:
            rc = "".join(ch * 2 for ch in rc)
        r = int(rc[0:2], 16)
        g = int(rc[2:4], 16)
        b = int(rc[4:6], 16)
        ring_color_tuple = (r, g, b, 255)
    else:
        ring_color_tuple = (200, 200, 200, 255)

    img = make_branded_qr(
        url=args.url,
        logo_path=args.logo_path,
        target_frac=args.target_frac,
        pad_frac=args.pad_frac,
        smooth_sigma=args.smooth_sigma,
        ring_thickness=args.ring_thickness,
        ring_color=ring_color_tuple,
        qr_scale=args.qr_scale,
        border_modules=args.border_modules,
        error=args.error,
        data_dark=args.data_dark,
        finder_from_logo=args.finder_from_logo,
        finder_dark_color=args.finder_dark_color,
        module_shape=args.module_shape,
        edge_clearance=args.edge_clearance,
        save_path=args.save_path,
        university=args.university,
    )

    # If not saving, save to default
    if not args.save_path:
        img.save("branded_qr.png")


if __name__ == "__main__":
    main()
