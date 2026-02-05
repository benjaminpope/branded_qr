# Branded QR

Generate QR codes with a circular logo inset, high error correction, and customizable styling. Aimed at Australian university branding for use on slides and posters.

## Quick Start

1. Install the package locally:

```bash
pip install .
```

2. Generate a QR via CLI:

```bash
branded-qr "https://example.com" path/to/logo.png -o example_qr.png
```

3. Or use it from Python:

```python
from branded_qr import make_branded_qr
img = make_branded_qr(url="https://example.com", logo_path="logo.png")
img.save("example_qr.png")
```

## UniSQ Branding

For colleagues at UniSQ, set the finder dark color to:

```text
finder_dark_color = "#3c2d4d"
```

## Features

- High error correction (`error='h'`) via Segno
- Circular white inset with optional ring
- Finder squares, circular data modules
- Logo centroid alignment for visual centering

## Disclaimer

This is not subscribed to any official branding guidelines. Use at your own risk and feel free to contribute improvements.