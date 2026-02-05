# Usage

## CLI

```bash
branded-qr "https://example.com" path/to/logo.png \
  --target-frac 0.18 --pad-frac 0.28 --smooth-sigma 1.2 \
  --qr-scale 10 --border-modules 4 --error h \
  --module-shape circle --edge-clearance 1.0 \
  --finder-dark-color "#3c2d4d" -o branded_qr.png
```

## Python API

```python
from branded_qr import make_branded_qr
img = make_branded_qr(
    url="https://example.com",
    logo_path="logo.png",
    finder_dark_color="#3c2d4d"
)
img.save("branded_qr.png")
```

## Build Docs Locally

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

Open http://127.0.0.1:8000 in your browser.
