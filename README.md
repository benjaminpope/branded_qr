# Branded QR

Generate branded QR codes with a circular logo inset, high error correction, and customizable styling.

## Install

```bash
pip install .
```

## CLI

```bash
branded-qr "https://example.com" path/to/logo.png -o example_qr.png
```

Preset branding:

```bash
# MQ preset (uses data/mq_colour.png)
branded-qr --university mq "https://example.com" -o data/aas_QR.png

# UniSQ preset (uses data/unisq_shield_plain.png and finder color #3c2d4d)
branded-qr --university unisq "https://example.com" -o data/aas_QR.png
```

## Python API

```python
from branded_qr import make_branded_qr
img = make_branded_qr(url="https://example.com", logo_path="logo.png")
img.save("example_qr.png")
```

Using presets via `university`:

```python
from branded_qr import make_branded_qr
img = make_branded_qr(url="https://example.com", university="mq")
img.save("data/aas_QR.png")
```

## UniSQ Branding

Set finder dark color to `#3c2d4d` for UniSQ brand consistency.

## Docs

Local docs:

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

## Continuous Deployment

A GitHub Actions workflow builds and deploys the MkDocs site to GitHub Pages on pushes to `main`. After creating the GitHub repo, enable Pages and push this folder.
