# Branded QR

[![Tests](https://github.com/benjaminpope/branded_qr/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/benjaminpope/branded_qr/actions/workflows/tests.yml)
[![Docs](https://github.com/benjaminpope/branded_qr/actions/workflows/deploy-docs.yml/badge.svg?branch=main)](https://github.com/benjaminpope/branded_qr/actions/workflows/deploy-docs.yml)

[![MQ QR linking to repo](docs/assets/examples/mq_repo_qr.png)](https://github.com/benjaminpope/branded_qr)

Generate branded QR codes with a circular logo inset, high error correction, and customizable styling.

## Install

```bash
uv pip install .

# Or with pip
pip install .
```

## CLI

```bash
uv run branded-qr "https://example.com" path/to/logo.png -o example_qr.png
```

Preset branding:

```bash
# MQ preset (uses data/mq_colour.png)
uv run branded-qr --university mq "https://example.com" -o data/aas_QR.png

# UniSQ preset (uses data/unisq_shield_plain.png and finder color #3c2d4d)
uv run branded-qr --university unisq "https://example.com" -o data/aas_QR.png

# UQ preset (uses data/uqlogo.png)
uv run branded-qr --university uq "https://example.com" -o data/aas_QR.png
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