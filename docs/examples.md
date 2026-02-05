# Examples

Below are example QR codes for a few university logos.

> Tip: For colleagues at UniSQ, set `finder_dark_color` to `#3c2d4d`.

## Macquarie University (MQ)

- Website: https://mq.edu.au
- Logo shown:

![MQ Logo](assets/logos/mq_colour.png)

Example code:

```python
from branded_qr import make_branded_qr
img = make_branded_qr(
    url="https://mq.edu.au",
    logo_path="docs/assets/logos/mq_colour.png"
)
img.save("mq_qr.png")
```

## University of Southern Queensland (UniSQ)

- Website: https://unisq.edu.au
- Logo shown:

![UniSQ Logo](assets/logos/unisq_shield_plain.png)

Example code:

```python
from branded_qr import make_branded_qr
img = make_branded_qr(
    url="https://unisq.edu.au",
    logo_path="docs/assets/logos/unisq_shield_plain.png",
    finder_dark_color="#3c2d4d"
)
img.save("unisq_qr.png")
```
