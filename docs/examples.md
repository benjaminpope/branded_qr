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
    university="mq"
)
img.save("mq_qr.png")
```

Rendered example:

![MQ QR](assets/examples/mq_qr.png)

## University of Southern Queensland (UniSQ)

- Website: https://unisq.edu.au
- Logo shown:

![UniSQ Logo](assets/logos/unisq_shield_plain.png)

Example code:

```python
from branded_qr import make_branded_qr
img = make_branded_qr(
    url="https://unisq.edu.au",
    university="unisq"
)
img.save("unisq_qr.png")
```

Rendered example:

![UniSQ QR](assets/examples/unisq_qr.png)

## University of Sydney

- Website: https://www.sydney.edu.au/
- Logo shown: (requires `data/sydlogo.png` in your repo)

Example code:

```python
from branded_qr import make_branded_qr
img = make_branded_qr(
    url="https://www.sydney.edu.au/",
    university="sydney"
)
img.save("sydney_qr.png")
```

Rendered example:

![Sydney QR](assets/examples/sydney_qr.png)
