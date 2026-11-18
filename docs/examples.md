# Examples

Below are example QR codes for a few university logos.

> Tip: For colleagues at UniSQ, set `finder_dark_color` to `#3c2d4d`.

## Macquarie University (MQ)

- Website: https://mq.edu.au

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

## University of Queensland (UQ)

- Website: https://uq.edu.au

Example code:

```python
from branded_qr import make_branded_qr
img = make_branded_qr(
    url="https://uq.edu.au",
    university="uq"
)
img.save("uq_qr.png")
```

Rendered example:

![UQ QR](assets/examples/uq_qr.png)

## arXiv

- Website: https://arxiv.org/archive/astro-ph

Example code:

```python
from branded_qr import make_branded_qr
img = make_branded_qr(
    url="https://arxiv.org/archive/astro-ph",
    university="arxiv"
)
img.save("arxiv_qr.png")
```

Rendered example:

![arXiv QR](assets/examples/arxiv_qr.png)

## Schmidt Sciences (Eric and Wendy Schmidt Observatory System)

- Website: https://www.schmidtsciences.org/schmidt-observatory-system/

The logo is the coloured part of the Observatory System mark, without its text. Modules follow the mark's diagonal gradient, from teal (`#0FB5B5`) at bottom left to purple (`#774D9F`) at top right.

Example code:

```python
from branded_qr import make_branded_qr
img = make_branded_qr(
    url="https://www.schmidtsciences.org/schmidt-observatory-system/",
    university="schmidt"
)
img.save("schmidt_qr.png")
```

Rendered example:

![Schmidt QR](assets/examples/schmidt_qr.png)
