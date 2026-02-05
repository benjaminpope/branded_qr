import pytest
from PIL import Image

from branded_qr import make_branded_qr


@pytest.mark.parametrize(
    "university,url",
    [
        ("mq", "https://mq.edu.au"),
        ("unisq", "https://unisq.edu.au"),
        ("sydney", "https://www.sydney.edu.au/"),
        ("uq", "https://uq.edu.au"),
    ],
)
def test_make_branded_qr_university(university: str, url: str):
    img = make_branded_qr(url=url, university=university)
    assert isinstance(img, Image.Image)
    w, h = img.size
    assert w > 0 and h > 0
