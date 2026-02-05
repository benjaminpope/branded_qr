import pytest

from PIL import Image

from branded_qr import make_branded_qr


def _decode_pyzbar(img: Image.Image):
    try:
        from pyzbar.pyzbar import decode as _decode
    except Exception:
        pytest.skip("pyzbar/zbar not available; skipping decode test")
    return _decode(img)


@pytest.mark.parametrize(
    "university,url",
    [
        ("mq", "https://mq.edu.au"),
        ("unisq", "https://unisq.edu.au"),
        ("sydney", "https://www.sydney.edu.au/"),
        ("uq", "https://uq.edu.au"),
    ],
)
def test_round_trip_decode(university: str, url: str):
    img = make_branded_qr(
        url=url,
        university=university,
        verify_decode=False,
    )
    res = _decode_pyzbar(img)
    assert any(sym.data.decode("utf-8", "ignore") == url for sym in res)
