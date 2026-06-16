import random

from PIL import Image, ImageDraw

from thaiocr.data.synthetic import add_paper, augment, list_fonts


def _make_img() -> Image.Image:
    img = Image.new("RGB", (200, 60), "white")
    ImageDraw.Draw(img).rectangle([20, 20, 180, 40], fill=(0, 0, 0))
    return img


def test_list_fonts_empty(tmp_path):
    assert list_fonts(str(tmp_path)) == []


def test_augment_returns_rgb_image():
    out = augment(_make_img(), seed=1)
    assert out.mode == "RGB"
    assert out.size[0] > 0 and out.size[1] > 0


def test_augment_deterministic_with_seed():
    a = augment(_make_img(), seed=7)
    b = augment(_make_img(), seed=7)
    assert list(a.getdata()) == list(b.getdata())


def test_augment_changes_image():
    # augment ต้องเปลี่ยนภาพจริง (ไม่ใช่ no-op)
    src = _make_img()
    out = augment(src, seed=3)
    assert list(out.getdata()) != list(src.getdata())


def test_add_paper_keeps_size():
    img = _make_img()
    out = add_paper(img, random.Random(0))
    assert out.size == img.size
