"""Synthetic Thai handwriting generator.

เรนเดอร์ข้อความไทยด้วยฟอนต์ลายมือ + augmentation → ได้ภาพ+เฉลย (ground-truth ฟรี)
ใช้เสริมข้อมูลจริง 13.5k เพื่อเพิ่มความครอบคลุม/ทนทาน

ข้อกำหนด:
- วางไฟล์ฟอนต์ลายมือไทย (.ttf/.otf) ไว้ในโฟลเดอร์ `fonts/`
  แหล่ง: https://www.freethaifont.com (หมวด Thai Handwriting), https://www.f0nt.com
- Pillow ต้อง build รองรับ raqm เพื่อจัดวางสระบน/ล่าง+วรรณยุกต์ให้ถูก
  เช็ค: python -c "from PIL import features; print(features.check('raqm'))"
  ถ้า False → สระลอย/วรรณยุกต์เพี้ยน (ข้อมูลปลอมจะหลอกโมเดล) → ต้องแก้ก่อนใช้จริง
"""
import glob
import json
import os
import random

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from thaiocr.data.format import to_conversation


def raqm_ok() -> bool:
    """True ถ้า Pillow จัดวาง Thai combining marks ได้ถูก (ต้องมี raqm)."""
    from PIL import features
    return features.check("raqm")


def list_fonts(font_dir: str) -> list[str]:
    """คืน path ของฟอนต์ .ttf/.otf ทั้งหมดในโฟลเดอร์ (เรียงแน่นอน)."""
    return sorted(
        glob.glob(os.path.join(font_dir, "*.ttf"))
        + glob.glob(os.path.join(font_dir, "*.otf"))
    )


def render_line(text: str, font_path: str, font_size: int = 48,
                pad: int = 16, rng: random.Random | None = None) -> Image.Image:
    """เรนเดอร์ข้อความ 1 บรรทัดด้วยฟอนต์ที่กำหนด บนพื้นขาว."""
    rng = rng or random.Random()
    font = ImageFont.truetype(font_path, font_size)
    measure = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    bbox = measure.textbbox((0, 0), text, font=font)
    w = (bbox[2] - bbox[0]) + 2 * pad
    h = (bbox[3] - bbox[1]) + 2 * pad
    img = Image.new("RGB", (max(w, 1), max(h, 1)), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    ink = rng.randint(0, 60)  # สีหมึกเข้มสุ่ม (ดำ–เทาเข้ม)
    draw.text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=(ink, ink, ink))
    return img


def add_paper(img: Image.Image, rng: random.Random) -> Image.Image:
    """ใส่พื้นหลังกระดาษมีเส้นบรรทัดจางๆ (แบบสมุดไทย)."""
    bg = Image.new("RGB", img.size, (255, 255, 255))
    draw = ImageDraw.Draw(bg)
    gap = rng.randint(28, 48)
    color = (205, 218, 236)  # เส้นฟ้าจาง
    offset = rng.randint(0, gap)
    for y in range(offset, img.height, gap):
        draw.line([(0, y), (img.width, y)], fill=color, width=1)
    # multiply: ขาวคงเดิม หมึกเข้มทับเส้นได้
    return ImageChops.multiply(img, bg)


def augment(img: Image.Image, seed: int | None = None) -> Image.Image:
    """สุ่ม augmentation ให้เหมือนเขียน/สแกนจริง (deterministic ตาม seed)."""
    r = random.Random(seed)
    gen = np.random.default_rng(seed)

    # หมุนเล็กน้อย
    angle = r.uniform(-4.0, 4.0)
    img = img.rotate(angle, expand=True, fillcolor=(255, 255, 255),
                     resample=Image.BICUBIC)

    # เบลอบางครั้ง (เหมือนสแกน/ปากกาเลอะ)
    if r.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(r.uniform(0.3, 1.2)))

    # noise
    if r.random() < 0.6:
        arr = np.asarray(img).astype(np.float32)
        noise = gen.normal(0.0, r.uniform(3.0, 12.0), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


def generate_sample(text: str, font_paths: list[str],
                    seed: int | None = None) -> tuple[Image.Image, str]:
    """สร้าง 1 ตัวอย่าง: เลือกฟอนต์/ขนาดสุ่ม → render → กระดาษ → augment."""
    r = random.Random(seed)
    font_path = r.choice(font_paths)
    size = r.randint(36, 64)
    img = render_line(text, font_path, font_size=size, rng=r)
    if r.random() < 0.5:
        img = add_paper(img, r)
    img = augment(img, seed=r.randint(0, 2**31 - 1))
    return img, text


def generate_dataset(texts, font_dir: str, out_jsonl: str,
                     img_dir: str, seed: int = 42) -> int:
    """สร้าง dataset สังเคราะห์เป็น jsonl (ms-swift format) + images. คืนจำนวน."""
    fonts = list_fonts(font_dir)
    if not fonts:
        raise FileNotFoundError(
            f"ไม่พบฟอนต์ใน '{font_dir}/' — ใส่ไฟล์ .ttf ลายมือไทยก่อน "
            "(ดู docstring ของ synthetic.py)"
        )
    if not raqm_ok():
        print("WARNING: Pillow ไม่มี raqm → สระ/วรรณยุกต์ไทยอาจวางผิด "
              "(แก้ก่อนใช้ข้อมูลจริง)")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    r = random.Random(seed)
    n = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            text = str(text).strip()
            if not text:
                continue
            img, gt = generate_sample(text, fonts, seed=r.randint(0, 2**31 - 1))
            p = os.path.abspath(os.path.join(img_dir, f"syn_{i}.png"))
            img.save(p)
            f.write(json.dumps(to_conversation(p, gt), ensure_ascii=False) + "\n")
            n += 1
    return n


def _load_texts(max_n: int = 200000) -> list[str]:
    """อ่าน corpus: ใช้ data/corpus/thai_lines.txt (ปริมาณ) ถ้ามี ไม่งั้น fall back iapp."""
    corpus = "data/corpus/thai_lines.txt"
    if os.path.exists(corpus):
        with open(corpus, encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()][:max_n]
    print("ไม่พบ data/corpus/thai_lines.txt → fall back ใช้ข้อความจาก iapp "
          "(รัน scripts/fetch_corpus.py เพื่อได้ volume เยอะกว่า)")
    from datasets import load_dataset
    ds = load_dataset("iapp/thai_handwriting_dataset", split="train")
    return [str(ds[i]["text"]) for i in range(min(len(ds), 5000))]


if __name__ == "__main__":
    texts = _load_texts()
    count = generate_dataset(texts, "fonts", "data/swift/synth.jsonl",
                             "data/swift/images/synth")
    print(f"generated {count} synthetic samples -> data/swift/synth.jsonl")
