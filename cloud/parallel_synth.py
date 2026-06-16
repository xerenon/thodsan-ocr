"""สร้าง synthetic แบบขนาน (multiprocessing) — เร็วกว่า single-thread มากบนเครื่องหลาย core."""
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, "src")
from thaiocr.data.synthetic import generate_sample, list_fonts
from thaiocr.data.format import to_conversation

_FONTS = None
_IMG_DIR = None


def _init(fonts, img_dir):
    global _FONTS, _IMG_DIR
    _FONTS, _IMG_DIR = fonts, img_dir


def _work(args):
    i, text = args
    img, gt = generate_sample(text, _FONTS, seed=i)
    p = os.path.abspath(os.path.join(_IMG_DIR, f"syn_{i}.png"))
    img.save(p)
    return json.dumps(to_conversation(p, gt), ensure_ascii=False)


if __name__ == "__main__":
    n_proc = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    fonts = list_fonts("fonts")
    img_dir = os.path.abspath("data/swift/images/synth")
    os.makedirs(img_dir, exist_ok=True)
    texts = [ln.strip() for ln in open("data/corpus/thai_lines.txt", encoding="utf-8") if ln.strip()]
    items = list(enumerate(texts))
    with Pool(n_proc, initializer=_init, initargs=(fonts, img_dir)) as pool:
        recs = pool.map(_work, items, chunksize=16)
    with open("data/swift/synth.jsonl", "w", encoding="utf-8") as f:
        f.write("\n".join(recs) + "\n")
    print("generated", len(recs), "synthetic samples")
