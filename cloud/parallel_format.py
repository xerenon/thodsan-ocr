"""แตกรูปจาก parquet แบบขนาน (datasets.map num_proc) — เร็วกว่า format.py เดิมมากบนเครื่องหลาย core.
ใช้ parquet ที่ prepare.py สร้างไว้แล้ว (ไม่ต้องโหลด/split ใหม่)
"""
import io
import json
import os
import sys

from PIL import Image
from datasets import load_dataset

sys.path.insert(0, "src")
from thaiocr.data.format import to_conversation


def _save(ex, idx, img_dir):
    img = ex["image"]
    p = os.path.join(img_dir, f"{idx}.png")
    if isinstance(img, Image.Image):
        img.convert("RGB").save(p)
    else:
        data = img["bytes"] if isinstance(img, dict) else img
        Image.open(io.BytesIO(data)).convert("RGB").save(p)
    return {"path": os.path.abspath(p), "text": str(ex["text"])}


def make_split(split: str, num_proc: int = 64) -> int:
    pq = f"data/splits/{split}.parquet"
    if not os.path.exists(pq):
        print(f"skip {pq}")
        return 0
    img_dir = os.path.abspath(f"data/swift/images/{split}")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs("data/swift", exist_ok=True)
    ds = load_dataset("parquet", data_files=pq, split="train")
    out = ds.map(
        _save, with_indices=True, num_proc=num_proc,
        fn_kwargs={"img_dir": img_dir}, remove_columns=ds.column_names,
        desc=f"extract {split}",
    )
    with open(f"data/swift/{split}.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(to_conversation(r["path"], r["text"]), ensure_ascii=False) + "\n")
    return len(out)


if __name__ == "__main__":
    np = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    for s in ["train", "val", "test"]:
        print(s, "->", make_split(s, np))
