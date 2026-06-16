"""ดาวน์โหลด + รวม dataset ลายมือไทย (real) จาก Hugging Face เป็น pool เดียว.

คลังข้อมูล (สำรวจ 2026-06): real line-level ภาษาไทยมีน้อยมาก —
`iapp` คือตัวหลักจริงๆ ที่เหลือเป็น subset/eval/char-level
ปริมาณจริงต้องมาจาก synthetic (ดู src/thaiocr/data/synthetic.py)

Burapha-TH (268k char/syllable) ต้องขอ access จากผู้วิจัยเอง (paper: doi 10.3390/app12084083)
→ ใส่เองใน data/external/ แล้วเพิ่ม Source เข้า CATALOG ได้
"""
import argparse
import os
from dataclasses import dataclass


@dataclass
class Source:
    name: str
    hf_id: str
    split: str
    use: str   # "train" | "eval"
    note: str


CATALOG = [
    Source("iapp", "iapp/thai_handwriting_dataset", "train", "train",
           "13.5k real line-level (BEST2019+Wang), Apache-2.0 — แกนหลัก"),
    Source("openthaigpt_eval", "openthaigpt/thai-ocr-evaluation", "test", "eval",
           "104 mixed (มี handwritten), CC-BY-SA-4.0"),
    Source("suchut_eval", "suchut/thaitrocr-eval-dataset-beta", "train", "eval",
           "eval beta"),
    # Thinnaphat/TH-HANDWRITTEN-CPE-OPH2025 = subset ของ iapp → ข้าม (กันซ้ำ)
]


def _find_columns(ds):
    """หา (image_col, text_col) แบบยืดหยุ่นจาก features ของ dataset."""
    from datasets import Image as HFImage

    img_col = None
    for c in ds.column_names:
        feat = ds.features[c]
        if isinstance(feat, HFImage) or c.lower() in ("image", "img", "picture"):
            img_col = c
            break
    text_keys = ("text", "label", "gt", "ground_truth", "transcription",
                 "sentence", "caption")
    txt_col = next((c for c in ds.column_names if c.lower() in text_keys), None)
    return img_col, txt_col


def download_source(src: Source, out_dir: str, limit: int = 0) -> dict:
    """โหลด 1 source → normalize เป็น {image, text} → เซฟ parquet. คืน metadata."""
    from datasets import load_dataset

    try:
        ds = load_dataset(src.hf_id, split=src.split)
    except Exception as e:
        return {"name": src.name, "ok": False, "error": str(e)[:200]}

    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    img_col, txt_col = _find_columns(ds)
    if img_col is None or txt_col is None:
        return {"name": src.name, "ok": False,
                "error": f"หา column ไม่เจอ (cols={ds.column_names})"}

    ds = ds.select_columns([img_col, txt_col]).rename_columns(
        {img_col: "image", txt_col: "text"})

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{src.name}.parquet")
    ds.to_parquet(path)
    return {"name": src.name, "ok": True, "n": len(ds), "use": src.use,
            "path": path, "note": src.note}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pool")
    ap.add_argument("--limit", type=int, default=0, help="0 = ทั้งหมด; ใส่เลขเพื่อ smoke")
    ap.add_argument("--only", default="", help="ชื่อ source เดียว (เช่น openthaigpt_eval)")
    args = ap.parse_args()

    sources = [s for s in CATALOG if not args.only or s.name == args.only]
    results = [download_source(s, args.out, args.limit) for s in sources]

    print("\n=== สรุปการดาวน์โหลด ===")
    total_train = total_eval = 0
    for r in results:
        if r["ok"]:
            print(f"  ✅ {r['name']:18} n={r['n']:>6}  [{r['use']}]  {r['note']}")
            if r["use"] == "train":
                total_train += r["n"]
            else:
                total_eval += r["n"]
        else:
            print(f"  ❌ {r['name']:18} {r['error']}")
    print(f"\nรวม: train={total_train} | eval={total_eval}")
    print("→ ปริมาณหลักให้เสริมด้วย synthetic: python -m thaiocr.data.synthetic")


if __name__ == "__main__":
    main()
