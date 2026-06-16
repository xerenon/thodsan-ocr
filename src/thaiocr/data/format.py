"""จัดข้อมูลเป็น conversation format (ms-swift / Qwen2.5-VL)."""

# instruction สำหรับงานอ่านลายมือ (เฟส 0 อาจปรับให้ตรง prompt ทางการของ typhoon-ocr)
OCR_INSTRUCTION = "อ่านข้อความลายมือภาษาไทยในรูปภาพ แล้วพิมพ์ข้อความออกมาให้ถูกต้อง"


def to_conversation(image_path: str, text: str) -> dict:
    """คืน record รูปแบบ ms-swift: {'messages': [...], 'images': [...]}."""
    return {
        "messages": [
            {"role": "user", "content": f"<image>{OCR_INSTRUCTION}"},
            {"role": "assistant", "content": text},
        ],
        "images": [image_path],
    }


def parquet_to_swift_jsonl(parquet_path: str, out_jsonl: str, img_dir: str) -> int:
    """อ่าน split parquet → เซฟภาพเป็นไฟล์ + เขียน jsonl สำหรับ ms-swift. คืนจำนวน record."""
    import io
    import json
    import os
    import pandas as pd
    from PIL import Image

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    df = pd.read_parquet(parquet_path)
    n = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            img = row["image"]
            # HF parquet เก็บภาพเป็น dict {'bytes':..., 'path':...}
            data = img["bytes"] if isinstance(img, dict) else img
            p = os.path.abspath(os.path.join(img_dir, f"{i}.png"))
            Image.open(io.BytesIO(data)).convert("RGB").save(p)
            rec = to_conversation(p, str(row["text"]))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        c = parquet_to_swift_jsonl(
            f"data/splits/{split}.parquet",
            f"data/swift/{split}.jsonl",
            f"data/swift/images/{split}",
        )
        print(f"{split}: {c} records")
