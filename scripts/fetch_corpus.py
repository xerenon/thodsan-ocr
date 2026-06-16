"""ดึงคลังข้อความไทยขนาดใหญ่ → บรรทัดสั้นๆ สำหรับ feed synthetic engine.

default: Thai Wikipedia (HF streaming — ไม่ต้องโหลดทั้งก้อน)
output: data/corpus/thai_lines.txt (1 บรรทัด/บรรทัด, dedupe, กรองให้มีอักษรไทย + กรองความยาว)

ตัวอย่าง:
    python scripts/fetch_corpus.py --n 200000
    python scripts/fetch_corpus.py --input_txt my_thai.txt --n 50000   # ใช้ไฟล์ local
"""
import argparse
import os
import re

THAI = re.compile(r"[฀-๿]")


def chunk_text(text: str, min_len: int, max_len: int):
    """ตัดข้อความยาวเป็นท่อนสั้น ๆ ที่ขอบช่องว่าง (ให้พอดีความยาวบรรทัดลายมือ)."""
    for raw in text.split("\n"):
        words = raw.strip().split(" ")
        cur = ""
        for w in words:
            if not w:
                continue
            if len(cur) + len(w) + 1 > max_len:
                if min_len <= len(cur) <= max_len:
                    yield cur.strip()
                cur = w
            else:
                cur = (cur + " " + w).strip()
        if min_len <= len(cur) <= max_len:
            yield cur.strip()


def iter_wikipedia():
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.th", split="train",
                      streaming=True)
    for row in ds:
        yield row["text"]


def iter_file(path: str):
    with open(path, encoding="utf-8") as f:
        yield f.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000, help="จำนวนบรรทัดเป้าหมาย")
    ap.add_argument("--min_len", type=int, default=6)
    ap.add_argument("--max_len", type=int, default=120)
    ap.add_argument("--out", default="data/corpus/thai_lines.txt")
    ap.add_argument("--input_txt", default="", help="ใช้ไฟล์ local แทน Wikipedia")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    source = iter_file(args.input_txt) if args.input_txt else iter_wikipedia()

    seen: set[str] = set()
    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for text in source:
            for line in chunk_text(text, args.min_len, args.max_len):
                if not THAI.search(line) or line in seen:
                    continue
                seen.add(line)
                f.write(line + "\n")
                written += 1
                if written >= args.n:
                    print(f"done: {written} lines -> {args.out}")
                    return
    print(f"done: {written} lines -> {args.out}")


if __name__ == "__main__":
    main()
