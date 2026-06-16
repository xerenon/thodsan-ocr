"""สร้างชุดทดสอบ "ตัวพิมพ์" (clean printed) ด้วยฟอนต์มาตรฐาน — ไม่ augment
เพื่อเช็คว่าโมเดลที่ fine-tune ลายมือ ยังอ่านตัวพิมพ์ได้ดีไหม
"""
import json
import os
import random
import sys

sys.path.insert(0, "src")
from thaiocr.data.synthetic import render_line
from thaiocr.data.format import to_conversation

font = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 150
texts = [ln.strip() for ln in open("data/corpus/thai_lines.txt", encoding="utf-8") if ln.strip()][:n]
os.makedirs("data/swift/images/printed", exist_ok=True)
with open("data/swift/printed_test.jsonl", "w", encoding="utf-8") as f:
    for i, t in enumerate(texts):
        img = render_line(t, font, font_size=40, rng=random.Random(i))  # clean, no augment
        p = os.path.abspath(f"data/swift/images/printed/{i}.png")
        img.save(p)
        f.write(json.dumps(to_conversation(p, t), ensure_ascii=False) + "\n")
print("printed test made:", len(texts))
