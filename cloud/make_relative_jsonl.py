"""สร้าง jsonl ที่ใช้ path สัมพัทธ์ (forward-slash) สำหรับรันบน cloud box (Linux).

ของเดิม (Windows) เก็บ path แบบ absolute เช่น C:\\Work\\...\\images\\train\\0.png
→ แปลงเป็น data/swift/images/train/0.png ให้พกพาข้ามเครื่องได้
ไม่แตะไฟล์ต้นฉบับ (สร้าง *.cloud.jsonl ใหม่ — กันกระทบ job ที่รันอยู่)
"""
import json
import os

PREFIX = "C:/Work/orc-finetune-handwrite/"


def relativize(path: str) -> str:
    p = path.replace("\\", "/")
    if p.startswith(PREFIX):
        p = p[len(PREFIX):]
    return p


def main():
    for split in ["train", "val", "test", "synth"]:
        src = f"data/swift/{split}.jsonl"
        dst = f"data/swift/{split}.cloud.jsonl"
        if not os.path.exists(src):
            print(f"skip {src} (ไม่มี)")
            continue
        n = 0
        with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as o:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["images"] = [relativize(x) for x in rec.get("images", [])]
                o.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        print(f"{split}: {n} -> {dst}")


if __name__ == "__main__":
    main()
