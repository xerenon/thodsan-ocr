# Thai Handwriting OCR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune `typhoon-ocr-3b` ให้อ่านลายมือไทยได้แม่นกว่า baseline แล้วเตรียมพร้อม deploy บน RunPod serverless

**Architecture:** เทรน QLoRA บน RTX 5070 Ti (16GB) ด้วย dataset `iapp/thai_handwriting_dataset` (13.5k ภาพมีเฉลย) → วัดผลด้วย CER/WER บน test set จริงที่กันไว้ → merge เป็นโมเดลเต็มเตรียม serve. โปรเจกต์แบ่งโมดูลชัด: data / eval / train / serve

**Tech Stack:** Python 3.11, PyTorch (CUDA 12.8 / Blackwell sm_120), Hugging Face transformers + datasets + peft, ms-swift (training), bitsandbytes (4-bit), jiwer (CER/WER), vLLM + RunPod (serve)

**ขอบเขตแผนนี้:** เฟส 0 (go/no-go) + เฟส 1 (fine-tune จนชนะ baseline). เฟส 2 (synthetic data) และเฟส 3 (deploy เต็มรูปแบบ) จะเป็นแผนแยกภายหลัง

> **หมายเหตุสภาพแวดล้อม:** stack เทรน (ms-swift/bitsandbytes/flash-attn) ทำงานบน Linux ได้ลื่นสุด — บนเครื่อง Windows 5070 Ti แนะนำใช้ **WSL2 (Ubuntu)** คำสั่งในแผนนี้เป็น bash (รันใน WSL2 หรือ Linux ได้เลย)

---

## File Structure

```
orc-finetune-handwrite/
├── requirements.txt              # deps (Task 1)
├── .gitignore                    # (Task 1)
├── README.md                     # (Task 1)
├── src/thaiocr/
│   ├── __init__.py               # (Task 1)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── prepare.py            # โหลด iapp + split ไม่ให้ writer รั่ว (Task 4)
│   │   └── format.py             # จัดเป็น chat format ของ Qwen2.5-VL (Task 5)
│   ├── eval/
│   │   ├── __init__.py
│   │   └── metrics.py            # CER/WER + normalize ภาษาไทย (Task 3)
│   └── serve/
│       ├── __init__.py
│       └── client.py             # client เรียก endpoint (Task 7)
├── scripts/
│   ├── check_env.py              # ตรวจ Blackwell/CUDA/bnb (Task 2)
│   ├── baseline_eval.py          # วัด baseline + วัดหลัง fine-tune (Task 6, 9)
│   ├── train.sh                  # คำสั่ง QLoRA (Task 8)
│   └── merge.py                  # merge LoRA → merged model (Task 10)
├── tests/
│   ├── test_metrics.py           # (Task 3)
│   ├── test_prepare.py           # (Task 4)
│   └── test_format.py            # (Task 5)
└── data/                         # (gitignored) splits + dataset cache
```

---

## Task 1: Project scaffold

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/thaiocr/__init__.py`, `src/thaiocr/data/__init__.py`, `src/thaiocr/eval/__init__.py`, `src/thaiocr/serve/__init__.py`

- [ ] **Step 1: สร้าง `requirements.txt`**

```txt
# PyTorch ต้องลงแยกด้วย index cu128 (Blackwell sm_120) — ดู README
transformers>=4.49.0
datasets>=3.0.0
peft>=0.14.0
accelerate>=1.3.0
bitsandbytes>=0.45.0
ms-swift>=3.0.0
qwen-vl-utils>=0.0.8
jiwer>=3.0.0
pillow>=10.0.0
wandb>=0.18.0
pytest>=8.0.0
```

- [ ] **Step 2: สร้าง `.gitignore`**

```gitignore
__pycache__/
*.pyc
.venv/
venv/
data/
outputs/
checkpoints/
*.safetensors
*.bin
wandb/
.env
```

- [ ] **Step 3: สร้าง `README.md`** (สรุป setup)

```markdown
# Thai Handwriting OCR

Fine-tune typhoon-ocr-3b เพื่ออ่านลายมือไทย → deploy บน RunPod serverless

## Setup (WSL2 Ubuntu / Linux)
\`\`\`bash
python3.11 -m venv .venv && source .venv/bin/activate
# PyTorch สำหรับ Blackwell (RTX 5070 Ti, sm_120):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
\`\`\`

## ตรวจ environment ก่อนเริ่ม (สำคัญ)
\`\`\`bash
python scripts/check_env.py
\`\`\`

ดู design: docs/superpowers/specs/2026-06-16-thai-handwriting-ocr-design.md
```

- [ ] **Step 4: สร้าง package `__init__.py` ทั้ง 4 ไฟล์ (ว่างได้)**

```bash
mkdir -p src/thaiocr/data src/thaiocr/eval src/thaiocr/serve scripts tests data
touch src/thaiocr/__init__.py src/thaiocr/data/__init__.py src/thaiocr/eval/__init__.py src/thaiocr/serve/__init__.py
```

- [ ] **Step 5: ติดตั้ง deps และยืนยัน import**

Run:
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import transformers, datasets, peft, jiwer, PIL; print('imports OK')"
```
Expected: `imports OK` (ไม่มี error)

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore README.md src/
git commit -m "chore: scaffold project structure and dependencies"
```

---

## Task 2: Environment verification (เฟส 0 — Go/No-Go) ⛳

ตรวจว่าเครื่อง Blackwell รัน stack เทรนได้จริง **ก่อน** ลงแรง

**Files:**
- Create: `scripts/check_env.py`

- [ ] **Step 1: เขียน `scripts/check_env.py`**

```python
"""ตรวจ environment สำหรับเทรน QLoRA บน Blackwell GPU (เฟส 0 go/no-go)."""
import sys

def main() -> int:
    ok = True

    import torch
    print(f"torch: {torch.__version__}")
    cuda = torch.cuda.is_available()
    print(f"cuda available: {cuda}")
    if not cuda:
        print("FAIL: ไม่เจอ CUDA GPU")
        return 1

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"gpu: {name}  sm_{cap[0]}{cap[1]}  VRAM={vram_gb:.1f}GB")
    if cap[0] < 9:
        print(f"WARN: compute capability {cap} ต่ำกว่าที่คาด (Blackwell=12.0)")

    # ทดสอบ matmul บน GPU จริง
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    y = (x @ x).float().sum().item()
    print(f"bf16 matmul on GPU: OK ({y:.1f})")

    # ทดสอบ bitsandbytes 4-bit (หัวใจของ QLoRA)
    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes: {bnb.__version__}")
        lin = bnb.nn.Linear4bit(1024, 1024, bias=False).cuda()
        out = lin(x)
        print(f"bnb Linear4bit forward: OK (shape={tuple(out.shape)})")
    except Exception as e:
        print(f"FAIL: bitsandbytes 4-bit ใช้ไม่ได้บน GPU นี้: {e}")
        ok = False

    print("\nRESULT:", "PASS ✅ ไปต่อได้" if ok else "FAIL ❌ ต้องแก้ก่อน")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: รันตรวจ**

Run: `python scripts/check_env.py`
Expected: เห็น `gpu: NVIDIA GeForce RTX 5070 Ti  sm_120 VRAM≈16.0GB`, `bnb Linear4bit forward: OK`, `RESULT: PASS ✅`

> ถ้า FAIL ที่ bitsandbytes/torch บน sm_120 → นี่คือ go/no-go ที่แท้จริง: ลองอัปเดต torch cu128 + bitsandbytes รุ่นล่าสุด; ถ้ายังไม่ได้ ให้พิจารณาเทรนบน cloud GPU (RunPod Pod) ชั่วคราว แล้วค่อยกลับมา deploy เหมือนเดิม **อย่าข้ามไป Task อื่นจนกว่าจะ PASS**

- [ ] **Step 3: Commit**

```bash
git add scripts/check_env.py
git commit -m "feat: add Blackwell/CUDA/bitsandbytes environment check"
```

---

## Task 3: CER/WER metrics (TDD)

**Files:**
- Create: `src/thaiocr/eval/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: เขียน failing test**

```python
# tests/test_metrics.py
from thaiocr.eval.metrics import normalize_thai, cer, wer

def test_normalize_strips_and_collapses_space():
    assert normalize_thai("  สวัสดี   ครับ ") == "สวัสดี ครับ"

def test_normalize_removes_zero_width():
    # zero-width space (U+200B) ต้องถูกลบ
    assert normalize_thai("ก​ข") == "กข"

def test_cer_perfect_match_is_zero():
    assert cer(["สวัสดี"], ["สวัสดี"]) == 0.0

def test_cer_counts_one_char_error():
    # เฉลย "กข" ทำนาย "กค" → ผิด 1 จาก 2 ตัว = 0.5
    assert abs(cer(["กข"], ["กค"]) - 0.5) < 1e-6

def test_wer_perfect_match_is_zero():
    assert wer(["ไป ตลาด"], ["ไป ตลาด"]) == 0.0
```

- [ ] **Step 2: รันให้เห็นว่า fail**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError` หรือ `ImportError: cannot import name 'normalize_thai'`

- [ ] **Step 3: เขียน implementation**

```python
# src/thaiocr/eval/metrics.py
"""CER/WER สำหรับภาษาไทย พร้อม normalization."""
import re
import jiwer

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿"), None)

def normalize_thai(text: str) -> str:
    """ลบ zero-width chars, ตัดช่องว่างหัวท้าย, ยุบช่องว่างซ้ำ."""
    text = text.translate(_ZERO_WIDTH)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def cer(refs: list[str], hyps: list[str]) -> float:
    """Character Error Rate (0.0 = สมบูรณ์แบบ)."""
    refs = [normalize_thai(r) for r in refs]
    hyps = [normalize_thai(h) for h in hyps]
    return float(jiwer.cer(refs, hyps))

def wer(refs: list[str], hyps: list[str]) -> float:
    """Word Error Rate (แบ่งคำด้วยช่องว่าง)."""
    refs = [normalize_thai(r) for r in refs]
    hyps = [normalize_thai(h) for h in hyps]
    return float(jiwer.wer(refs, hyps))
```

- [ ] **Step 4: รันให้ผ่าน**

Run: `PYTHONPATH=src pytest tests/test_metrics.py -v`
Expected: PASS ทั้ง 5 tests

- [ ] **Step 5: Commit**

```bash
git add src/thaiocr/eval/metrics.py tests/test_metrics.py
git commit -m "feat: add Thai-aware CER/WER metrics with TDD"
```

---

## Task 4: Data preparation + no-leakage split (TDD)

dataset มี train split เดียว ไม่มี test → ต้องตัด test เอง **โดยไม่ให้ writer/source เดียวกันรั่วข้าม split** (ไม่งั้นตัวเลข eval หลอก)

**Files:**
- Create: `src/thaiocr/data/prepare.py`
- Test: `tests/test_prepare.py`

- [ ] **Step 1: เขียน failing test (เทสฟังก์ชัน split ด้วยข้อมูลปลอม ไม่ต้องโหลดจริง)**

```python
# tests/test_prepare.py
from thaiocr.data.prepare import split_by_group

def test_split_no_group_overlap():
    # 10 ตัวอย่าง จาก 5 กลุ่ม (กลุ่มละ 2)
    items = [{"id": i, "group": f"g{i//2}"} for i in range(10)]
    train, val, test = split_by_group(items, val_frac=0.2, test_frac=0.2, seed=42)
    g = lambda xs: {x["group"] for x in xs}
    # ไม่มีกลุ่มไหนโผล่ข้าม split
    assert g(train).isdisjoint(g(val))
    assert g(train).isdisjoint(g(test))
    assert g(val).isdisjoint(g(test))

def test_split_covers_all_items():
    items = [{"id": i, "group": f"g{i//2}"} for i in range(10)]
    train, val, test = split_by_group(items, val_frac=0.2, test_frac=0.2, seed=42)
    assert len(train) + len(val) + len(test) == 10

def test_split_deterministic_with_seed():
    items = [{"id": i, "group": f"g{i%4}"} for i in range(20)]
    a = split_by_group(items, 0.2, 0.2, seed=1)
    b = split_by_group(items, 0.2, 0.2, seed=1)
    assert [x["id"] for x in a[0]] == [x["id"] for x in b[0]]
```

- [ ] **Step 2: รันให้ fail**

Run: `PYTHONPATH=src pytest tests/test_prepare.py -v`
Expected: FAIL — `cannot import name 'split_by_group'`

- [ ] **Step 3: เขียน implementation**

```python
# src/thaiocr/data/prepare.py
"""โหลด iapp/thai_handwriting_dataset และตัด split แบบไม่ให้กลุ่มรั่ว."""
import random
from typing import Any

def split_by_group(
    items: list[dict[str, Any]],
    val_frac: float,
    test_frac: float,
    seed: int = 42,
    group_key: str = "group",
) -> tuple[list, list, list]:
    """แบ่ง train/val/test โดยจัดทั้งกลุ่มไปอยู่ split เดียว (กัน leakage)."""
    groups: dict[str, list] = {}
    for it in items:
        groups.setdefault(it[group_key], []).append(it)

    keys = list(groups.keys())
    random.Random(seed).shuffle(keys)

    n = len(keys)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_keys = set(keys[:n_test])
    val_keys = set(keys[n_test:n_test + n_val])

    train, val, test = [], [], []
    for k in keys:
        bucket = test if k in test_keys else val if k in val_keys else train
        bucket.extend(groups[k])
    return train, val, test


def load_and_split(out_dir: str = "data/splits", seed: int = 42):
    """โหลด dataset จริงจาก HF แล้วเซฟ split เป็น parquet. คืนจำนวนต่อ split."""
    from datasets import load_dataset
    import os

    ds = load_dataset("iapp/thai_handwriting_dataset", split="train")
    # ใช้ label_file เป็น group proxy (ตัวบ่งแหล่ง/ผู้เขียน) เพื่อกัน leakage
    items = [{"idx": i, "group": str(ds[i]["label_file"])} for i in range(len(ds))]
    train, val, test = split_by_group(items, val_frac=0.05, test_frac=0.10, seed=seed)

    os.makedirs(out_dir, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        idxs = [x["idx"] for x in part]
        ds.select(idxs).to_parquet(f"{out_dir}/{name}.parquet")
    return {"train": len(train), "val": len(val), "test": len(test)}


if __name__ == "__main__":
    print(load_and_split())
```

- [ ] **Step 4: รันให้ผ่าน**

Run: `PYTHONPATH=src pytest tests/test_prepare.py -v`
Expected: PASS ทั้ง 3 tests

- [ ] **Step 5: โหลด dataset จริง + สร้าง split**

Run: `PYTHONPATH=src python -m thaiocr.data.prepare`
Expected: พิมพ์ `{'train': ~11500, 'val': ~700, 'test': ~1300}` (ตัวเลขประมาณ) และมีไฟล์ `data/splits/{train,val,test}.parquet`

> ⚠️ ถ้า `label_file` มี cardinality ต่ำมาก (กลุ่มใหญ่ไม่กี่กลุ่ม) จน split ไม่สมดุล → ปรับ `group_key` ไปใช้ field อื่นที่บ่งผู้เขียนได้ละเอียดกว่า แล้วรันใหม่

- [ ] **Step 6: Commit**

```bash
git add src/thaiocr/data/prepare.py tests/test_prepare.py
git commit -m "feat: load iapp dataset with leakage-safe group split"
```

---

## Task 5: Prompt formatting for Qwen2.5-VL (TDD)

แปลง (image, text) → conversation format ที่ ms-swift/Qwen2.5-VL คาดหวัง

**Files:**
- Create: `src/thaiocr/data/format.py`
- Test: `tests/test_format.py`

- [ ] **Step 1: เขียน failing test**

```python
# tests/test_format.py
from thaiocr.data.format import OCR_INSTRUCTION, to_conversation

def test_conversation_has_image_and_instruction():
    msg = to_conversation("path/to/img.png", "สวัสดี")
    user = msg["messages"][0]
    assert user["role"] == "user"
    assert OCR_INSTRUCTION in user["content"]
    assert msg["images"] == ["path/to/img.png"]

def test_conversation_assistant_is_ground_truth():
    msg = to_conversation("x.png", "ข้อความเฉลย")
    assert msg["messages"][1] == {"role": "assistant", "content": "ข้อความเฉลย"}
```

- [ ] **Step 2: รันให้ fail**

Run: `PYTHONPATH=src pytest tests/test_format.py -v`
Expected: FAIL — `cannot import name 'to_conversation'`

- [ ] **Step 3: เขียน implementation**

```python
# src/thaiocr/data/format.py
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
    import json, os
    import pandas as pd
    from PIL import Image
    import io

    os.makedirs(img_dir, exist_ok=True)
    df = pd.read_parquet(parquet_path)
    n = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            img = row["image"]
            # HF parquet เก็บภาพเป็น dict {'bytes':..., 'path':...}
            data = img["bytes"] if isinstance(img, dict) else img
            p = os.path.join(img_dir, f"{i}.png")
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
```

- [ ] **Step 4: รันให้ผ่าน**

Run: `PYTHONPATH=src pytest tests/test_format.py -v`
Expected: PASS ทั้ง 2 tests

- [ ] **Step 5: แปลง split จริงเป็น jsonl + images**

Run: `PYTHONPATH=src python -m thaiocr.data.format`
Expected: `train: ~11500 records`, `val: ...`, `test: ...` และมีโฟลเดอร์ `data/swift/`

- [ ] **Step 6: Commit**

```bash
git add src/thaiocr/data/format.py tests/test_format.py
git commit -m "feat: format dataset into ms-swift conversation jsonl"
```

---

## Task 6: Baseline evaluation (เฟส 0 — รู้ว่าเริ่มจากตรงไหน) ⛳

วัด CER/WER ของ **base typhoon-ocr-3b (ยังไม่ fine-tune)** บน test set → นี่คือ baseline ที่ต้องเอาชนะ

**Files:**
- Create: `scripts/baseline_eval.py`

- [ ] **Step 1: เขียน `scripts/baseline_eval.py`**

```python
"""วัด CER/WER ของโมเดลบน test split. ใช้ได้ทั้ง base และ fine-tuned."""
import argparse, json, sys, io
import pandas as pd
from PIL import Image
import torch

sys.path.insert(0, "src")
from thaiocr.eval.metrics import cer, wer
from thaiocr.data.format import OCR_INSTRUCTION

def load_model(model_path: str):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    return model, proc

def predict(model, proc, pil_img) -> str:
    from qwen_vl_utils import process_vision_info
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_img},
        {"type": "text", "text": OCR_INSTRUCTION},
    ]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(messages)
    inputs = proc(text=[text], images=imgs, videos=vids, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    trimmed = out[:, inputs.input_ids.shape[1]:]
    return proc.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="typhoon-ai/typhoon-ocr-3b")
    ap.add_argument("--test", default="data/splits/test.parquet")
    ap.add_argument("--limit", type=int, default=0, help="0 = ทั้งหมด; ใส่เลขเพื่อ smoke test")
    ap.add_argument("--out", default="outputs/baseline_report.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.test)
    if args.limit:
        df = df.head(args.limit)
    model, proc = load_model(args.model)

    refs, hyps = [], []
    for _, row in df.iterrows():
        img = row["image"]
        data = img["bytes"] if isinstance(img, dict) else img
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        hyps.append(predict(model, proc, pil))
        refs.append(str(row["text"]))

    report = {"model": args.model, "n": len(refs),
              "cer": cer(refs, hyps), "wer": wer(refs, hyps)}
    import os; os.makedirs("outputs", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test ก่อน (10 ตัวอย่าง) ให้ pipeline เดินได้**

Run: `python scripts/baseline_eval.py --limit 10 --out outputs/baseline_smoke.json`
Expected: พิมพ์ JSON มี `cer`/`wer` เป็นตัวเลข (ไม่ crash). ค่าจะสูง (base ยังไม่เก่งลายมือ) — ปกติ

> ถ้าโหลดโมเดลด้วย `Qwen2_5_VLForConditionalGeneration` ไม่ได้ ให้ลอง `AutoModelForImageTextToText` แทน หรืออัปเดต transformers

- [ ] **Step 3: รัน baseline เต็ม test set**

Run: `python scripts/baseline_eval.py --out outputs/baseline_report.json`
Expected: ได้ `outputs/baseline_report.json` มี baseline CER/WER เต็ม — **บันทึกเลขนี้ไว้ นี่คือเส้นที่ต้องชนะ**

- [ ] **Step 4: Commit**

```bash
git add scripts/baseline_eval.py outputs/baseline_report.json
git commit -m "feat: baseline eval of base typhoon-ocr-3b on test set"
```

---

## Task 7: RunPod base-model smoke deploy + วัด latency/cost (เฟส 0) ⛳

ยืนยันว่า deploy path ใช้ได้จริง **ก่อน** ทุ่มเทรน — deploy โมเดล base ขึ้น RunPod แล้ววัดเวลา/ราคา

**Files:**
- Create: `src/thaiocr/serve/client.py`

- [ ] **Step 1: ตั้ง RunPod Serverless endpoint (ทำบนเว็บ RunPod)**

ขั้นตอน (บันทึกผลลง `outputs/runpod_phase0.md`):
1. RunPod Console → Serverless → New Endpoint → เลือก **vLLM** worker template
2. ตั้ง model = `typhoon-ai/typhoon-ocr-3b`, GPU = **L4 (24GB)** หรือ A5000, Flex workers (max 1)
3. เปิด **FlashBoot**
4. คัดลอก **Endpoint ID** + สร้าง **API Key**

- [ ] **Step 2: เขียน `src/thaiocr/serve/client.py`**

```python
"""Client เรียก RunPod serverless endpoint (วัด latency)."""
import base64, time, os
import requests

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def ocr_via_runpod(image_path: str, endpoint_id: str | None = None,
                   api_key: str | None = None) -> tuple[str, float]:
    """ส่งภาพไป endpoint, คืน (ข้อความ, latency วินาที)."""
    endpoint_id = endpoint_id or os.environ["RUNPOD_ENDPOINT_ID"]
    api_key = api_key or os.environ["RUNPOD_API_KEY"]
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    b64 = encode_image(image_path)
    payload = {"input": {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": "อ่านข้อความลายมือภาษาไทยในรูปภาพ แล้วพิมพ์ออกมา"},
    ]}], "max_tokens": 256}}
    t0 = time.time()
    r = requests.post(url, json=payload, headers={"Authorization": f"Bearer {api_key}"}, timeout=300)
    r.raise_for_status()
    dt = time.time() - t0
    data = r.json()
    text = data["output"]["choices"][0]["message"]["content"]
    return text, dt

if __name__ == "__main__":
    import sys
    txt, dt = ocr_via_runpod(sys.argv[1])
    print(f"latency={dt:.1f}s\ntext={txt}")
```

- [ ] **Step 3: ทดสอบ + วัด (ครั้งแรก = cold start, ครั้งสอง = warm)**

Run:
```bash
export RUNPOD_ENDPOINT_ID=... RUNPOD_API_KEY=...
python src/thaiocr/serve/client.py data/swift/images/test/0.png   # cold
python src/thaiocr/serve/client.py data/swift/images/test/0.png   # warm
```
Expected: ครั้งแรก latency สูง (cold ~15–60s), ครั้งสองต่ำ (warm ไม่กี่วินาที) — บันทึกทั้งสองค่า + ประมาณ cost/หน้า ลง `outputs/runpod_phase0.md`

> นี่คือ go/no-go ของ deploy: ถ้า warm latency/cost ยอมรับได้ → เดินหน้า. ถ้าไม่ → ทบทวน (Active worker, GPU เล็กลง, หรือ self-host 5070 Ti)

- [ ] **Step 4: Commit**

```bash
git add src/thaiocr/serve/client.py outputs/runpod_phase0.md
git commit -m "feat: RunPod serverless client + phase-0 latency/cost check"
```

---

## Task 8: QLoRA fine-tune (เฟส 1)

**Files:**
- Create: `scripts/train.sh`

- [ ] **Step 1: เขียน `scripts/train.sh`**

```bash
#!/usr/bin/env bash
# QLoRA fine-tune typhoon-ocr-3b บน RTX 5070 Ti (16GB) ด้วย ms-swift
set -euo pipefail

swift sft \
  --model typhoon-ai/typhoon-ocr-3b \
  --dataset data/swift/train.jsonl \
  --val_dataset data/swift/val.jsonl \
  --train_type lora \
  --quant_method bnb \
  --quant_bits 4 \
  --bnb_4bit_compute_dtype bfloat16 \
  --torch_dtype bfloat16 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --freeze_vit true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --num_train_epochs 3 \
  --max_length 2048 \
  --save_steps 200 \
  --eval_steps 200 \
  --logging_steps 10 \
  --output_dir outputs/qlora \
  --report_to wandb
```

- [ ] **Step 2: Smoke test (จำกัด step เพื่อยืนยันว่าเทรนเดินได้ + fit VRAM)**

Run:
```bash
chmod +x scripts/train.sh
# ลองสั้นๆ ก่อน: เพิ่ม --max_steps 20 ชั่วคราว
swift sft --model typhoon-ai/typhoon-ocr-3b --dataset data/swift/train.jsonl \
  --train_type lora --quant_method bnb --quant_bits 4 --torch_dtype bfloat16 \
  --target_modules all-linear --freeze_vit true \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --max_steps 20 --output_dir outputs/qlora_smoke
```
Expected: เทรนเดิน 20 steps ไม่ OOM, loss พิมพ์ออกมา, มี adapter ใน `outputs/qlora_smoke`

> ถ้า **OOM** บน 16GB: ลด `--max_length` (เช่น 1536), ยืนยัน `--freeze_vit true`, ลด `--lora_rank` เป็น 8, หรือเปิด `--gradient_checkpointing true`

- [ ] **Step 3: รันเทรนเต็ม**

Run: `./scripts/train.sh`
Expected: เทรนจบ 3 epochs, eval loss ลดลง, adapter สุดท้ายอยู่ใน `outputs/qlora` (ดู curve ใน W&B)

- [ ] **Step 4: Commit**

```bash
git add scripts/train.sh
git commit -m "feat: QLoRA fine-tuning script for typhoon-ocr-3b"
```

---

## Task 9: Post-finetune eval + เทียบ baseline (เฟส 1)

**Files:**
- ใช้ `scripts/baseline_eval.py` (Task 6) ซ้ำ — ชี้ไป adapter

- [ ] **Step 1: รัน eval บนโมเดลที่ fine-tune แล้ว**

Run:
```bash
# ms-swift มี swift infer/eval ที่รับ adapter ได้ตรงๆ:
swift infer --adapters outputs/qlora --val_dataset data/swift/test.jsonl \
  --max_new_tokens 256 --result_path outputs/ft_predictions.jsonl
```
Expected: ได้ไฟล์ predictions `outputs/ft_predictions.jsonl`

- [ ] **Step 2: คำนวณ CER/WER จาก predictions แล้วเทียบ baseline**

เขียน `scripts/score_predictions.py`:
```python
"""อ่าน predictions jsonl (มี label + response) → คำนวณ CER/WER เทียบ baseline."""
import json, sys
sys.path.insert(0, "src")
from thaiocr.eval.metrics import cer, wer

refs, hyps = [], []
with open(sys.argv[1], encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        refs.append(d.get("label") or d["messages"][-1]["content"])
        hyps.append(d["response"])

ft = {"cer": cer(refs, hyps), "wer": wer(refs, hyps), "n": len(refs)}
base = json.load(open("outputs/baseline_report.json", encoding="utf-8"))
print("baseline:", base["cer"], "| fine-tuned:", ft["cer"])
print("CER improvement:", round((base["cer"] - ft["cer"]) * 100, 2), "points")
json.dump(ft, open("outputs/ft_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
```

Run: `python scripts/score_predictions.py outputs/ft_predictions.jsonl`
Expected: `fine-tuned cer` **ต่ำกว่า** `baseline cer` อย่างชัดเจน (CER improvement เป็นบวก) — นี่คือเกณฑ์สำเร็จเฟส 1

> ถ้า **ไม่ดีขึ้น/แย่ลง:** ตรวจ (1) prompt format ตรงกับที่โมเดลคาดหวังไหม (เทียบ repo scb-10x/typhoon-ocr), (2) learning rate, (3) epochs น้อยไป, (4) test set รั่วหรือ label เพี้ยน

- [ ] **Step 3: Commit**

```bash
git add scripts/score_predictions.py outputs/ft_report.json outputs/ft_predictions.jsonl
git commit -m "feat: post-finetune eval, beats baseline CER"
```

---

## Task 10: Merge LoRA → merged model (เฟส 1 → พร้อม deploy)

**Files:**
- Create: `scripts/merge.py`

- [ ] **Step 1: เขียน `scripts/merge.py`**

```python
"""Merge LoRA adapter เข้า base → โมเดลเต็มพร้อม serve."""
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", default="outputs/qlora")
    ap.add_argument("--out", default="outputs/typhoon-ocr-3b-thaihw-merged")
    args = ap.parse_args()
    from swift.llm import export_main, ExportArguments
    export_main(ExportArguments(adapters=args.adapters, merge_lora=True, output_dir=args.out))
    print("merged ->", args.out)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: รัน merge**

Run: `python scripts/merge.py`
Expected: มีโฟลเดอร์ `outputs/typhoon-ocr-3b-thaihw-merged/` ที่มี config + safetensors ครบ

> ทางเลือก: ถ้า `export_main` API ต่างเวอร์ชัน ใช้ CLI แทน: `swift export --adapters outputs/qlora --merge_lora true --output_dir outputs/typhoon-ocr-3b-thaihw-merged`

- [ ] **Step 3: ยืนยัน merged model ใช้ได้ (eval ซ้ำสั้นๆ ต้องได้ผลใกล้เคียง fine-tuned)**

Run: `python scripts/baseline_eval.py --model outputs/typhoon-ocr-3b-thaihw-merged --limit 50 --out outputs/merged_smoke.json`
Expected: CER ใกล้เคียงกับ `outputs/ft_report.json` (ยืนยัน merge ไม่ทำพัง)

- [ ] **Step 4: Commit**

```bash
git add scripts/merge.py outputs/merged_smoke.json
git commit -m "feat: merge LoRA into deployable merged model"
```

---

## ผลลัพธ์เมื่อจบแผนนี้

- ✅ ยืนยันแล้วว่าเครื่อง Blackwell เทรนได้ + RunPod รันได้ (เฟส 0 ผ่าน)
- ✅ มี baseline CER/WER ที่วัดจริง
- ✅ มีโมเดล fine-tune ที่ **CER ดีกว่า baseline** บน test set จริง
- ✅ มี merged model พร้อม push ขึ้น RunPod (เฟส 3)

**แผนถัดไป (เขียนแยก):**
- เฟส 2: Synthetic Data Engine (ฟอนต์ลายมือไทย + augmentation) → ดัน accuracy ต่อ
- เฟส 3: Deploy เต็มรูปแบบ (push merged model → RunPod endpoint จริง → client บน EPYC → end-to-end)
- เฟส 4: ขยายโดเมน (ใบสั่งยา/ฟอร์ม) + active learning
