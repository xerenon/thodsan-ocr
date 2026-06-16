# 🖋️ Thai Handwriting OCR — ต่อยอด Typhoon ให้อ่านลายมือไทย

> Fine-tuned **`typhoon-ocr-3b`** to read **handwritten Thai** — slashing character error rate by **66%**.

<p>
<img alt="base" src="https://img.shields.io/badge/base-typhoon--ocr--3b%20(Qwen2.5--VL)-5b6ee1">
<img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-green">
<img alt="cer" src="https://img.shields.io/badge/CER-35.3%25%20→%2011.97%25-brightgreen">
<img alt="task" src="https://img.shields.io/badge/task-Thai%20Handwriting%20OCR-orange">
</p>

ภาษาไทยเขียนด้วยลายมือคือหนึ่งในงาน OCR ที่ **หินที่สุด** — สระบน/ล่าง วรรณยุกต์ซ้อน ตัวเขียนติดกัน ไม่เว้นวรรคระหว่างคำ โปรเจกต์นี้พิสูจน์ว่าเราเอาโมเดล OCR ไทยแบบเปิด (Typhoon) มา fine-tune ต่อให้ **อ่านลายมือเก่งขึ้นแบบก้าวกระโดด** ได้จริง

---

## 🏆 ผลลัพธ์ (Results)

| Metric | Baseline (typhoon ดิบ) | **Fine-tuned (ตัวนี้)** | Improvement |
|:---|:---:|:---:|:---:|
| **CER** (Character Error Rate) | 35.3% | **11.97%** | **−66%** ⬇️ |
| อ่านผิดโดยเฉลี่ย | ~1 ใน 3 ตัวอักษร | **~1 ใน 8 ตัวอักษร** | |
| Exact-match บรรทัด | — | **25%** (อ่านถูกทั้งบรรทัด) | |

> เทรน QLoRA 2 epochs บน `iapp/thai_handwriting_dataset` (13,550 บรรทัดลายมือจริง จาก 2,026 ผู้เขียน)

### 🔍 อ่านจริง: เฉลย vs โมเดล (ตัวอย่างจากชุดทดสอบ)

```
✅ ถูกเป๊ะทั้งประโยค
   เฉลย : ในเมื่อรัฐยังต้องให้เงินอุดหนุนแก่นักศึกษาในระดับปริญญาตรีในอัตราสูง
   โมเดล: ในเมื่อรัฐยังต้องให้เงินอุดหนุนแก่นักศึกษาในระดับปริญญาตรีในอัตราสูง

✅ เกือบเป๊ะ (CER 1.3%) — ประโยคยาว ผิดตัวเดียว
   เฉลย : ...จะเห็นว่ามีความสัมพันธ์กับกลอนหรือคำขับ
   โมเดล: ...จะเห็นว่ามีความสัมพันธ์กับกลอนหรือค้ำขับ
```

---

## ✨ Features

- 🇹🇭 **อ่านลายมือไทยระดับประโยค** — รับภาพบรรทัดลายมือ → คืนข้อความ
- 🧠 **ต่อยอดจาก Typhoon OCR** (Qwen2.5-VL, Apache-2.0) — เก็บความรู้ภาษาไทยเดิมไว้ครบ เติมความเก่งลายมือเข้าไป
- ⚙️ **เทรนบน GPU เดียว** — QLoRA 4-bit, รันได้ตั้งแต่ RTX 5070 Ti (16GB) ถึง RTX PRO 6000 (96GB)
- 🏭 **Synthetic Data Engine** — ปั๊มภาพลายมือสังเคราะห์จากฟอนต์ลายมือไทย 12 แบบ (OFL) + augmentation
- ☁️ **Cloud-ready** — kit ครบสำหรับเทรนบน RunPod (setup → train → eval → merge)

---

## 🏗️ Architecture

```
[ภาพลายมือ] → typhoon-ocr-3b + LoRA adapter (fine-tuned) → [ข้อความไทย]

Train:  iapp 13.5k (+synthetic) ──QLoRA 4-bit──► adapter ──merge──► โมเดลเต็ม
Serve:  EPYC (API/คิว) ──► RunPod serverless (vLLM) ──► คืนข้อความ
```

**Stack:** PyTorch (CUDA 12.8 / Blackwell sm_120) · Transformers · ms-swift · PEFT · bitsandbytes · jiwer

---

## 🚀 Quickstart

```bash
# 1) Setup (แนะนำ WSL2 / Linux)
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts/check_env.py            # ตรวจ GPU + bitsandbytes

# 2) Data
python -m thaiocr.data.prepare         # โหลด iapp + ตัด train/val/test
python -m thaiocr.data.format          # → jsonl + images

# 3) Baseline → Train → Eval
python scripts/baseline_eval.py        # วัด baseline
./scripts/train.sh                     # QLoRA fine-tune
python scripts/score_predictions.py outputs/ft_predictions.jsonl

# 4) ลองดูผลอ่านจริง
python scripts/demo_show.py outputs/typhoon-thaihw-merged 6
```

☁️ เทรนบน cloud (RunPod RTX 6000): ดู [`cloud/RUNBOOK.md`](cloud/RUNBOOK.md)
🧩 ปั๊มข้อมูลสังเคราะห์ + ฟอนต์: ดู [`fonts/README.md`](fonts/README.md)

---

## 📂 โครงสร้าง

```
src/thaiocr/        data (prepare/format/synthetic) · eval (CER/WER) · serve
scripts/            check_env · baseline_eval · train.sh · merge · demo_show
cloud/              RunPod kit (setup/train/RUNBOOK) + parallel extractor
docs/               design spec, plan, WSL2+raqm setup
```

---

## ⚖️ ความซื่อสัตย์ (Honest notes)

- **Public Thai handwriting data หายากมาก** — `iapp` (13.5k) คือชุดระดับบรรทัดที่ใช้ได้จริงเกือบชุดเดียว ปริมาณที่เหลือมาจาก synthetic
- โมเดล **plateau ที่ข้อมูล ไม่ใช่ epoch** (0.4 epoch ได้ 12.82% แล้ว) → จะดันต่ำกว่า 11.97% ต้องเพิ่มข้อมูล (synthetic + ลายมือจริง) ไม่ใช่เทรนนานขึ้น
- ลายมือเละ/แหวกมากๆ ยังเป็นโจทย์ (บางเคส CER ยังสูง)

## 🗺️ Roadmap
- [ ] ผสม synthetic 5k + เก็บลายมือจริงเพิ่ม → ดัน CER < 10%
- [ ] ขยายโดเมน: ใบสั่งยาแพทย์ / ฟอร์ม / เอกสารราชการ
- [ ] Deploy: push → HF → RunPod serverless endpoint

---

<sub>Base model: <a href="https://huggingface.co/typhoon-ai/typhoon-ocr-3b">typhoon-ai/typhoon-ocr-3b</a> · Dataset: <a href="https://huggingface.co/datasets/iapp/thai_handwriting_dataset">iapp/thai_handwriting_dataset</a> · Fonts: Google Fonts (OFL)</sub>
