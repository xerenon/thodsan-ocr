# Thai Handwriting OCR

Fine-tune `typhoon-ocr-3b` เพื่ออ่านลายมือไทย → deploy บน RunPod serverless

## เอกสาร
- Design: `docs/superpowers/specs/2026-06-16-thai-handwriting-ocr-design.md`
- Plan: `docs/superpowers/plans/2026-06-16-thai-handwriting-ocr.md`

## Setup (แนะนำ WSL2 Ubuntu / Linux — stack เทรนทำงานลื่นสุด)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
# PyTorch สำหรับ Blackwell (RTX 5070 Ti, sm_120):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## ลำดับการรัน

```bash
# เฟส 0 — ตรวจ environment (go/no-go)
python scripts/check_env.py

# เตรียมข้อมูล
python -m thaiocr.data.prepare          # โหลด iapp dataset + ตัด split
python -m thaiocr.data.format           # แปลงเป็น jsonl + images สำหรับ ms-swift

# เฟส 0 — baseline (รู้ว่าเริ่มจากตรงไหน)
python scripts/baseline_eval.py --out outputs/baseline_report.json

# เฟส 1 — fine-tune + eval
./scripts/train.sh
swift infer --adapters outputs/qlora --val_dataset data/swift/test.jsonl \
  --max_new_tokens 256 --result_path outputs/ft_predictions.jsonl
python scripts/score_predictions.py outputs/ft_predictions.jsonl

# เฟส 1 — merge เป็นโมเดลพร้อม deploy
python scripts/merge.py
```

## รัน tests (โมดูล Python ล้วน ไม่ต้องใช้ GPU)

```bash
pytest -v
```
