# Cloud Training Runbook — RTX 6000 (48-96GB)

เป้าหมาย: เทรน fine-tune typhoon-ocr-3b ให้เสร็จเร็ว (~30-50 นาที/epoch) บน GPU เช่า

## ของที่เตรียมไว้แล้ว (ในเครื่อง)
- `thaiocr_data.tar` — ข้อมูลพร้อมเทรน (jsonl path สัมพัทธ์ + รูป real+synth)
- `cloud/setup_cloud.sh` — ลง env
- `cloud/train_cloud.sh` — คำสั่งเทรน (bf16, batch ใหญ่, 3 epoch)

---

## ขั้นตอนบน box (Linux GPU)

### 1. เช่า box + เลือก GPU
- RunPod / Vast → เลือก **RTX 6000 Ada / A6000 (48GB)** หรือ **RTX PRO 6000 (96GB)**
- เลือก image ที่มี CUDA 12.x + Python 3.10/3.11

### 2. clone repo + อัปข้อมูล
```bash
git clone <your-repo-url> orc-finetune-handwrite   # หรือ scp โค้ดขึ้นไป
cd orc-finetune-handwrite
# อัป tarball ขึ้น box: ใช้ scp / runpodctl send / หรือ HF dataset
# แล้ว extract ที่ root repo:
tar xf thaiocr_data.tar           # ได้ data/swift/{*.cloud.jsonl, images/}
```

### 3. setup env (ครั้งเดียว ~5-10 นาที)
```bash
bash cloud/setup_cloud.sh
source .venv/bin/activate
```

### 4. เทรน (~30-50 นาที/epoch × 3 ≈ 1.5-2.5 ชม. หรือลด epoch ถ้ารีบ)
```bash
bash cloud/train_cloud.sh
# ดู loss สด: ดู outputs/qlora_cloud/v*/logging.jsonl
```
> รีบ 1 ชม.: แก้ `--num_train_epochs 1` ใน train_cloud.sh → ~30-50 นาที

### 5. eval วัด CER (เทียบ baseline 35.3%)
```bash
LATEST=$(ls -td outputs/qlora_cloud/v*/checkpoint-* | head -1)
USE_HF=1 USE_TF=0 swift infer --adapters "$LATEST" \
  --val_dataset data/swift/test.cloud.jsonl \
  --max_new_tokens 256 --result_path outputs/ft_pred.jsonl
python scripts/score_predictions.py outputs/ft_pred.jsonl   # โชว์ CER
```

### 6. merge + ดึงโมเดลกลับ
```bash
USE_HF=1 swift export --adapters "$LATEST" --merge_lora true \
  --output_dir outputs/typhoon-ocr-3b-thaihw-merged
# ดึงกลับ: scp / runpodctl send / push ขึ้น HF (private) เพื่อใช้ deploy ต่อ
```

---

## เคล็ดลับให้คุ้มชั่วโมง
- **เตรียม tarball ไว้ก่อน** (ทำแล้ว) → อัปไฟล์เดียวจบ ไม่ต้อง regen ข้อมูลบน box
- โหลด model typhoon (~8GB) บน box เร็วกว่าอัปจากบ้าน → ปล่อยให้ box โหลดเอง
- ปิด box ทันทีหลังดึงโมเดลกลับ (อย่าลืม — เสียเงินต่อชั่วโมง)
