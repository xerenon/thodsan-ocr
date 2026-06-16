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

---

# Deploy → RunPod Serverless (v2: typhoon-ocr1.5-2b + LoRA)

สถาปัตยกรรม: client / EPYC frontend → RunPod serverless endpoint (GPU) → คืนข้อความไทย
worker เป็น custom transformers handler (`cloud/serverless/handler.py`) โหลด base+adapter ครั้งเดียวตอน
cold start. โมเดล bake เข้า image แล้ว (base โหลดตอน build, adapter `COPY` เข้าไป) — ไม่ต้องใช้ HF token ตอนรัน.

ของที่อยู่ใน `cloud/serverless/`: `handler.py`, `Dockerfile`, `requirements.txt`, `build_and_push.sh`,
`test_handler.py`. I/O ตรงกับ `src/thaiocr/serve/client.py` (OpenAI-style `messages` → `choices[0].message.content`).

### 1. (ออปชัน) smoke test handler ในเครื่องก่อน — ไม่เสียค่า GPU endpoint
ต้องมี GPU + adapter ในเครื่อง (ครั้งแรกจะโหลด base ~5GB จาก HF):
```bash
ADAPTER_DIR=outputs/v2_2b_ckpt1063 THAIOCR_SRC=src \
  python cloud/serverless/test_handler.py        # หรือใส่ path รูปเองเป็น argv[1]
```
ได้ `OK — text: ...` = handler โหลดโมเดล + คืน contract ถูกต้อง.

วิธีที่แนะนำ: ให้ **RunPod build image จาก GitHub repo เอง** (ไม่ต้องมี Docker ในเครื่อง, ไม่ต้องอัป image
ใหญ่จากเน็ตบ้าน) โดย adapter ดึงจาก **HF private repo** ตอน build. ของหนัก (base 5GB, build image ~10GB)
เกิดบน cloud หมด — เครื่องเราอัปแค่โค้ด + adapter 67MB.

### 2. push adapter ขึ้น HF (ครั้งเดียว, ~67MB)
อัปแค่ไฟล์ที่ใช้ตอน serve (ข้าม optimizer.pt/scheduler ฯลฯ ที่เป็น training state):
```bash
hf auth login                       # paste write token
hf upload xerenon/thodsan-ocr outputs/v2_2b_ckpt1063 . \
  --include "adapter_config.json" "adapter_model.safetensors"
```
repo: `xerenon/thodsan-ocr` (private) — ใช้เป็นทั้ง source ตอน build และ backup ของ adapter (local-only).

### 3. push โค้ดขึ้น GitHub + สร้าง endpoint
```bash
git add -A; git commit -m "feat: runpod serverless worker"; git push origin main
```
RunPod Console → Serverless → New Endpoint → **Deploy from a GitHub repository** → เชื่อม repo
`xerenon/thodsan-ocr` → ชี้ Dockerfile = `cloud/serverless/Dockerfile`:
- **Build env / Build arg: `HF_TOKEN` = <read token>** ← จำเป็น (ดึง adapter จาก HF private ตอน build)
- GPU: คลาส **16GB** (A4000 / A4500 / RTX 4000) — โมเดล 2B bf16 พอสบาย, ~$0.0002/s
- **Max workers** เล็กๆ (เช่น 3), **Active workers = 0** (scale-to-zero ประหยัด)
- Container disk ใหญ่พอเก็บ weight ที่ bake มา (~15GB)

> ทางเลือก (ไม่อยาก build บน cloud): build เอง แล้ว push ขึ้น registry —
> `docker build -f cloud/serverless/Dockerfile --build-arg HF_TOKEN=hf_xxx -t <reg>/thaiocr-serverless:v2 .`
> (`build_and_push.sh` ช่วยได้ แต่ต้องส่ง HF_TOKEN เอง) แล้วเลือก "Deploy from a Docker image" แทน

> หมายเหตุเวอร์ชัน: pin `transformers==5.10.4` (เวอร์ชันที่เซฟโมเดลบน pod — local 4.57 โหลดไม่ขึ้น)
> + `peft==0.19.1`. ถ้า build/load พังเรื่อง Qwen3-VL ให้เลื่อน transformers เป็นตัวล่าสุดที่มี Qwen3-VL.

### 4. ทดสอบ endpoint จริง
```bash
export RUNPOD_ENDPOINT_ID=<id> RUNPOD_API_KEY=<key>
python -m thaiocr.serve.client <handwriting.png>     # พิมพ์ latency + ข้อความไทย
```
เช็ค regression: ยิงรูป "ตัวพิมพ์" ด้วย — ควรออกมาเกือบเป๊ะ (ตรงกับ printed CER 0.02%)
ยืนยันว่าโมเดลที่ deploy = ตัวที่ eval.

### gotchas
- adapter `adapter_config.json` ชี้ base เป็น path บน pod (`/dev/shm/...`) — ไม่เป็นไร handler โหลด base
  จาก HF id เองแล้วค่อย attach adapter (PEFT ใช้ base object ที่ส่งให้ ไม่สน path ใน config)
- cold start แรกจะช้า (โหลด weight ขึ้น GPU) — warm แล้วเร็ว; ตั้ง active worker > 0 ถ้าอยากเลี่ยง cold start
- HF_TOKEN ตอน build เป็น **read** token พอ; อย่า bake token ไว้ใน image (ใช้ build arg/secret เท่านั้น)
