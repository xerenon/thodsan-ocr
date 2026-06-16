# Design: Thai Handwriting OCR — fine-tune typhoon-ocr-3b

**วันที่:** 2026-06-16
**สถานะ:** Approved (design) — รอเขียน implementation plan
**เจ้าของ:** Aek

---

## 1. เป้าหมาย (Goal)

สร้างระบบ OCR ที่อ่าน **ลายมือภาษาไทย** ได้แม่นที่สุดเท่าที่เป็นไปได้ โดย **fine-tune ต่อยอดจาก `typhoon-ocr-3b`** (โมเดล OCR ภาษาไทยแบบเปิด) แล้ว deploy ใช้งานจริงผ่าน RunPod serverless

หลักการที่ยึดตลอดโปรเจกต์: **โมเดลเป็นของถูก ข้อมูลคือของแพง** — สิ่งที่ทำให้ "เก่ง" คือคุณภาพและความครอบคลุมของข้อมูล ไม่ใช่ขนาดโมเดล

---

## 2. ขอบเขต (Scope)

### v1 — ทำในรอบนี้
- อ่าน **ลายมือไทยระดับบรรทัด/ประโยคทั่วไป** ตามที่ dataset ครอบคลุม (BEST2019 + Wang)
- input = ภาพบรรทัด/ประโยคลายมือ → output = ข้อความ (text transcription)
- เป้า accuracy: **ดัน CER/WER ให้ดีสุด** เอาชนะ baseline (base model zero-shot) ให้มากที่สุด — ยังไม่ผูกตัวเลขตายตัว

### Out of scope (เฟสหลัง)
- โดเมนเฉพาะทาง: **ใบสั่งยาแพทย์, ฟอร์มราชการ, เอกสารโบราณ** → ต้องเก็บข้อมูลเฉพาะโดเมนเพิ่ม ทำเป็นเฟสแยก
- การอ่าน **เอกสารทั้งหน้า + layout ซับซ้อน** (ตาราง, multi-column) → typhoon-ocr ทำได้อยู่แล้วสำหรับงานพิมพ์ ส่วนลายมือทั้งหน้าเป็นงานต่อยอด
- Active learning loop / การเก็บข้อมูลจริงจาก production → เฟส 4

---

## 3. การตัดสินใจที่ล็อกแล้ว (Locked Decisions)

| ประเด็น | การตัดสินใจ | เหตุผล |
|---|---|---|
| **Base model** | `typhoon-ocr-3b` (4B params, Qwen2.5-VL-Instruct, Apache-2.0, BF16) | specialized ไทย+OCR, license สะอาด, เล็กพอ fit ทั้งเทรนและ deploy |
| **Data** | `iapp/thai_handwriting_dataset` (13,550 ภาพมีเฉลย, Apache-2.0) + synthetic augmentation | มีของพร้อมใช้ทันที + ขยายความครอบคลุมด้วย synthetic |
| **เครื่องเทรน** | RTX 5070 Ti (16GB VRAM), 32GB RAM, i7 gen14 | มีอยู่แล้ว, QLoRA 4B fit ได้ |
| **วิธีเทรน** | QLoRA (4-bit) | 4B บน 16GB ต้อง quantize ตอนเทรน |
| **Deploy** | RunPod serverless (vLLM serve) | จ่ายตามใช้, scale-to-zero, ไม่ต้องดูแลเครื่อง |
| **App/Frontend** | เครื่อง EPYC 8-vCPU (เดิม) เป็น API/คิว เท่านั้น | CPU-only รันโมเดลเองไม่ไหว → ทำหน้าบ้านแทน |

### ทำไมไม่รันโมเดลบนเครื่อง EPYC เอง
เครื่อง deploy จริงเป็น **AMD EPYC-Rome (VPS), 8 vCPU @2.0GHz, RAM 15GB, flags มีแค่ avx2/f16c (ไม่มี AVX-512/AMX)** → รัน VLM 4B บน CPU นี้ = หลักสิบวินาที–เป็นนาที/หน้า + RAM ตึง → **ไม่เวิร์กเป็น runtime** จึงแยกโมเดลไปรันบน GPU (RunPod)

---

## 4. สถาปัตยกรรมรวม

```
[ฝึก] RTX 5070 Ti:
    data prep → QLoRA fine-tune typhoon-ocr-3b → eval (CER/WER) → merge & export

                              │ push model (HF private / network volume)
                              ▼
[รัน]
    [ผู้ใช้/แอป]
        │ อัปโหลดรูป
        ▼
    [EPYC 8-vCPU]  ── HTTP ส่งรูป ──►  [RunPod Serverless (vLLM + โมเดล)]
     หน้าบ้าน: API,คิว,เก็บผล  ◄── ข้อความ ──  ปลุก GPU → อ่าน → ดับเอง
```

---

## 5. องค์ประกอบ (4 โมดูล แยกหน้าที่ชัด)

### 5.1 Data module
- **หน้าที่:** เตรียมข้อมูลให้พร้อมเทรน + กัน test set ไว้วัดผล
- **Input:** `iapp/thai_handwriting_dataset` (parquet: `image`, `text`, `label_file`), ฟอนต์ลายมือไทย (synthetic)
- **Output:** train / val / **test split** ในรูปแบบ prompt ที่ typhoon-ocr คาดหวัง
- **รายละเอียด:**
  - dataset มี **train split เดียว ไม่มี test** → ต้องตัด test set กันไว้เอง (เช่น hold out ~5–10% โดยแบ่งตาม writer/source ไม่ให้รั่ว)
  - **เฟส 2:** Synthetic Data Engine — เรนเดอร์ข้อความไทยด้วยฟอนต์ลายมือหลายแบบ + augmentation (เอียง, เส้นสั่น, พื้นหลังกระดาษ, เบลอ, noise, รอยขีดฆ่า) → ได้ภาพ+เฉลยเพิ่มไม่จำกัด
  - ต้องจัดรูป input ให้ตรง format ของ typhoon-ocr (olmocr-style prompt; ภาพ resize ตามที่โมเดลคาดหวัง ~1800px width สำหรับเอกสาร — สำหรับบรรทัดเดี่ยวต้องทดลองขนาดที่เหมาะ)
- **Dependencies:** huggingface datasets, ฟอนต์ลายมือไทย (freethaifont.com / fontspace), PIL/augmentation lib

### 5.2 Train module
- **หน้าที่:** QLoRA fine-tune typhoon-ocr-3b บน 5070 Ti
- **Input:** train/val split, base model
- **Output:** LoRA adapter → merge เป็นโมเดลเต็ม
- **รายละเอียด:**
  - QLoRA 4-bit, LoRA บน language layers (พิจารณา vision layers ด้วย)
  - batch เล็ก + gradient accumulation (VRAM จำกัด)
  - Framework: **ms-swift / LLaMA-Factory / Unsloth** (ทั้งหมดรองรับ Qwen2.5-VL VLM)
  - ⚠️ **ความเสี่ยง compatibility:** 5070 Ti = Blackwell (sm_120) ใหม่มาก → ต้องใช้ CUDA 12.8+/PyTorch รุ่นใหม่ + bitsandbytes/flash-attn ที่ build รองรับ ต้อง **verify ในเฟส 0**
- **Dependencies:** PyTorch (Blackwell-compatible), transformers, peft, bitsandbytes, training framework, W&B (logging)

### 5.3 Eval module
- **หน้าที่:** วัดผลแบบเชื่อถือได้
- **Input:** test set จริง (ไม่ใช่ synthetic), โมเดลที่จะวัด
- **Output:** CER, WER + รายงานเทียบ
- **รายละเอียด:** เทียบ 3 จุด — (1) base typhoon-ocr-3b zero-shot = **baseline**, (2) หลัง fine-tune, (3) อ้างอิง prior art (`Aekanun/thai-handwriting-llm`)
- **กฎเหล็ก:** เทรนด้วย synthetic ได้ แต่ **วัดผลด้วยของจริงเสมอ**

### 5.4 Serve module
- **หน้าที่:** รันโมเดล production บน RunPod + ต่อกับ EPYC
- **Input:** merged model, ภาพจาก client
- **Output:** API endpoint คืนข้อความ
- **รายละเอียด:**
  - RunPod Serverless Endpoint รัน **vLLM** (รองรับ Qwen2.5-VL vision input) ชี้ไปที่โมเดล (HF private repo หรือ network volume)
  - จัดการ **cold start**: เลือก Flex (scale-to-zero, ถูก, มี cold start ~15–60s) vs Active (warm ≥1, ตอบเร็ว, จ่าย idle); ใช้ **FlashBoot** + network volume ลดเวลาโหลด
  - Client บน EPYC: ส่งภาพ → endpoint → รับข้อความ
- **Dependencies:** RunPod account, vLLM worker, EPYC client (Python/Node)

---

## 6. ลำดับเฟส (Phases)

### เฟส 0 — พิสูจน์ว่าทำได้ (Go/No-Go Gate) ⛳ สำคัญสุด
ทำ **ก่อน** ลงแรงหนัก เพื่อกันเจ็บ:
1. ตั้ง env บน 5070 Ti — **verify PyTorch/CUDA/bitsandbytes รันบน Blackwell sm_120 ได้**
2. โหลด iapp dataset → ตัด test set
3. รัน base typhoon-ocr-3b zero-shot → วัด **baseline CER/WER** บน test (รู้ว่าเริ่มจากตรงไหน)
4. ลอง QLoRA แบบ smoke test (ข้อมูลนิดเดียว) → ยืนยัน fit บน 16GB
5. ลอง deploy โมเดล base ขึ้น RunPod → วัด latency/cost จริงต่อหน้า
- **ผ่านแล้วค่อยไปต่อ** — ถ้า Blackwell มีปัญหา หรือ RunPod ไม่คุ้ม รู้ตั้งแต่ต้น

### เฟส 1 — fine-tune
- QLoRA บน iapp 13.5k → eval → **ต้องชนะ baseline**
- ปรับ hyperparameter / prompt format จนได้ผลดีสุดบนข้อมูลจริง

### เฟส 2 — ดันความแม่น
- สร้าง Synthetic Data Engine → ผสม synthetic + real → re-train → re-eval
- จัดการ failure cases (ลายมือแบบที่โมเดลพลาดบ่อย)

### เฟส 3 — deploy
- merge + push model → ตั้ง RunPod serverless endpoint → ต่อ client บน EPYC → ทดสอบ end-to-end

### เฟส 4 (later) — ขยาย
- โดเมนเฉพาะทาง (ใบสั่งยา/ฟอร์ม) ด้วยข้อมูลใหม่
- active learning จาก production

---

## 7. ความเสี่ยง & ข้อควรระวัง

| ความเสี่ยง | ผลกระทบ | การรับมือ |
|---|---|---|
| Blackwell (sm_120) compatibility | เทรนไม่ได้ | verify เฟส 0 ก่อน, เตรียม fallback (cloud GPU เทรน) |
| ข้อมูลจริงแค่ 13.5k อาจไม่พอ "เก่งสุดๆ" | accuracy ตัน | synthetic augmentation (เฟส 2) + เก็บเพิ่มภายหลัง |
| prompt format ของ typhoon-ocr ไม่ตรง | fine-tune ไม่ขยับ | ศึกษา format จาก repo scb-10x/typhoon-ocr + olmocr |
| cold start RunPod ช้า/แพง | UX แย่ / บิลบาน | FlashBoot + network volume + เลือกโหมด worker ตาม volume |
| test set รั่ว (writer เดียวกันอยู่ทั้ง train/test) | ตัวเลขหลอก | แบ่ง split ตาม writer/source |

---

## 8. ทางเลือก/ตัวเลือกที่บันทึกไว้ (Notes)

- **`typhoon-ocr1.5-2b` (2B)** มีอยู่ — เล็กกว่า เร็ว/ถูกกว่าบน RunPod และ fit เทรนง่ายกว่า → พิจารณาเป็น base ทางเลือกเทียบในเฟส 0
- **Deploy ทางเลือก:** ถ้า volume สม่ำเสมอทั้งวัน → RunPod Pod (รายชม.) อาจคุ้ม/ง่ายกว่า serverless ทบทวนตอนใกล้ deploy
- **Self-host บน 5070 Ti** เป็น deploy ทางเลือกตอน volume ต่ำ (compute ฟรี ไม่มี cold start)

---

## 9. เกณฑ์ความสำเร็จ (Success Criteria)

- เฟส 0: env รันได้ + ได้ baseline CER + ยืนยัน RunPod รันไหว
- เฟส 1: โมเดล fine-tune **CER/WER ดีกว่า baseline อย่างมีนัยสำคัญ** บน test set จริง
- เฟส 3: ระบบ end-to-end (EPYC → RunPod → คืนข้อความ) ใช้งานได้จริง โดย latency/cost อยู่ในระดับยอมรับได้
