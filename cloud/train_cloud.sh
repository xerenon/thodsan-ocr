#!/usr/bin/env bash
# เทรนบน RTX 6000 (48-96GB): bf16 full LoRA (ไม่ quant) + batch ใหญ่ + 3 epoch
# เร็วกว่า 5070 Ti หลายเท่าเพราะ VRAM เยอะ → ไม่ติด memory เหมือน 16GB
set -euo pipefail

export USE_HF=1 USE_TF=0 USE_FLAX=0 PYTHONIOENCODING=utf-8
# cap token/รูป → เร็วขึ้น + กัน memory พุ่งจากรูปใหญ่ (ลายมือบรรทัดเดียวไม่ต้องใช้เยอะ)
export IMAGE_MAX_TOKEN_NUM=4096

swift sft \
  --model typhoon-ai/typhoon-ocr-3b \
  --template qwen2_5_vl \
  --dataset data/swift/train.cloud.jsonl data/swift/synth.cloud.jsonl \
  --val_dataset data/swift/val.cloud.jsonl \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --lora_rank 32 --lora_alpha 64 \
  --target_modules all-linear \
  --freeze_vit true \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --num_train_epochs 3 \
  --max_length 2048 \
  --dataloader_num_workers 4 \
  --save_steps 300 --eval_steps 300 --logging_steps 10 \
  --report_to tensorboard \
  --output_dir outputs/qlora_cloud

# ปรับ batch ตาม VRAM:
#   48GB (Ada/A6000): batch 4-8 ปลอดภัย
#   96GB (PRO 6000):  batch 8-16 ได้
#   ถ้า OOM: ลด per_device_train_batch_size ลงครึ่ง
