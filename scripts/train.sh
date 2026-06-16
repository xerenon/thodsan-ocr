#!/usr/bin/env bash
# QLoRA fine-tune typhoon-ocr-3b บน RTX 5070 Ti (16GB) ด้วย ms-swift 4.x
# flags ผ่าน smoke test แล้ว (memory ~8GB ที่ batch=1)
set -euo pipefail

# env สำคัญ (เครื่องนี้): ใช้ HuggingFace hub + ปิด TF/Flax (กัน Keras3 ชน)
export USE_HF=1 USE_TF=0 USE_FLAX=0 PYTHONIOENCODING=utf-8

swift sft \
  --model typhoon-ai/typhoon-ocr-3b \
  --template qwen2_5_vl \
  --dataset data/swift/train.jsonl data/swift/synth.jsonl \
  --val_dataset data/swift/val.jsonl \
  --tuner_type lora \
  --quant_method bnb --quant_bits 4 \
  --torch_dtype bfloat16 \
  --lora_rank 16 --lora_alpha 32 \
  --target_modules all-linear \
  --freeze_vit true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --num_train_epochs 1 \
  --max_length 2048 \
  --dataloader_num_workers 0 \
  --save_steps 200 \
  --eval_steps 200 \
  --logging_steps 10 \
  --report_to tensorboard \
  --output_dir outputs/qlora

# หมายเหตุ:
# - ใช้ real (train.jsonl) + synthetic (synth.jsonl) รวมกัน
# - 1 epoch บน ~17k samples ≈ 5 ชม. (0.9 samples/s) — เพิ่ม epoch/batch ได้ถ้าต้องการ
# - ไม่ใส่ --gradient_checkpointing (vision frozen + memory เหลือเยอะ; ใส่แล้วจะ warning)
