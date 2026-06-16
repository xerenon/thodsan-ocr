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
  --gradient_checkpointing true \
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
