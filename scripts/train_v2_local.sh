#!/usr/bin/env bash
# v2: fine-tune typhoon-ocr1.5-2b (Qwen3-VL) บน RTX 5070 Ti 16GB — real + synthetic
# memory-safe: cap image tokens + grad checkpointing + max_length สั้นลง (กัน spillover เกิน 16GB)
set -e
export USE_HF=1 USE_TF=0 USE_FLAX=0 PYTHONIOENCODING=utf-8
export IMAGE_MAX_TOKEN_NUM=1024
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

swift sft \
  --model models/typhoon-ocr1.5-2b --model_type qwen3_vl --template qwen3_vl \
  --dataset data/swift/train.jsonl data/swift/synth.jsonl \
  --val_dataset data/swift/val.jsonl \
  --tuner_type lora --quant_method bnb --quant_bits 4 --torch_dtype bfloat16 \
  --lora_rank 16 --lora_alpha 32 --target_modules all-linear --freeze_vit true \
  --gradient_checkpointing true \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 --num_train_epochs 1 --max_length 1536 \
  --dataloader_num_workers 0 \
  --save_steps 300 --eval_steps 300 --logging_steps 10 \
  --report_to tensorboard \
  --output_dir outputs/qlora_v2_2b
