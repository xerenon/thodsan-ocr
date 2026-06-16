#!/usr/bin/env bash
# เทรนบน RTX PRO 6000 Blackwell 96GB — bf16 full LoRA, batch ใหญ่, real 12k
set -e
export USE_HF=1 USE_TF=0 USE_FLAX=0 PYTHONIOENCODING=utf-8
export HF_HOME=/dev/shm/hf PYTHONPATH=src IMAGE_MAX_TOKEN_NUM=4096
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/orc

swift sft \
  --model typhoon-ai/typhoon-ocr-3b \
  --template qwen2_5_vl \
  --dataset data/swift/train.jsonl \
  --val_dataset data/swift/val.jsonl \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --lora_rank 32 --lora_alpha 64 \
  --target_modules all-linear \
  --freeze_vit true \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing true \
  --learning_rate 1e-4 \
  --num_train_epochs 2 \
  --max_length 2048 \
  --dataloader_num_workers 8 \
  --save_steps 200 --eval_steps 200 --logging_steps 5 \
  --report_to tensorboard \
  --output_dir outputs/qlora_cloud
