#!/usr/bin/env bash
# v2: fine-tune typhoon-ocr1.5-2b (Qwen3-VL) บน RTX PRO 6000 96GB — real + synthetic, batch ใหญ่
set -e
export USE_HF=1 USE_TF=0 USE_FLAX=0 PYTHONIOENCODING=utf-8
export HF_HOME=/dev/shm/hf PYTHONPATH=src IMAGE_MAX_TOKEN_NUM=1024
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/orc

swift sft \
  --model typhoon-ai/typhoon-ocr1.5-2b --model_type qwen3_vl --template qwen3_vl \
  --dataset data/swift/train.jsonl data/swift/synth.jsonl \
  --val_dataset data/swift/val.jsonl \
  --tuner_type lora --torch_dtype bfloat16 \
  --lora_rank 16 --lora_alpha 32 --target_modules all-linear --freeze_vit true \
  --gradient_checkpointing true \
  --per_device_train_batch_size 16 --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 --num_train_epochs 1 --max_length 1536 \
  --dataloader_num_workers 8 \
  --save_steps 200 --eval_steps 200 --logging_steps 5 \
  --report_to tensorboard \
  --output_dir outputs/qlora_v2_2b
