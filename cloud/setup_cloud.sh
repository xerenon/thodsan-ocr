#!/usr/bin/env bash
# รันบน cloud GPU box (Linux) ครั้งเดียว หลัง: git clone repo + extract thaiocr_data.tar.gz ที่ root repo
set -euo pipefail

python3 -m venv .venv && source .venv/bin/activate
pip install -U pip

# torch cu128 รองรับทั้ง RTX 6000 Ada (sm_89) และ RTX PRO 6000 Blackwell (sm_120)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ms-swift transformers accelerate datasets jiwer qwen-vl-utils peft pillow numpy pandas tensorboard

echo "=== GPU ==="
python -c "import torch; p=torch.cuda.get_device_properties(0); print(p.name, round(p.total_memory/1e9,1),'GB', 'sm_%d%d'%torch.cuda.get_device_capability(0))"
echo "=== data check ==="
wc -l data/swift/train.cloud.jsonl data/swift/synth.cloud.jsonl data/swift/val.cloud.jsonl data/swift/test.cloud.jsonl 2>&1
echo "setup done. ต่อ: bash cloud/train_cloud.sh"
