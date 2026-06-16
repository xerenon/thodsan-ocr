# Thai Handwriting OCR

Fine-tuning [`typhoon-ocr-3b`](https://huggingface.co/typhoon-ai/typhoon-ocr-3b) (Qwen2.5-VL, 4B) to read **handwritten Thai**, reducing character error rate from 35.3% to 11.97%.

![base](https://img.shields.io/badge/base-typhoon--ocr--3b-5b6ee1)
![license](https://img.shields.io/badge/license-Apache--2.0-green)
![CER](https://img.shields.io/badge/CER-35.3%25%20→%2011.97%25-brightgreen)

Handwritten Thai is a hard OCR target: stacked vowels and tone marks, connected strokes, and no spaces between words. Typhoon OCR is trained for *printed* Thai documents and reads handwriting poorly out of the box. This repo fine-tunes it on real handwriting data to close most of that gap on a single GPU.

## Results

Evaluated on a held-out test split of `iapp/thai_handwriting_dataset`.

| Metric | Baseline (zero-shot) | Fine-tuned | Δ |
|---|---:|---:|---:|
| CER | 35.3% | **11.97%** | -66% |
| Avg. error | ~1 in 3 chars | ~1 in 8 chars | |
| Exact-line match | — | 25% | |

Training: QLoRA (4-bit), 2 epochs, on 13,550 real handwritten lines from 2,026 writers.

### Sample predictions

```
Exact match:
  ground truth: ในเมื่อรัฐยังต้องให้เงินอุดหนุนแก่นักศึกษาในระดับปริญญาตรีในอัตราสูง
  prediction  : ในเมื่อรัฐยังต้องให้เงินอุดหนุนแก่นักศึกษาในระดับปริญญาตรีในอัตราสูง

Near-perfect (CER 1.3%, one wrong character on a long line):
  ground truth: ...มีความสัมพันธ์กับกลอนหรือคำขับ
  prediction  : ...มีความสัมพันธ์กับกลอนหรือค้ำขับ
```

## What's here

- Fine-tuned Thai-handwriting reader built on Typhoon OCR (Apache-2.0; commercial use OK).
- QLoRA training that fits a single GPU, from a 16 GB RTX 5070 Ti up to a 96 GB RTX PRO 6000.
- A synthetic data engine: renders Thai text with 12 OFL handwriting fonts plus augmentation.
- A RunPod kit for cloud training (setup, train, eval, merge).

## Architecture

```
image (handwriting)  ->  typhoon-ocr-3b + LoRA adapter  ->  text

Train:  iapp 13.5k (+ optional synthetic)  --QLoRA 4-bit-->  adapter  --merge-->  full model
Serve:  API/queue host  ->  RunPod serverless (vLLM)  ->  text
```

Stack: PyTorch (CUDA 12.8 / Blackwell sm_120), Transformers, ms-swift, PEFT, bitsandbytes, jiwer.

## Quickstart

```bash
# Setup (WSL2 / Linux recommended)
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts/check_env.py            # verify GPU + bitsandbytes

# Data
python -m thaiocr.data.prepare         # download iapp, split train/val/test
python -m thaiocr.data.format          # -> jsonl + images

# Baseline, train, eval
python scripts/baseline_eval.py
./scripts/train.sh                     # QLoRA fine-tune
python scripts/score_predictions.py outputs/ft_predictions.jsonl

# Inspect predictions
python scripts/demo_show.py outputs/typhoon-thaihw-merged 6
```

Cloud training on RunPod: see [`cloud/RUNBOOK.md`](cloud/RUNBOOK.md).
Synthetic data and fonts: see [`fonts/README.md`](fonts/README.md).

## Layout

```
src/thaiocr/   data (prepare/format/synthetic), eval (CER/WER), serve
scripts/       check_env, baseline_eval, train.sh, merge, demo_show
cloud/         RunPod kit (setup/train/RUNBOOK) + parallel extractor
docs/          design spec, plan, WSL2 + raqm setup
```

## Limitations

- Public line-level Thai handwriting data is scarce; `iapp` (13.5k) is essentially the only sizeable set. Volume beyond that comes from synthetic rendering.
- Accuracy plateaus on *data*, not epochs — 0.4 of an epoch already reached 12.82% CER. Lowering CER further needs more/better data, not longer training.
- Very messy or unusual handwriting still produces high per-line error.

## Roadmap

- Mix in synthetic data and collect more real samples, targeting CER < 10%.
- Extend to specific domains (medical prescriptions, forms, government documents).
- Deploy: push the merged model to Hugging Face, serve on a RunPod serverless endpoint.
