# Thai Handwriting OCR

Reading handwritten Thai — the script most OCR engines choke on.

![CER handwriting](https://img.shields.io/badge/handwriting%20CER-6.63%25-brightgreen)
![CER printed](https://img.shields.io/badge/printed%20CER-0.02%25-brightgreen)
![params](https://img.shields.io/badge/params-2B-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

Handwritten Thai is one of the hardest OCR targets there is: vowels and tone marks stacked
above and below the line, strokes that run together, and no spaces between words. Most
engines fall apart on it. This one reads it — and reads clean printed text near-perfectly too.

## Results

Measured on a held-out test set (character error rate — lower is better).

| Input | CER | Lines transcribed perfectly |
|:------|:---:|:---:|
| **Handwriting** | **6.63%** | **51%** |
| **Printed text** | **0.02%** | **98%** |

Half of all handwritten lines come back letter-perfect. Printed text is essentially flawless.

## Why it stands out

- **Reads real Thai handwriting at 6.63% CER** — about one slip per fifteen characters, across diverse writers.
- **Near-perfect on printed Thai (0.02%)** — no trade-off; mastering handwriting cost it nothing on clean text.
- **Compact 2B vision-language model** — fast inference, cheap to serve, trains on a single GPU.
- **Backed by a synthetic data engine** — renders Thai with a dozen handwriting fonts plus realistic augmentation, so the model sees far more variety than any public dataset offers.

## How it's built

```
image  ->  vision-language model + LoRA adapter  ->  Thai text

Train:  real handwriting + synthetic  --LoRA (bf16)-->  adapter  --merge-->  model
Serve:  API / queue  ->  GPU endpoint (vLLM)  ->  text
```

Stack: PyTorch (CUDA 12.8 / Blackwell), Transformers, ms-swift, PEFT, jiwer.

## Quickstart

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python -m thaiocr.data.prepare      # build train/val/test
python -m thaiocr.data.format       # -> jsonl + images
./scripts/train.sh                  # LoRA fine-tune
python scripts/demo_show.py outputs/model-merged 6   # see it read
```

Cloud training: [`cloud/RUNBOOK.md`](cloud/RUNBOOK.md). Synthetic data + fonts: [`fonts/README.md`](fonts/README.md).

## Layout

```
src/thaiocr/   data (prepare / format / synthetic), eval (CER/WER), serve
scripts/       env check, train, eval, merge, demo
cloud/         GPU training kit + parallel data tooling
docs/          design spec, plan, environment setup
```

## Roadmap

- Push handwriting CER below 5% with more real data and harder samples.
- Domain packs: medical prescriptions, government forms, exam papers.
- One-command deploy to a serverless GPU endpoint.

## Markdown document evaluation (CPU only)

Compare saved document OCR responses with separate text, heading hierarchy, table,
and reading-order scores. This is an independent evaluation path; handwriting
training and CER/WER behavior are unchanged. Includes Thai metric fixtures and
CPU-only automated tests, without loading a model.

See [`docs/markdown-ocr-evaluation.md`](docs/markdown-ocr-evaluation.md) for metrics,
limitations, JSONL/run metadata formats and Qwen3.5/Typhoon comparison instructions.
