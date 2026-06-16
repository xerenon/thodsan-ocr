# RunPod serverless worker — Thai handwriting OCR (typhoon-ocr1.5-2b + LoRA).
# Base + adapter weights are baked into the image at build time. The adapter is
# pulled from a private HF repo, so this builds with NO local files beyond the repo
# itself — RunPod can build it straight from the GitHub repo.
#
# Build context = repo root. Needs a HF token (read) to fetch the private adapter:
#   RunPod GitHub build: add HF_TOKEN as a build environment variable.
#   Local build:  docker build -f cloud/serverless/Dockerfile \
#                   --build-arg HF_TOKEN=hf_xxx -t <registry>/thaiocr-serverless:v2 .
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    THAIOCR_SRC=/app/src \
    ADAPTER_DIR=/adapter \
    BASE_MODEL=typhoon-ai/typhoon-ocr1.5-2b \
    ADAPTER_REPO=xerenon/thodsan-ocr

WORKDIR /app

COPY cloud/serverless/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the base model (public) into the image — no runtime download, deterministic cold starts.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('typhoon-ai/typhoon-ocr1.5-2b')"

# Pull the LoRA adapter from the private HF repo into /adapter (only adapter_* files needed).
ARG HF_TOKEN
RUN python -c "import os; from huggingface_hub import snapshot_download; \
snapshot_download(os.environ['ADAPTER_REPO'], local_dir='/adapter', token='${HF_TOKEN}', \
allow_patterns=['adapter_config.json','adapter_model.safetensors'])"

# Source (for OCR_INSTRUCTION) and the handler.
COPY src/ /app/src/
COPY cloud/serverless/handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
