# RunPod serverless worker — Thai handwriting OCR (typhoon-ocr1.5-2b + LoRA).
# The public base model is baked into the image at build time (no token needed, so
# RunPod can build straight from the GitHub repo). The private LoRA adapter (67MB) is
# pulled at RUNTIME by handler.py using the HF_TOKEN endpoint env var — keeps secrets
# out of image layers and out of the build entirely.
#
# Build context = repo root (COPY paths are repo-relative).
#   Local build:  docker build -f cloud/serverless/Dockerfile -t <registry>/thaiocr-serverless:v2 .
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

# Source (for OCR_INSTRUCTION) and the handler. The adapter is fetched at runtime
# (see handler._ensure_adapter) using the HF_TOKEN env var set on the endpoint.
COPY src/ /app/src/
COPY cloud/serverless/handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
