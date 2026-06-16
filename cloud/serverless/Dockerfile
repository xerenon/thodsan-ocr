# RunPod serverless worker — Thai handwriting OCR (typhoon-ocr1.5-2b + LoRA).
# The public base model is baked into the image at build time (no token needed, so
# RunPod can build straight from the GitHub repo). The private LoRA adapter (67MB) is
# pulled at RUNTIME by handler.py using the HF_TOKEN endpoint env var — keeps secrets
# out of image layers and out of the build entirely.
#
# Build context = repo root (COPY paths are repo-relative).
#   Local build:  docker build -f cloud/serverless/Dockerfile -t <registry>/thaiocr-serverless:v2 .
# torch >= 2.7 is REQUIRED: transformers 5.10.4 references torch.float8_e8m0fnu (added in
# 2.7) at import. cuda12.8 matches the training pod (cu128). torch 2.8 runs fine on the
# 16GB inference GPU (Ampere/Ada).
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

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
