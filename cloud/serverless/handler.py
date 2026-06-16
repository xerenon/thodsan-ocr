"""RunPod serverless handler — Thai handwriting OCR (typhoon-ocr1.5-2b + LoRA).

Loads the base model + LoRA adapter ONCE at cold start (module scope); every job
reuses them. Request/response shape matches src/thaiocr/serve/client.py:
  in :  {"input": {"messages": [{role:"user", content:[{image_url|image}, {text}]}], "max_tokens": N}}
  out:  {"choices": [{"message": {"role":"assistant", "content": "<thai text>"}}]}
RunPod wraps the return value under "output", so the client reads
output.choices[0].message.content unchanged.
"""
import base64
import io
import os
import sys

import torch
from PIL import Image

# src/ is COPY'd into the image at /app/src (THAIOCR_SRC); locally pass THAIOCR_SRC=src
sys.path.insert(0, os.environ.get("THAIOCR_SRC", "/app/src"))
from thaiocr.data.format import OCR_INSTRUCTION  # noqa: E402

BASE_MODEL = os.environ.get("BASE_MODEL", "typhoon-ai/typhoon-ocr1.5-2b")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "/adapter")
ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "xerenon/thodsan-ocr")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))


def _ensure_adapter() -> str:
    """Return a local dir holding the adapter. If ADAPTER_DIR isn't already populated
    (e.g. baked into the image or mounted), pull it from the private HF repo using the
    runtime HF_TOKEN env var. Only the two inference files are fetched (~67MB)."""
    if os.path.exists(os.path.join(ADAPTER_DIR, "adapter_config.json")):
        return ADAPTER_DIR
    from huggingface_hub import snapshot_download

    return snapshot_download(
        ADAPTER_REPO,
        local_dir=ADAPTER_DIR,
        token=os.environ.get("HF_TOKEN"),
        allow_patterns=["adapter_config.json", "adapter_model.safetensors"],
    )


# ---- cold-start load (once per worker) ----
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402
from peft import PeftModel  # noqa: E402

_processor = AutoProcessor.from_pretrained(BASE_MODEL)
# AutoModelForImageTextToText is class-agnostic — resolves to the Qwen3-VL class for
# this checkpoint (NOT Qwen2_5_VL, which is the v1/3b model in scripts/baseline_eval.py).
_model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
)
# Attach the adapter explicitly. adapter_config.json records a pod-local base path,
# but PEFT uses the base object passed here, so that path is ignored.
_model = PeftModel.from_pretrained(_model, _ensure_adapter())
_model.eval()


def _decode_b64(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _image_from_content(content) -> Image.Image:
    """Pull the image out of an OpenAI-style content list.

    Supports {"type":"image_url","image_url":{"url": "data:...;base64,..."}} (and plain
    http(s) URLs), plus {"type":"image","image": "<base64>"}.
    """
    for part in content:
        t = part.get("type")
        if t == "image_url":
            url = part["image_url"]["url"]
            if url.startswith("data:"):
                return _decode_b64(url.split(",", 1)[1])
            import requests  # only needed for remote URLs
            return Image.open(io.BytesIO(requests.get(url, timeout=30).content)).convert("RGB")
        if t == "image":
            return _decode_b64(part["image"])
    raise ValueError("no image found in message content")


def _ocr(pil_img: Image.Image, max_new_tokens: int) -> str:
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_img},
        {"type": "text", "text": OCR_INSTRUCTION},
    ]}]
    text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(messages)
    inputs = _processor(text=[text], images=imgs, videos=vids, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[:, inputs.input_ids.shape[1]:]
    return _processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def handler(job):
    inp = job.get("input") or {}
    msgs = inp.get("messages") or []
    user_msg = next((m for m in msgs if m.get("role") == "user"), msgs[-1] if msgs else None)
    if user_msg is None:
        return {"error": "no messages provided"}
    content = user_msg.get("content")
    if not isinstance(content, list):
        return {"error": "expected content to be a list containing an image"}
    try:
        pil = _image_from_content(content)
    except Exception as e:  # noqa: BLE001
        return {"error": f"could not read image: {e}"}
    max_new = int(inp.get("max_tokens") or MAX_NEW_TOKENS)
    text = _ocr(pil, max_new)
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
