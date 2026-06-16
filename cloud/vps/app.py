"""Thai Handwriting OCR — VPS frontend API.

A thin, CPU-only FastAPI service that proxies image OCR requests to the RunPod
serverless GPU endpoint. Runs anywhere (EPYC / Xeon VPS, no GPU needed) — it only
forwards images and returns the transcribed Thai text. It submits the job and polls
RunPod's /status so a worker cold start doesn't surface as a half-finished response.

Config (env):
  RUNPOD_ENDPOINT_ID   RunPod serverless endpoint id              (required)
  RUNPOD_API_KEY       RunPod API key                             (required)
  API_KEY              key clients must send as X-API-Key header  (optional; enforced if set)
  OCR_MAX_TOKENS       default output cap (full A4 Thai page)      (default 4096)
  OCR_TIMEOUT          max seconds to wait for a job              (default 300)
  CORS_ORIGINS         comma-separated allowed origins            (default *)

Run:  uvicorn app:app --host 0.0.0.0 --port 8000
"""
import base64
import os
import time

import requests
from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

RUNPOD_ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]
RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
API_KEY = os.environ.get("API_KEY")
DEFAULT_MAX_TOKENS = int(os.environ.get("OCR_MAX_TOKENS", "4096"))
OCR_TIMEOUT = int(os.environ.get("OCR_TIMEOUT", "300"))
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",")]

_BASE = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"
_HDR = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}

app = FastAPI(title="Thai Handwriting OCR", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"],
)


def _check_key(x_api_key: str | None) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")


def _ocr_b64(image_b64: str, max_tokens: int) -> str:
    """Submit one image to RunPod and poll until it finishes. Raises HTTPException
    on worker failure / timeout so the client gets a clean status code."""
    payload = {"input": {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_b64}},
        {"type": "text", "text": "ocr"},
    ]}], "max_tokens": max_tokens}}
    try:
        r = requests.post(f"{_BASE}/run", json=payload, headers=_HDR, timeout=60)
        r.raise_for_status()
        jid = r.json()["id"]
        t0 = time.time()
        while True:
            s = requests.get(f"{_BASE}/status/{jid}", headers=_HDR, timeout=60).json()
            st = s.get("status")
            if st == "COMPLETED":
                out = s.get("output") or {}
                if "choices" in out:
                    return out["choices"][0]["message"]["content"]
                raise HTTPException(status_code=502, detail=f"worker returned no text: {out}")
            if st in ("FAILED", "CANCELLED", "TIMED_OUT"):
                raise HTTPException(status_code=502, detail=f"ocr job {st}: {s.get('output')}")
            if time.time() - t0 > OCR_TIMEOUT:
                raise HTTPException(status_code=504, detail="ocr timed out (cold start or page too long)")
            time.sleep(2)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"runpod request failed: {e}")


class OcrJson(BaseModel):
    image_base64: str | None = None
    image_url: str | None = None
    max_tokens: int | None = None


@app.get("/health")
def health():
    return {"status": "ok", "endpoint": RUNPOD_ENDPOINT_ID}


@app.post("/ocr")
async def ocr_upload(
    file: UploadFile = File(...),
    max_tokens: int | None = Query(None),
    x_api_key: str | None = Header(None),
):
    """OCR an uploaded image file (multipart/form-data, field name 'file')."""
    _check_key(x_api_key)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    b64 = base64.b64encode(data).decode()
    return {"text": _ocr_b64(b64, max_tokens or DEFAULT_MAX_TOKENS)}


@app.post("/ocr-json")
def ocr_json(body: OcrJson, x_api_key: str | None = Header(None)):
    """OCR an image given as base64 (image_base64) or a URL (image_url)."""
    _check_key(x_api_key)
    if body.image_base64:
        b64 = body.image_base64.split(",", 1)[-1]  # tolerate a data: URL prefix
    elif body.image_url:
        try:
            img = requests.get(body.image_url, timeout=30)
            img.raise_for_status()
            b64 = base64.b64encode(img.content).decode()
        except requests.RequestException as e:
            raise HTTPException(status_code=400, detail=f"could not fetch image_url: {e}")
    else:
        raise HTTPException(status_code=400, detail="provide image_base64 or image_url")
    return {"text": _ocr_b64(b64, body.max_tokens or DEFAULT_MAX_TOKENS)}
