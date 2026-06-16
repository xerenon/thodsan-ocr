# Thai Handwriting OCR — VPS frontend API

A thin, CPU-only FastAPI service that proxies image OCR requests to the RunPod
serverless GPU endpoint. Deploy it on any VPS (EPYC / Xeon, no GPU) — it forwards
images to the GPU worker and returns the transcribed Thai text. It submits the job
and polls RunPod's `/status`, so a worker cold start never returns a partial result.

```
client  -->  this API (VPS, CPU)  -->  RunPod serverless (GPU)  -->  Thai text
```

## Configure

```bash
cp .env.example .env
# edit .env: set RUNPOD_API_KEY, and API_KEY (a long random string clients must send)
```

| Env | Required | Default | Notes |
|-----|----------|---------|-------|
| `RUNPOD_ENDPOINT_ID` | yes | — | RunPod serverless endpoint id |
| `RUNPOD_API_KEY` | yes | — | RunPod API key (keep secret) |
| `API_KEY` | no | — | If set, clients must send it as the `X-API-Key` header |
| `OCR_MAX_TOKENS` | no | 4096 | Output cap; 4096 covers a full A4 page of Thai |
| `OCR_TIMEOUT` | no | 300 | Max seconds to wait for a job (cold start + long pages) |
| `CORS_ORIGINS` | no | `*` | Comma-separated allowed origins |

## Run

**Docker Compose (recommended):**
```bash
docker compose up -d --build      # serves on :8000
```

**Plain Python:**
```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
set -a; . ./.env; set +a               # load env
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2
```

**systemd (no Docker):** put the app in `/opt/thaiocr-api` (with `.venv` and `.env`), then
```bash
sudo cp thaiocr-api.service /etc/systemd/system/
sudo systemctl enable --now thaiocr-api
```

## Endpoints

- `GET /health` -> `{"status":"ok", ...}`
- `POST /ocr` -> multipart upload, field `file`; optional `?max_tokens=`
- `POST /ocr-json` -> JSON body `{"image_base64": "..."}` or `{"image_url": "..."}`, optional `max_tokens`

Interactive docs at `/docs`.

## Examples

```bash
# file upload
curl -s -X POST http://localhost:8000/ocr \
  -H "X-API-Key: $API_KEY" \
  -F "file=@page.jpg"

# longer cap for a dense page
curl -s -X POST "http://localhost:8000/ocr?max_tokens=4096" \
  -H "X-API-Key: $API_KEY" -F "file=@page.jpg"

# JSON with a URL
curl -s -X POST http://localhost:8000/ocr-json \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/handwriting.png"}'
```

Response: `{"text": "<transcribed Thai>"}`

## Notes

- First request after idle is slow (~1-3 min) while a GPU worker cold-starts; subsequent
  requests are fast. The API waits up to `OCR_TIMEOUT` seconds.
- Put this behind nginx/Caddy with TLS for public use; keep `API_KEY` set so the endpoint
  isn't open. The VPS needs only outbound HTTPS to `api.runpod.ai`.
- This service holds no model and needs no GPU — all inference runs on RunPod.
