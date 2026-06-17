# Thai Handwriting OCR API — Usage

How to call the OCR API from client code. Send an image, get back transcribed Thai text.
Formal spec: [`openapi.yaml`](openapi.yaml). Interactive docs: `http://<host>:8000/docs`.

## Base URL and auth

```
Base URL:  http://46.62.242.183:8000
Auth:      header  X-API-Key: <your key>
```

Every request except `GET /health` must include the `X-API-Key` header. A missing or wrong
key returns `401`.

## Endpoints

| Method | Path | Body | Purpose |
|--------|------|------|---------|
| GET  | `/health`   | — | Liveness check (no auth) |
| POST | `/ocr`      | multipart `file` | OCR an uploaded image file |
| POST | `/ocr-json` | JSON `{image_base64 \| image_url}` | OCR from base64 or a URL |

Optional `max_tokens` (default **4096**, covers a full A4 page of Thai) — query param on
`/ocr`, or a field in the `/ocr-json` body.

Success response (all OCR endpoints):
```json
{ "text": "<transcribed Thai text>" }
```

## Examples

### curl

```bash
KEY=YOUR_API_KEY
BASE=http://46.62.242.183:8000

# health
curl $BASE/health

# upload a file
curl -X POST $BASE/ocr -H "X-API-Key: $KEY" -F "file=@page.jpg"

# upload with a higher cap for a dense page
curl -X POST "$BASE/ocr?max_tokens=4096" -H "X-API-Key: $KEY" -F "file=@page.jpg"

# from an image URL
curl -X POST $BASE/ocr-json -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/handwriting.png"}'
```

### Python (requests)

```python
import requests

BASE = "http://46.62.242.183:8000"
KEY = "YOUR_API_KEY"

# file upload
with open("page.jpg", "rb") as f:
    r = requests.post(f"{BASE}/ocr", headers={"X-API-Key": KEY},
                      files={"file": f}, timeout=300)
r.raise_for_status()
print(r.json()["text"])

# base64
import base64
b64 = base64.b64encode(open("page.jpg", "rb").read()).decode()
r = requests.post(f"{BASE}/ocr-json", headers={"X-API-Key": KEY},
                  json={"image_base64": b64, "max_tokens": 4096}, timeout=300)
print(r.json()["text"])
```

### JavaScript (browser / fetch)

```js
const BASE = "http://46.62.242.183:8000";
const KEY = "YOUR_API_KEY";

const fd = new FormData();
fd.append("file", fileInput.files[0]);

const res = await fetch(`${BASE}/ocr`, {
  method: "POST",
  headers: { "X-API-Key": KEY },   // do NOT set Content-Type; the browser sets the boundary
  body: fd,
});
const { text } = await res.json();
console.log(text);
```

### Node.js (fetch + FormData)

```js
import fs from "node:fs";

const fd = new FormData();
fd.append("file", new Blob([fs.readFileSync("page.jpg")]), "page.jpg");

const res = await fetch("http://46.62.242.183:8000/ocr", {
  method: "POST",
  headers: { "X-API-Key": process.env.OCR_API_KEY },
  body: fd,
});
console.log((await res.json()).text);
```

## Errors

Errors are JSON: `{ "detail": "<message>" }`.

| Code | Meaning | Fix |
|------|---------|-----|
| 400 | Empty/invalid file, or no image given | Send a valid `file`, or one of `image_base64` / `image_url` |
| 401 | Missing or wrong `X-API-Key` | Send the correct key |
| 502 | GPU worker failed / returned no text | Retry; check the endpoint if it persists |
| 504 | Timed out (cold start or very long page) | Retry — the worker was warming up; next call is fast |

## Notes

- **Cold start:** the first request after an idle period takes ~1-3 minutes while a GPU
  worker spins up. Subsequent requests are fast. Use a client timeout of at least 300s.
- **max_tokens** is an output cap, not a fixed cost — the model stops at end-of-text, so a
  high value only adds headroom for long pages and does not slow short ones.
- **Accuracy:** ~6.6% CER on single-line Thai handwriting; clear writing often comes back
  letter-perfect. Full multi-line pages also work well.
- **Image formats:** common raster formats (png, jpg, webp). Send a reasonably sharp image;
  very low-resolution scans reduce accuracy.
