"""Client เรียก RunPod serverless endpoint (วัด latency)."""
import base64
import os
import sys
import time

import requests


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def ocr_via_runpod(image_path: str, endpoint_id: str | None = None,
                   api_key: str | None = None) -> tuple[str, float]:
    """ส่งภาพไป endpoint, คืน (ข้อความ, latency วินาที)."""
    endpoint_id = endpoint_id or os.environ["RUNPOD_ENDPOINT_ID"]
    api_key = api_key or os.environ["RUNPOD_API_KEY"]
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    b64 = encode_image(image_path)
    payload = {"input": {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": "อ่านข้อความลายมือภาษาไทยในรูปภาพ แล้วพิมพ์ออกมา"},
    ]}], "max_tokens": 256}}
    t0 = time.time()
    r = requests.post(url, json=payload,
                      headers={"Authorization": f"Bearer {api_key}"}, timeout=300)
    r.raise_for_status()
    dt = time.time() - t0
    data = r.json()
    text = data["output"]["choices"][0]["message"]["content"]
    return text, dt


if __name__ == "__main__":
    txt, dt = ocr_via_runpod(sys.argv[1])
    print(f"latency={dt:.1f}s\ntext={txt}")
