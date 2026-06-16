"""Local smoke test for the serverless handler — no GPU endpoint cost.

Validates that the model loads (correct transformers + Qwen3-VL class + PEFT adapter)
and that handler() returns text in the shape serve/client.py expects.

Run from repo root (needs the adapter + a GPU with the base model available):
  ADAPTER_DIR=outputs/v2_2b_ckpt1063 THAIOCR_SRC=src \
    python cloud/serverless/test_handler.py [image.png]
Defaults to the first image under data/swift/images/test/ if no path is given.
"""
import base64
import glob
import os
import sys

os.environ.setdefault("ADAPTER_DIR", "outputs/v2_2b_ckpt1063")
os.environ.setdefault("THAIOCR_SRC", "src")

sys.path.insert(0, "cloud/serverless")
import handler as h  # noqa: E402  (loads the model on import)


def main():
    img = sys.argv[1] if len(sys.argv) > 1 else None
    if not img:
        cands = sorted(glob.glob("data/swift/images/test/*.png"))
        if not cands:
            sys.exit("no test image found; pass a path as argv[1]")
        img = cands[0]

    b64 = base64.b64encode(open(img, "rb").read()).decode()
    job = {"input": {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": "ocr"},
    ]}], "max_tokens": 256}}

    out = h.handler(job)
    print("image :", img)
    print("output:", out)
    assert out.get("choices"), f"unexpected output shape: {out}"
    print("OK — text:", out["choices"][0]["message"]["content"][:120])


if __name__ == "__main__":
    main()
