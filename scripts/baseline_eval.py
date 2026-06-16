"""วัด CER/WER ของโมเดลบน test split. ใช้ได้ทั้ง base และ fine-tuned/merged."""
import argparse
import io
import json
import os
import sys

import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, "src")
from thaiocr.eval.metrics import cer, wer
from thaiocr.data.format import OCR_INSTRUCTION


def load_model(model_path: str):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    return model, proc


def predict(model, proc, pil_img) -> str:
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_img},
        {"type": "text", "text": OCR_INSTRUCTION},
    ]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(messages)
    inputs = proc(text=[text], images=imgs, videos=vids, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    trimmed = out[:, inputs.input_ids.shape[1]:]
    return proc.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="typhoon-ai/typhoon-ocr-3b")
    ap.add_argument("--test", default="data/splits/test.parquet")
    ap.add_argument("--limit", type=int, default=0, help="0 = ทั้งหมด; ใส่เลขเพื่อ smoke test")
    ap.add_argument("--out", default="outputs/baseline_report.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.test)
    if args.limit:
        df = df.head(args.limit)
    model, proc = load_model(args.model)

    refs, hyps = [], []
    for _, row in df.iterrows():
        img = row["image"]
        data = img["bytes"] if isinstance(img, dict) else img
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        hyps.append(predict(model, proc, pil))
        refs.append(str(row["text"]))

    report = {"model": args.model, "n": len(refs),
              "cer": cer(refs, hyps), "wer": wer(refs, hyps)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
