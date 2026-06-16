"""โชว์ผลอ่านลายมือ: เฉลย vs โมเดลทำนาย (ใช้ transformers ตรงๆ เหมือน baseline_eval)."""
import io
import sys

import pandas as pd
from PIL import Image

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from baseline_eval import load_model, predict
from thaiocr.eval.metrics import cer

N = int(sys.argv[2]) if len(sys.argv) > 2 else 6
model_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/typhoon-thaihw-merged"

df = pd.read_parquet("data/splits/test.parquet").head(N)
model, proc = load_model(model_path)

rows = []
for i, row in df.iterrows():
    img = row["image"]
    data = img["bytes"] if isinstance(img, dict) else img
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    pred = predict(model, proc, pil)
    gt = str(row["text"])
    c = cer([gt], [pred])
    mark = "ถูกเป๊ะ" if gt.strip() == pred.strip() else f"CER {round(c * 100, 1)}%"
    rows.append(f"{len(rows)+1}. [{mark}]\n   เฉลย  : {gt}\n   โมเดล : {pred}")

with open("outputs/demo_show.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(rows))
print("\n".join(rows))
