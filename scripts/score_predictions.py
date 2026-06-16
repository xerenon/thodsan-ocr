"""อ่าน predictions jsonl (มี label + response) → คำนวณ CER/WER เทียบ baseline."""
import json
import sys

sys.path.insert(0, "src")
from thaiocr.eval.metrics import cer, wer


def main(pred_path: str):
    refs, hyps = [], []
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ref = d.get("label")
            if ref is None:
                ref = d["messages"][-1]["content"]
            refs.append(str(ref))
            hyps.append(str(d["response"]))

    ft = {"cer": cer(refs, hyps), "wer": wer(refs, hyps), "n": len(refs)}
    try:
        base = json.load(open("outputs/baseline_report.json", encoding="utf-8"))
        print("baseline cer:", round(base["cer"], 4), "| fine-tuned cer:", round(ft["cer"], 4))
        print("CER improvement:", round((base["cer"] - ft["cer"]) * 100, 2), "points")
    except FileNotFoundError:
        print("fine-tuned:", ft)

    json.dump(ft, open("outputs/ft_report.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main(sys.argv[1])
