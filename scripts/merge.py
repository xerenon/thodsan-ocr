"""Merge LoRA adapter เข้า base → โมเดลเต็มพร้อม serve."""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", default="outputs/qlora")
    ap.add_argument("--out", default="outputs/typhoon-ocr-3b-thaihw-merged")
    args = ap.parse_args()

    # ms-swift export API (ถ้าเวอร์ชันต่างกัน ใช้ CLI:
    #   swift export --adapters <a> --merge_lora true --output_dir <out>)
    from swift.llm import export_main, ExportArguments

    export_main(ExportArguments(
        adapters=args.adapters, merge_lora=True, output_dir=args.out,
    ))
    print("merged ->", args.out)


if __name__ == "__main__":
    main()
