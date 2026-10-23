"""Compare saved OCR responses on identical page IDs. Never loads a model."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
from importlib.metadata import version

from thaiocr.eval.markdown import METRIC_VERSION, score_markdown


def _load_jsonl(path: Path, field: str) -> dict[str, str]:
    records = {}
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                    or not row["id"].strip() or not isinstance(row.get(field), str)):
                raise ValueError(f"{path}:{number}: expected non-empty string id and string {field}")
            if row["id"] in records:
                raise ValueError(f"{path}:{number}: duplicate id {row['id']}")
            records[row["id"]] = row[field]
    if not records:
        raise ValueError(f"{path}: empty dataset")
    return records


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _aggregate(pages: list[dict]) -> dict:
    # Macro-average per page; undefined categories do not become perfect scores.
    output = {}
    for category, keys in {
        "text": ("cer", "wer_whitespace", "verbatim_markdown_cer"),
        "headings": ("hierarchy_f1", "level_f1"),
        "tables": ("shape_similarity", "cell_exact_accuracy", "cell_cer"),
        "reading_order": ("pair_accuracy", "anchor_coverage_f1", "score"),
    }.items():
        output[category] = {}
        for key in keys:
            values = [p["scores"][category][key] for p in pages
                      if p["scores"][category][key] is not None]
            output[category][key] = {
                "macro_mean": sum(values) / len(values) if values else None,
                "scored_pages": len(values), "total_pages": len(pages),
            }
    return output


def compare(reference_path: Path, runs_path: Path) -> dict:
    refs = _load_jsonl(reference_path, "markdown")
    runs = json.loads(runs_path.read_text(encoding="utf-8"))
    if not isinstance(runs, list) or not runs:
        raise ValueError("runs must be a non-empty list")
    report = {"metric_version": METRIC_VERSION,
              "environment": {"python": platform.python_version(),
                              "markdown-it-py": version("markdown-it-py"),
                              "jiwer": version("jiwer")},
              "reference_sha256": _hash(reference_path), "page_ids": sorted(refs), "runs": []}
    names = set()
    for run in runs:
        required = ("name", "model", "revision", "prompt", "decoding", "predictions")
        if (not isinstance(run, dict) or any(key not in run for key in required)
                or any(not isinstance(run[key], str) or not run[key].strip()
                       for key in required if key != "decoding")
                or not isinstance(run["decoding"], dict)):
            raise ValueError(f"each run requires string metadata and decoding object: {required}")
        if run["name"] in names:
            raise ValueError(f"duplicate run name: {run['name']}")
        names.add(run["name"])
        predictions_path = runs_path.parent / run["predictions"]
        hyps = _load_jsonl(predictions_path, "response")
        if refs.keys() != hyps.keys():
            raise ValueError(f"{run['name']}: ID mismatch; missing={sorted(refs.keys() - hyps.keys())}, "
                             f"unknown={sorted(hyps.keys() - refs.keys())}")
        pages = [{"id": page_id, "scores": score_markdown(refs[page_id], hyps[page_id])}
                 for page_id in sorted(refs)]
        report["runs"].append({"metadata": run, "predictions_sha256": _hash(predictions_path),
                               "pages": pages, "summary": _aggregate(pages)})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = compare(args.references, args.runs)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(f"Scored {len(report['page_ids'])} pages across {len(report['runs'])} runs: {args.output}")


if __name__ == "__main__":
    main()
