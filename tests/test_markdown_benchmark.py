import json
from pathlib import Path
import subprocess
import sys

import pytest

from thaiocr.eval.markdown_benchmark import compare, main

FIXTURES = Path(__file__).parent / "fixtures" / "markdown"


def test_fixture_comparison_and_macro_coverage():
    report = compare(FIXTURES / "references.jsonl", FIXTURES / "runs.json")
    perfect, damaged = report["runs"]
    assert len(report["reference_sha256"]) == 64
    assert perfect["summary"]["text"]["cer"]["macro_mean"] == 0
    assert perfect["summary"]["tables"]["shape_similarity"]["scored_pages"] == 1
    assert damaged["summary"]["headings"]["hierarchy_f1"]["macro_mean"] < 1
    assert damaged["summary"]["tables"]["shape_similarity"]["macro_mean"] < 1
    assert damaged["summary"]["reading_order"]["score"]["macro_mean"] < 1


def setup_run(tmp_path, rows):
    refs = tmp_path / "refs.jsonl"
    refs.write_text('{"id":"page","markdown":"# ก"}\n', encoding="utf-8")
    (tmp_path / "pred.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    runs = tmp_path / "runs.json"
    runs.write_text(json.dumps([{"name":"test","model":"test","revision":"v1",
                                "prompt":"test","decoding":{},"predictions":"pred.jsonl"}]))
    return refs, runs


@pytest.mark.parametrize("rows,message", [
    ([{"id":"page","response":"ก"},{"id":"page","response":"ข"}], "duplicate"),
    ([{"id":"unknown","response":"ก"}], "ID mismatch"),
    ([{"id":"page","response":None}], "string response"),
    ([{"id":"page","response":123}], "string response"),
    ([{"id":"","response":"ก"}], "non-empty"),
    ([], "empty dataset"),
])
def test_bad_predictions_fail_closed(tmp_path, rows, message):
    refs, runs = setup_run(tmp_path, rows)
    with pytest.raises(ValueError, match=message):
        compare(refs, runs)


def test_prediction_order_is_joined_by_id(tmp_path):
    refs, runs = setup_run(tmp_path, [{"id":"other","response":"ข"},{"id":"page","response":"# ก"}])
    refs.write_text(refs.read_text()+ '{"id":"other","markdown":"ข"}\n')
    report = compare(refs, runs)
    assert report["runs"][0]["summary"]["text"]["cer"]["macro_mean"] == 0


def test_duplicate_reference_rejected(tmp_path):
    refs, runs = setup_run(tmp_path, [{"id":"page","response":"ก"}])
    refs.write_text(refs.read_text()*2)
    with pytest.raises(ValueError, match="duplicate"):
        compare(refs, runs)


def test_run_metadata_required_and_names_unique(tmp_path):
    refs, runs = setup_run(tmp_path, [{"id":"page","response":"ก"}])
    config = json.loads(runs.read_text())
    runs.write_text(json.dumps(config*2))
    with pytest.raises(ValueError, match="duplicate run"):
        compare(refs, runs)
    del config[0]["revision"]
    runs.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="requires"):
        compare(refs, runs)


def test_cli_writes_valid_utf8_json(tmp_path):
    output = tmp_path / "new" / "report.json"
    main(["--references", str(FIXTURES / "references.jsonl"),
          "--runs", str(FIXTURES / "runs.json"), "--output", str(output)])
    result = json.loads(output.read_text(encoding="utf-8"))
    assert len(result["runs"]) == 2
    assert "ภาษาไทย" in output.read_text(encoding="utf-8")


def test_eval_does_not_import_gpu_stack():
    subprocess.run([sys.executable, "-c", "from thaiocr.eval.markdown import score_markdown; "
                    "import sys; score_markdown('# ก', '# ก'); "
                    "assert not {'torch', 'transformers', 'swift', 'peft'} & sys.modules.keys()"],
                   env=__import__('os').environ | {"PYTHONPATH":"src"}, check=True)
