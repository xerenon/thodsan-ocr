from pathlib import Path

import pytest

from thaiocr.eval.markdown import parse_markdown, score_markdown
from thaiocr.eval.metrics import normalize_thai

FIXTURES = Path(__file__).parent / "fixtures" / "markdown"
TABLE = "| สินค้า | จำนวน |\n| --- | --- |\n| โต๊ะ | ๒ |\n| ตู้ | ๕ |\n"


@pytest.mark.parametrize("name", ["receipt", "procedure", "columns"])
def test_thai_fixture_identity(name):
    text = (FIXTURES / f"{name}.md").read_text(encoding="utf-8")
    result = score_markdown(text, text)
    assert result["text"]["cer"] == 0
    assert result["text"]["wer_whitespace"] == 0
    assert result["headings"]["hierarchy_f1"] == 1
    assert result["reading_order"]["score"] == 1
    if name == "receipt":
        assert result["tables"]["shape_similarity"] == 1
        assert result["tables"]["cell_exact_accuracy"] == 1
        assert result["tables"]["cell_cer"] == 0


def test_heading_level_error_separate_from_text():
    ref = "# รายงาน\n\n## หัวข้อ\n\nข้อความ"
    result = score_markdown(ref, ref.replace("## หัวข้อ", "### หัวข้อ"))
    assert result["text"]["cer"] == 0
    assert result["headings"]["hierarchy_f1"] < 1
    assert result["reading_order"]["score"] == 1


def test_heading_parent_path_matters():
    result = score_markdown("# ก\n\n## ข\n\n### ค", "# ก\n\n## ข\n\n## ค")
    assert result["headings"]["hierarchy_f1"] < 1


@pytest.mark.parametrize("hyp", ["", "# รายงาน\n\n# เพิ่ม", "รายงาน"])
def test_missing_extra_or_plain_heading(hyp):
    assert score_markdown("# รายงาน", hyp)["headings"]["hierarchy_f1"] < 1


def test_equivalent_setext_heading_and_emphasis():
    result = score_markdown("# รายงาน\n\n**พร้อม**", "รายงาน\n===\n\nพร้อม")
    assert result["headings"]["hierarchy_f1"] == 1
    assert result["text"]["cer"] == 0
    assert not result["text"]["raw_markdown_exact"]


def test_thai_tone_marks_are_not_discarded():
    result = score_markdown("น้ำ\n\nป้า", "นำ\n\nปา")
    assert result["text"]["cer"] > 0


def test_legacy_normalization_unchanged():
    assert normalize_thai("  ก\u200bข\n\n  ค ") == "กข ค"


def test_table_content_error_leaves_shape_intact():
    result = score_markdown(TABLE, TABLE.replace("๒", "๓"))
    assert result["tables"]["shape_similarity"] == 1
    assert result["tables"]["cell_cer"] > 0
    assert result["tables"]["cell_exact_accuracy"] < 1


@pytest.mark.parametrize("hyp", ["", TABLE + "\n" + TABLE,
    TABLE.replace("| ตู้ | ๕ |\n", ""), TABLE.replace("| โต๊ะ | ๒ |", "| โต๊ะ |"),
    TABLE.replace("| โต๊ะ | ๒ |", "| โต๊ะ | ๒ | เพิ่ม |")])
def test_table_missing_extra_ragged_rows(hyp):
    result = score_markdown(TABLE, hyp)
    assert result["tables"]["shape_similarity"] < 1
    assert result["tables"]["cell_exact_accuracy"] < 1


def test_cell_and_row_swaps_are_detected():
    swapped = TABLE.replace("| โต๊ะ | ๒ |\n| ตู้ | ๕ |", "| ตู้ | ๕ |\n| โต๊ะ | ๒ |")
    result = score_markdown(TABLE, swapped)
    assert result["tables"]["shape_similarity"] == 1
    assert result["tables"]["cell_exact_accuracy"] < 1


def test_table_spacing_alignment_and_escaped_pipes():
    a = "| รายการ | จำนวน |\n|---|---|\n| ก\\|ข | ๒ |"
    b = "รายการ|จำนวน\n:---|---:\nก\\|ข|๒"
    result = score_markdown(a, b)
    assert result["tables"]["cell_exact_accuracy"] == 1
    assert parse_markdown(a).tables[0][1][0] == "ก|ข"


def test_code_fence_does_not_create_headings_or_tables():
    result = parse_markdown("```md\n# หัวข้อ\n" + TABLE + "```\n")
    assert result.headings == []
    assert result.tables == []
    assert "# หัวข้อ" in result.text


def test_order_swap_has_all_anchors_but_wrong_pairs():
    result = score_markdown("ก\n\nข\n\nค", "ข\n\nก\n\nค")["reading_order"]
    assert result["anchor_coverage_f1"] == 1
    assert result["pair_accuracy"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("hyp", ["ก\n\nค", "ก\n\nข\n\nค\n\nง", "ก\n\nข\n\nค\n\nก"])
def test_order_missing_hallucinated_duplicate_blocks_penalize_coverage(hyp):
    result = score_markdown("ก\n\nข\n\nค", hyp)["reading_order"]
    assert result["score"] < 1


def test_duplicate_blocks_match_occurrences_and_report_ambiguity():
    text = "ก\n\nข\n\nก"
    result = score_markdown(text, text)["reading_order"]
    assert result["matched_blocks"] == 3
    assert result["score"] == 1
    assert result["ambiguous_duplicate_blocks"] == 1


def test_order_is_undefined_with_insufficient_exact_anchors():
    result = score_markdown("ก\n\nข", "ก\n\nค")["reading_order"]
    assert result["score"] is None
    assert result["comparable_pairs"] == 0
    assert result["anchor_coverage_f1"] == 0.5


def test_absent_structures_and_empty_reference_are_not_perfect_scores():
    result = score_markdown("", "เพิ่มข้อความ")
    assert result["text"]["cer"] is None
    assert result["text"]["prediction_characters"] > 0
    assert result["headings"]["hierarchy_f1"] is None
    assert result["tables"]["shape_similarity"] is None
    assert result["reading_order"]["score"] is None


def test_hallucinated_structure_when_reference_has_none():
    result = score_markdown("ข้อความ", "# เพิ่ม\n\n" + TABLE)
    assert result["headings"]["hierarchy_f1"] == 0
    assert result["tables"]["shape_similarity"] == 0


def test_whitespace_normalization_does_not_hide_heading_loss():
    result = score_markdown("# ก\n\nข", "ก\n\nข")
    assert result["text"]["cer"] == 0
    assert result["headings"]["hierarchy_f1"] == 0


def test_nested_table_rejected_explicitly():
    with pytest.raises(ValueError, match="Nested tables"):
        parse_markdown("> | ก | ข |\n> |---|---|\n> | ๑ | ๒ |")


def test_verbatim_score_detects_collapsed_whitespace():
    result = score_markdown("ก  ข\n\nค", "ก ข\nค")
    assert result["text"]["cer"] == 0
    assert result["text"]["verbatim_markdown_cer"] > 0


def test_heading_text_error_keeps_level_score():
    result = score_markdown("# รายงาน", "# รายงาย")
    assert result["headings"]["level_f1"] == 1
    assert result["headings"]["hierarchy_f1"] == 0


def test_empty_cells_do_not_disappear():
    table = "| ก | ข |\n|---|---|\n| | ๒ |"
    assert score_markdown(table, table)["tables"]["cell_cer"] == 0
    assert score_markdown(table, table.replace("| | ๒ |", "| ๒ |"))["tables"]["cell_exact_accuracy"] < 1


def test_table_flattening_cannot_get_perfect_structure_score():
    result = score_markdown(TABLE, "สินค้า จำนวน\nโต๊ะ ๒\nตู้ ๕")
    assert result["text"]["cer"] == 0
    assert result["tables"]["shape_similarity"] == 0


def test_missing_table_separator_is_detected():
    assert score_markdown(TABLE, TABLE.replace("| --- | --- |\n", ""))["tables"]["shape_similarity"] == 0
