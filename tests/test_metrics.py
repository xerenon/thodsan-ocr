from thaiocr.eval.metrics import normalize_thai, cer, wer


def test_normalize_strips_and_collapses_space():
    assert normalize_thai("  สวัสดี   ครับ ") == "สวัสดี ครับ"


def test_normalize_removes_zero_width():
    # zero-width space (U+200B) ต้องถูกลบ
    assert normalize_thai("ก​ข") == "กข"


def test_cer_perfect_match_is_zero():
    assert cer(["สวัสดี"], ["สวัสดี"]) == 0.0


def test_cer_counts_one_char_error():
    # เฉลย "กข" ทำนาย "กค" → ผิด 1 จาก 2 ตัว = 0.5
    assert abs(cer(["กข"], ["กค"]) - 0.5) < 1e-6


def test_wer_perfect_match_is_zero():
    assert wer(["ไป ตลาด"], ["ไป ตลาด"]) == 0.0
