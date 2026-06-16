"""CER/WER สำหรับภาษาไทย พร้อม normalization."""
import re
import jiwer

# zero-width chars: U+200B ZWSP, U+200C ZWNJ, U+200D ZWJ, U+FEFF BOM
_ZERO_WIDTH = {ord(c): None for c in ("​", "‌", "‍", "﻿")}


def normalize_thai(text: str) -> str:
    """ลบ zero-width chars, ตัดช่องว่างหัวท้าย, ยุบช่องว่างซ้ำ."""
    text = text.translate(_ZERO_WIDTH)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def cer(refs: list[str], hyps: list[str]) -> float:
    """Character Error Rate (0.0 = สมบูรณ์แบบ)."""
    refs = [normalize_thai(r) for r in refs]
    hyps = [normalize_thai(h) for h in hyps]
    return float(jiwer.cer(refs, hyps))


def wer(refs: list[str], hyps: list[str]) -> float:
    """Word Error Rate (แบ่งคำด้วยช่องว่าง)."""
    refs = [normalize_thai(r) for r in refs]
    hyps = [normalize_thai(h) for h in hyps]
    return float(jiwer.wer(refs, hyps))
