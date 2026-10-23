"""Offline Markdown OCR metrics, separate from the handwriting evaluation path.

Supported dialect: CommonMark headings/paragraphs/lists/code plus pipe tables.
HTML is treated as literal text; merged HTML cells are outside this benchmark.
"""
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from itertools import combinations
import unicodedata

from markdown_it import MarkdownIt
from markdown_it.rules_block.table import escapedSplit

from thaiocr.eval.metrics import cer, normalize_thai, wer

METRIC_VERSION = "markdown-ocr-v1"


@dataclass
class Document:
    text: str
    headings: list[tuple]
    tables: list[list[list[str]]]
    blocks: list[str]


def _text(token) -> str:
    """Visible inline text, excluding markup while retaining code/image alt text."""
    if token.children is None:
        return token.content
    return "".join(
        "\n" if child.type in {"softbreak", "hardbreak"} else
        child.content if child.type in {"text", "code_inline", "html_inline", "image"} else ""
        for child in token.children
    )


def parse_markdown(source: str) -> Document:
    # NFC is applied only to the new evaluator. Thai tone marks are preserved.
    source = unicodedata.normalize("NFC", source.replace("\r\n", "\n").replace("\r", "\n"))
    parser = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = parser.parse(source)
    lines = source.splitlines()
    text, headings, tables, blocks = [], [], [], []
    parents: list[tuple[int, str]] = []
    table = None
    row = None
    for i, token in enumerate(tokens):
        if token.type == "table_open":
            if token.level != 0:
                raise ValueError("Nested tables are not supported by markdown-ocr-v1")
            table = []
        elif token.type == "tr_open":
            # Preserve ragged rows; the renderer silently pads/truncates them.
            raw_cells = escapedSplit(lines[token.map[0]].strip())
            if raw_cells and raw_cells[0] == "":
                raw_cells.pop(0)
            if raw_cells and raw_cells[-1] == "":
                raw_cells.pop()
            row = [normalize_thai(_text(parser.parseInline(cell.strip())[0]))
                   for cell in raw_cells]
            text.extend(row)
        elif token.type == "tr_close":
            table.append(row)
        elif token.type == "table_close":
            tables.append(table)
            blocks.append(normalize_thai(" ".join(cell for r in table for cell in r)))
            table = None
        elif token.type == "inline":
            visible = _text(token)
            if table is None:
                text.append(visible)
                blocks.append(normalize_thai(visible))
                previous = tokens[i - 1]
                if previous.type == "heading_open":
                    level = int(previous.tag[1:])
                    while parents and parents[-1][0] >= level:
                        parents.pop()
                    name = normalize_thai(visible)
                    parents.append((level, name))
                    headings.append(tuple(parents))
        elif token.type in {"fence", "code_block"}:
            text.append(token.content)
            blocks.append(normalize_thai(token.content))
    return Document("\n".join(text), headings, tables, blocks)


def _distance(a, b) -> int:
    """Levenshtein for token sequences, linear auxiliary memory."""
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def _sequence_f1(a, b) -> float | None:
    # LCS: ordering, duplicates and insertion/deletion all matter.
    if not a and not b:
        return None
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if x == y else max(current[-1], previous[j]))
        previous = current
    return 2 * previous[-1] / (len(a) + len(b))


def _error_rate(reference: str, prediction: str, metric) -> float | None:
    # jiwer 3.x rejects an empty reference. Report undefined, with counts separately.
    if not normalize_thai(reference):
        return None
    return metric([reference], [prediction])


def _table_shape(tables):
    # No cell contents: this score isolates count, row boundaries and column count.
    return [item for table in tables for item in
            (["table"] + [("row", len(row)) for row in table] + ["/table"])]


def _table_cells(tables):
    return {(t, r, c): value for t, table in enumerate(tables)
            for r, row in enumerate(table) for c, value in enumerate(row)}


def _reading_order(ref: list[str], hyp: list[str]) -> dict:
    # Exact visible-text anchors: conservative and deterministic. Duplicate blocks
    # match by occurrence; their true identity cannot be inferred from text alone.
    positions = defaultdict(deque)
    for i, block in enumerate(ref):
        positions[block].append(i)
    matched = []
    for block in hyp:
        if positions[block]:
            matched.append(positions[block].popleft())
    n = len(matched)
    pairs = n * (n - 1) // 2
    correct = sum(a < b for a, b in combinations(matched, 2))
    coverage = 2 * n / (len(ref) + len(hyp)) if ref or hyp else None
    concordance = correct / pairs if pairs else None
    return {"pair_accuracy": concordance, "anchor_coverage_f1": coverage,
            "score": concordance * coverage if pairs else None,
            "matched_blocks": n, "reference_blocks": len(ref),
            "prediction_blocks": len(hyp), "comparable_pairs": pairs,
            "ambiguous_duplicate_blocks": sum(v - 1 for v in Counter(ref).values() if v > 1)}


def score_markdown(reference: str, prediction: str) -> dict:
    """No composite score. Rates are fractions; CER/WER can exceed one."""
    ref, hyp = parse_markdown(reference), parse_markdown(prediction)
    shape_ref, shape_hyp = _table_shape(ref.tables), _table_shape(hyp.tables)
    cells_ref, cells_hyp = _table_cells(ref.tables), _table_cells(hyp.tables)
    coordinates = cells_ref.keys() | cells_hyp.keys()
    exact_cells = sum(c in cells_ref and c in cells_hyp and cells_ref[c] == cells_hyp[c]
                      for c in coordinates)
    # Coordinate CER penalizes moving content into a different cell. One space is
    # used as the non-empty reference sentinel for absent/empty cells (documented).
    cell_refs = [cells_ref.get(c, "") or " " for c in sorted(coordinates)]
    cell_hyps = [cells_hyp.get(c, "") or " " for c in sorted(coordinates)]
    # Do not send sentinel strings through normalize_thai, which strips them.
    import jiwer
    return {
        "text": {"cer": _error_rate(ref.text, hyp.text, cer),
                 "wer_whitespace": _error_rate(ref.text, hyp.text, wer),
                 "reference_characters": len(normalize_thai(ref.text)),
                 "prediction_characters": len(normalize_thai(hyp.text)),
                 "visible_text_exact": normalize_thai(ref.text) == normalize_thai(hyp.text),
                 "raw_markdown_exact": reference == prediction,
                 "verbatim_markdown_cer": float(jiwer.cer([reference], [prediction]))
                 if reference else None},
        "headings": {"hierarchy_f1": _sequence_f1(ref.headings, hyp.headings),
                     "level_f1": _sequence_f1([h[-1][0] for h in ref.headings],
                                              [h[-1][0] for h in hyp.headings]),
                     "reference_count": len(ref.headings), "prediction_count": len(hyp.headings)},
        "tables": {"shape_similarity": max(0.0, 1 - _distance(shape_ref, shape_hyp) /
                                            max(len(shape_ref), len(shape_hyp)))
                   if shape_ref or shape_hyp else None,
                   "cell_exact_accuracy": exact_cells / len(coordinates) if coordinates else None,
                   "cell_cer": float(jiwer.cer(cell_refs, cell_hyps)) if coordinates else None,
                   "reference_count": len(ref.tables), "prediction_count": len(hyp.tables)},
        "reading_order": _reading_order(ref.blocks, hyp.blocks),
    }
