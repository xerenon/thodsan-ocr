# Markdown OCR evaluation (CPU only)

## ตรวจโค้ดเดิม

ตรวจจาก main commit `8678c43139397caa4d97400d25e29317fc7d1071` วันที่ 23 ตุลาคม 2026:

- `scripts/train.sh`: Typhoon OCR 3B, QLoRA 4-bit, ms-swift; template Qwen2.5-VL
- `src/thaiocr/data/format.py`: สร้าง image/conversation พร้อม prompt อ่านลายมือ ไม่ได้สร้าง Markdown ground truth
- `src/thaiocr/eval/metrics.py`: CER/WER โดยลบ zero-width characters และยุบ whitespace; WER แบ่งตามช่องว่าง ไม่ใช่การตัดคำภาษาไทย
- `scripts/score_predictions.py`: อ่าน `response` และ `label`/assistant message แล้วให้คะแนนข้อความ
- `tests/test_metrics.py`: ทดสอบ normalization และ CER/WER; ยังไม่มี heading/table/order metrics
- README ระบุผลลายมือ 6.63% และพิมพ์ 0.02%; ตัวเลขนี้ไม่ใช่ผล Markdown OCR และไม่ได้ยืนยันใหม่ในงานนี้

ตรวจไฟล์ทั้งหมดใน main, branch search (พบ main เท่านั้น), GitHub code search `markdown`,
issue/PR search `markdown` และ PR search `OCR` ไม่พบงานประเมิน Markdown OCR ที่ซ้ำกัน
ขอบเขตนี้ไม่ครอบคลุม private fork หรือการเปลี่ยนแปลงที่ยังไม่ได้ push

## ส่วนที่ใช้ซ้ำและขอบเขตแพตช์

ใช้ `normalize_thai`, `cer`, `wer` ของเดิมซ้ำเฉพาะ visible text; ใช้ package `thaiocr.eval`,
pytest configuration และรูปแบบ JSONL ที่มี `response` เหมือนเดิม เพิ่ม `id` เพื่อจับคู่หน้าโดยไม่อาศัยลำดับไฟล์

เพิ่ม `markdown.py` และ `markdown_benchmark.py` โดยไม่แก้ metrics, formatter, training script,
score_predictions หรือ assertions เดิม ไม่ใช้ prompt อ่านลายมือของ formatter กับงานเอกสาร Markdown
ไม่มีการโหลด torch/transformers, เรียก API, ฝึกโมเดล หรือใช้ GPU
ติดตั้ง evaluator แยกด้วย `requirements-eval.txt` ได้; ไม่ต้องติดตั้ง training requirements

## นิยามคะแนน v1

ค่าทั้งหมดเป็นสัดส่วน ไม่ใช่เปอร์เซ็นต์ และไม่มีคะแนนรวมที่รวมทุกด้านเข้าด้วยกัน

| กลุ่ม | ค่า | การคำนวณและการตีความ |
| --- | --- | --- |
| ข้อความ | `cer` | เอา Markdown syntax ออก แล้วเรียก CER เดิม; ต่ำดีกว่า อาจเกิน 1 |
| ข้อความ | `wer_whitespace` | WER เดิมสำหรับเทียบย้อนหลัง; ไม่ใช่ Thai word segmentation |
| รูปแบบดิบ | `verbatim_markdown_cer` | CER ของ source จริง ไม่ยุบ whitespace; ใช้ตรวจรูปแบบที่คะแนนข้อความละเลย แต่ syntax ที่เทียบเท่ากันก็ทำให้คะแนนลดได้ |
| หัวข้อ | `level_f1` | 2 × LCS ของลำดับระดับหัวข้อ / จำนวนหัวข้อรวมทั้งสองฝั่ง; แยกจากข้อความชื่อหัวข้อ |
| หัวข้อ | `hierarchy_f1` | LCS ของเส้นทางบรรพบุรุษ `(level, title)` แต่ละหัวข้อ; ตรวจทั้งระดับ ชื่อ parent ลำดับ และหัวข้อขาด/เกิน |
| ตาราง | `shape_similarity` | 1 − edit distance ของ token table-start/row-column-count/table-end หารด้วยความยาวฝั่งที่ยาวกว่า; ไม่มีข้อความเซลล์ในคะแนนนี้ |
| ตาราง | `cell_exact_accuracy` | เซลล์ตรงกัน / union ของพิกัด `(table,row,column)`; missing/extra cells ผิด แม้ว่าง |
| ตาราง | `cell_cer` | CER รวมที่พิกัดเซลล์เดียวกัน; จับคู่ตารางตามลำดับในหน้า โดยใช้ช่องว่าง 1 ตัวแทนพิกัดขาดหรือเซลล์ว่าง เพื่อรองรับ jiwer 3.x |
| ลำดับการอ่าน | `pair_accuracy` | สัดส่วนคู่บล็อกที่อยู่ลำดับเดียวกับ reference ในบล็อกที่จับคู่ได้ |
| ลำดับการอ่าน | `anchor_coverage_f1` | 2 × จำนวนบล็อกจับคู่ / จำนวนบล็อกรวมทั้งสองฝั่ง; penalize บล็อกหาย เกิน และซ้ำ |
| ลำดับการอ่าน | `score` | pair accuracy × anchor coverage; อ่านคู่กับจำนวน comparable pairs เสมอ |

Heading/table/order scores สูงดีกว่า; cell CER ต่ำดีกว่า

### Parsing และ normalization

ใช้ CommonMark ผ่าน markdown-it-py 3.0.0 เปิด pipe-table extension รองรับ ATX/Setext headings,
paragraph, list item, code block และ top-level pipe tables รวม escaped pipes
อ่าน inline text, code และ image alt text; ไม่ให้ emphasis/link syntax ปะปนใน text CER
แต่ละ inline block เป็นหนึ่ง reading-order block; ตารางทั้งตารางเป็นหนึ่ง block
เก็บจำนวนคอลัมน์จริงของ ragged rows เพราะ renderer ปกติอาจเติมหรือตัดเซลล์ให้เอง

NFC/CRLF normalization ใช้เฉพาะ parser ใหม่ ไม่ลบสระหรือวรรณยุกต์
ไม่ unwrap fenced response อัตโนมัติ: ถ้าโมเดลครอบเอกสารทั้งหมดใน code fence จะเสียคะแนนโครงสร้าง
HTML เป็น literal text; HTML table, rowspan/colspan, nested tables และ layout coordinate
อยู่นอก benchmark v1 (nested table ที่ parser ตรวจพบจะถูกปฏิเสธชัดเจน)
ต้องกำหนด reference ใน dialect เดียวกันก่อนเปรียบเทียบ

### ข้อจำกัดที่ต้องอ่านก่อนจัดอันดับโมเดล

- Text CER ยังไวต่อลำดับข้อความ จึงต้องอ่านแยกกับ structure/order metrics
- Heading hierarchy ใช้ชื่อ exact หลัง normalization: ชื่อผิดทำให้ hierarchy ลดด้วย ดู `level_f1` เพื่อแยก topology
- ตารางจับคู่ตาม index: ตารางหายหนึ่งตารางทำให้ตารางถัดไปเหลื่อม คะแนน cell ไม่ใช่ TEDS หรือ semantic matching
- Reading order จับคู่ exact normalized visible-text anchors เท่านั้น จึงวัดลำดับโดยแยกจาก heading level ได้ แต่ typo ทำให้ anchor หายได้
- บล็อกซ้ำจับคู่ตาม occurrence ไม่อาจระบุตัวตนจริงจากข้อความ; รายงาน `ambiguous_duplicate_blocks` เพื่อแสดงความกำกวม
- มี anchor ตรงกันน้อยกว่า 2 บล็อก: order pair accuracy/score เป็น `null` ไม่ให้เป็น 1 หรือ 0 โดยเดา
- Reference ไม่มีหัวข้อ/ตาราง และ prediction ไม่มีเช่นกัน: คะแนนกลุ่มนั้นเป็น `null`; prediction เพิ่มโครงสร้างเองจะได้ 0
- Reference text ว่าง: CER/WER เป็น `null` พร้อม character counts ไม่เอาหน้าว่างไปเพิ่มคะแนน perfect
- Summary เป็น macro average ต่อหน้า โดยไม่รวม null พร้อม `scored_pages/total_pages` ทุกค่า
  ถ้าจำนวนหน้าที่วัดได้ต่างกัน อย่าใช้ macro order score เพียงค่าเดียวตัดสินผู้ชนะ

## ใช้งาน fixtures โดยไม่รันโมเดล

จาก repository root:

```bash
python -m pip install -r requirements-eval.txt
python -m pytest -q tests/test_markdown.py tests/test_markdown_benchmark.py tests/test_metrics.py tests/test_format.py
PYTHONPATH=src python -m thaiocr.eval.markdown_benchmark \
  --references tests/fixtures/markdown/references.jsonl \
  --runs tests/fixtures/markdown/runs.json \
  --output outputs/markdown-fixture-report.json
```

Fixtures มีใบรับสินค้า (หัวข้อและตาราง), วิธีตรวจ QC (หัวข้อหลายระดับ),
รายงานสองฝ่าย (ลำดับอ่านและข้อความซ้ำ) พร้อมเลขไทยและวรรณยุกต์
ไฟล์ `perfect.jsonl` และ `damaged.jsonl` เป็นคำตอบจำลอง ไม่ใช่ output ของ Qwen หรือ Typhoon
กรณีเสียหายประกอบด้วยตัวเลขผิด แถวตารางหาย ระดับหัวข้อผิด และบล็อกสลับ
ใช้ `.md` เพื่ออ่านง่าย และ JSONL เป็น reference จริงที่ CLI ใช้
นี่คือ metric fixtures ไม่มีภาพเอกสาร จึงยังไม่ใช่ชุดทดสอบความสามารถ OCR จากภาพ

รัน tests ทั้งหมด (รวม synthetic image tests เดิม) โดยเพิ่ม CPU packages:

```bash
python -m pip install 'pillow>=10' 'numpy>=1.26'
python -m pytest -q
```

เพิ่ม GitHub Actions สำหรับคำสั่งนี้บน Python 3.10 และ 3.11; ไม่มีขั้นตอน inference/training

## เปรียบเทียบ Qwen3.5 กับ Typhoon OCR จริงภายหลัง

1. กำหนด model ID/revision ของ Qwen3.5 รุ่นที่มี vision และ Typhoon OCR รุ่นที่ต้องการให้แน่นอน
   train.sh เดิมระบุ 3B ไม่ใช่หลักฐานว่าเป็น Typhoon OCR 1.5
2. เตรียมภาพ held-out ชุดเดียวกัน พร้อม reference Markdown ที่คนตรวจแล้ว ให้แต่ละหน้าใช้ ID คงที่
   แยกจาก training data ตามเอกสารต้นฉบับ/แหล่งข้อมูล ป้องกันหน้าใกล้เคียงรั่วข้าม split
3. Reference JSONL: `{"id":"page-001","markdown":"# รายงาน\n..."}`
   เพิ่ม `image_sha256`, `document_id`, `source` ใน record เพื่อบันทึก provenance ได้;
   evaluator จะ hash manifest ทั้งไฟล์ แต่ยังไม่ตรวจไฟล์ภาพเอง
4. เก็บ output แต่ละโมเดลเป็น JSONL: `{"id":"page-001","response":"# รายงาน\n..."}`
   output ว่างใช้ empty string; หน้า missing, unknown, ID ซ้ำ หรือ response ไม่ใช่ string จะ error
   ไม่อนุญาตประเมินเฉพาะหน้าที่โมเดลตอบสำเร็จโดยเงียบ ๆ
5. กำหนด runs JSON ตามตัวอย่างด้านล่าง เก็บ prompt และ decoding จริง รวม preprocessing,
   image resolution, max tokens, truncation/retry policy ไว้ใน metadata เพิ่มเติม
   ถ้าต้องใช้ native prompt ต่างกัน ให้ระบุว่าเป็นการเทียบระบบพร้อม prompt ไม่ใช่ควบคุม prompt เท่ากัน
6. รัน CLI เดิมด้วย references/runs ของจริง รายงานมี per-page scores, macro scores,
   model metadata, reference/prediction SHA256, metric version และ dependency versions
7. ตรวจหน้าที่เสียคะแนนด้วยคน แยกประเภท heading/table/multi-column/printed/handwritten
   ก่อนสรุป; fixtures ในแพตช์นี้ยังไม่ให้ผลจัดอันดับโมเดลจริง

ตัวอย่าง runs (แทนค่าที่ขึ้นต้น `REPLACE_` ก่อนใช้):

```json
[
  {
    "name": "qwen35",
    "model": "REPLACE_EXACT_QWEN_VISION_MODEL_ID",
    "revision": "REPLACE_COMMIT_OR_VERSION",
    "prompt": "ถอดข้อความในภาพเป็น Markdown โดยคงหัวข้อ ตาราง และลำดับการอ่าน",
    "decoding": {"temperature": 0, "max_new_tokens": 8192},
    "predictions": "qwen35.jsonl"
  },
  {
    "name": "typhoon",
    "model": "REPLACE_EXACT_TYPHOON_MODEL_ID",
    "revision": "REPLACE_COMMIT_OR_VERSION",
    "prompt": "REPLACE_ACTUAL_PROMPT",
    "decoding": {"temperature": 0, "max_new_tokens": 8192},
    "predictions": "typhoon.jsonl"
  }
]
```

`predictions` paths อ้างอิงจากโฟลเดอร์ของ runs.json ไม่ใช่ current working directory
การเปรียบเทียบนี้ไม่ได้ผูกกับ inference engine จึงใช้ saved output จากระบบใดก็ได้
