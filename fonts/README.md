# Fonts — ลายมือไทยสำหรับ synthetic data

วางไฟล์ `.ttf` / `.otf` ในโฟลเดอร์นี้ แล้ว `synthetic.py` จะสุ่มใช้

## โหลดอัตโนมัติ (OFL — ปลอดภัยสุด, 12 ฟอนต์)
```bash
python scripts/fetch_fonts.py
```
ดึงฟอนต์ไทยสไตล์ลายมือ/กึ่งลายมือ license **OFL** จาก Google Fonts (Cadson Demak):
Itim, Charmonman, Sriracha, Mali (3 น้ำหนัก), Charm, **Playpen Sans Thai**, Kodchasan —
ใช้เชิงพาณิชย์ได้ แก้/ฝังได้ (แนบ OFL notice)

## เพิ่มลายมือ "จริง/เละ" กว่า — f0nt.com (ตรวจ license แล้ว ✅)

ฟอนต์ OFL ข้างบนยังค่อนข้างเรียบ (display) — เพิ่มฟอนต์ลายมือจริงเพื่อให้โมเดลทน
ตัวด้านล่างอยู่ในหมวด **"ใช้ทำมาหากินได้"** ของ f0nt.com (ผู้สร้างอนุญาตเชิงพาณิชย์):

| ฟอนต์ | สไตล์ | โหลด |
|---|---|---|
| **SOV_HuaHyai (หัวใหญ่ๆ)** | ลายมือหัวโตน่ารัก | f0nt.com (manual) |
| **Torsilp เทพนิยาย** | ลายมือ | f0nt.com (manual) |
| **Torsilp คิดถึงเธอ** | ลายมือ | f0nt.com (manual) |

- หมวดเต็ม: https://www.f0nt.com/type/commercial/
- ⚠️ ดาวน์โหลด **manual** (f0nt ไม่มี direct link เสถียร) → โหลดไฟล์มาวางใน `fonts/` เอง
- ⚠️ **ตรวจหน้าฟอนต์แต่ละตัวอีกครั้ง** ก่อนใช้พาณิชย์ — บางตัวมีเงื่อนไขเฉพาะ (เก็บไฟล์ license ไว้ด้วย)

## บัลก์เพิ่มเติม (open-source)
- **jeffmcneill/thai-font-collection** (Codeberg): 300+ ฟอนต์ free/OSS, มี .ttf/.otf ตรงๆ
  `git clone https://codeberg.org/jeffmcneill/thai-font-collection` แล้วคัดตัวลายมือมาใส่ `fonts/`
- **SIPA 13 ฟอนต์ราชการ** มี TH Charmonman (ลายมือ) — แต่ ≈ Charmonman ที่โหลดอัตโนมัติแล้ว

> เป้าหมาย: ยิ่งฟอนต์ลายมือหลากหลาย (10–30+ แบบ) synthetic ยิ่งช่วยให้โมเดล generalize
> ⚠️ ต้องมี Pillow ที่ build รองรับ **raqm** ไม่งั้นสระ/วรรณยุกต์ไทยวางผิด (ดู README หลัก)
