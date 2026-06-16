# Fonts — ลายมือไทยสำหรับ synthetic data

วางไฟล์ `.ttf` / `.otf` ในโฟลเดอร์นี้ แล้ว `synthetic.py` จะสุ่มใช้

## โหลดอัตโนมัติ (OFL — ปลอดภัยสุด)
```bash
python scripts/fetch_fonts.py
```
ดึงฟอนต์ไทยสไตล์ลายมือ license **OFL** จาก Google Fonts (Cadson Demak):
Itim, Charmonman, Sriracha, Mali, Charm — ใช้เชิงพาณิชย์ได้ แก้/ฝังได้ (แนบ OFL notice)

## เพิ่มความหลากหลาย (ลายมือ "จริง/เละ" กว่า)
ฟอนต์ OFL ข้างบนค่อนข้างเรียบ (display) — อยากให้โมเดลทนลายมือจริง ควรเพิ่มฟอนต์ลายมือหลายๆ แบบ:

- **f0nt.com** หมวดใช้เชิงพาณิชย์ได้: https://www.f0nt.com/type/commercial/
  (เช่น Hauhyak, Noo Ra Rada, TKCantaloupe — ⚠️ **ตรวจ license ต่อฟอนต์** บางตัวเป็น donation-ware/ห้ามพาณิชย์)
- **freethaifont.com** หมวด Thai Handwriting
- **jeffmcneill/thai-font-collection** (GitHub) — รวมฟอนต์ไทยฟรี/โอเพนซอร์ส

> เป้าหมาย: ยิ่งฟอนต์ลายมือหลากหลาย (10–30+ แบบ) synthetic ยิ่งช่วยให้โมเดล generalize
> ⚠️ ต้องมี Pillow ที่ build รองรับ **raqm** ไม่งั้นสระ/วรรณยุกต์ไทยวางผิด (ดู README หลัก)
