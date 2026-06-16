"""ดาวน์โหลดฟอนต์ลายมือไทย license OFL (ปลอดภัย ใช้เชิงพาณิชย์ได้) จาก google/fonts → fonts/

OFL = แก้ไข/ฝัง/ใช้เชิงพาณิชย์ได้ ขอแค่แนบ OFL notice ไปกับไฟล์
ฟอนต์กลุ่มนี้เป็น hand-drawn/handwriting ของ Cadson Demak — baseline ที่ปลอดภัย+หลากหลาย
อยากได้ลายมือ "เละ/จริง" กว่านี้ → ดู fonts/README.md (f0nt.com ฯลฯ ตรวจ license ต่อฟอนต์)
"""
import os
import urllib.request

BASE = "https://raw.githubusercontent.com/google/fonts/main/ofl"

# ฟอนต์ไทยสไตล์ลายมือ/เขียนมือ บน Google Fonts (ทั้งหมด OFL)
FONTS = [
    "itim/Itim-Regular.ttf",            # hand-drawn กลม อบอุ่น
    "charmonman/Charmonman-Regular.ttf",  # ลายมือหวัด (cursive)
    "charmonman/Charmonman-Bold.ttf",
    "sriracha/Sriracha-Regular.ttf",    # ลายมือพู่กัน/มาร์กเกอร์
    "mali/Mali-Regular.ttf",            # ลายมือ
    "mali/Mali-Medium.ttf",
    "mali/Mali-Light.ttf",
    "charm/Charm-Regular.ttf",          # ลายมือเอียง
    "charm/Charm-Bold.ttf",
    "playpensansthai/PlaypenSansThai[wght].ttf",  # ลายมือเด็ก/casual (variable)
    "kodchasan/Kodchasan-Regular.ttf",  # กึ่งลายมือ กลมๆ
    "kodchasan/Kodchasan-Medium.ttf",
]


def main(out_dir: str = "fonts"):
    os.makedirs(out_dir, exist_ok=True)
    ok = 0
    for rel in FONTS:
        name = rel.split("/")[-1]
        dst = os.path.join(out_dir, name)
        try:
            urllib.request.urlretrieve(f"{BASE}/{rel}", dst)
            print(f"  ✅ {name} ({os.path.getsize(dst) // 1024} KB)")
            ok += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
    print(f"\nได้ {ok}/{len(FONTS)} ฟอนต์ -> {out_dir}/")
    print("เพิ่มความหลากหลาย: โหลดฟอนต์ลายมือจาก f0nt.com ใส่ใน fonts/ (ดู fonts/README.md)")


if __name__ == "__main__":
    main()
