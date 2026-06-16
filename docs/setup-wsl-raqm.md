# Setup: WSL2 + raqm-Pillow (แก้บล็อกเกอร์ synthetic)

เป้าหมาย: ได้ Pillow ที่มี **raqm** เพื่อเรนเดอร์สระ/วรรณยุกต์ไทยให้ถูก
สถานะเครื่อง (เช็คแล้ว): WSL2 Ubuntu, Python 3.12, **ยังไม่มี pip/raqm**, GPU เห็นใน WSL ✅

> ข่าวดี: GPU เห็นใน WSL → ทำได้ทั้ง generate synthetic **และเทรน** ใน WSL2 ที่เดียวจบ
> (ไม่ต้องสลับไป Windows) แต่ถ้าจะเทรนบน Windows เหมือนเดิมก็ได้ — ใช้ WSL แค่ generate

## 1) เข้า WSL2 + ไปที่โปรเจกต์
```bash
wsl -d Ubuntu
cd /mnt/c/Work/orc-finetune-handwrite
```

## 2) ลง system libs (รวม libraqm) + เครื่องมือ build
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv python3-dev \
    libjpeg-dev zlib1g-dev libfreetype-dev \
    libraqm-dev libharfbuzz-dev libfribidi-dev
```

## 3) สร้าง venv + build Pillow จาก source (ให้ link raqm)
```bash
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
pip install --upgrade pip

# ลง deps อื่นก่อน (จะได้ pillow แบบ wheel ไม่มี raqm)
pip install numpy datasets jiwer pandas

# ⚠️ สำคัญ: rebuild pillow จาก source "เป็นขั้นสุดท้าย" เพื่อให้มี raqm
#    (ถ้าลง pillow/อย่างอื่นทับทีหลัง raqm จะหาย ต้อง build ใหม่)
pip install --force-reinstall --no-binary :all: pillow
```

## 4) ยืนยัน raqm = True (ด่านชี้ขาด)
```bash
python -c "from PIL import features; print('raqm:', features.check('raqm'))"
```
ต้องได้ `raqm: True` — ถ้ายัง `False` แปลว่า build ไม่เจอ libraqm:
ตรวจว่า `sudo apt install libraqm-dev` ผ่าน แล้ว `pip install --force-reinstall --no-binary :all: pillow` ใหม่

## 5) ปั๊ม synthetic ได้เลย
```bash
python scripts/fetch_fonts.py                 # ฟอนต์ OFL 12 ตัว
python scripts/fetch_corpus.py --n 200000     # ข้อความไทย (ต้องมี datasets + เน็ต)
python -m thaiocr.data.synthetic              # → data/swift/synth.jsonl + images
```

> 💡 หมายเหตุความเร็ว: เขียนภาพจำนวนมากลง `/mnt/c` (ดิสก์ Windows) **ช้า**
> ถ้าจะ generate เป็นแสนภาพ ให้แก้ path output ไปที่ home ของ WSL (เช่น `~/synth`) จะเร็วกว่ามาก แล้วค่อย copy

## (ถ้าจะเทรนใน WSL2 ด้วย)
GPU ใช้ได้ใน WSL → ลง torch cu128 ใน venv เดียวกันแล้วเทรนได้เลย:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# ⚠️ requirements มี pillow → จะลงทับ! หลังลงเสร็จ rebuild raqm อีกที:
pip install --force-reinstall --no-binary :all: pillow
python scripts/check_env.py                    # ยืนยัน GPU+bnb ใน WSL
```
