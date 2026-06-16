# ติดตั้ง Thai OCR API บน VPS

คู่มือติดตั้งตัว frontend API (`cloud/vps/`) บน VPS ตั้งแต่เครื่องเปล่า ตัว API เป็น CPU ล้วน
ทำหน้าที่ proxy ไป RunPod serverless (GPU) — VPS ไม่ต้องมีการ์ดจอ ต้องการแค่ต่อ internet ออกไป
`api.runpod.ai` ได้ (รายละเอียด endpoint/usage ดูที่ `README.md`)

ภาพรวม: `client -> VPS API (CPU) -> RunPod serverless (GPU) -> ข้อความไทย`

## สิ่งที่ต้องเตรียม
- VPS Linux (แนะนำ Ubuntu 22.04/24.04), RAM 1GB+ ก็พอ
- RunPod **Endpoint ID** = `wtoixvuklw37td` และ **API key** (`rpa_...`)
- ตั้ง **API_KEY** ของตัวเอง (string สุ่มยาวๆ) ไว้ให้ client ส่งมาทาง header `X-API-Key`
  - สุ่มได้ด้วย: `openssl rand -hex 24`
- (ถ้าจะเปิด public ด้วย https) โดเมนที่ชี้ A record มาที่ IP ของ VPS

---

## วิธี A — Docker (แนะนำ)

### 1. ลง Docker (ครั้งเดียว)
```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker   # ใช้ docker โดยไม่ต้อง sudo
```

### 2. clone repo + เข้าโฟลเดอร์
```bash
git clone git@github.com:xerenon/thodsan-ocr.git
cd thodsan-ocr/cloud/vps
```
> ถ้า VPS ยังไม่มี SSH key กับ GitHub ใช้ https แทน: `git clone https://github.com/xerenon/thodsan-ocr.git`

### 3. ตั้งค่า .env
```bash
cp .env.example .env
nano .env      # แก้ RUNPOD_API_KEY และตั้ง API_KEY เป็น string สุ่ม
```

### 4. รัน
```bash
docker compose up -d --build      # เปิดที่พอร์ต 8000
docker compose logs -f            # ดู log (Ctrl+C ออก ไม่ได้หยุด container)
```

### 5. ทดสอบ
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ocr -H "X-API-Key: <API_KEY ของคุณ>" -F "file=@/path/to/page.jpg"
```

---

## วิธี B — systemd (ไม่ใช้ Docker)

```bash
# 1. ลง python + git
sudo apt update && sudo apt install -y python3.11 python3.11-venv git

# 2. วางโค้ดที่ /opt/thaiocr-api
sudo git clone https://github.com/xerenon/thodsan-ocr.git /tmp/ocr
sudo mkdir -p /opt/thaiocr-api
sudo cp /tmp/ocr/cloud/vps/app.py /tmp/ocr/cloud/vps/requirements.txt /opt/thaiocr-api/
cd /opt/thaiocr-api

# 3. venv + deps
sudo python3.11 -m venv .venv
sudo .venv/bin/pip install -r requirements.txt

# 4. ตั้ง .env
sudo cp /tmp/ocr/cloud/vps/.env.example .env
sudo nano .env      # แก้ RUNPOD_API_KEY + API_KEY

# 5. ติดตั้ง service
sudo cp /tmp/ocr/cloud/vps/thaiocr-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now thaiocr-api
sudo systemctl status thaiocr-api        # เช็คว่ารันอยู่
journalctl -u thaiocr-api -f             # ดู log
```

---

## เปิด firewall
ถ้าต่อพอร์ต 8000 ตรงๆ:
```bash
sudo ufw allow 8000/tcp
```
แต่แนะนำให้ซ่อนหลัง nginx + https (ข้างล่าง) แล้วเปิดแค่ 80/443 แทน

---

## (แนะนำ) nginx + HTTPS

```bash
sudo apt install -y nginx
sudo tee /etc/nginx/sites-available/thaiocr <<'EOF'
server {
    listen 80;
    server_name ocr.example.com;          # << เปลี่ยนเป็นโดเมนคุณ
    client_max_body_size 25M;             # เผื่อรูปหน้าใหญ่
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 360s;          # เผื่อ cold start / หน้ายาว
    }
}
EOF
sudo ln -s /etc/nginx/sites-available/thaiocr /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo ufw allow 'Nginx Full'

# ออก TLS ฟรีด้วย Let's Encrypt
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d ocr.example.com
```
จากนั้นเรียกผ่าน `https://ocr.example.com/ocr` ได้เลย (ไม่ต้องเปิดพอร์ต 8000 สู่ public)

---

## ดูแลรักษา

| งาน | Docker | systemd |
|-----|--------|---------|
| ดู log | `docker compose logs -f` | `journalctl -u thaiocr-api -f` |
| restart | `docker compose restart` | `sudo systemctl restart thaiocr-api` |
| หยุด | `docker compose down` | `sudo systemctl stop thaiocr-api` |
| อัปเดตโค้ด | `git pull && docker compose up -d --build` | `git pull` แล้ว copy `app.py` ใหม่ + `systemctl restart` |

อัปเดต/หมุน key: แก้ `.env` แล้ว restart ตามตารางข้างบน

---

## เช็กลิสต์ความปลอดภัย
- ตั้ง `API_KEY` ใน `.env` เสมอ — ไม่งั้นใครก็ยิงได้ = เปลือง GPU
- อย่า commit `.env` (มี `.gitignore` กันไว้ให้แล้ว)
- เปิด public ให้ผ่าน https (nginx + certbot) อย่าเปิดพอร์ต 8000 ตรงๆ
- RunPod API key เก็บไว้แค่ใน `.env` บน VPS เท่านั้น
- จะจำกัด origin ที่เรียกได้ก็ตั้ง `CORS_ORIGINS` ใน `.env` (default `*`)
