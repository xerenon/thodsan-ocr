# Thai OCR VPS frontend — CPU-only, proxies to the RunPod GPU endpoint.
# Build (context = this dir):  docker build -t thaiocr-api cloud/vps
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
