# Используем более полный образ для ML
FROM python:3.10-slim

# Установка необходимых системных пакетов для CV (OpenCV, FFmpeg, Render)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 libxext6 libxrender-dev \
    libgl1 libglib2.0-0 \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем и устанавливаем ML-зависимости
COPY requirements_ml.txt .
RUN pip install --no-cache-dir -r requirements_ml.txt

# Копирование кода ML-сервиса
# Предполагаем, что ML-код находится в папке ml_service
COPY ml_service/ /app/

# Порт для связи с Backend (web-сервисом)
EXPOSE 8000

# Команда запуска ML-FastAPI, если она есть
CMD ["sh", "-lc", "uvicorn main:app --host 0.0.0.0 --port 8000 --workers ${ML_WORKERS:-1}"]