# Используем образ Python, как предложено в homomap.docx
FROM python:3.10-slim

# Установка необходимых системных пакетов
# (Необходимо для ML-части и других библиотек, которые могут быть скомпилированы)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Установка зависимостей
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование остального кода
COPY . .

# Порт для FastAPI
EXPOSE 8000

# Команда запуска будет в docker-compose.yml
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]