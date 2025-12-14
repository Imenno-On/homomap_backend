from fastapi import FastAPI
from app.db.database import engine, Base
from app.db import models  # Важно импортировать models, чтобы Base "узнал" о них
from contextlib import asynccontextmanager
import uvicorn
import os
import logging
from app.api.endpoints import users, homography, projects

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Автоматическое создание папки для загружаемых файлов
UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 1. Lifespan context manager для запуска и остановки приложения
@asynccontextmanager
async def lifespan(app: FastAPI):
    # При старте: Создание таблиц в БД
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully or already exist.")
    yield
    # При остановке: можно добавить логику очистки, если нужно

# 2. Инициализация FastAPI приложения
app = FastAPI(
    title="Homomap Backend API",
    description="API для проекта 'Составление карты гомографии помещения'.",
    version="1.0.0",
    lifespan=lifespan
)

# 3. Добавление маршрутов API
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(homography.router, prefix="/api/v1/homography", tags=["homography"])
app.include_router(projects.router, prefix="/api/v1/projects", tags=["projects"])

@app.get("/")
def read_root():
    return {"message": "Homomap API is running"}

# Для локального запуска (если не через uvicorn)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)