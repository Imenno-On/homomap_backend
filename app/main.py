from fastapi import FastAPI
from app.db.database import engine, Base
from app.db import models  # Важно импортировать models, чтобы Base "узнал" о них
from app.db.migrations import run_migrations
from contextlib import asynccontextmanager
import uvicorn
import os
import logging
from app.api.endpoints import users, homography, projects
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Автоматическое создание папки для загружаемых файлов
UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)
# Создание папки для превью изображений
PREVIEWS_DIR = "previews"
os.makedirs(PREVIEWS_DIR, exist_ok=True)

# 1. Lifespan context manager для запуска и остановки приложения
@asynccontextmanager
async def lifespan(app: FastAPI):
    # При старте: Создание таблиц в БД
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully or already exist.")
    # Выполняем автоматическую миграцию для добавления новых колонок
    run_migrations()
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

# Статические маршруты для доступа к видео и превью
app.mount("/api/v1/videos", StaticFiles(directory=UPLOAD_DIR), name="videos")
app.mount("/api/v1/previews", StaticFiles(directory=PREVIEWS_DIR), name="previews")

# CORS (разрешаем запросы от фронтенда при разработке)
_origins_env = os.getenv("FRONTEND_ORIGINS")
if _origins_env:
    if _origins_env.strip() == "*":
        _allow_origins = ["*"]
        _allow_credentials = False
    else:
        _allow_origins = [o.strip() for o in _origins_env.split(',') if o.strip()]
        _allow_credentials = True
else:
    # По-умолчанию разрешаем локальный Vite / CRA dev server
    _allow_origins = ["http://localhost:5173", "http://localhost:3000"]
    _allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Homomap API is running"}

# Для локального запуска (если не через uvicorn)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)