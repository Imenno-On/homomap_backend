import os
import httpx
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.api.dependencies import get_current_user
from app.db import crud
from app.schemas.project import HomographyResult
from app.db.models import User
import logging

# Настройка логирования
logger = logging.getLogger(__name__)

# Получаем URL ML-сервиса из переменных окружения
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml:8000")
router = APIRouter()


@router.post("/process_video", response_model=HomographyResult, status_code=status.HTTP_201_CREATED)
async def process_video_on_ml(
        video_file: UploadFile = File(..., description="Видеофайл для анализа"),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    Эндпоинт, который принимает видео от фронтенда и отправляет его в ML-сервис.
    """
    ml_endpoint = f"{ML_SERVICE_URL}/process_video"

    logger.info(f"Processing video: {video_file.filename} for user ID: {current_user.id}")

    # httpx.AsyncClient используется для асинхронных запросов, таймаут увеличен для ML-задач
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            # Пересылаем файл в ML-сервис (важно: передавать его как 'files')
            files = {'video_file': (video_file.filename, video_file.file, video_file.content_type)}

            logger.info(f"Sending file '{video_file.filename}' to ML-service: {ml_endpoint}")

            response = await client.post(ml_endpoint, files=files)
            response.raise_for_status()  # Вызывает ошибку для 4xx/5xx статусов

            # Получение и обработка результата
            ml_result_data = response.json()
            homography_result = HomographyResult(**ml_result_data)  # Проверяем данные по схеме

            # Используем имя файла как заголовок проекта
            project_title = video_file.filename
            video_path = f"/videos/{current_user.id}_{video_file.filename}"
            preview_path = None

            # Сохраняем проект в БД
            project = crud.create_project(
                db=db,
                user_id=current_user.id,
                title=project_title,
                video_path=video_path,
                homography_matrix=ml_result_data.get("homography_matrix"),
                trajectory_points=ml_result_data.get("trajectory_points"),
                preview_path=preview_path
            )

            logger.info(f"Project created with ID: {project.id}")
            return homography_result

        except httpx.HTTPStatusError as e:
            # Ошибка от ML-сервиса
            logger.error(f"ML service error: {e.response.text}")
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка обработки видео ML-сервисом: {e.response.text}"
            )
        except httpx.ConnectError:
            # ML-сервис недоступен
            logger.error("ML service is unavailable")
            raise HTTPException(
                status_code=503,
                detail="ML-сервис недоступен. Проверьте, запущен ли контейнер 'ml'."
            )
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Неожиданная ошибка: {str(e)}")