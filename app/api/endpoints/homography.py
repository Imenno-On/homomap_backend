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

    # Сохраняем загруженное видео локально, чтобы выдавать его позже и для удобства
    uploads_dir = "uploaded_videos"
    try:
        os.makedirs(uploads_dir, exist_ok=True)
    except Exception:
        pass

    # httpx.AsyncClient используется для асинхронных запросов, таймаут можно задавать через ML_TIMEOUT
    timeout_seconds = float(os.getenv("ML_TIMEOUT", "900"))
    # Увеличиваем таймаут, логируем начало запроса
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        try:
            # Пересылаем файл в ML-сервис (важно: передавать его как 'files')
            # Сохраняем файл на диск
            saved_video_filename = f"{current_user.id}_{os.path.basename(video_file.filename)}"
            saved_video_path = os.path.join(uploads_dir, saved_video_filename)
            # Переходим в начало файла и записываем содержимое
            try:
                video_file.file.seek(0)
            except Exception:
                pass
            with open(saved_video_path, 'wb') as out_f:
                out_f.write(await video_file.read())

            f = open(saved_video_path, 'rb')
            try:
                files = {'video_file': (video_file.filename, f, video_file.content_type)}

                logger.info(f"Sending file '{video_file.filename}' to ML-service: {ml_endpoint}")

                response = await client.post(ml_endpoint, files=files)
            finally:
                try:
                    f.close()
                except Exception:
                    pass
            # Log elapsed time if available
            try:
                elapsed = getattr(response, 'elapsed', None)
                if elapsed is not None:
                    logger.info(f"ML-service responded in {elapsed.total_seconds():.2f}s, status={response.status_code}")
                else:
                    logger.info(f"ML-service responded with status={response.status_code}")
            except Exception:
                logger.info(f"ML-service responded with status={response.status_code}")

            response.raise_for_status()  # Вызывает ошибку для 4xx/5xx статусов

            # Получение и обработка результата
            ml_result_data = response.json()
            
            # Логируем данные от ML-сервиса для отладки
            logger.info(f"ML service response keys: {list(ml_result_data.keys())}")
            logger.info(f"Trajectory points count: {len(ml_result_data.get('trajectory_points', []))}")
            logger.info(f"Step peaks count: {len(ml_result_data.get('step_peaks', []))}")
            logger.info(f"Has scale_info: {ml_result_data.get('scale_info') is not None}")
            logger.info(f"Has room_dimensions: {ml_result_data.get('room_dimensions') is not None}")
            if ml_result_data.get('scale_info'):
                logger.info(f"Scale info: {ml_result_data.get('scale_info')}")
            if ml_result_data.get('trajectory_points') and len(ml_result_data.get('trajectory_points', [])) > 0:
                logger.info(f"First trajectory point: {ml_result_data.get('trajectory_points')[0]}")

            # Если ML вернул preview (base64), декодируем и сохраняем его
            preview_b64 = ml_result_data.get("preview_image")
            preview_path = None
            if preview_b64:
                try:
                    import base64
                    previews_dir = "previews"
                    os.makedirs(previews_dir, exist_ok=True)
                    safe_name = f"{current_user.id}_{os.path.basename(video_file.filename)}"
                    # расширение .jpg
                    preview_filename = f"{os.path.splitext(safe_name)[0]}_preview.jpg"
                    preview_full_path = os.path.join(previews_dir, preview_filename)
                    with open(preview_full_path, 'wb') as pf:
                        pf.write(base64.b64decode(preview_b64))

                    # Сохраняем относительный путь, который frontend умеет преобразовывать
                    preview_path = f"/previews/{preview_filename}"
                except Exception as e:
                    logger.warning(f"Failed to save preview image: {e}")

            # Используем имя файла как заголовок проекта
            project_title = video_file.filename
            video_path = f"/videos/{saved_video_filename}"

            # Сохраняем проект в БД с preview_path (если есть) и всеми новыми полями
            project = crud.create_project(
                db=db,
                user_id=current_user.id,
                title=project_title,
                video_path=video_path,
                homography_matrix=ml_result_data.get("homography_matrix"),
                scaled_homography_matrix=ml_result_data.get("scaled_homography_matrix"),
                floor_polygons=ml_result_data.get("floor_polygons"),
                wall_polygons=ml_result_data.get("wall_polygons"),
                trajectory_points=ml_result_data.get("trajectory_points"),
                step_peaks=ml_result_data.get("step_peaks"),
                scale_info=ml_result_data.get("scale_info"),
                room_dimensions=ml_result_data.get("room_dimensions"),
                processing_time=ml_result_data.get("processing_time"),
                preview_image=ml_result_data.get("preview_image"),  # Сохраняем base64 для быстрого доступа
                preview_path=preview_path
            )

            logger.info(f"Project created with ID: {project.id}")
            
            # Логируем, что сохранили в БД
            logger.info(f"Saved to DB - trajectory_points: {len(project.trajectory_points) if project.trajectory_points else 0} points")
            logger.info(f"Saved to DB - step_peaks: {len(project.step_peaks) if project.step_peaks else 0} peaks")
            logger.info(f"Saved to DB - scale_info: {project.scale_info is not None}")
            logger.info(f"Saved to DB - room_dimensions: {project.room_dimensions is not None}")

            # Добавим project_id к результату для удобства фронтенда
            ml_result_data["project_id"] = project.id

            # Формируем ответ (валидируем поля, возвращаем HomographyResult)
            homography_result = HomographyResult(**ml_result_data)
            return homography_result

        except httpx.HTTPStatusError as e:
            # Ошибка от ML-сервиса (4xx/5xx)
            logger.error(f"ML service error (status {e.response.status_code}): {e.response.text}")
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка обработки видео ML-сервисом: {e.response.text}"
            )
        except httpx.ReadTimeout as e:
            logger.error(f"Timeout while waiting for ML service after {timeout_seconds}s: {str(e)}")
            raise HTTPException(status_code=504, detail=f"ML service timeout after {timeout_seconds} seconds")
        except httpx.ConnectError:
            # ML-сервис недоступен
            logger.error("ML service is unavailable")
            raise HTTPException(
                status_code=503,
                detail="ML-сервис недоступен. Проверьте, запущен ли контейнер 'ml'."
            )
        except Exception as e:
            logger.exception("Unexpected error while processing video")
            raise HTTPException(status_code=500, detail=f"Неожиданная ошибка: {repr(e)}")