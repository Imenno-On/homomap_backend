import os
import tempfile
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from homography import Homography
import cv2
import numpy as np

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ml_service")

app = FastAPI(title="Homomap ML Service", version="1.0.0")

# Инициализация ML-модели при запуске
logger.info("Loading ML models...")
ml_processor = Homography()
logger.info("ML models loaded successfully")


@app.get("/health")
def health_check():
    """Проверка работоспособности сервиса"""
    return {"status": "healthy", "message": "ML service is running"}


@app.post("/process_video")
async def process_video(video_file: UploadFile = File(...)):
    """
    Эндпоинт для обработки видео и получения матрицы гомографии
    """
    logger.info(f"Received video file: {video_file.filename}")

    # Проверка типа файла
    if not video_file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        logger.warning(f"Invalid file format: {video_file.filename}")
        raise HTTPException(status_code=400, detail="Invalid video file format. Supported formats: mp4, avi, mov, mkv")

    temp_video_path = None
    temp_img_path = None

    try:
        # Сохранение видео во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.filename)[1]) as temp_file:
            temp_video_path = temp_file.name
            content = await video_file.read()
            temp_file.write(content)

        logger.info(f"Video saved to temporary file: {temp_video_path}")

        # Открытие видео для получения первого кадра
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            logger.error("Failed to open video file")
            raise HTTPException(status_code=500, detail="Failed to open video file")

        # Чтение первого кадра
        ret, frame = cap.read()
        cap.release()

        if not ret:
            logger.error("Failed to read frame from video")
            raise HTTPException(status_code=500, detail="Failed to read frame from video")

        logger.info("First frame extracted successfully")

        # Создание временного файла для изображения
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_img_file:
            temp_img_path = temp_img_file.name
            # Конвертация BGR (OpenCV) в RGB и сохранение
            cv2.imwrite(temp_img_path, frame)

        logger.info(f"First frame saved to: {temp_img_path}")

        # Обработка изображения с помощью Homography
        try:
            logger.info("Starting homography computation...")
            H, floor_polygons, wall_polygons = ml_processor.compute_homography(
                temp_img_path,
                ml_processor.processor_seg,
                ml_processor.model_seg,
                ml_processor.processor_depth,
                ml_processor.model_depth
            )
            logger.info("Homography computation completed successfully")
        except Exception as e:
            logger.error(f"Error during homography computation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"ML processing error: {str(e)}")

        # Подготовка результата
        result = {
            "homography_matrix": H.tolist(),
            "floor_polygons": floor_polygons,
            "wall_polygons": wall_polygons,
            "trajectory_points": []  # Временно пусто, нужно добавить логику трекинга
        }

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # Очистка временных файлов
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
            logger.info(f"Temporary video file cleaned up: {temp_video_path}")
        if temp_img_path and os.path.exists(temp_img_path):
            os.unlink(temp_img_path)
            logger.info(f"Temporary image file cleaned up: {temp_img_path}")