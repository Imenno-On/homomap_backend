import os
import tempfile
import logging
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from homography import Homography, StepAnalyzer
import asyncio
import traceback
import cv2
import numpy as np

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ml_service")

app = FastAPI(title="Homomap ML Service", version="1.0.0")

# Инициализация ML-моделей при запуске
logger.info("Loading ML models...")
ml_processor = Homography()
step_analyzer = StepAnalyzer(model_path='yolov8n-pose.pt')  # Инициализация анализатора шагов
logger.info("ML models loaded successfully")


@app.get("/health")
def health_check():
    """Проверка работоспособности сервиса"""
    return {"status": "healthy", "message": "ML service is running"}


# Старый обработчик удален - используем новый с анализом шагов (ниже)

@app.post("/process_video")
async def process_video(video_file: UploadFile = File(...)):
    """
    Эндпоинт для обработки видео и получения матрицы гомографии с калибровкой по шагам
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

        # === ЧАСТЬ 1: Обработка первого кадра для матрицы гомографии ===
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
            cv2.imwrite(temp_img_path, frame)

        logger.info(f"First frame saved to: {temp_img_path}")

        # Обработка первого кадра для получения матрицы гомографии
        try:
            logger.info("Starting homography computation in background thread...")
            t0 = time.time()

            H, floor_polygons, wall_polygons = await asyncio.to_thread(
                ml_processor.compute_homography,
                temp_img_path,
                ml_processor.processor_seg,
                ml_processor.model_seg,
                ml_processor.processor_depth,
                ml_processor.model_depth
            )

            homography_time = time.time() - t0
            logger.info(f"Homography computation completed successfully in {homography_time:.2f}s")
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Error during homography computation: {e}\n{tb}")
            raise HTTPException(status_code=500, detail=f"ML processing error: {str(e)}")

        # === ЧАСТЬ 2: Анализ шагов на всем видео ===
        try:
            logger.info("Starting step analysis...")
            t0 = time.time()

            # Анализируем видео для определения шагов
            step_analysis = await asyncio.to_thread(
                step_analyzer.analyze_video_steps,
                temp_video_path,
                max_frames=300  # Ограничиваем количество кадров для скорости
            )

            # Вычисляем масштабный коэффициент
            scale_info = step_analyzer.calculate_scale_factor()

            step_analysis_time = time.time() - t0
            logger.info(f"Step analysis completed in {step_analysis_time:.2f}s")
            logger.info(f"Found {len(step_analysis['peak_distances'])} step peaks")

            if scale_info:
                logger.info(f"Scale factor calculated: {scale_info['scale_factor']:.4f} cm/px")
            else:
                logger.warning("Could not calculate scale factor - not enough step peaks detected")
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Error during step analysis: {e}\n{tb}")
            scale_info = None
            step_analysis = {'trajectory_points': [], 'peak_distances': []}

        # === ЧАСТЬ 3: Комбинирование результатов ===
        # Если есть информация о масштабе, применяем ее к матрице гомографии
        scaled_homography = H.copy()
        room_dimensions = None

        if scale_info and scale_info['scale_factor']:
            # scale_info['scale_factor'] уже в см/пиксель
            scale_factor_cm_per_px = scale_info['scale_factor']  # см/пиксель
            scale_factor_m_per_px = scale_factor_cm_per_px / 100.0  # метры/пиксель

            # Масштабируем матрицу гомографии для получения координат в метрах
            # Масштабируем только масштабные коэффициенты матрицы
            scaled_homography[0, 0] *= scale_factor_m_per_px
            scaled_homography[0, 1] *= scale_factor_m_per_px
            scaled_homography[1, 0] *= scale_factor_m_per_px
            scaled_homography[1, 1] *= scale_factor_m_per_px
            scaled_homography[0, 2] *= scale_factor_m_per_px
            scaled_homography[1, 2] *= scale_factor_m_per_px

            # Рассчитываем примерные размеры комнаты
            # Используем исходную матрицу H и масштаб, так как масштабированная матрица уже в метрах
            room_dimensions = calculate_room_dimensions(
                floor_polygons,
                H,  # Используем исходную матрицу
                scale_factor_m_per_px  # Масштаб для конвертации пикселей в метры
            )

            logger.info(f"Room dimensions estimated: {room_dimensions}")

        # Сохранение превью (первого кадра) в base64
        preview_b64 = None
        try:
            import base64
            with open(temp_img_path, 'rb') as f:
                preview_b64 = base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            logger.warning(f"Failed to encode preview image: {e}")

        # Подготовка результата
        result = {
            "homography_matrix": H.tolist(),
            "scaled_homography_matrix": scaled_homography.tolist() if scale_info else None,
            "floor_polygons": floor_polygons,
            "wall_polygons": wall_polygons,
            "trajectory_points": step_analysis['trajectory_points'] if step_analysis else [],
            "step_peaks": step_analysis['peak_distances'] if step_analysis else [],
            "scale_info": scale_info,
            "room_dimensions": room_dimensions,
            "preview_image": preview_b64,
            "processing_time": {
                "homography": homography_time,
                "step_analysis": step_analysis_time if 'step_analysis_time' in locals() else 0,
                "total": homography_time + (step_analysis_time if 'step_analysis_time' in locals() else 0)
            }
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

def calculate_room_dimensions(floor_polygons, homography_matrix, scale_factor_m_per_px):
    """
    Расчет размеров комнаты в метрах на основе матрицы гомографии и масштаба

    Args:
        floor_polygons: список полигонов пола (координаты в пикселях)
        homography_matrix: исходная матрица гомографии (в пикселях)
        scale_factor_m_per_px: масштаб в метрах на пиксель
    """
    if not floor_polygons or len(floor_polygons) == 0:
        return None

    try:
        # Берем первый (главный) полигон пола
        main_polygon = floor_polygons[0]
        if len(main_polygon) < 4:
            return None

        # Преобразуем точки полигона с помощью матрицы гомографии (получаем координаты в пикселях)
        transformed_points = []
        for point in main_polygon:
            x, y = point
            # Применяем гомографию
            transformed = np.dot(homography_matrix, np.array([x, y, 1]))
            transformed = transformed / transformed[2]
            transformed_points.append((transformed[0], transformed[1]))

        # Находим bounding box в пикселях
        xs = [p[0] for p in transformed_points]
        ys = [p[1] for p in transformed_points]

        width_px = abs(max(xs) - min(xs))
        length_px = abs(max(ys) - min(ys))

        # Конвертируем пиксели в метры используя масштаб
        width_m = width_px * scale_factor_m_per_px
        length_m = length_px * scale_factor_m_per_px
        area_sq_m = width_m * length_m

        logger.info(f"Calculated dimensions: width_px={width_px:.1f}, length_px={length_px:.1f}, scale={scale_factor_m_per_px:.6f} m/px")
        logger.info(f"Calculated dimensions: width={width_m:.2f}m, length={length_m:.2f}m, area={area_sq_m:.2f}m²")

        return {
            "width_m": round(width_m * 10, 2),
            "length_m": round(length_m * 10, 2),
            "area_sq_m": round(area_sq_m * 100, 2)
        }
    except Exception as e:
        logger.error(f"Error calculating room dimensions: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None