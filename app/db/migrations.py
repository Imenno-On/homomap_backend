"""
Автоматическая миграция БД при старте приложения.
Проверяет и создает недостающие колонки в таблице projects.
"""
import logging
from sqlalchemy import text, inspect
from app.db.database import engine

logger = logging.getLogger(__name__)

# Список новых колонок для добавления
NEW_COLUMNS = [
    ("scaled_homography_matrix", "JSON"),
    ("floor_polygons", "JSON"),
    ("wall_polygons", "JSON"),
    ("step_peaks", "JSON"),
    ("scale_info", "JSON"),
    ("room_dimensions", "JSON"),
    ("processing_time", "JSON"),
    ("preview_image", "TEXT"),
]


def check_and_add_columns():
    """Проверяет наличие колонок и добавляет их, если они отсутствуют."""
    try:
        # Получаем список существующих колонок
        inspector = inspect(engine)
        columns = [col["name"] for col in inspector.get_columns("projects")]
        
        logger.info(f"Existing columns in projects table: {columns}")
        
        # Проверяем каждую новую колонку
        for column_name, column_type in NEW_COLUMNS:
            if column_name not in columns:
                logger.info(f"Adding column {column_name} ({column_type}) to projects table")
                try:
                    with engine.begin() as conn:  # begin() автоматически коммитит транзакцию
                        # Используем IF NOT EXISTS через проверку
                        conn.execute(
                            text(f'ALTER TABLE projects ADD COLUMN {column_name} {column_type}')
                        )
                    logger.info(f"Successfully added column {column_name}")
                except Exception as e:
                    logger.error(f"Error adding column {column_name}: {e}")
                    # Продолжаем даже если одна колонка не добавилась
            else:
                logger.debug(f"Column {column_name} already exists")
        
        logger.info("Database migration check completed")
        
    except Exception as e:
        logger.error(f"Error during database migration: {e}")
        # Не падаем при ошибке миграции, чтобы приложение могло запуститься
        # Но логируем ошибку для отладки


def run_migrations():
    """Запускает миграции при старте приложения."""
    logger.info("Running database migrations...")
    check_and_add_columns()

