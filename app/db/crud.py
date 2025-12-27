from sqlalchemy.orm import Session
from app.db import models
from app.schemas import user as user_schemas
from app.core.security import get_password_hash
from app.db.models import Project, User
from app.schemas.project import ProjectCreate, HomographyResult
import logging

logger = logging.getLogger(__name__)


# --- Функции для работы с пользователями (User CRUD) ---

def get_user_by_email(db: Session, email: str):
    """Находит пользователя по email."""
    return db.query(models.User).filter(models.User.email == email).first()


def get_user_by_id(db: Session, user_id: int):
    """Возвращает пользователя по ID."""
    return db.query(models.User).filter(models.User.id == user_id).first()


def create_user(db: Session, user: user_schemas.UserCreate):
    """Создает нового пользователя."""
    # Хешируем пароль перед сохранением
    hashed_password = get_password_hash(user.password)

    db_user = models.User(
        email=user.email,
        hashed_password=hashed_password
    )

    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


# --- Функции для работы с проектами (Project CRUD) ---

def create_project(
        db: Session,
        user_id: int,
        title: str,
        video_path: str,
        homography_matrix=None,
        scaled_homography_matrix=None,
        floor_polygons=None,
        wall_polygons=None,
        trajectory_points=None,
        step_peaks=None,
        scale_info=None,
        room_dimensions=None,
        processing_time=None,
        preview_image=None,
        preview_path: str | None = None
):
    """Создает новый проект для пользователя."""
    logger.info(f"Creating project for user ID: {user_id}, title: {title}")

    db_project = Project(
        owner_id=user_id,
        title=title,
        video_path=video_path,
        preview_path=preview_path,
        homography_matrix=homography_matrix,
        scaled_homography_matrix=scaled_homography_matrix,
        floor_polygons=floor_polygons,
        wall_polygons=wall_polygons,
        trajectory_points=trajectory_points,
        step_peaks=step_peaks,
        scale_info=scale_info,
        room_dimensions=room_dimensions,
        processing_time=processing_time,
        preview_image=preview_image,
        status="completed"
    )

    db.add(db_project)
    db.commit()
    db.refresh(db_project)

    logger.info(f"Project created with ID: {db_project.id}")
    return db_project


def get_user_projects(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    """Возвращает список проектов для конкретного пользователя с пагинацией."""
    logger.info(f"Fetching projects for user ID: {user_id}, skip: {skip}, limit: {limit}")
    return db.query(Project).filter(Project.owner_id == user_id).offset(skip).limit(limit).all()


def get_project(db: Session, project_id: int):
    """Находит проект по ID."""
    logger.info(f"Fetching project with ID: {project_id}")
    return db.query(Project).filter(Project.id == project_id).first()


def delete_project(db: Session, project_id: int):
    """Удаляет проект по ID."""
    logger.info(f"Deleting project with ID: {project_id}")
    project = db.query(Project).filter(Project.id == project_id).first()

    if project:
        # Удаляем связанные файлы (видео и превью) если они существуют
        try:
            import os
            # video_path и preview_path хранятся в виде "/videos/filename" и "/previews/filename"
            if project.video_path:
                video_fname = os.path.basename(project.video_path)
                video_full = os.path.join("uploaded_videos", video_fname)
                if os.path.exists(video_full):
                    os.remove(video_full)
                    logger.info(f"Removed video file: {video_full}")
            if project.preview_path:
                preview_fname = os.path.basename(project.preview_path)
                preview_full = os.path.join("previews", preview_fname)
                if os.path.exists(preview_full):
                    os.remove(preview_full)
                    logger.info(f"Removed preview file: {preview_full}")
        except Exception as e:
            logger.warning(f"Failed to remove project files for {project_id}: {e}")

        db.delete(project)
        db.commit()
        logger.info(f"Project {project_id} successfully deleted")
        return True

    logger.warning(f"Project {project_id} not found for deletion")
    return False