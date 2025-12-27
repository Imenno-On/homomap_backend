from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.db.database import get_db
from app.api.dependencies import get_current_user
from app.db import crud
from app.schemas.project import ProjectResponse
from app.db.models import User
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=List[ProjectResponse])
def read_user_projects(
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
        skip: int = 0,
        limit: int = 100
):
    """Получить список всех проектов текущего пользователя."""
    logger.info(f"Fetching projects for user ID: {current_user.id}")
    projects = crud.get_user_projects(
        db=db,
        user_id=current_user.id,
        skip=skip,
        limit=limit
    )
    return projects


@router.get("/{project_id}", response_model=ProjectResponse)
def read_project(
        project_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Получить детальную информацию о проекте по ID."""
    logger.info(f"Fetching project details for project ID: {project_id}")
    project = crud.get_project(db, project_id=project_id)

    if project is None:
        logger.warning(f"Project not found: {project_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Проект не найден"
        )

    # Проверка, что проект принадлежит текущему пользователю
    if project.owner_id != current_user.id:
        logger.warning(f"Unauthorized access attempt to project {project_id} by user {current_user.id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="У вас нет доступа к этому проекту"
        )

    # Логируем данные проекта для отладки
    logger.info(f"Returning project {project_id} data:")
    logger.info(f"  - trajectory_points: {len(project.trajectory_points) if project.trajectory_points else 0} points")
    logger.info(f"  - step_peaks: {len(project.step_peaks) if project.step_peaks else 0} peaks")
    logger.info(f"  - scale_info: {project.scale_info is not None}")
    logger.info(f"  - room_dimensions: {project.room_dimensions is not None}")
    logger.info(f"  - floor_polygons: {len(project.floor_polygons) if project.floor_polygons else 0} polygons")
    logger.info(f"  - wall_polygons: {len(project.wall_polygons) if project.wall_polygons else 0} polygons")

    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(
        project_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Удалить проект по ID."""
    logger.info(f"Deleting project ID: {project_id} by user ID: {current_user.id}")
    project = crud.get_project(db, project_id=project_id)

    if project is None:
        logger.warning(f"Project not found for deletion: {project_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Проект не найден"
        )

    # Проверка владельца
    if project.owner_id != current_user.id:
        logger.warning(f"Unauthorized delete attempt on project {project_id} by user {current_user.id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="У вас нет прав для удаления этого проекта"
        )

    crud.delete_project(db, project_id=project_id)
    logger.info(f"Project {project_id} successfully deleted")
    return