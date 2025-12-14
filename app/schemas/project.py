from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


# Схема для создания проекта (при загрузке видео)
class ProjectCreate(BaseModel):
    video_path: str  # Только временный путь к файлу перед обработкой


# Схема для отображения результата (Матрица и точки)
class HomographyResult(BaseModel):
    homography_matrix: List[List[float]]
    trajectory_points: List[List[float]]


class ProjectResponse(BaseModel):
    id: int
    title: Optional[str] = None
    description: Optional[str] = None
    owner_id: int

    homography_matrix: Optional[Any] = None  # JSON (List[List[float]])
    trajectory_points: Optional[Any] = None  # JSON
    accuracy_score: Optional[float] = None

    video_path: str
    preview_path: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True