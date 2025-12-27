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
    project_id: Optional[int] = None


class ProjectResponse(BaseModel):
    id: int
    title: Optional[str] = None
    description: Optional[str] = None
    owner_id: int

    homography_matrix: Optional[Any] = None  # JSON (List[List[float]])
    scaled_homography_matrix: Optional[Any] = None  # JSON (List[List[float]])
    floor_polygons: Optional[Any] = None  # JSON (List[List[List[float]]])
    wall_polygons: Optional[Any] = None  # JSON (List[List[List[float]]])
    trajectory_points: Optional[Any] = None  # JSON (List[List[float]] или List[List[float, float, float]])
    step_peaks: Optional[Any] = None  # JSON (List[float])
    scale_info: Optional[Any] = None  # JSON (dict)
    room_dimensions: Optional[Any] = None  # JSON (dict)
    processing_time: Optional[Any] = None  # JSON (dict)
    preview_image: Optional[str] = None  # base64 строка
    accuracy_score: Optional[float] = None

    video_path: str
    preview_path: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True