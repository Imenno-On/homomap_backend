from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.database import Base


class User(Base):
    """
    Модель пользователя для авторизации.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Связь с проектами (один ко многим: один пользователь - много проектов)
    projects = relationship("Project", back_populates="owner")

    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}')>"


class Project(Base):
    """
    Модель проекта гомографии.
    Хранит информацию о загруженном видео и результатах обработки.
    """
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True, nullable=True)
    description = Column(String, nullable=True)

    # Путь или имя файла загруженного видео.
    # Файлы будут храниться не в БД, а на диске/S3, здесь - только ссылка.
    video_path = Column(String, nullable=False)
    preview_path = Column(String, nullable=True)
    # Статус обработки: pending, processing, completed, failed
    status = Column(String, default="pending", nullable=False)

    # --- Результаты ML-части ---
    # Матрица гомографии (3x3) хранится как JSON (например, [[a,b,c],[d,e,f],[g,h,i]])
    homography_matrix = Column(JSON, nullable=True)
    trajectory_points = Column(JSON, nullable=True)
    # Дополнительная информация, например, точность (из документации)
    accuracy_score = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Внешний ключ, связывающий проект с пользователем
    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="projects")

    def __repr__(self):
        return f"<Project(id={self.id}, title='{self.title}', status='{self.status}')>"