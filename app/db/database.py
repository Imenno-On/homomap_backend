from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

# 1. Создание URL для подключения к БД (берется из .env через settings)
# PydanticSettings автоматически подставит значение
SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL

# 2. Создание движка (Engine) SQLAlchemy
# connect_args={"check_same_thread": False} - только для SQLite,
# для PostgreSQL это не нужно
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# 3. Настройка сессии (SessionLocal)
# Эта сессия будет использоваться для создания/получения/изменения данных
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. Базовый класс для моделей (Base)
# Все наши модели БД будут наследоваться от него
Base = declarative_base()

# 5. Функция-зависимость для получения сессии в FastAPI
# Используется в каждом маршруте, где нужен доступ к БД (Dependency Injection)
def get_db():
    """Возвращает сессию БД, которая автоматически закрывается."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()