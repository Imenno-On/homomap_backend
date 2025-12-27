from datetime import datetime, timedelta, timezone
from typing import Optional
from passlib.context import CryptContext
from jose import jwt, JWTError
# from app.core.config import settings
import os
from fastapi.security import OAuth2PasswordBearer

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-that-should-be-kept-secret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
# Контекст для хеширования паролей (используем bcrypt)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# 1. Хеширование пароля
def get_password_hash(password: str) -> str:
    """Хеширует пароль."""
    return pwd_context.hash(password)


# 2. Проверка хеша пароля
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Проверяет соответствие открытого пароля хешу."""
    return pwd_context.verify(plain_password, hashed_password)


# 3. Создание JWT токена
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Создает JWT Access Token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        # Берем время жизни токена из .env
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})

    # Добавляем субъект токена (например, email пользователя)
    # to_encode.update({"sub": str(data["email"])})
    user_id = data.get("id")
    if user_id is None:
        raise ValueError("User ID must be passed to create_access_token under the 'id' key.")
    to_encode.update({"sub": str(user_id)})
    # Создание самого токена
    encoded_jwt = jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    return encoded_jwt


# 4. Декодирование токена (будет использоваться для проверки)
def decode_access_token(token: str) -> dict:
    """Декодирует JWT токен."""
    return jwt.decode(
        token,
        SECRET_KEY,
        algorithms=[ALGORITHM]
    )

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/users/login")