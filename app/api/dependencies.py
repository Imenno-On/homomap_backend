from typing import Annotated
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from sqlalchemy.orm import Session
import os
from app.core.security import SECRET_KEY, ALGORITHM
from app.db.database import get_db
from app.db.models import User
from app.db import crud  # Предполагаем, что у вас есть функция crud.get_user_by_id


SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-that-should-be-kept-secret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Настройка схемы OAuth2 для получения токена из заголовков (Bearer)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/users/login")


async def get_current_user(
        db: Session = Depends(get_db),
        token: Annotated[str, Depends(oauth2_scheme)] = None
) -> User:
    """
    Извлекает и декодирует токен, находит пользователя в БД.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Не удалось проверить учетные данные",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if token is None:
        raise credentials_exception
    token = token.strip()
    try:
        # 1. Декодирование токена
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # "sub" (subject) - это обычно ID пользователя, который мы кодировали
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception

    except JWTError:
        # Неверный токен (истек, изменен и т.д.)
        raise credentials_exception

    # 2. Поиск пользователя в БД
    user = crud.get_user_by_id(db, user_id=int(user_id))

    if user is None:
        raise credentials_exception

    return user


def get_current_active_user(current_user: Annotated[User, Depends(get_current_user)]) -> User:
    """
    Проверяет, что пользователь активен (если у вас есть поле is_active).
    """
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Неактивный пользователь")
    return current_user