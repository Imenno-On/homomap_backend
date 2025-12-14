from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from app.db import crud
from app.db.database import get_db
from app.schemas import user as user_schemas
from app.core import security
#from app.core.config import settings
from app.api.dependencies import get_current_user # <-- Импортируем ОДНУ ПРАВИЛЬНУЮ версию
from app.core.security import ACCESS_TOKEN_EXPIRE_MINUTES

router = APIRouter()


# --- Вспомогательная функция для получения текущего пользователя (Зависимость) ---
# Эта функция будет использоваться в других эндпоинтах, требующих авторизации
# def get_current_user(
#         db: Session = Depends(get_db),
#         token: str = Depends(security.oauth2_scheme)  # Используем стандартную схему OAuth2
# ):
#     """
#     Проверяет JWT токен и возвращает объект пользователя.
#     Используется как зависимость (Depends).
#     """
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     try:
#         # 1. Декодируем токен
#         payload = security.decode_access_token(token)
#         email: str = payload.get("sub")  # Получаем 'sub' (email)
#
#         if email is None:
#             raise credentials_exception
#
#         token_data = user_schemas.TokenData(email=email)
#
#     except security.JWTError:
#         raise credentials_exception
#
#     # 2. Ищем пользователя в БД
#     user = crud.get_user_by_email(db, email=token_data.email)
#
#     if user is None:
#         raise credentials_exception
#
#     return user


# --- Эндпоинты ---

@router.post("/register", response_model=user_schemas.UserOut, status_code=status.HTTP_201_CREATED)
def register_user(user: user_schemas.UserCreate, db: Session = Depends(get_db)):
    """Регистрация нового пользователя."""
    db_user = crud.get_user_by_email(db, email=user.email)

    if db_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    return crud.create_user(db=db, user=user)


@router.post("/login", response_model=user_schemas.Token)
def login_for_access_token(
        form_data: OAuth2PasswordRequestForm = Depends(),
        db: Session = Depends(get_db)
):
    """Аутентификация пользователя и выдача JWT токена."""

    user = crud.get_user_by_email(db, email=form_data.username)  # username - это email

    if not user or not security.verify_password(form_data.password, user.hashed_password):
        # Одинаковая ошибка для email не найден и пароль не верен
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Создаем токен
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        data={"id": user.id, "email": user.email},
        expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=user_schemas.UserOut)
def read_users_me(current_user: user_schemas.UserOut = Depends(get_current_user)):
    """Получить данные текущего авторизованного пользователя."""
    return current_user