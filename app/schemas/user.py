from pydantic import BaseModel, EmailStr, Field
from typing import Optional

# 1. Схема для создания нового пользователя (регистрации)
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)

# 2. Схема, используемая для чтения данных пользователя (возвращается клиенту)
class UserOut(BaseModel):
    id: int
    email: EmailStr
    is_active: bool

    # Позволяет Pydantic работать с ORM-объектами SQLAlchemy
    class Config:
        from_attributes = True

# 3. Схема для токена авторизации
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

# 4. Схема для данных, хранящихся внутри JWT-токена
class TokenData(BaseModel):
    email: Optional[str] = None