from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Класс для загрузки всех настроек приложения из файла .env.
    """
    # Настройки базы данных
    DATABASE_URL: str
    DB_HOST: str
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str
    DB_PORT: int

    # Настройки безопасности (JWT)
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int

    ML_SERVICE_URL: str = "http://ml:8000"
    # Указываем, что настройки нужно читать из файла .env
    model_config = SettingsConfigDict(env_file='.env')

# Создаем единственный экземпляр настроек, который будет использоваться во всем приложении
settings = Settings()