# Homomap Backend
## Система получения матрицы гомографии помещения на основе видео с камеры.

### Стек технологий
- Python 3.10
- FastAPI
- PostgreSQL
- Docker
- PyTorch (ML часть)

### Как запустить проект

- Склонируйте репозиторий:
```bash
git clone https://github.com/Imenno-On/homomap_backend.git
cd homomap_backend
```

- Создайте файл .env:
```bash
cp .env.example .env
```

- Запустите Docker-контейнеры:
```bash
docker-compose up --build -d
```

- Документация будет доступна по адресу: http://localhost:8000/docs

### Основные функции
- Регистрация и авторизация пользователей
- Загрузка видео для обработки
- Получение матрицы гомографии
- Просмотр истории обработанных проектов
- Удаление своих проектов

### Эндпоинты API
- POST /api/v1/users/register - Регистрация
- POST /api/v1/users/login - Вход
- POST /api/v1/homography/process_video - Обработка видео
- GET /api/v1/projects/ - Список проектов
- DELETE /api/v1/projects/{id} - Удаление проекта