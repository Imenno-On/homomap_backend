# Миграция БД: Добавление новых полей

Эта миграция добавляет новые поля в таблицу `projects` для поддержки анализа шагов и расширенной визуализации.

## Выполнение миграции

### Вариант 1: Через psql (рекомендуется)

```bash
# Подключитесь к контейнеру БД
docker exec -i homomap_db psql -U postgres -d homomap < migrations/add_new_fields.sql
```

### Вариант 2: Через docker exec

```bash
docker exec -i homomap_db psql -U postgres -d homomap << EOF
ALTER TABLE projects 
ADD COLUMN IF NOT EXISTS scaled_homography_matrix JSON,
ADD COLUMN IF NOT EXISTS floor_polygons JSON,
ADD COLUMN IF NOT EXISTS wall_polygons JSON,
ADD COLUMN IF NOT EXISTS step_peaks JSON,
ADD COLUMN IF NOT EXISTS scale_info JSON,
ADD COLUMN IF NOT EXISTS room_dimensions JSON,
ADD COLUMN IF NOT EXISTS processing_time JSON,
ADD COLUMN IF NOT EXISTS preview_image TEXT;
EOF
```

### Вариант 3: Через pgAdmin или другой клиент БД

Выполните SQL из файла `add_new_fields.sql` в вашем клиенте БД.

## Проверка миграции

После выполнения миграции проверьте, что новые колонки добавлены:

```bash
docker exec -i homomap_db psql -U postgres -d homomap -c "\d projects"
```

Вы должны увидеть новые колонки:
- `scaled_homography_matrix`
- `floor_polygons`
- `wall_polygons`
- `step_peaks`
- `scale_info`
- `room_dimensions`
- `processing_time`
- `preview_image`

## Откат миграции (если нужно)

Если нужно откатить изменения:

```sql
ALTER TABLE projects 
DROP COLUMN IF EXISTS scaled_homography_matrix,
DROP COLUMN IF EXISTS floor_polygons,
DROP COLUMN IF EXISTS wall_polygons,
DROP COLUMN IF EXISTS step_peaks,
DROP COLUMN IF EXISTS scale_info,
DROP COLUMN IF EXISTS room_dimensions,
DROP COLUMN IF EXISTS processing_time,
DROP COLUMN IF EXISTS preview_image;
```

