-- Миграция для добавления новых полей в таблицу projects
-- Выполните этот скрипт в вашей БД PostgreSQL

ALTER TABLE projects 
ADD COLUMN IF NOT EXISTS scaled_homography_matrix JSON,
ADD COLUMN IF NOT EXISTS floor_polygons JSON,
ADD COLUMN IF NOT EXISTS wall_polygons JSON,
ADD COLUMN IF NOT EXISTS step_peaks JSON,
ADD COLUMN IF NOT EXISTS scale_info JSON,
ADD COLUMN IF NOT EXISTS room_dimensions JSON,
ADD COLUMN IF NOT EXISTS processing_time JSON,
ADD COLUMN IF NOT EXISTS preview_image TEXT;

