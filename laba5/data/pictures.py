"""Уровень данных: прямое взаимодействие с SQLite через DB-API."""

import sqlite3
import cv2
import numpy as np
from datetime import datetime
from data.init import get_db_connection
from model.pictures import Picture


def row_to_model(row: tuple) -> Picture | None:
    """
    Преобразует строку из БД в объект Picture.
    
    Args:
        row: Кортеж с данными из БД (id, name, description, image_data, created_at)
        
    Returns:
        Picture | None: Объект Picture или None, если строка пуста
    """
    if row is None:
        return None
    
    id_, name, description, image_blob, created_at_str = row
    
    # Преобразуем BLOB обратно в numpy.ndarray
    nparr = np.frombuffer(image_blob, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Преобразуем строку даты в объект datetime
    created_at = datetime.strptime(created_at_str, '%Y-%m-%d %H:%M:%S')
    
    return Picture(
        name=name,
        img=img,
        description=description or "",
        dt=created_at
    )


def get_all() -> list[Picture]:
    """
    Получить все изображения из базы данных.
    
    Returns:
        list[Picture]: Список всех изображений
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, description, image_data, created_at FROM pictures")
        rows = cursor.fetchall()
        return [row_to_model(row) for row in rows if row_to_model(row) is not None]


def get_one(name: str) -> Picture | None:
    """
    Получить одно изображение по имени.
    
    Args:
        name: Имя изображения
        
    Returns:
        Picture | None: Объект Picture или None, если изображение не найдено
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        qry = "SELECT id, name, description, image_data, created_at FROM pictures WHERE name = ?"
        cursor.execute(qry, (name,))
        row = cursor.fetchone()
        return row_to_model(row)


def add_one(picture: Picture) -> int:
    """
    Сохранить объект Picture в базу данных (np.ndarray → BLOB).
    
    Args:
        picture: Объект Picture для сохранения
        
    Returns:
        int: ID новой записи
        
    Raises:
        ValueError: Если не удалось закодировать изображение
    """
    # Кодируем numpy-массив в PNG формат
    success, encoded_img = cv2.imencode('.png', picture.img)
    if not success:
        raise ValueError("Не удалось закодировать изображение")
    
    image_blob = encoded_img.tobytes()
    created_at_str = picture.dt.strftime('%Y-%m-%d %H:%M:%S')
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO pictures (name, description, image_data, created_at)
            VALUES (?, ?, ?, ?)
        ''', (picture.name, picture.description, image_blob, created_at_str))
        conn.commit()
        return cursor.lastrowid


def delete_one(name: str) -> bool:
    """
    Удалить изображение из базы данных по имени.
    
    Args:
        name: Имя изображения для удаления
        
    Returns:
        bool: True если запись была удалена, False если запись не найдена
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM pictures WHERE name = ?", (name,))
        conn.commit()
        return cursor.rowcount > 0


def update_description(name: str, new_description: str) -> bool:
    """
    Обновить описание изображения по имени.
    
    Args:
        name: Имя изображения
        new_description: Новое описание
        
    Returns:
        bool: True если запись была обновлена, False если запись не найдена
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE pictures 
            SET description = ? 
            WHERE name = ?
        ''', (new_description, name))
        conn.commit()
        return cursor.rowcount > 0


def update_one(picture: Picture) -> bool:
    """
    Полностью обновить изображение в базе данных.
    
    Args:
        picture: Объект Picture с новыми данными
        
    Returns:
        bool: True если запись была обновлена, False если запись не найдена
    """
    # Кодируем numpy-массив в PNG формат
    success, encoded_img = cv2.imencode('.png', picture.img)
    if not success:
        raise ValueError("Не удалось закодировать изображение")
    
    image_blob = encoded_img.tobytes()
    created_at_str = picture.dt.strftime('%Y-%m-%d %H:%M:%S')
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE pictures 
            SET description = ?, image_data = ?, created_at = ?
            WHERE name = ?
        ''', (picture.description, image_blob, created_at_str, picture.name))
        conn.commit()
        return cursor.rowcount > 0