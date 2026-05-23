# data/init.py
"""Инициализация подключения к SQLite."""

import sqlite3
from contextlib import contextmanager

DB_PATH = "db/images.db"

@contextmanager
def get_db_connection():
    """
    Контекстный менеджер для подключения к SQLite.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()