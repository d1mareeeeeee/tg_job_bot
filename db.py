"""Слой хранения на стандартном sqlite3.

Две таблицы:
  * seen_posts — какие посты уже обработаны (чтобы не обрабатывать дважды);
  * matches    — подходящие вакансии, которые мы отправили пользователю.

Соединение открывается один раз (init_db) и переиспользуется. Запросы быстрые,
поэтому вызываем их из async-кода напрямую, без отдельного пула.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Глобальное соединение модуля. Инициализируется в init_db().
_conn: sqlite3.Connection | None = None


def _now_iso() -> str:
    """Текущее время в ISO-8601 (UTC) — для полей processed_at / sent_at."""
    return datetime.now(timezone.utc).isoformat()


def init_db(path: str) -> None:
    """Открывает соединение и создаёт таблицы, если их ещё нет."""
    global _conn
    _conn = sqlite3.connect(path, check_same_thread=False)
    # WAL — устойчивее к конкурентному чтению/записи.
    _conn.execute("PRAGMA journal_mode=WAL;")
    _conn.execute(
        """
        CREATE TABLE IF NOT EXISTS seen_posts (
            channel      TEXT    NOT NULL,
            msg_id       INTEGER NOT NULL,
            processed_at TEXT    NOT NULL,
            PRIMARY KEY (channel, msg_id)
        );
        """
    )
    _conn.execute(
        """
        CREATE TABLE IF NOT EXISTS matches (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            channel      TEXT    NOT NULL,
            msg_id       INTEGER NOT NULL,
            score        INTEGER NOT NULL,
            reason       TEXT,
            role         TEXT,
            vacancy_text TEXT,
            sent_at      TEXT    NOT NULL
        );
        """
    )
    _conn.commit()
    logger.info("База данных инициализирована: %s", path)


def _require_conn() -> sqlite3.Connection:
    """Возвращает соединение или бросает ошибку, если init_db не вызван."""
    if _conn is None:
        raise RuntimeError("init_db() не вызван — база данных не инициализирована.")
    return _conn


def is_seen(channel: str, msg_id: int) -> bool:
    """Проверяет, обрабатывался ли уже этот пост."""
    conn = _require_conn()
    cur = conn.execute(
        "SELECT 1 FROM seen_posts WHERE channel = ? AND msg_id = ? LIMIT 1;",
        (channel, msg_id),
    )
    return cur.fetchone() is not None


def mark_seen(channel: str, msg_id: int) -> None:
    """Помечает пост как обработанный. Идемпотентно (INSERT OR IGNORE)."""
    conn = _require_conn()
    conn.execute(
        "INSERT OR IGNORE INTO seen_posts (channel, msg_id, processed_at) VALUES (?, ?, ?);",
        (channel, msg_id, _now_iso()),
    )
    conn.commit()


def save_match(
    channel: str,
    msg_id: int,
    score: int,
    reason: str,
    role: str,
    vacancy_text: str,
) -> None:
    """Сохраняет подходящую вакансию, которую отправили пользователю."""
    conn = _require_conn()
    conn.execute(
        """
        INSERT INTO matches (channel, msg_id, score, reason, role, vacancy_text, sent_at)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        (channel, msg_id, score, reason, role, vacancy_text, _now_iso()),
    )
    conn.commit()


def close_db() -> None:
    """Закрывает соединение (при graceful shutdown)."""
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None
