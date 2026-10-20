import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from app.config import KNOWLEDGE_BASE_DB_PATH


@dataclass(frozen=True)
class KnowledgeBase:
    id: str
    name: str
    status: str
    created_at: str


@contextmanager
def _connection():
    path = Path(KNOWLEDGE_BASE_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path, timeout=5)
    conn.row_factory = sqlite3.Row

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_registry():
    with _connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_bases (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'inactive')),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def get_knowledge_base(
    knowledge_base_id: str,
) -> KnowledgeBase | None:
    initialize_registry()

    with _connection() as conn:
        row = conn.execute(
            """
            SELECT id, name, status, created_at
            FROM knowledge_bases
            WHERE id = ?
            """,
            (knowledge_base_id,),
        ).fetchone()

    if row is None:
        return None

    return KnowledgeBase(**dict(row))


def list_knowledge_bases(
    status: str | None = None,
) -> list[KnowledgeBase]:
    initialize_registry()

    with _connection() as conn:
        if status is None:
            rows = conn.execute(
                """
                SELECT id, name, status, created_at
                FROM knowledge_bases
                ORDER BY id
                """
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, name, status, created_at
                FROM knowledge_bases
                WHERE status = ?
                ORDER BY id
                """,
                (status,),
            ).fetchall()

    return [
        KnowledgeBase(**dict(row))
        for row in rows
    ]


def register_knowledge_base(
    knowledge_base_id: str,
    name: str,
):
    if not re.fullmatch(
        r"[a-z0-9][a-z0-9-]{0,62}",
        knowledge_base_id,
    ):
        raise ValueError(
            "Knowledge-base ID must be a lowercase "
            "slug containing letters, digits, and hyphens"
        )

    initialize_registry()

    with _connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO knowledge_bases
                (id, name, status)
            VALUES (?, ?, 'active')
            """,
            (knowledge_base_id, name),
        )

    return get_knowledge_base(knowledge_base_id)
