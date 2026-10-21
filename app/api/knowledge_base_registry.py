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


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    knowledge_base_id: str
    filename: str
    status: str
    size_bytes: int
    chunk_count: int
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

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                knowledge_base_id TEXT NOT NULL
                    REFERENCES knowledge_bases(id),
                filename TEXT NOT NULL,
                storage_path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
                status TEXT NOT NULL DEFAULT 'processing'
                    CHECK (status IN ('processing', 'indexed', 'failed')),
                chunk_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (knowledge_base_id, sha256)
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_documents_kb_created
            ON documents (knowledge_base_id, created_at)
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

    return [KnowledgeBase(**dict(row)) for row in rows]


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


def _document_from_row(row) -> DocumentRecord | None:
    if row is None:
        return None

    return DocumentRecord(
        id=row["id"],
        knowledge_base_id=row["knowledge_base_id"],
        filename=row["filename"],
        status=row["status"],
        size_bytes=row["size_bytes"],
        chunk_count=row["chunk_count"],
        created_at=row["created_at"],
    )


def create_document_record(
    document_id: str,
    knowledge_base_id: str,
    filename: str,
    storage_path: str,
    sha256: str,
    size_bytes: int,
) -> DocumentRecord:
    initialize_registry()

    with _connection() as conn:
        conn.execute(
            """
            INSERT INTO documents (
                id, knowledge_base_id, filename,
                storage_path, sha256, size_bytes, status
            )
            VALUES (?, ?, ?, ?, ?, ?, 'processing')
            """,
            (
                document_id,
                knowledge_base_id,
                filename,
                storage_path,
                sha256,
                size_bytes,
            ),
        )

    record = get_document_record(document_id)
    if record is None:
        raise RuntimeError("Document record was not created")

    return record


def get_document_record(
    document_id: str,
) -> DocumentRecord | None:
    initialize_registry()

    with _connection() as conn:
        row = conn.execute(
            """
            SELECT id, knowledge_base_id, filename,
                   status, size_bytes, chunk_count, created_at
            FROM documents
            WHERE id = ?
            """,
            (document_id,),
        ).fetchone()

    return _document_from_row(row)


def update_document_record_status(
    document_id: str,
    status: str,
    chunk_count: int = 0,
) -> DocumentRecord:
    if status not in {"processing", "indexed", "failed"}:
        raise ValueError("Invalid document status")

    initialize_registry()

    with _connection() as conn:
        cursor = conn.execute(
            """
            UPDATE documents
            SET status = ?, chunk_count = ?
            WHERE id = ?
            """,
            (status, chunk_count, document_id),
        )

        if cursor.rowcount != 1:
            raise ValueError("Document record not found")

    record = get_document_record(document_id)
    if record is None:
        raise RuntimeError("Document record disappeared")

    return record


def list_document_records(
    knowledge_base_id: str,
) -> list[DocumentRecord]:
    initialize_registry()

    with _connection() as conn:
        rows = conn.execute(
            """
            SELECT id, knowledge_base_id, filename,
                   status, size_bytes, chunk_count, created_at
            FROM documents
            WHERE knowledge_base_id = ?
            ORDER BY created_at DESC, id
            """,
            (knowledge_base_id,),
        ).fetchall()

    return [
        DocumentRecord(
            id=row["id"],
            knowledge_base_id=row["knowledge_base_id"],
            filename=row["filename"],
            status=row["status"],
            size_bytes=row["size_bytes"],
            chunk_count=row["chunk_count"],
            created_at=row["created_at"],
        )
        for row in rows
    ]
