import sqlite3
from contextlib import contextmanager
from typing import Tuple, Optional, Dict, Any, List

DB_PATH = "northstar_cache.db"
LOCK_TTL_SECONDS = 300  # 5 minutes before a 'PROCESSING' lock is considered a zombie


@contextmanager
def get_db_connection():
    """Yields a concurrency-safe SQLite connection configured with WAL mode."""
    conn = sqlite3.connect(DB_PATH, timeout=5.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA foreign_keys=ON;")

        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Initializes the local schema for Phases 1-5 (State Machine, Vector Index, Asymmetric Memory)."""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lecture_cache (
                file_hash TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'PROCESSING',
                raw_extracted_text TEXT,
                taxonomy_classification TEXT,
                routing_payload TEXT,
                generated_media_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Phase 5: Local Vector Store table for slide paragraph embeddings
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lecture_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                embedding_json TEXT NOT NULL
            );
        """)

        # Phase 5: Asymmetric Memory Buffer state (Summarized Knowledge State + Chat Log)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS socratic_memory (
                file_hash TEXT PRIMARY KEY,
                knowledge_summary TEXT NOT NULL DEFAULT 'Student has just started reviewing this lecture.',
                messages_json TEXT NOT NULL DEFAULT '[]',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)


def acquire_extraction_lock(
    file_hash: str, ttl_seconds: int = LOCK_TTL_SECONDS
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    ttl_modifier = f"-{ttl_seconds} seconds"

    with get_db_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO lecture_cache (file_hash, status, updated_at)
            VALUES (?, 'PROCESSING', CURRENT_TIMESTAMP)
            ON CONFLICT(file_hash) DO UPDATE SET
                status = 'PROCESSING',
                updated_at = CURRENT_TIMESTAMP
            WHERE lecture_cache.status = 'PROCESSING'
              AND lecture_cache.updated_at <= datetime('now', ?);
        """, (file_hash, ttl_modifier))

        if cursor.rowcount == 1:
            return True, None

        existing = conn.execute(
            "SELECT * FROM lecture_cache WHERE file_hash = ?",
            (file_hash,)
        ).fetchone()

        return False, dict(existing) if existing else None


def save_extracted_text(file_hash: str, extracted_text: str) -> None:
    with get_db_connection() as conn:
        conn.execute("""
            UPDATE lecture_cache
            SET status = 'EXTRACTED',
                raw_extracted_text = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE file_hash = ?;
        """, (extracted_text, file_hash))


def save_routing_decision(
    file_hash: str, classification: str, routing_payload: str
) -> None:
    with get_db_connection() as conn:
        conn.execute("""
            UPDATE lecture_cache
            SET status = 'ROUTED',
                taxonomy_classification = ?,
                routing_payload = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE file_hash = ?;
        """, (classification, routing_payload, file_hash))


def mark_generating_media(file_hash: str) -> None:
    """Transitions status to 'GENERATING' while Phase 3 media workers execute."""
    with get_db_connection() as conn:
        conn.execute("""
            UPDATE lecture_cache
            SET status = 'GENERATING',
                updated_at = CURRENT_TIMESTAMP
            WHERE file_hash = ?;
        """, (file_hash,))


def save_generated_media(
    file_hash: str, media_path: str, updated_routing_payload: Optional[str] = None
) -> None:
    """Saves Phase 3 media output path and resolves task status to 'COMPLETE'."""
    with get_db_connection() as conn:
        if updated_routing_payload:
            conn.execute("""
                UPDATE lecture_cache
                SET status = 'COMPLETE',
                    generated_media_path = ?,
                    routing_payload = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE file_hash = ?;
            """, (media_path, updated_routing_payload, file_hash))
        else:
            conn.execute("""
                UPDATE lecture_cache
                SET status = 'COMPLETE',
                    generated_media_path = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE file_hash = ?;
            """, (media_path, file_hash))


def release_failed_lock(file_hash: str) -> None:
    """Deletes a locked row on caught exceptions so retries are instant."""
    with get_db_connection() as conn:
        conn.execute(
            "DELETE FROM lecture_cache WHERE file_hash = ? AND status IN ('PROCESSING', 'EXTRACTED', 'ROUTED', 'GENERATING');",
            (file_hash,)
        )