"""PostgreSQL faceprint store (pgvector when available, else double precision[]).

Falls back to an in-memory store when the database is unavailable.
Also manages face_match_providers (Polygon=1, Azure=2, Internal=3).
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

import asyncpg

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5432/liveness_db",
)

_pool: Optional[asyncpg.Pool] = None
_db_available = False
_has_pgvector = False

# In-memory fallback used when Postgres is unreachable
_memory_faceprints: dict[str, list[dict]] = {}
_memory_providers: list[dict] = [
    {"id": 1, "code": "polygon", "name": "Polygon / Bank API", "enabled": True, "is_active": False, "priority": 2, "config": {}},
    {"id": 2, "code": "azure", "name": "Azure Face", "enabled": True, "is_active": False, "priority": 3, "config": {}},
    {"id": 3, "code": "internal", "name": "Internal MTCNN+ArcFace", "enabled": True, "is_active": True, "priority": 1, "config": {}},
]

# Cache active provider ~30s
_provider_cache: dict[str, Any] = {"provider": None, "expires_at": 0.0}
_PROVIDER_CACHE_TTL = 30.0


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10)
    return _pool


async def _ensure_faceprint_columns(conn) -> None:
    """Add multi-face / source columns if missing (safe on existing DBs)."""
    for col, ddl in [
        ("source", "ALTER TABLE faceprints ADD COLUMN IF NOT EXISTS source VARCHAR(32)"),
        ("face_index", "ALTER TABLE faceprints ADD COLUMN IF NOT EXISTS face_index INT DEFAULT 0"),
        ("provider_code", "ALTER TABLE faceprints ADD COLUMN IF NOT EXISTS provider_code VARCHAR(32)"),
        ("bbox", "ALTER TABLE faceprints ADD COLUMN IF NOT EXISTS bbox JSONB"),
    ]:
        try:
            await conn.execute(ddl)
        except Exception as e:
            logger.warning("Could not add faceprints.%s: %s", col, e)


async def _ensure_providers_table(conn) -> None:
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS face_match_providers (
            id INT PRIMARY KEY,
            code VARCHAR(32) NOT NULL UNIQUE,
            name VARCHAR(128) NOT NULL,
            enabled BOOLEAN NOT NULL DEFAULT TRUE,
            is_active BOOLEAN NOT NULL DEFAULT FALSE,
            priority INT NOT NULL DEFAULT 100,
            config JSONB DEFAULT '{}'::jsonb,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    # Exactly one active provider (expression unique index)
    await conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_face_match_providers_one_active
        ON face_match_providers ((TRUE))
        WHERE is_active = TRUE;
    """)
    rows = [
        (1, "polygon", "Polygon / Bank API", True, False, 2),
        (2, "azure", "Azure Face", True, False, 3),
        (3, "internal", "Internal MTCNN+ArcFace", True, True, 1),
    ]
    for pid, code, name, enabled, is_active, priority in rows:
        await conn.execute(
            """
            INSERT INTO face_match_providers (id, code, name, enabled, is_active, priority)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (id) DO NOTHING
            """,
            pid, code, name, enabled, is_active, priority,
        )
    # Ensure at least one active
    active = await conn.fetchval(
        "SELECT COUNT(*) FROM face_match_providers WHERE is_active = TRUE"
    )
    if not active:
        await conn.execute(
            "UPDATE face_match_providers SET is_active = TRUE, updated_at = NOW() WHERE id = 3"
        )


async def init_db() -> bool:
    """Create tables and extensions if not exists. Returns True if DB is ready."""
    global _db_available, _has_pgvector
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                _has_pgvector = True
                logger.info("pgvector extension enabled")
            except Exception as e:
                _has_pgvector = False
                logger.warning("pgvector not available (%s) — using double precision[] embeddings", e)

            if _has_pgvector:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS faceprints (
                        id SERIAL PRIMARY KEY,
                        account_id VARCHAR(64) NOT NULL,
                        embedding vector(512) NOT NULL,
                        reference_image_b64 TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        source_image_hash VARCHAR(64),
                        source VARCHAR(32),
                        face_index INT DEFAULT 0,
                        provider_code VARCHAR(32),
                        bbox JSONB
                    );
                """)
                row_count = await conn.fetchval("SELECT COUNT(*) FROM faceprints")
                if row_count and row_count > 0:
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_faceprints_embedding
                        ON faceprints USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                    """)
            else:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS faceprints (
                        id SERIAL PRIMARY KEY,
                        account_id VARCHAR(64) NOT NULL,
                        embedding DOUBLE PRECISION[] NOT NULL,
                        reference_image_b64 TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        source_image_hash VARCHAR(64),
                        source VARCHAR(32),
                        face_index INT DEFAULT 0,
                        provider_code VARCHAR(32),
                        bbox JSONB
                    );
                """)

            await _ensure_faceprint_columns(conn)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_faceprints_account
                ON faceprints (account_id);
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(64) NOT NULL,
                    account_id VARCHAR(64) NOT NULL,
                    action VARCHAR(32) NOT NULL,
                    result JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            await _ensure_providers_table(conn)
            from app.db.training_store import ensure_training_tables
            await ensure_training_tables(conn)
        _db_available = True
        _provider_cache["provider"] = None
        _provider_cache["expires_at"] = 0.0
        logger.info(
            "PostgreSQL ready (pgvector=%s) url_host=%s",
            _has_pgvector,
            DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL,
        )
        return True
    except Exception as e:
        _db_available = False
        _has_pgvector = False
        logger.warning("Database unavailable (%s) — using in-memory faceprint store", e)
        return False


def _row_to_provider(r) -> dict:
    cfg = r["config"]
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg)
        except Exception:
            cfg = {}
    return {
        "id": int(r["id"]),
        "code": r["code"],
        "name": r["name"],
        "enabled": bool(r["enabled"]),
        "is_active": bool(r["is_active"]),
        "priority": int(r["priority"] or 100),
        "config": cfg or {},
    }


async def list_face_match_providers() -> list[dict]:
    if _db_available:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, code, name, enabled, is_active, priority, config "
                "FROM face_match_providers ORDER BY id"
            )
            return [_row_to_provider(r) for r in rows]
    return list(_memory_providers)


async def get_active_face_match_provider() -> dict:
    """Return the active provider (cached ~30s). Defaults to internal."""
    now = time.time()
    if _provider_cache["provider"] and _provider_cache["expires_at"] > now:
        return _provider_cache["provider"]

    provider = None
    if _db_available:
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT id, code, name, enabled, is_active, priority, config "
                    "FROM face_match_providers WHERE is_active = TRUE LIMIT 1"
                )
                if row:
                    provider = _row_to_provider(row)
        except Exception as e:
            logger.warning("get_active_face_match_provider failed: %s", e)

    if provider is None:
        provider = next((p for p in _memory_providers if p["is_active"]), _memory_providers[-1])

    _provider_cache["provider"] = provider
    _provider_cache["expires_at"] = now + _PROVIDER_CACHE_TTL
    return provider


async def set_active_face_match_provider(provider_id: int) -> dict:
    """Activate one provider by id; deactivate others."""
    global _memory_providers
    if _db_available:
        pool = await get_pool()
        async with pool.acquire() as conn:
            exists = await conn.fetchval(
                "SELECT id FROM face_match_providers WHERE id = $1 AND enabled = TRUE",
                provider_id,
            )
            if exists is None:
                raise ValueError(f"Provider {provider_id} not found or disabled")
            async with conn.transaction():
                await conn.execute(
                    "UPDATE face_match_providers SET is_active = FALSE, updated_at = NOW() "
                    "WHERE is_active = TRUE"
                )
                await conn.execute(
                    "UPDATE face_match_providers SET is_active = TRUE, updated_at = NOW() "
                    "WHERE id = $1",
                    provider_id,
                )
            row = await conn.fetchrow(
                "SELECT id, code, name, enabled, is_active, priority, config "
                "FROM face_match_providers WHERE id = $1",
                provider_id,
            )
            provider = _row_to_provider(row)
    else:
        found = False
        for p in _memory_providers:
            p["is_active"] = p["id"] == provider_id and p["enabled"]
            if p["is_active"]:
                found = True
                provider = dict(p)
        if not found:
            raise ValueError(f"Provider {provider_id} not found or disabled")

    _provider_cache["provider"] = provider
    _provider_cache["expires_at"] = time.time() + _PROVIDER_CACHE_TTL
    return provider


async def list_enabled_providers_fallback_order() -> list[dict]:
    """Enabled providers ordered for outage fallback: 3 → 1 → 2."""
    preferred = [3, 1, 2]
    all_p = await list_face_match_providers()
    by_id = {p["id"]: p for p in all_p if p.get("enabled")}
    ordered = [by_id[i] for i in preferred if i in by_id]
    # Append any other enabled not in preferred
    for p in all_p:
        if p.get("enabled") and p["id"] not in preferred:
            ordered.append(p)
    return ordered


async def store_faceprint(
    account_id: str,
    embedding: list[float],
    reference_image_b64: Optional[str] = None,
    source_image_hash: Optional[str] = None,
    source: Optional[str] = None,
    face_index: int = 0,
    provider_code: Optional[str] = "internal",
    bbox: Optional[dict | list] = None,
):
    """Store a new faceprint for an account."""
    bbox_json = json.dumps(bbox) if bbox is not None else None

    if _db_available:
        pool = await get_pool()
        async with pool.acquire() as conn:
            if _has_pgvector:
                vec_str = "[" + ",".join(str(v) for v in embedding) + "]"
                await conn.execute(
                    """INSERT INTO faceprints
                       (account_id, embedding, reference_image_b64, source_image_hash,
                        source, face_index, provider_code, bbox)
                       VALUES ($1, $2::vector, $3, $4, $5, $6, $7, $8::jsonb)""",
                    account_id, vec_str, reference_image_b64, source_image_hash,
                    source, face_index, provider_code, bbox_json,
                )
            else:
                await conn.execute(
                    """INSERT INTO faceprints
                       (account_id, embedding, reference_image_b64, source_image_hash,
                        source, face_index, provider_code, bbox)
                       VALUES ($1, $2::double precision[], $3, $4, $5, $6, $7, $8::jsonb)""",
                    account_id, embedding, reference_image_b64, source_image_hash,
                    source, face_index, provider_code, bbox_json,
                )
        return

    _memory_faceprints.setdefault(account_id, []).append({
        "embedding": embedding,
        "reference_image_b64": reference_image_b64,
        "source_image_hash": source_image_hash,
        "source": source,
        "face_index": face_index,
        "provider_code": provider_code,
        "bbox": bbox,
        "created_at": datetime.now(timezone.utc),
    })


async def replace_account_faceprints(
    account_id: str,
    source: str,
    rows: list[dict],
) -> int:
    """Replace all faceprints for account+source with new rows.

    Each row: embedding, reference_image_b64?, face_index?, bbox?, provider_code?
    """
    if _db_available:
        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM faceprints WHERE account_id = $1 AND source = $2",
                    account_id, source,
                )
                for i, row in enumerate(rows):
                    emb = row["embedding"]
                    bbox = row.get("bbox")
                    bbox_json = json.dumps(bbox) if bbox is not None else None
                    face_index = int(row.get("face_index", i))
                    provider_code = row.get("provider_code") or "internal"
                    ref_b64 = row.get("reference_image_b64")
                    if _has_pgvector:
                        vec_str = "[" + ",".join(str(v) for v in emb) + "]"
                        await conn.execute(
                            """INSERT INTO faceprints
                               (account_id, embedding, reference_image_b64, source,
                                face_index, provider_code, bbox)
                               VALUES ($1, $2::vector, $3, $4, $5, $6, $7::jsonb)""",
                            account_id, vec_str, ref_b64, source,
                            face_index, provider_code, bbox_json,
                        )
                    else:
                        await conn.execute(
                            """INSERT INTO faceprints
                               (account_id, embedding, reference_image_b64, source,
                                face_index, provider_code, bbox)
                               VALUES ($1, $2::double precision[], $3, $4, $5, $6, $7::jsonb)""",
                            account_id, emb, ref_b64, source,
                            face_index, provider_code, bbox_json,
                        )
        return len(rows)

    existing = _memory_faceprints.get(account_id, [])
    kept = [fp for fp in existing if fp.get("source") != source]
    for i, row in enumerate(rows):
        kept.append({
            "embedding": row["embedding"],
            "reference_image_b64": row.get("reference_image_b64"),
            "source": source,
            "face_index": int(row.get("face_index", i)),
            "provider_code": row.get("provider_code") or "internal",
            "bbox": row.get("bbox"),
            "created_at": datetime.now(timezone.utc),
        })
    _memory_faceprints[account_id] = kept
    return len(rows)


async def list_faceprints_for_account(
    account_id: str,
    source: Optional[str] = None,
) -> list[dict]:
    """List stored templates for an account (optionally filtered by source)."""
    if _db_available:
        pool = await get_pool()
        async with pool.acquire() as conn:
            if source:
                rows = await conn.fetch(
                    """SELECT id, account_id, embedding, reference_image_b64, source,
                              face_index, provider_code, bbox, created_at
                       FROM faceprints WHERE account_id = $1 AND source = $2
                       ORDER BY face_index, created_at""",
                    account_id, source,
                )
            else:
                rows = await conn.fetch(
                    """SELECT id, account_id, embedding, reference_image_b64, source,
                              face_index, provider_code, bbox, created_at
                       FROM faceprints WHERE account_id = $1
                       ORDER BY source, face_index, created_at""",
                    account_id,
                )
            out = []
            for r in rows:
                emb = r["embedding"]
                if hasattr(emb, "tolist"):
                    emb = emb.tolist()
                elif isinstance(emb, str):
                    # pgvector may come as "[1,2,...]"
                    emb = [float(x) for x in emb.strip("[]").split(",") if x.strip()]
                else:
                    emb = list(emb or [])
                bbox = r["bbox"]
                if isinstance(bbox, str):
                    try:
                        bbox = json.loads(bbox)
                    except Exception:
                        pass
                out.append({
                    "id": r["id"],
                    "account_id": r["account_id"],
                    "embedding": emb,
                    "reference_image_b64": r["reference_image_b64"],
                    "source": r["source"],
                    "face_index": r["face_index"],
                    "provider_code": r["provider_code"],
                    "bbox": bbox,
                    "created_at": r["created_at"],
                })
            return out

    prints = _memory_faceprints.get(account_id, [])
    if source:
        prints = [p for p in prints if p.get("source") == source]
    return list(prints)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    n1 = math.sqrt(sum(x * x for x in a))
    n2 = math.sqrt(sum(y * y for y in b))
    return dot / (n1 * n2 + 1e-8)


async def find_nearest(embedding: list[float], top_k: int = 5) -> list[dict]:
    """ANN / brute-force search for nearest faceprints."""
    if _db_available:
        pool = await get_pool()
        async with pool.acquire() as conn:
            if _has_pgvector:
                vec_str = "[" + ",".join(str(v) for v in embedding) + "]"
                rows = await conn.fetch(
                    """SELECT account_id, 1 - (embedding <=> $1::vector) as similarity
                       FROM faceprints
                       ORDER BY embedding <=> $1::vector
                       LIMIT $2""",
                    vec_str, top_k,
                )
                return [
                    {"account_id": r["account_id"], "similarity": float(r["similarity"])}
                    for r in rows
                ]

            rows = await conn.fetch("SELECT account_id, embedding FROM faceprints")
            scored = []
            for r in rows:
                emb = list(r["embedding"] or [])
                scored.append({"account_id": r["account_id"], "similarity": _cosine(embedding, emb)})
            scored.sort(key=lambda x: x["similarity"], reverse=True)
            return scored[:top_k]

    results = []
    for account_id, prints in _memory_faceprints.items():
        for fp in prints:
            results.append({
                "account_id": account_id,
                "similarity": _cosine(embedding, fp.get("embedding") or []),
            })
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


async def match_embedding_to_account(
    account_id: str,
    embedding: list[float],
    sources: Optional[list[str]] = None,
) -> dict:
    """Best cosine match of live embedding against account templates."""
    templates = await list_faceprints_for_account(account_id)
    if sources:
        templates = [t for t in templates if t.get("source") in sources]

    best = None
    best_sim = -1.0
    for t in templates:
        sim = _cosine(embedding, t.get("embedding") or [])
        if sim > best_sim:
            best_sim = sim
            best = t

    if best is None:
        return {
            "matched": False,
            "cosine_similarity": 0.0,
            "matched_source": None,
            "matched_face_index": None,
            "template_count": 0,
        }
    return {
        "matched": True,  # caller applies threshold
        "cosine_similarity": best_sim,
        "matched_source": best.get("source"),
        "matched_face_index": best.get("face_index"),
        "template_count": len(templates),
        "best_template_id": best.get("id"),
    }


async def get_reference_image_b64(account_id: str) -> Optional[str]:
    """Get the reference image for face matching."""
    if _db_available:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT reference_image_b64 FROM faceprints
                   WHERE account_id = $1 AND reference_image_b64 IS NOT NULL
                   ORDER BY created_at DESC LIMIT 1""",
                account_id,
            )
        if row and row["reference_image_b64"]:
            return row["reference_image_b64"]
        return None

    prints = _memory_faceprints.get(account_id, [])
    for fp in reversed(prints):
        if fp.get("reference_image_b64"):
            return fp["reference_image_b64"]
    return None
