"""Training sample storage in PostgreSQL (no local folder dependency)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


async def ensure_training_tables(conn) -> None:
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS training_sessions (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(64) NOT NULL,
            account_id VARCHAR(64) NOT NULL,
            source VARCHAR(64),
            auto_label VARCHAR(16),
            label VARCHAR(16),
            temporal_label VARCHAR(16),
            result JSONB,
            face_match_meta JSONB,
            frame_count INT,
            used_in_training_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_training_sessions_label
        ON training_sessions (label);
    """)
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_training_sessions_unused
        ON training_sessions (used_in_training_at)
        WHERE used_in_training_at IS NULL;
    """)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS training_images (
            id SERIAL PRIMARY KEY,
            session_pk INT REFERENCES training_sessions(id) ON DELETE CASCADE,
            kind VARCHAR(32) NOT NULL,
            frame_index INT,
            blur_score DOUBLE PRECISION,
            width INT,
            height INT,
            image_jpeg BYTEA NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_training_images_session
        ON training_images (session_pk);
    """)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS training_videos (
            id SERIAL PRIMARY KEY,
            session_pk INT REFERENCES training_sessions(id) ON DELETE CASCADE,
            kind VARCHAR(32) NOT NULL,
            fps INT,
            frame_count INT,
            video_mp4 BYTEA NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(64) NOT NULL,
            sample_count INT,
            metrics JSONB,
            notes TEXT,
            started_at TIMESTAMPTZ,
            finished_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def encode_mp4(frames: list[np.ndarray], fps: int = 8) -> bytes:
    import tempfile
    from pathlib import Path

    if not frames:
        return b""
    h, w = frames[0].shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fr in frames:
        if fr.shape[0] != h or fr.shape[1] != w:
            fr = cv2.resize(fr, (w, h))
        writer.write(fr)
    writer.release()
    data = Path(path).read_bytes()
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass
    return data


async def store_training_session(
    *,
    session_id: str,
    account_id: str,
    source: str,
    frames: list[np.ndarray],
    reference_image: np.ndarray | None,
    face_match_frame: np.ndarray | None,
    face_match_meta: dict,
    result: dict,
    auto_label: str,
    temporal_label: str,
    nid_image: np.ndarray | None = None,
    cbs_image: np.ndarray | None = None,
) -> Optional[int]:
    """Insert one training session + images + video into Postgres. Returns session PK."""
    from app.db.pgvector import get_pool, _db_available

    if not _db_available:
        logger.warning("DB unavailable — cannot store training session")
        return None

    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                INSERT INTO training_sessions
                    (session_id, account_id, source, auto_label, label, temporal_label,
                     result, face_match_meta, frame_count)
                VALUES ($1,$2,$3,$4,$4,$5,$6::jsonb,$7::jsonb,$8)
                RETURNING id
                """,
                session_id,
                account_id,
                source,
                auto_label,
                temporal_label,
                json.dumps(result, default=str),
                json.dumps(face_match_meta, default=str),
                len(frames),
            )
            pk = int(row["id"])

            async def _ins_image(kind: str, img: np.ndarray, idx: int | None = None, blur: float | None = None):
                if img is None or img.size == 0:
                    return
                h, w = img.shape[:2]
                await conn.execute(
                    """
                    INSERT INTO training_images
                        (session_pk, kind, frame_index, blur_score, width, height, image_jpeg)
                    VALUES ($1,$2,$3,$4,$5,$6,$7)
                    """,
                    pk, kind, idx, blur, w, h, encode_jpeg(img),
                )

            if reference_image is not None:
                await _ins_image("reference", reference_image)
            if face_match_frame is not None:
                await _ins_image(
                    "face_match",
                    face_match_frame,
                    face_match_meta.get("frame_index"),
                    face_match_meta.get("blur_score"),
                )
            if nid_image is not None:
                await _ins_image("nid", nid_image)
            if cbs_image is not None:
                await _ins_image("cbs_sign", cbs_image)

            # Keep a sparse set of frames for FAS training (not every frame)
            if frames:
                step = max(1, len(frames) // 12)
                for i, fr in enumerate(frames[::step][:12]):
                    await _ins_image("frame", fr, i * step)

            video_bytes = encode_mp4(frames, fps=8)
            if video_bytes:
                await conn.execute(
                    """
                    INSERT INTO training_videos (session_pk, kind, fps, frame_count, video_mp4)
                    VALUES ($1,'clip',8,$2,$3)
                    """,
                    pk, len(frames), video_bytes,
                )

    print(
        f"[TRAINING-DB] stored session_pk={pk} session={session_id} "
        f"label={auto_label}/{temporal_label} frames={len(frames)} "
        f"nid={'yes' if nid_image is not None else 'no'} "
        f"cbs={'yes' if cbs_image is not None else 'no'}",
        flush=True,
    )
    logger.info("Stored training session pk=%s label=%s", pk, auto_label)
    return pk


async def set_session_label(session_id: str, label: str) -> bool:
    from app.db.pgvector import get_pool, _db_available

    if not _db_available:
        return False
    label = label.lower().strip()
    if label not in ("real", "spoof"):
        raise ValueError("label must be real or spoof")
    temporal = "fake" if label == "spoof" else "real"
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE training_sessions
            SET label=$2, temporal_label=$3, used_in_training_at=NULL
            WHERE session_id=$1
            """,
            session_id, label, temporal,
        )
    return result.endswith("1") or "UPDATE" in result


async def fetch_untrained_batch(limit: int = 64) -> list[dict[str, Any]]:
    """Return untrained sessions with face_match / frame images for fine-tuning."""
    from app.db.pgvector import get_pool, _db_available

    if not _db_available:
        return []
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, session_id, label, temporal_label
            FROM training_sessions
            WHERE used_in_training_at IS NULL
              AND label IN ('real','spoof')
            ORDER BY id ASC
            LIMIT $1
            """,
            limit,
        )
        batch = []
        for r in rows:
            imgs = await conn.fetch(
                """
                SELECT kind, image_jpeg FROM training_images
                WHERE session_pk=$1 AND kind IN ('face_match','frame','reference')
                ORDER BY id ASC
                """,
                r["id"],
            )
            batch.append({
                "id": r["id"],
                "session_id": r["session_id"],
                "label": r["label"],
                "temporal_label": r["temporal_label"],
                "images": [{"kind": i["kind"], "jpeg": bytes(i["image_jpeg"])} for i in imgs],
            })
        return batch


async def mark_trained(session_pks: list[int], metrics: dict | None = None) -> None:
    from app.db.pgvector import get_pool, _db_available

    if not _db_available or not session_pks:
        return
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE training_sessions
            SET used_in_training_at=NOW()
            WHERE id = ANY($1::int[])
            """,
            session_pks,
        )
        await conn.execute(
            """
            INSERT INTO training_runs (model_name, sample_count, metrics, started_at, finished_at, notes)
            VALUES ('swin_uad_incremental', $1, $2::jsonb, NOW(), NOW(), 'background worker')
            """,
            len(session_pks),
            json.dumps(metrics or {}, default=str),
        )


async def training_stats() -> dict:
    from app.db.pgvector import get_pool, _db_available

    if not _db_available:
        return {"available": False}
    pool = await get_pool()
    async with pool.acquire() as conn:
        total = await conn.fetchval("SELECT COUNT(*) FROM training_sessions")
        by_label = await conn.fetch(
            "SELECT label, COUNT(*) c FROM training_sessions GROUP BY label ORDER BY 1"
        )
        unused = await conn.fetchval(
            "SELECT COUNT(*) FROM training_sessions WHERE used_in_training_at IS NULL"
        )
        imgs = await conn.fetchval("SELECT COUNT(*) FROM training_images")
        vids = await conn.fetchval("SELECT COUNT(*) FROM training_videos")
    return {
        "available": True,
        "sessions": total,
        "unused": unused,
        "images": imgs,
        "videos": vids,
        "by_label": {r["label"]: r["c"] for r in by_label},
    }
