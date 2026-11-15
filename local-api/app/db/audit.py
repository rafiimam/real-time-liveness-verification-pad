"""Audit trail logging for compliance."""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_memory_audit: list[dict] = []


async def log_verification(
    session_id: str,
    account_id: str,
    result: dict[str, Any],
    *,
    action: str = "verification",
):
    """Log a verification decision to the audit table (or memory fallback)."""
    # Enrich with reason codes for compliance / FAR-FRR reviews
    enriched = dict(result)
    enriched.setdefault("reason_codes", result.get("reason_codes") or [])
    enriched.setdefault("capture_trust", result.get("capture_trust"))

    try:
        from app.db.pgvector import get_pool, _db_available
        if _db_available:
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO audit_log (session_id, account_id, action, result)
                       VALUES ($1, $2, $3, $4::jsonb)""",
                    session_id, account_id, action, json.dumps(enriched),
                )
            logger.info(
                "AUDIT session=%s account=%s approved=%s reasons=%s",
                session_id,
                account_id,
                enriched.get("is_approved"),
                enriched.get("reason_codes"),
            )
            return
    except Exception as e:
        logger.warning("Audit DB write failed: %s", e)

    _memory_audit.append({
        "session_id": session_id,
        "account_id": account_id,
        "action": action,
        "result": enriched,
    })
    logger.info(
        "AUDIT(memory) session=%s account=%s approved=%s reasons=%s",
        session_id,
        account_id,
        enriched.get("is_approved"),
        enriched.get("reason_codes"),
    )


async def get_audit_trail(account_id: str, limit: int = 50) -> list[dict]:
    """Retrieve audit trail for an account."""
    try:
        from app.db.pgvector import get_pool, _db_available
        if _db_available:
            pool = await get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT session_id, action, result, created_at
                       FROM audit_log WHERE account_id = $1
                       ORDER BY created_at DESC LIMIT $2""",
                    account_id, limit,
                )
            return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("Audit DB read failed: %s", e)

    return [a for a in reversed(_memory_audit) if a["account_id"] == account_id][:limit]
