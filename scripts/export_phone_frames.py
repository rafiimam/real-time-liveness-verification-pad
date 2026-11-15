"""Export labeled phone frames from Postgres for UAD / temporal training.

Usage:
    cd local-api
    ..\\.venv\\Scripts\\python.exe ..\\scripts\\export_phone_frames.py --out data/phone_fas
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / "local-api" / ".env")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/phone_fas")
    parser.add_argument("--max-per-session", type=int, default=8)
    args = parser.parse_args()

    out = Path(args.out)
    real_dir = out / "real"
    spoof_dir = out / "spoof"
    real_dir.mkdir(parents=True, exist_ok=True)
    spoof_dir.mkdir(parents=True, exist_ok=True)

    conn = await asyncpg.connect(os.getenv("DATABASE_URL"))
    rows = await conn.fetch(
        """
        SELECT ts.id AS session_pk,
               COALESCE(ts.label, ts.auto_label) AS lab,
               ti.id AS image_id,
               ti.frame_index,
               ti.image_jpeg
        FROM training_images ti
        JOIN training_sessions ts ON ts.id = ti.session_pk
        WHERE ti.kind = 'frame'
          AND COALESCE(ts.label, ts.auto_label) IN ('real', 'spoof')
        ORDER BY ts.id, ti.frame_index NULLS LAST, ti.id
        """
    )
    await conn.close()

    per_session: dict[int, int] = {}
    n_real = n_spoof = 0
    for row in rows:
        pk = int(row["session_pk"])
        count = per_session.get(pk, 0)
        if count >= args.max_per_session:
            continue
        per_session[pk] = count + 1
        lab = row["lab"]
        dest = real_dir if lab == "real" else spoof_dir
        path = dest / f"phone_{pk}_{row['image_id']}.jpg"
        path.write_bytes(bytes(row["image_jpeg"]))
        if lab == "real":
            n_real += 1
        else:
            n_spoof += 1

    print(f"exported real={n_real} spoof={n_spoof} -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
