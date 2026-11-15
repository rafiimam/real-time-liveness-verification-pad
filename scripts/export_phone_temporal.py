"""Export training videos from Postgres to a folder for GPU temporal training."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / "local-api" / ".env")


async def main() -> None:
    out = Path(os.getenv("EXPORT_DIR", "data/phone_temporal"))
    real_dir = out / "real"
    fake_dir = out / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    conn = await asyncpg.connect(os.getenv("DATABASE_URL"))
    rows = await conn.fetch(
        """
        SELECT s.id, s.label, v.video_mp4
        FROM training_sessions s
        JOIN training_videos v ON v.session_pk = s.id
        WHERE s.label IN ('real', 'spoof')
        ORDER BY s.id
        """
    )
    n_real = n_fake = 0
    for row in rows:
        label = row["label"]
        dest_dir = real_dir if label == "real" else fake_dir
        path = dest_dir / f"phone_{row['id']}.mp4"
        path.write_bytes(bytes(row["video_mp4"]))
        if label == "real":
            n_real += 1
        else:
            n_fake += 1
        print(f"wrote {path} ({label})")
    await conn.close()
    print(f"exported real={n_real} spoof/fake={n_fake} -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
