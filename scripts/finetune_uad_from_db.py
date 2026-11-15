"""Fine-tune spatial UAD on labeled phone frames from Postgres (with SPSC).

Usage:
    cd local-api
    ..\\.venv\\Scripts\\python.exe ..\\scripts\\finetune_uad_from_db.py --promote
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "local-api"))
sys.path.insert(0, str(ROOT / "scripts"))
load_dotenv(ROOT / "local-api" / ".env")


async def load_samples(per_class: int, use_spsc: bool) -> list[tuple[np.ndarray, int]]:
    import asyncpg
    from spsc_augment import augment_for_label

    conn = await asyncpg.connect(os.getenv("DATABASE_URL"))
    await conn.execute("SET statement_timeout = '120s'")
    rows = await conn.fetch(
        """
        SELECT ti.image_jpeg,
               CASE
                 WHEN COALESCE(ts.label, ts.auto_label) = 'spoof' THEN 1
                 ELSE 0
               END AS y
        FROM training_images ti
        JOIN training_sessions ts ON ts.id = ti.session_pk
        WHERE ti.kind = 'frame'
          AND COALESCE(ts.label, ts.auto_label) IN ('real', 'spoof')
        ORDER BY ti.id DESC
        """
    )
    await conn.close()

    real: list[np.ndarray] = []
    spoof: list[np.ndarray] = []
    for row in rows:
        arr = np.frombuffer(bytes(row["image_jpeg"]), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        if int(row["y"]) == 1:
            spoof.append(img)
        else:
            real.append(img)

    random.shuffle(real)
    random.shuffle(spoof)
    real = real[:per_class]
    spoof = spoof[:per_class]

    samples: list[tuple[np.ndarray, int]] = []
    for im, y in [(im, 0) for im in real] + [(im, 1) for im in spoof]:
        if use_spsc:
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            pil = augment_for_label(pil, y)
            im = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        samples.append((im, y))
    random.shuffle(samples)
    print(f"loaded real={len(real)} spoof={len(spoof)} total={len(samples)} spsc={use_spsc}")
    return samples


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--no-spsc", action="store_true")
    args = parser.parse_args()

    from app.services import gpu_client

    samples = await load_samples(args.per_class, use_spsc=not args.no_spsc)
    if len(samples) < 8:
        raise SystemExit("not enough labeled frames")

    result = await gpu_client.finetune_uad(
        samples,
        epochs=args.epochs,
        lr=args.lr,
        promote=args.promote,
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
