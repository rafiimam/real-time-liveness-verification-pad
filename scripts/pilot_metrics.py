"""Pilot FAR/FRR metrics on labeled training sessions.

Ground-truth labels from training_sessions.label (real|spoof).
Re-runs screen/spatial/temporal gates (no face-match) and reports:

  FRR = real sessions wrongly rejected (false reject)
  FAR = spoof sessions wrongly approved (false accept)

Usage (from local-api/):
  .venv\\Scripts\\python.exe ..\\scripts\\pilot_metrics.py
  .venv\\Scripts\\python.exe ..\\scripts\\pilot_metrics.py --ids 4,5,7,24,25,26,27,28
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import httpx
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "local-api"))
load_dotenv(ROOT / "local-api" / ".env")

import asyncpg  # noqa: E402

from app.services.quality import filter_sharp_frames, select_spatial_samples  # noqa: E402
from app.services.screen_replay import analyze_screen_replay  # noqa: E402

GPU = os.getenv("GPU_SERVER_URL", "http://localhost:8000")
SPATIAL_THR = float(os.getenv("SPATIAL_LIVENESS_THRESHOLD", "0.5"))
TEMPORAL_THR = float(os.getenv("TEMPORAL_LIVENESS_THRESHOLD", "0.45"))
REQUIRE_SPATIAL = os.getenv("REQUIRE_SPATIAL", "true").strip().lower() in ("1", "true", "yes", "on")
REQUIRE_TEMPORAL = os.getenv("REQUIRE_TEMPORAL", "true").strip().lower() in ("1", "true", "yes", "on")


async def eval_session(conn, sid: int, client: httpx.AsyncClient) -> dict:
    row = await conn.fetchrow(
        "SELECT id, label, result FROM training_sessions WHERE id=$1", sid
    )
    if not row:
        return {"id": sid, "error": "missing"}
    vid = await conn.fetchrow(
        "SELECT video_mp4 FROM training_videos WHERE session_pk=$1", sid
    )
    if not vid or not vid["video_mp4"]:
        return {"id": sid, "label": row["label"], "error": "no_video"}

    path = Path(tempfile.gettempdir()) / f"pilot_metrics_{sid}.mp4"
    path.write_bytes(bytes(vid["video_mp4"]))
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()

    screen = analyze_screen_replay(frames)
    screen_ok = not screen["is_screen_replay"]

    spatial_samples = select_spatial_samples(frames, 10)
    spoofs = []
    for f in spatial_samples:
        ok, buf = cv2.imencode(".jpg", f)
        if not ok:
            continue
        import base64

        r = await client.post(
            f"{GPU}/spoof/frame",
            json={"image_base64": base64.b64encode(buf.tobytes()).decode()},
        )
        spoofs.append(float(r.json()["spoof_score"]))
    mean_spoof = float(np.mean(spoofs)) if spoofs else 1.0
    spatial_ok = mean_spoof < SPATIAL_THR

    sharp = filter_sharp_frames(frames)
    step = max(1, len(sharp) // 60) if sharp else 1
    temporal_frames = (sharp or frames)[::step][:60]
    import base64

    b64s = []
    for f in temporal_frames:
        ok, buf = cv2.imencode(".jpg", f)
        if ok:
            b64s.append(base64.b64encode(buf.tobytes()).decode())
    temporal = (
        await client.post(f"{GPU}/spoof/temporal", json={"frames_base64": b64s})
    ).json()
    temp_score = float(temporal.get("liveness_score", 0.0))
    temporal_ok = temp_score > TEMPORAL_THR

    live_ok = True
    if REQUIRE_SPATIAL:
        live_ok = live_ok and spatial_ok
    live_ok = live_ok and screen_ok
    if REQUIRE_TEMPORAL:
        live_ok = live_ok and temporal_ok

    label = (row["label"] or "").lower()
    return {
        "id": sid,
        "label": label,
        "approve_liveness": live_ok,
        "spatial_spoof": mean_spoof,
        "spatial_ok": spatial_ok,
        "temporal": temp_score,
        "temporal_ok": temporal_ok,
        "screen": screen["screen_spoof_score"],
        "screen_ok": screen_ok,
        "n_frames": len(frames),
    }


async def main():
    parser = argparse.ArgumentParser(description="Pilot FAR/FRR on labeled sessions")
    parser.add_argument(
        "--ids",
        default="",
        help="Comma-separated session PKs (default: all labeled real/spoof with video)",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "docs" / "pilot_metrics_latest.json"),
        help="Write JSON report path",
    )
    args = parser.parse_args()

    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    if args.ids.strip():
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
    else:
        rows = await conn.fetch(
            """
            SELECT s.id FROM training_sessions s
            JOIN training_videos v ON v.session_pk = s.id
            WHERE lower(s.label) IN ('real', 'spoof')
            ORDER BY s.id
            """
        )
        ids = [int(r["id"]) for r in rows]

    results = []
    async with httpx.AsyncClient(timeout=180.0) as client:
        for sid in ids:
            print(f"Evaluating session {sid}…", flush=True)
            try:
                results.append(await eval_session(conn, sid, client))
            except Exception as e:
                results.append({"id": sid, "error": f"{type(e).__name__}: {e}"})
                print(f"  ERROR {e}", flush=True)

    await conn.close()

    reals = [r for r in results if r.get("label") == "real" and "approve_liveness" in r]
    spoofs = [r for r in results if r.get("label") == "spoof" and "approve_liveness" in r]
    fr_count = sum(1 for r in reals if not r["approve_liveness"])
    fa_count = sum(1 for r in spoofs if r["approve_liveness"])
    frr = fr_count / len(reals) if reals else None
    far = fa_count / len(spoofs) if spoofs else None

    report = {
        "n_real": len(reals),
        "n_spoof": len(spoofs),
        "false_rejects": fr_count,
        "false_accepts": fa_count,
        "FRR": frr,
        "FAR": far,
        "thresholds": {
            "spatial": SPATIAL_THR,
            "temporal": TEMPORAL_THR,
            "require_spatial": REQUIRE_SPATIAL,
            "require_temporal": REQUIRE_TEMPORAL,
        },
        "sessions": results,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n========== PILOT METRICS ==========")
    print(f"real={len(reals)} spoof={len(spoofs)}")
    print(f"false_rejects={fr_count}  FRR={frr}")
    print(f"false_accepts={fa_count}  FAR={far}")
    print(f"wrote {out}")
    for r in results:
        if r.get("error"):
            print(f"  id={r['id']} ERROR {r['error']}")
            continue
        pred = "LIVE" if r["approve_liveness"] else "REJECT"
        mark = "OK" if (
            (r["label"] == "real" and r["approve_liveness"])
            or (r["label"] == "spoof" and not r["approve_liveness"])
        ) else "BAD"
        print(
            f"  {mark} id={r['id']} label={r['label']} pred={pred} "
            f"screen={r['screen']:.3f} spat={r['spatial_spoof']:.3f} temp={r['temporal']:.3f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
