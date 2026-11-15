"""Background incremental training from Postgres training samples."""

from __future__ import annotations

import asyncio
import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BACKGROUND_TRAINING = os.getenv("BACKGROUND_TRAINING", "true").strip().lower() in (
    "1", "true", "yes", "on",
)
TRAIN_INTERVAL_SEC = int(os.getenv("TRAIN_INTERVAL_SEC", "300"))
TRAIN_MIN_SAMPLES = int(os.getenv("TRAIN_MIN_SAMPLES", "8"))
TRAIN_BATCH_LIMIT = int(os.getenv("TRAIN_BATCH_LIMIT", "64"))
BG_TRAIN_MAX_IMAGES = int(os.getenv("BG_TRAIN_MAX_IMAGES", "96"))
BG_TRAIN_EPOCHS = int(os.getenv("BG_TRAIN_EPOCHS", "2"))


async def _run_incremental_uad(batch: list[dict]) -> dict:
    """Export labeled frames and fine-tune UAD on the GPU inference server."""
    from app.services import gpu_client

    real_imgs: list[np.ndarray] = []
    spoof_imgs: list[np.ndarray] = []
    for sample in batch:
        bucket = real_imgs if sample["label"] == "real" else spoof_imgs
        for img in sample["images"]:
            arr = np.frombuffer(img["jpeg"], np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            bucket.append(bgr)

    # Cap and balance so we don't ship huge payloads
    per_class = max(2, BG_TRAIN_MAX_IMAGES // 2)
    real_imgs = real_imgs[:per_class]
    spoof_imgs = spoof_imgs[:per_class]

    metrics = {
        "n_real_images": len(real_imgs),
        "n_spoof_images": len(spoof_imgs),
        "sessions": len(batch),
        "status": "exported",
    }
    print(
        f"[BG-TRAIN] exported batch sessions={len(batch)} "
        f"real_imgs={len(real_imgs)} spoof_imgs={len(spoof_imgs)}",
        flush=True,
    )

    if len(real_imgs) < 2 or len(spoof_imgs) < 2:
        metrics["status"] = "skipped_unbalanced"
        print("[BG-TRAIN] need both real and spoof samples — skip fine-tune", flush=True)
        return metrics

    samples = [(im, 0) for im in real_imgs] + [(im, 1) for im in spoof_imgs]
    try:
        print(
            f"[BG-TRAIN] sending {len(samples)} images to GPU "
            f"{gpu_client.GPU_SERVER_URL}/spoof/finetune ...",
            flush=True,
        )
        result = await gpu_client.finetune_uad(samples, epochs=BG_TRAIN_EPOCHS)
        metrics.update(result)
        status = result.get("status", "unknown")
        if status == "trained":
            metrics["status"] = "trained_remote"
            print(
                f"[BG-TRAIN] GPU fine-tune done loss={result.get('mean_loss')} "
                f"weights={result.get('weights')}",
                flush=True,
            )
        else:
            metrics["status"] = f"remote_{status}"
            print(f"[BG-TRAIN] GPU train status={status} detail={result}", flush=True)
    except Exception as e:
        logger.exception("Remote GPU incremental train failed")
        metrics["status"] = f"error:{type(e).__name__}"
        metrics["error"] = str(e)
        print(f"[BG-TRAIN] error: {e}", flush=True)

    return metrics


async def training_loop(stop_event: asyncio.Event) -> None:
    from app.db import training_store

    print(
        f"[BG-TRAIN] loop started interval={TRAIN_INTERVAL_SEC}s min_samples={TRAIN_MIN_SAMPLES}",
        flush=True,
    )
    while not stop_event.is_set():
        try:
            if BACKGROUND_TRAINING:
                batch = await training_store.fetch_untrained_batch(TRAIN_BATCH_LIMIT)
                print(f"[BG-TRAIN] unused batch size={len(batch)}", flush=True)
                if len(batch) >= TRAIN_MIN_SAMPLES:
                    labels = {s["label"] for s in batch}
                    if labels >= {"real", "spoof"}:
                        metrics = await _run_incremental_uad(batch)
                        if metrics.get("status") == "trained_remote":
                            await training_store.mark_trained(
                                [s["id"] for s in batch], metrics
                            )
                        else:
                            print(
                                f"[BG-TRAIN] not marking trained (status={metrics.get('status')})",
                                flush=True,
                            )
                    else:
                        print(f"[BG-TRAIN] waiting for both classes, have={labels}", flush=True)
                else:
                    stats = await training_store.training_stats()
                    print(f"[BG-TRAIN] not enough unused samples yet stats={stats}", flush=True)
        except Exception:
            logger.exception("Background training iteration failed")

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=TRAIN_INTERVAL_SEC)
        except asyncio.TimeoutError:
            pass

    print("[BG-TRAIN] loop stopped", flush=True)
