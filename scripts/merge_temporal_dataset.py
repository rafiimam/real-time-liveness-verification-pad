#!/usr/bin/env python3
"""Merge temporal (video) datasets into train_temporal.py layout.

Expected output:
    <out>/
        real/   # genuine webcam / live videos
        fake/   # replay / deepfake videos

Usage:
    python scripts/merge_temporal_dataset.py \\
        --sources /data/replay_attack /data/fakeavceleb \\
        --output /data/temporal_merged
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

REAL_KEYS = ("live", "real", "genuine", "bonafide", "client", "positive")
FAKE_KEYS = ("fake", "spoof", "attack", "replay", "deepfake", "manipulated", "print", "screen")


def classify(path: Path) -> str | None:
    blob = "/".join(p.lower() for p in path.parts) + "/" + path.name.lower()
    if any(k in blob for k in FAKE_KEYS):
        return "fake"
    if any(k in blob for k in REAL_KEYS):
        return "real"
    return None


def unique_name(src: Path, dest_dir: Path) -> Path:
    digest = hashlib.md5(str(src).encode()).hexdigest()[:10]
    return dest_dir / f"{src.stem}_{digest}{src.suffix.lower()}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge temporal video datasets")
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--symlink", action="store_true")
    parser.add_argument("--max-per-class", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.output)
    real_dir = out / "real"
    fake_dir = out / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    counts = {"real": 0, "fake": 0, "skipped": 0}

    for src_root in args.sources:
        root = Path(src_root)
        if not root.is_dir():
            print(f"WARN: missing source {root}")
            continue
        videos = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
        print(f"Scanning {root}: {len(videos)} videos")
        for vid in videos:
            label = classify(vid)
            if label is None:
                counts["skipped"] += 1
                continue
            if args.max_per_class and counts[label] >= args.max_per_class:
                continue
            dest_dir = real_dir if label == "real" else fake_dir
            dest = unique_name(vid, dest_dir)
            if dest.exists():
                continue
            if args.symlink:
                os.symlink(vid.resolve(), dest)
            else:
                shutil.copy2(vid, dest)
            counts[label] += 1

    print(
        f"Done → {out}\n"
        f"  real={counts['real']}  fake={counts['fake']}  skipped={counts['skipped']}"
    )


if __name__ == "__main__":
    main()
