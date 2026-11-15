#!/usr/bin/env python3
"""Merge face anti-spoofing image datasets into train_uad.py layout.

Expected output:
    <out>/
        real/   # live face images
        spoof/  # print / replay / mask attacks

Usage:
    python scripts/merge_fas_dataset.py \\
        --sources /data/celeba_spoof /data/casia_fasd \\
        --output /data/fas_merged

Heuristics (case-insensitive path/filename keywords):
    real  <- live, real, genuine, bonafide, client
    spoof <- spoof, fake, attack, print, replay, mask, paper, screen
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

REAL_KEYS = ("live", "real", "genuine", "bonafide", "client", "positive")
SPOOF_KEYS = ("spoof", "fake", "attack", "print", "replay", "mask", "paper", "screen", "imposter", "negative")


def classify(path: Path) -> str | None:
    parts = "/".join(p.lower() for p in path.parts)
    name = path.name.lower()
    blob = parts + "/" + name
    # Prefer more specific spoof keywords first
    if any(k in blob for k in SPOOF_KEYS):
        return "spoof"
    if any(k in blob for k in REAL_KEYS):
        return "real"
    return None


def unique_name(src: Path, dest_dir: Path) -> Path:
    digest = hashlib.md5(str(src).encode()).hexdigest()[:10]
    return dest_dir / f"{src.stem}_{digest}{src.suffix.lower()}"


def collect_images(root: Path) -> list[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge FAS image datasets")
    parser.add_argument("--sources", nargs="+", required=True, help="Source dataset roots")
    parser.add_argument("--output", required=True, help="Output root with real/ and spoof/")
    parser.add_argument("--symlink", action="store_true", help="Symlink instead of copy")
    parser.add_argument("--max-per-class", type=int, default=0, help="Cap images per class (0=all)")
    args = parser.parse_args()

    out = Path(args.output)
    real_dir = out / "real"
    spoof_dir = out / "spoof"
    real_dir.mkdir(parents=True, exist_ok=True)
    spoof_dir.mkdir(parents=True, exist_ok=True)

    counts = {"real": 0, "spoof": 0, "skipped": 0}

    for src_root in args.sources:
        root = Path(src_root)
        if not root.is_dir():
            print(f"WARN: missing source {root}")
            continue
        images = collect_images(root)
        print(f"Scanning {root}: {len(images)} images")
        for img in images:
            label = classify(img)
            if label is None:
                counts["skipped"] += 1
                continue
            if args.max_per_class and counts[label] >= args.max_per_class:
                continue
            dest_dir = real_dir if label == "real" else spoof_dir
            dest = unique_name(img, dest_dir)
            if dest.exists():
                continue
            if args.symlink:
                os.symlink(img.resolve(), dest)
            else:
                shutil.copy2(img, dest)
            counts[label] += 1

    print(
        f"Done → {out}\n"
        f"  real={counts['real']}  spoof={counts['spoof']}  skipped={counts['skipped']}"
    )


if __name__ == "__main__":
    main()
