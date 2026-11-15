#!/bin/bash
# Dataset download helper for liveness model training.
# NOTE: Most academic datasets require registration. This script provides
# structure and placeholder URLs. Replace with actual download links after
# obtaining access from dataset authors.

set -e

DATA_DIR="${1:-./data}"
mkdir -p "$DATA_DIR"

echo "=== Liveness/Anti-Spoofing Datasets ==="

# CelebA-Spoof (request access at https://github.com/ZhangYuanhan-AI/CelebA-Spoof)
echo "[1/5] CelebA-Spoof — requires manual download after access approval"
mkdir -p "$DATA_DIR/celeba_spoof"

# CASIA-FASD (http://www.cbsr.ia.ac.cn/english/FASDB.asp)
echo "[2/5] CASIA-FASD — requires registration at CBSR"
mkdir -p "$DATA_DIR/casia_fasd"

# Replay-Attack (https://www.idiap.ch/en/dataset/replayattack)
echo "[3/5] Replay-Attack — requires Idiap registration"
mkdir -p "$DATA_DIR/replay_attack"

# OULU-NPU (https://sites.google.com/site/aborzhao/dataset)
echo "[4/5] OULU-NPU — requires email request to authors"
mkdir -p "$DATA_DIR/oulu_npu"

# SiW (http://cvlab.cse.msu.edu/siw-spoof-in-the-wild-database.html)
echo "[5/5] SiW — requires MSU CVLAB agreement"
mkdir -p "$DATA_DIR/siw"

echo ""
echo "=== Face Recognition Datasets ==="

# MS1MV2 (cleaned MS-Celeb-1M)
echo "[1/2] MS1MV2 — download from InsightFace model zoo"
mkdir -p "$DATA_DIR/ms1mv2"

# Glint360K
echo "[2/2] Glint360K — download from InsightFace"
mkdir -p "$DATA_DIR/glint360k"

echo ""
echo "=== Deepfake/Temporal Datasets ==="

# FakeAVCeleb
echo "[1/3] FakeAVCeleb — https://github.com/ArshalanQuasain/FakeAVCeleb"
mkdir -p "$DATA_DIR/fakeavceleb"

# DFDC (Deepfake Detection Challenge)
echo "[2/3] DFDC — https://ai.meta.com/datasets/dfdc/"
mkdir -p "$DATA_DIR/dfdc"

# FaceForensics++
echo "[3/3] FaceForensics++ — https://github.com/ondyari/FaceForensics"
mkdir -p "$DATA_DIR/faceforensics"

echo ""
echo "Dataset directories created at: $DATA_DIR"
echo "Please download datasets manually after obtaining access."
