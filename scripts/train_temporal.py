"""Training script for 3D ResNet temporal liveness model.

Fine-tunes r3d_18 on video datasets (FakeAVCeleb, DFDC, FF++) for
binary classification: real video (0) vs fake/replay (1).
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-server"))
from app.models.temporal_3dcnn import TemporalLivenessModel, CHUNK_FRAMES


class VideoChunkDataset(Dataset):
    """Video dataset that extracts 30-frame chunks.

    Expects:
        root/
            real/
                video001.mp4
                ...
            fake/
                video001.mp4
                ...
    """

    def __init__(self, root_dir: str, frames_per_clip: int = 30, clips_per_video: int = 3):
        self.frames_per_clip = frames_per_clip
        self.clips_per_video = clips_per_video
        self.samples = []

        for label, subdir in [(0, "real"), (1, "fake")]:
            dir_path = os.path.join(root_dir, subdir)
            if not os.path.isdir(dir_path):
                continue
            for f in os.listdir(dir_path):
                if f.lower().endswith((".mp4", ".avi", ".mov")):
                    self.samples.append((os.path.join(dir_path, f), label))

    def __len__(self):
        return len(self.samples) * self.clips_per_video

    def __getitem__(self, idx):
        video_idx = idx // self.clips_per_video
        clip_idx = idx % self.clips_per_video
        path, label = self.samples[video_idx]

        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= self.frames_per_clip:
            start = 0
        else:
            segment_len = total_frames // self.clips_per_video
            start = clip_idx * segment_len

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(self.frames_per_clip):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (112, 112))
            frames.append(frame)
        cap.release()

        # Pad if needed
        while len(frames) < self.frames_per_clip:
            frames.append(np.zeros((112, 112, 3), dtype=np.uint8))

        # (T, H, W, C) -> (C, T, H, W)
        tensor = torch.from_numpy(np.array(frames, dtype=np.float32) / 255.0)
        tensor = tensor.permute(3, 0, 1, 2)

        return tensor, label


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    dataset = VideoChunkDataset(args.data_dir, frames_per_clip=CHUNK_FRAMES)
    print(f"Dataset: {len(dataset)} clips from {len(dataset.samples)} videos")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = TemporalLivenessModel(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.BCELoss()

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for clips, labels in train_loader:
            clips = clips.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            scores = model(clips)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (scores > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for clips, labels in val_loader:
                clips = clips.to(device)
                labels = labels.float().to(device)
                scores = model(clips)
                preds = (scores > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, "temporal_3dcnn_best.pth"))
            print(f"  Saved best model (val_acc={val_acc:.4f})")

    torch.save(model.state_dict(), os.path.join(args.output_dir, "temporal_3dcnn_final.pth"))
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D ResNet temporal liveness model")
    parser.add_argument("--data-dir", required=True, help="Path to video dataset (real/ and fake/ subdirs)")
    parser.add_argument("--output-dir", default="./weights")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train(args)
