"""Training script for UAD (Unified Attack Detection) module.

Trains the Swin backbone + UAD head on face anti-spoofing datasets.
Binary classification: real (0) vs spoof (1).
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "gpu-server"))
from app.models.swin_backbone import SwinLivenessModel


class SpoofDataset(Dataset):
    """Generic spoof dataset loader.

    Expects directory structure:
        root/
            real/
                img001.jpg
                ...
            spoof/
                img001.jpg
                ...
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
        face_crop: bool = False,
        use_spsc: bool = True,
    ):
        self.transform = transform
        self.face_crop = face_crop
        self.use_spsc = use_spsc
        self._detector = None
        self.samples = []

        real_dir = os.path.join(root_dir, "real")
        spoof_dir = os.path.join(root_dir, "spoof")

        if os.path.isdir(real_dir):
            for f in os.listdir(real_dir):
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.samples.append((os.path.join(real_dir, f), 0))

        if os.path.isdir(spoof_dir):
            for f in os.listdir(spoof_dir):
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.samples.append((os.path.join(spoof_dir, f), 1))

    def _maybe_crop(self, img: Image.Image) -> Image.Image:
        if not self.face_crop:
            return img
        if self._detector is None:
            from app.models.mtcnn_detector import get_detector
            self._detector = get_detector()
        import numpy as np
        import cv2

        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        crop = self._detector.get_best_face(bgr)
        if crop is None:
            return img
        arr = ((crop.detach().cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self._maybe_crop(img)
        if self.use_spsc:
            from spsc_augment import augment_for_label
            img = augment_for_label(img, label)
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(input_size: int = 224, train: bool = True):
    ops = [transforms.Resize((input_size, input_size))]
    if train:
        ops.extend([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transforms.Compose(ops)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = SwinLivenessModel(pretrained=True).to(device)

    # Freeze backbone initially, train only UAD head
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.uad.parameters():
        param.requires_grad = True

    transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    dataset = SpoofDataset(
        args.data_dir,
        transform=transform,
        face_crop=args.face_crop,
        use_spsc=not args.no_spsc,
    )
    print(f"Dataset size: {len(dataset)} samples (face_crop={args.face_crop} spsc={not args.no_spsc})")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    # Validation without stochastic SPSC: wrap indices with a plain loader path
    val_base = SpoofDataset(
        args.data_dir,
        transform=val_transform,
        face_crop=args.face_crop,
        use_spsc=False,
    )
    val_indices = val_set.indices if hasattr(val_set, "indices") else list(range(train_size, len(dataset)))
    val_set = torch.utils.data.Subset(val_base, val_indices)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    optimizer = torch.optim.AdamW(model.uad.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCELoss()

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            output = model(images)
            spoof_scores = output["spoof_score"]
            loss = criterion(spoof_scores, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (spoof_scores > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device)
                output = model(images)
                preds = (output["spoof_score"] > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, "swin_uad_best.pth"))
            print(f"  Saved best model (val_acc={val_acc:.4f})")

    # Phase 2: light Stage-3 unfreeze + UAD (prefer texture layers over full backbone)
    if args.finetune_epochs > 0:
        print("\n--- Phase 2: Fine-tuning Stage-3 + UAD ---")
        for param in model.backbone.parameters():
            param.requires_grad = False
        # Unfreeze stage-3 blocks when resolvable
        blocks = getattr(model, "_stage3_blocks", None)
        if blocks is not None:
            for param in blocks.parameters():
                param.requires_grad = True
            trainable = list(blocks.parameters()) + list(model.uad.parameters())
        else:
            for param in model.backbone.parameters():
                param.requires_grad = True
            trainable = list(model.parameters())
        for param in model.uad.parameters():
            param.requires_grad = True
        optimizer = torch.optim.AdamW(trainable, lr=args.lr * 0.1, weight_decay=1e-4)

        for epoch in range(args.finetune_epochs):
            model.train()
            total_loss = 0.0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.float().to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output["spoof_score"], labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Fine-tune Epoch {epoch+1}/{args.finetune_epochs} | Loss: {total_loss/len(train_loader):.4f}")

        torch.save(model.state_dict(), os.path.join(args.output_dir, "swin_uad_finetuned.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UAD spoof detection model")
    parser.add_argument("--data-dir", required=True, help="Path to dataset (real/ and spoof/ subdirs)")
    parser.add_argument("--output-dir", default="./weights", help="Output directory for weights")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--face-crop",
        action="store_true",
        help="Run MTCNN face crop before transform (for full-scene FAS images)",
    )
    parser.add_argument(
        "--no-spsc",
        action="store_true",
        help="Disable SPSC / phone-domain augmentation",
    )
    args = parser.parse_args()
    train(args)
