"""Training script for ArcFace FRM head on face recognition datasets.

Fine-tunes the Swin backbone's FRM projection head with ArcFace loss
on MS1MV2 or Glint360K for discriminative 512-D face embeddings.
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
from app.models.swin_backbone import SwinLivenessModel, ArcFaceLoss


class FaceIdentityDataset(Dataset):
    """Face identity dataset.

    Expects directory structure:
        root/
            person_001/
                img001.jpg
                ...
            person_002/
                img001.jpg
                ...
    """

    def __init__(self, root_dir: str, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        for idx, person_dir in enumerate(sorted(os.listdir(root_dir))):
            person_path = os.path.join(root_dir, person_dir)
            if not os.path.isdir(person_path):
                continue
            self.class_to_idx[person_dir] = idx
            for f in os.listdir(person_path):
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.samples.append((os.path.join(person_path, f), idx))

        self.num_classes = len(self.class_to_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = FaceIdentityDataset(args.data_dir, transform=transform)
    print(f"Dataset: {len(dataset)} images, {dataset.num_classes} identities")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    model = SwinLivenessModel(pretrained=True).to(device)
    arcface_loss = ArcFaceLoss(
        embedding_dim=512,
        num_classes=dataset.num_classes,
        s=64.0,
        m=0.5,
    ).to(device)

    # Freeze UAD, train backbone + FRM
    for param in model.uad.parameters():
        param.requires_grad = False

    optimizer = torch.optim.SGD(
        list(model.backbone.parameters()) + list(model.frm_pool.parameters()) +
        list(model.frm_proj.parameters()) + list(arcface_loss.parameters()),
        lr=args.lr, momentum=0.9, weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 14, 18], gamma=0.1)
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            embeddings = output["embedding"]
            logits = arcface_loss(embeddings, labels)
            loss = ce_loss(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        acc = correct / total
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(loader):.4f} | Acc: {acc:.4f}")

        if (epoch + 1) % 5 == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"swin_arcface_epoch{epoch+1}.pth"))

    torch.save(model.state_dict(), os.path.join(args.output_dir, "swin_arcface_final.pth"))
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ArcFace FRM head")
    parser.add_argument("--data-dir", required=True, help="Path to identity dataset (person_id/ subdirs)")
    parser.add_argument("--output-dir", default="./weights")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()
    train(args)
