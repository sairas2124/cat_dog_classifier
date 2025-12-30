import argparse
import os
import time
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

from utils import ensure_dir, save_json

def accuracy(pred_logits, y):
    preds = pred_logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data/processed",
                    help="Must contain train/ and val/ subfolders with class dirs.")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--num-workers", type=int, default=0, help="0 is safest on Windows")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    if args.device in ["cpu", "cuda"]:
        device = args.device
    print("Using device:", device)

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise SystemExit("Expected data-dir to contain train/ and val/. Run src/prepare_data.py first.")

    # Data transforms
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(args.img_size + 32),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device=="cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device=="cuda"))

    num_classes = len(train_ds.classes)
    print("Classes:", train_ds.classes)

    # Model (transfer learning)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=args.lr)

    ensure_dir("models")
    best_path = os.path.join("models", "best.pt")
    metadata_path = os.path.join("models", "metadata.json")

    best_val_acc = 0.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses, train_accs = [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", unit="batch")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            acc = accuracy(logits.detach(), y)
            train_losses.append(loss.item())
            train_accs.append(acc)
            pbar.set_postfix(loss=sum(train_losses)/len(train_losses), acc=sum(train_accs)/len(train_accs))

        # Validation
        model.eval()
        val_losses, val_accs = [], []
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]", unit="batch")
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                acc = accuracy(logits, y)
                val_losses.append(loss.item())
                val_accs.append(acc)
                pbar.set_postfix(loss=sum(val_losses)/len(val_losses), acc=sum(val_accs)/len(val_accs))

        val_acc = sum(val_accs)/len(val_accs)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": train_ds.classes,
                "img_size": args.img_size
            }, best_path)
            save_json({
                "classes": train_ds.classes,
                "class_to_idx": train_ds.class_to_idx,
                "img_size": args.img_size,
                "arch": "resnet18",
                "best_val_acc": best_val_acc,
            }, metadata_path)
            print(f"Saved new best model to {best_path} (val_acc={best_val_acc:.4f})")

    elapsed = time.time() - start
    print(f"Training complete. Best val acc: {best_val_acc:.4f}. Time: {elapsed/60:.1f} min")
    print("Best weights:", os.path.abspath(best_path))

if __name__ == "__main__":
    main()
