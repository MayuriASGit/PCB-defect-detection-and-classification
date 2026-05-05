"""
PCB Defect Classifier Training Script
======================================
Trains EfficientNet-B0 on the DeepPCB dataset (or any folder of labeled defect ROIs).

Expected data directory layout:
    data_dir/
        train/
            missing_hole/    *.png
            mouse_bite/      *.png
            open_circuit/    *.png
            short/           *.png
            spur/            *.png
            spurious_copper/ *.png
        val/
            ...same structure...

Usage:
    python train.py --data_dir /path/to/data --epochs 50 --batch_size 32 --output model_checkpoint.pth
"""

import argparse
import os
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

from pipeline.model import PCBDefectClassifier, DEFECT_CLASSES


def get_transforms(augment: bool = True):
    if augment:
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


def plot_training_curves(history: dict, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["train_loss"], label="Train Loss", marker="o", markersize=4)
    axes[0].plot(history["val_loss"], label="Val Loss", marker="s", markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot([a * 100 for a in history["train_acc"]], label="Train Acc", marker="o", markersize=4)
    axes[1].plot([a * 100 for a in history["val_acc"]], label="Val Acc", marker="s", markersize=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=120)
    plt.close(fig)
    print(f"Training curves saved to {output_dir}/training_curves.png")


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, ax=ax,
        xticklabels=class_names, yticklabels=class_names,
        annot=True, fmt="d", cmap="Blues",
        linewidths=0.5
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=120)
    plt.close(fig)
    print(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")


def main():
    parser = argparse.ArgumentParser(description="Train PCB Defect Classifier")
    parser.add_argument("--data_dir", required=True, help="Root of dataset with train/ and val/ subdirs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="model_checkpoint.pth")
    parser.add_argument("--output_dir", default="training_outputs")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    train_dataset = datasets.ImageFolder(train_dir, transform=get_transforms(augment=True))
    val_dataset = datasets.ImageFolder(val_dir, transform=get_transforms(augment=False))

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    num_classes = len(train_dataset.classes)
    model = PCBDefectClassifier(num_classes=num_classes, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_epoch = 0

    print(f"\nTraining for {args.epochs} epochs...\n")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "classes": train_dataset.classes,
            }, args.output)
            print(f"  ✓ Best model saved (val_acc={val_acc:.2%})")

    print(f"\nBest val accuracy: {best_val_acc:.2%} at epoch {best_epoch}")

    print("\nRunning final evaluation on validation set...")
    checkpoint = torch.load(args.output, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, final_acc, all_preds, all_labels = evaluate(model, val_loader, criterion, device)

    report = classification_report(all_labels, all_preds, target_names=train_dataset.classes, digits=4)
    print("\nClassification Report:")
    print(report)

    report_path = os.path.join(args.output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    metrics = {
        "best_epoch": best_epoch,
        "best_val_accuracy": round(best_val_acc, 6),
        "final_val_accuracy": round(final_acc, 6),
        "epochs_trained": args.epochs,
        "classes": train_dataset.classes,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plot_training_curves(history, args.output_dir)
    plot_confusion_matrix(all_labels, all_preds, train_dataset.classes, args.output_dir)

    print(f"\nAll outputs saved to: {args.output_dir}/")
    print(f"Model checkpoint saved to: {args.output}")
    print(f"Final val accuracy: {final_acc:.2%}")


if __name__ == "__main__":
    main()
