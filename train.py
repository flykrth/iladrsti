"""
Fixed-seed Training Pipeline for Ilādṛṣṭi Mandatory Baseline Model.
Optimizes ResNet-50 U-Net using BCEWithLogitsLoss as the mandatory comparison floor.
Checkpoints best model strictly based on validation Dice score.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.dataset.multispectral_dataset import get_dataloaders
from src.models.baseline_unet import WildfireBaselineUNet
from src.utils.metrics import MetricTracker, compute_dice, compute_iou
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ilādṛṣṭi Baseline Training Pipeline - Track 6 Hackathon"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base path to data folder containing train/val/test splits.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW optimizer.",
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=4,
        choices=[3, 4],
        help="Number of input channels: 3 for RGB, 4 for RGB+NIR.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save best checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save training metrics and logs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="DataLoader worker subprocesses.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda' or 'cpu'). Auto-detected if None.",
    )
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """Train the model for one epoch."""
    model.train()
    tracker = MetricTracker()

    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch:02d} [Train]",
        leave=False,
        dynamic_ncols=True,
    )
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images, return_logits=True)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        # Compute batch metrics
        with torch.no_grad():
            dice = compute_dice(logits, masks)
            iou = compute_iou(logits, masks)

        tracker.update(loss.item(), dice, iou, batch_size=images.size(0))
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Dice": f"{dice.item():.4f}"})

    return tracker.compute()


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Val",
) -> dict[str, float]:
    """Evaluate model on a specific split without gradient tracking."""
    model.eval()
    tracker = MetricTracker()

    pbar = tqdm(
        dataloader,
        desc=f"[{desc}]",
        leave=False,
        dynamic_ncols=True,
    )
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images, return_logits=True)
        loss = criterion(logits, masks)

        dice = compute_dice(logits, masks)
        iou = compute_iou(logits, masks)

        tracker.update(loss.item(), dice, iou, batch_size=images.size(0))

    return tracker.compute()


def main():
    args = parse_args()

    # 1. Strictly lock deterministic seeds
    set_seed(args.seed)
    print(f"[Ilādṛṣṭi] Deterministic seed locked to {args.seed}")

    # 2. Setup Device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Ilādṛṣṭi] Using compute device: {device}")
    if device.type == "cuda":
        print(f"            Device Name: {torch.cuda.get_device_name(0)}")

    # 3. Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # 4. Prepare DataLoaders
    print(
        f"[Ilādṛṣṭi] Initializing data loaders (Bands: {args.in_channels}, Batch size: {args.batch_size})..."
    )
    loaders = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        in_channels=args.in_channels,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    print(
        f"            Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}"
    )

    # 5. Initialize Model
    print(
        f"[Ilādṛṣṭi] Building WildfireBaselineUNet (ResNet-50 backbone, in_channels={args.in_channels})..."
    )
    model = WildfireBaselineUNet(
        in_channels=args.in_channels,
        num_classes=1,
        pretrained=True,
    ).to(device)

    # 6. Loss and Optimizer
    # Mandatory Baseline: Binary Cross Entropy with Logits Loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 7. Training Loop with Validation Checkpointing
    best_val_dice = -1.0
    best_checkpoint_path = Path(args.checkpoint_dir) / "best_baseline_model.pt"
    history = []

    print("\n" + "=" * 78)
    print(
        f"{'Epoch':<7} | {'Train Loss':<11} | {'Train Dice':<11} | {'Val Loss':<11} | {'Val Dice':<11} | {'Val IoU':<9} | {'Status'}"
    )
    print("=" * 78)

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_metrics = evaluate_split(
            model, val_loader, criterion, device, desc=f"Epoch {epoch:02d} [Val]"
        )

        scheduler.step()

        # Check for checkpoint improvement strictly on validation Dice
        is_best = val_metrics["dice"] > best_val_dice
        status_str = ""
        if is_best:
            best_val_dice = val_metrics["dice"]
            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": best_val_dice,
                "val_iou": val_metrics["iou"],
                "val_loss": val_metrics["loss"],
                "in_channels": args.in_channels,
                "args": vars(args),
            }
            torch.save(checkpoint_payload, best_checkpoint_path)
            status_str = "⭐ Best Model Saved"

        # Log formatted summary line
        print(
            f"{epoch:02d}/{args.epochs:02d}   | "
            f"{train_metrics['loss']:<11.4f} | "
            f"{train_metrics['dice']:<11.4f} | "
            f"{val_metrics['loss']:<11.4f} | "
            f"{val_metrics['dice']:<11.4f} | "
            f"{val_metrics['iou']:<9.4f} | "
            f"{status_str}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "lr": optimizer.param_groups[0]["lr"],
        })

    elapsed_time = time.time() - start_time
    print("=" * 78)
    print(
        f"[Ilādṛṣṭi] Training completed in {elapsed_time/60:.2f} mins. Best Validation Dice: {best_val_dice:.4f}"
    )
    print(f"[Ilādṛṣṭi] Checkpoint saved to: {best_checkpoint_path}")

    # Save training log
    log_path = Path(args.output_dir) / "training_history.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Ilādṛṣṭi] Training history logged to: {log_path}\n")


if __name__ == "__main__":
    main()
