"""
Optimization Training Pipeline for Ilādṛṣṭi (Phase 2).
Supports Focal-Tversky Loss with configurable alpha, beta, and gamma hyperparameters,
alongside baseline BCE and other loss formulations.
Strictly checkpoints best model based on validation Dice score.
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
from src.utils.losses import get_loss
from src.utils.logger import log_benchmark_result
from src.utils.metrics import MetricTracker, compute_dice, compute_iou
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ilādṛṣṭi Phase 2 Optimization Training Pipeline"
    )
    # Dataset & Model options
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base path to data folder containing train/val/test splits.",
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=4,
        choices=[3, 4],
        help="Number of input spectral channels: 3 for RGB, 4 for RGB+NIR.",
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
        help="Mini-batch size for training and validation.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW optimizer.",
    )

    # Loss configuration & Hyperparameters
    parser.add_argument(
        "--loss",
        type=str,
        default="focal_tversky",
        choices=["focal_tversky", "tversky", "bce", "dice", "bce_dice"],
        help="Objective loss function. Default: focal_tversky.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Tversky False Negative penalty weight (alpha). Higher boosts recall. Default: 0.7.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="Tversky False Positive penalty weight (beta). Higher boosts precision. Default: 0.3.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.333,
        help="Focal-Tversky focusing parameter (gamma). Penalizes hard examples. Default: 1.333.",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=1e-6,
        help="Smoothing epsilon for Tversky index stability. Default: 1e-6.",
    )

    # Environment & Checkpoint options
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=None,
        help="Filename for best checkpoint (e.g., best_optimized_model.pt).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save training logs and histories.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for full reproducibility.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="DataLoader worker subprocess count.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda' or 'cpu'). Auto-detected if None.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Custom experiment identifier for logging.",
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

        # Batch metrics calculation
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
    """Evaluate model on a specific split without gradient calculation."""
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
    print(f"[Ilādṛṣṭi Optimization] Deterministic seed locked to {args.seed}")

    # 2. Setup Device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Ilādṛṣṭi Optimization] Using compute device: {device}")
    if device.type == "cuda":
        print(f"                         Device Name: {torch.cuda.get_device_name(0)}")

    # 3. Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # 4. Prepare DataLoaders
    print(
        f"[Ilādṛṣṭi Optimization] Initializing data loaders (Channels: {args.in_channels}, Batch size: {args.batch_size})..."
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
        f"                         Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}"
    )

    # 5. Initialize Model Architecture
    print(
        f"[Ilādṛṣṭi Optimization] Initializing WildfireBaselineUNet (ResNet-50, in_channels={args.in_channels})..."
    )
    model = WildfireBaselineUNet(
        in_channels=args.in_channels,
        num_classes=1,
        pretrained=True,
    ).to(device)

    # 6. Initialize Configured Loss Function
    criterion = get_loss(
        loss_name=args.loss,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        smooth=args.smooth,
        from_logits=True,
    )
    print(f"[Ilādṛṣṭi Optimization] Loss Function configured: {args.loss.upper()}")
    if args.loss in ("focal_tversky", "tversky"):
        print(
            f"                         Hyperparameters: alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}, smooth={args.smooth}"
        )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 7. Checkpointing setup
    best_val_dice = -1.0
    if args.checkpoint_name:
        checkpoint_filename = args.checkpoint_name
    else:
        checkpoint_filename = f"best_{args.loss}_{args.in_channels}band_model.pt"
    best_checkpoint_path = Path(args.checkpoint_dir) / checkpoint_filename

    history = []

    print("\n" + "=" * 82)
    print(
        f"{'Epoch':<7} | {'Train Loss':<11} | {'Train Dice':<11} | {'Val Loss':<11} | {'Val Dice':<11} | {'Val IoU':<9} | {'Status'}"
    )
    print("=" * 82)

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_metrics = evaluate_split(
            model, val_loader, criterion, device, desc=f"Epoch {epoch:02d} [Val]"
        )

        scheduler.step()

        # Checkpoint strictly on validation Dice improvement
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
                "loss": args.loss,
                "alpha": args.alpha,
                "beta": args.beta,
                "gamma": args.gamma,
                "args": vars(args),
            }
            torch.save(checkpoint_payload, best_checkpoint_path)
            status_str = "⭐ Best Model Saved"

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
    print("=" * 82)
    print(
        f"[Ilādṛṣṭi Optimization] Training complete in {elapsed_time/60:.2f} mins. "
        f"Best Val Dice: {best_val_dice:.4f}"
    )
    print(f"[Ilādṛṣṭi Optimization] Best checkpoint saved to: {best_checkpoint_path}")

    # Save training history
    history_filename = (
        f"optimization_{args.loss}_{args.in_channels}band_history.json"
    )
    history_path = Path(args.output_dir) / history_filename
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Ilādṛṣṭi Optimization] Training history logged to: {history_path}\n")


if __name__ == "__main__":
    main()
