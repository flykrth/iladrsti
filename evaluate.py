"""
Decoupled Test Evaluation Pipeline for Ilādṛṣṭi Baseline.
Loads the best trained baseline checkpoint and evaluates exclusively
on the held-out test split, reporting Test Loss, Test Dice, and Test IoU.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from src.dataset.multispectral_dataset import WildfireDataset
from src.models.baseline_unet import WildfireBaselineUNet
from src.utils.logger import log_benchmark_result
from src.utils.losses import get_loss
from src.utils.metrics import MetricTracker, compute_dice, compute_iou
from src.utils.seed import seed_worker, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ilādṛṣṭi Decoupled Test Evaluation - Track 6 Hackathon"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base path to data folder containing test split.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_baseline_model.pt",
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=4,
        choices=[3, 4],
        help="Input channels (3 or 4). Overridden by checkpoint if recorded.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for test inference.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for probability map.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default=None,
        help="Loss function name. Auto-detected from checkpoint if None.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name for logging. Auto-detected from checkpoint if None.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="outputs/evaluation_results.json",
        help="Path to save evaluation summary metrics.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda' or 'cpu'). Auto-detected if None.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic evaluation seed.",
    )
    return parser.parse_args()


def print_summary_table(results: dict[str, float], checkpoint_path: str, num_samples: int) -> None:
    """Print clean formatted summary table of decoupled test evaluation."""
    border = "+---------------------------------------+--------------------+"
    header = "| Metric / Parameter                    | Value              |"

    print("\n" + border)
    print("|      ILĀDṚṢṬI TEST SET BENCHMARK EVALUATION (DECOUPLED)     |")
    print(border)
    print(header)
    print(border)
    print(f"| {'Held-Out Test Samples':<37} | {num_samples:<18} |")
    print(f"| {'Checkpoint Evaluated':<37} | {Path(checkpoint_path).name:<18} |")
    print(f"| {'Input Channels (Bands)':<37} | {results.get('in_channels', 4):<18} |")
    print(f"| {'Loss Function':<37} | {results.get('loss_function', 'BCE'):<18} |")
    print(f"| {'Decision Threshold':<37} | {results.get('threshold', 0.5):<18.2f} |")
    print(border)
    print(f"| {'Test Loss':<37} | {results['test_loss']:<18.4f} |")
    print(f"| {'Test Dice Coefficient (F1)':<37} | {results['test_dice']:<18.4f} |")
    print(f"| {'Test Mean IoU (Jaccard)':<37} | {results['test_iou']:<18.4f} |")
    print(border + "\n")


def main():
    args = parse_args()

    # 1. Strictly lock seed for deterministic evaluation
    set_seed(args.seed)

    # 2. Setup Device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # 3. Load checkpoint
    print(f"[Ilādṛṣṭi] Loading baseline checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Detect in_channels & loss from checkpoint if available
    in_channels = checkpoint.get("in_channels", args.in_channels)
    saved_epoch = checkpoint.get("epoch", "Unknown")
    saved_val_dice = checkpoint.get("val_dice", "N/A")
    loss_name = args.loss or checkpoint.get("loss", "bce")
    alpha = checkpoint.get("alpha", 0.7)
    beta = checkpoint.get("beta", 0.3)
    gamma = checkpoint.get("gamma", 1.333)

    print(f"            Trained Epoch: {saved_epoch} | Validation Dice: {saved_val_dice}")
    print(f"            Loss Function: {loss_name.upper()} | Bands: {in_channels}")

    # 4. Instantiate Model Architecture
    model = WildfireBaselineUNet(
        in_channels=in_channels,
        num_classes=1,
        pretrained=False,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 5. Build Held-Out Test DataLoader
    test_dataset = WildfireDataset(
        data_dir=args.data_dir,
        split="test",
        in_channels=in_channels,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        drop_last=False,
    )

    criterion = get_loss(
        loss_name=loss_name,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        from_logits=True,
    )
    tracker = MetricTracker()

    print(f"[Ilādṛṣṭi] Running decoupled inference on {len(test_dataset)} test samples...")

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="[Test Split]", dynamic_ncols=True):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images, return_logits=True)
            loss = criterion(logits, masks)

            dice = compute_dice(logits, masks, threshold=args.threshold)
            iou = compute_iou(logits, masks, threshold=args.threshold)

            tracker.update(loss.item(), dice, iou, batch_size=images.size(0))

    metrics = tracker.compute()
    results = {
        "checkpoint": str(checkpoint_path),
        "in_channels": in_channels,
        "loss_function": loss_name.upper(),
        "test_samples": len(test_dataset),
        "threshold": args.threshold,
        "test_loss": metrics["loss"],
        "test_dice": metrics["dice"],
        "test_iou": metrics["iou"],
    }

    # 6. Print formatted summary table
    print_summary_table(results, str(checkpoint_path), len(test_dataset))

    # 7. Save output json
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Ilādṛṣṭi] Results saved to: {out_path}")

    # 8. Safely log to benchmark results table (CSV & Markdown)
    val_dice = checkpoint.get("val_dice", 0.0)
    val_iou = checkpoint.get("val_iou", 0.0)
    exp_name = args.experiment_name or checkpoint.get("args", {}).get("experiment_name")
    if not exp_name:
        if in_channels == 4 and loss_name.lower() in ("bce", "bcewithlogitsloss"):
            exp_name = "Mandatory Baseline"
        else:
            exp_name = f"Optimized {in_channels}-band ({loss_name.upper()})"

    log_benchmark_result(
        experiment_name=exp_name,
        architecture="Standard U-Net",
        backbone="ResNet-50 (Pretrained)",
        in_channels=in_channels,
        loss_function=loss_name.upper(),
        val_dice=float(val_dice) if isinstance(val_dice, (float, int)) else 0.0,
        val_iou=float(val_iou) if isinstance(val_iou, (float, int)) else 0.0,
        test_loss=metrics["loss"],
        test_dice=metrics["dice"],
        test_iou=metrics["iou"],
        seed=args.seed,
        notes=f"Evaluation on held-out test split (Seed locked {args.seed})",
    )


if __name__ == "__main__":
    main()

