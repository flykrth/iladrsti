"""
Visual Dashboard and Video Demo Generator for Ilādṛṣṭi (Phase 2).
Generates side-by-side inference visual dashboards comparing:
- Original Satellite Input (True-Color RGB)
- False-Color Infrared Composite (NIR-R-G)
- Ground Truth Burned Area Mask
- Model Predicted Mask & Probability Heatmap
- Spatial Confusion Map (True Positives, False Positives, False Negatives)

Also compiles frames into an MP4 video and animated GIF for hackathon presentation demos.
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from src.dataset.multispectral_dataset import WildfireDataset
from src.models.baseline_unet import WildfireBaselineUNet
from src.utils.metrics import compute_dice, compute_iou
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ilādṛṣṭi Visual Dashboard & Video Demo Generator"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_baseline_model_fp16.pt",
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base path to dataset directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate and visualize.",
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=None,
        choices=[3, 4],
        help="Input channels (3 or 4). If None, auto-detected from checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/demo_visualizations",
        help="Directory to save generated visual dashboards and video demo.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=12,
        help="Number of samples to visualize (-1 for all samples in split).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for binary segmentation.",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        default=True,
        help="Compile dashboard frames into MP4 video and animated GIF.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Frame rate for video demo (frames per second).",
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


def enhance_rgb(img_rgb: np.ndarray) -> np.ndarray:
    """
    Apply percentile-based contrast stretching and gamma correction
    to convert raw satellite reflectances into crisp display imagery.
    """
    enhanced = np.zeros_like(img_rgb)
    for c in range(3):
        channel = img_rgb[:, :, c]
        p_low, p_high = np.percentile(channel, (2.0, 98.0))
        if p_high > p_low:
            stretched = np.clip((channel - p_low) / (p_high - p_low), 0.0, 1.0)
        else:
            stretched = np.clip(channel, 0.0, 1.0)
        enhanced[:, :, c] = stretched
    # Mild gamma adjustment for shadow penetration
    return np.power(enhanced, 0.85)


def build_confusion_map(pred_bin: np.ndarray, gt_bin: np.ndarray) -> np.ndarray:
    """
    Construct color-coded spatial confusion map:
    - Green = True Positive (TP: Correct fire detection)
    - Red = False Positive (FP: Commission error / false alarm)
    - Blue = False Negative (FN: Omission error / missed scar)
    - Dark Grey = True Negative (TN: Correct background)
    """
    h, w = pred_bin.shape
    cmap = np.zeros((h, w, 3), dtype=np.float32)
    # Background: dark charcoal grey
    cmap[:] = [0.15, 0.17, 0.20]

    tp = (pred_bin == 1.0) & (gt_bin == 1.0)
    fp = (pred_bin == 1.0) & (gt_bin == 0.0)
    fn = (pred_bin == 0.0) & (gt_bin == 1.0)

    cmap[tp] = [0.10, 0.85, 0.35]  # Bright Green (TP)
    cmap[fp] = [0.95, 0.20, 0.20]  # Bright Red (FP)
    cmap[fn] = [0.20, 0.55, 1.00]  # Electric Blue (FN)

    return cmap


def render_dashboard(
    sample_name: str,
    img_tensor: torch.Tensor,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    pred_prob: np.ndarray,
    dice: float,
    iou: float,
    precision: float,
    recall: float,
    in_channels: int,
    output_path: Path,
) -> None:
    """
    Render a 5-panel side-by-side scientific dashboard for a single satellite tile.
    """
    # 1. Prepare visual layers
    # Extract RGB channels
    # Dataset order: [Red, Green, Blue, (NIR)]
    img_np = img_tensor.cpu().numpy()
    rgb_raw = np.transpose(img_np[:3, :, :], (1, 2, 0))
    rgb_disp = enhance_rgb(rgb_raw)

    # Prepare False-Color NIR Composite (NIR as R, Red as G, Green as B)
    has_nir = (in_channels >= 4 and img_np.shape[0] >= 4)
    if has_nir:
        nir_channel = img_np[3, :, :]
        cir_raw = np.stack([nir_channel, img_np[0, :, :], img_np[1, :, :]], axis=-1)
        cir_disp = enhance_rgb(cir_raw)
    else:
        cir_disp = rgb_disp

    # Spatial Confusion Map
    confusion_map = build_confusion_map(pred_mask, gt_mask)

    # Overlay Prediction on RGB
    overlay_rgb = rgb_disp.copy()
    fire_pixels = pred_mask == 1.0
    overlay_rgb[fire_pixels] = 0.5 * overlay_rgb[fire_pixels] + 0.5 * np.array([1.0, 0.25, 0.0])

    # Burn scar statistics
    scar_pixels = int(np.sum(pred_mask))
    gt_pixels = int(np.sum(gt_mask))
    scar_pct = (scar_pixels / (pred_mask.shape[0] * pred_mask.shape[1])) * 100.0

    # 2. Create Figure Layout (Dark Theme)
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(22, 5.8), dpi=150)
    gs = fig.add_gridspec(
        1, 5,
        left=0.03, right=0.97,
        top=0.82, bottom=0.08,
        wspace=0.15
    )

    # Supertitle Banner
    fig.suptitle(
        f"ILĀDṚṢṬI WILDFIRE SEGMENTATION DASHBOARD  |  Sample: {sample_name}  |  "
        f"Dice (F1): {dice:.4f}  |  IoU: {iou:.4f}  |  Precision: {precision:.4f}  |  Recall: {recall:.4f}",
        fontsize=14,
        fontweight="bold",
        color="#F3F4F6",
        y=0.94,
    )

    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]

    # Panel 1: Sentinel-2 True Color
    axes[0].imshow(rgb_disp)
    axes[0].set_title("1. Sentinel-2 RGB (True Color)", fontsize=11, color="#E5E7EB", pad=8)
    axes[0].axis("off")

    # Panel 2: False Color NIR or Overlay
    if has_nir:
        axes[1].imshow(cir_disp)
        axes[1].set_title("2. False-Color Infrared (NIR-R-G)", fontsize=11, color="#E5E7EB", pad=8)
    else:
        axes[1].imshow(overlay_rgb)
        axes[1].set_title("2. Prediction Overlay on RGB", fontsize=11, color="#E5E7EB", pad=8)
    axes[1].axis("off")

    # Panel 3: Ground Truth Mask
    axes[2].imshow(gt_mask, cmap="inferno", vmin=0, vmax=1)
    axes[2].set_title(
        f"3. Ground Truth Scar ({gt_pixels:,} px)",
        fontsize=11,
        color="#E5E7EB",
        pad=8,
    )
    axes[2].axis("off")

    # Panel 4: Predicted Probability & Binary Mask
    axes[3].imshow(pred_prob, cmap="magma", vmin=0.0, vmax=1.0)
    axes[3].set_title(
        f"4. Model Prediction ({scar_pct:.1f}% Scar)",
        fontsize=11,
        color="#E5E7EB",
        pad=8,
    )
    axes[3].axis("off")

    # Panel 5: Confusion Map (TP, FP, FN)
    axes[4].imshow(confusion_map)
    axes[4].set_title("5. Error Map (TP / FP / FN)", fontsize=11, color="#E5E7EB", pad=8)
    axes[4].axis("off")

    # Add legend beneath the error map
    legend_elements = [
        matplotlib.patches.Patch(facecolor=[0.10, 0.85, 0.35], label="TP (Hit)"),
        matplotlib.patches.Patch(facecolor=[0.95, 0.20, 0.20], label="FP (Alarm)"),
        matplotlib.patches.Patch(facecolor=[0.20, 0.55, 1.00], label="FN (Miss)"),
    ]
    axes[4].legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(1.0, -0.04),
        ncol=3,
        fontsize=8,
        frameon=False,
    )

    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def compile_video(frames_dir: Path, output_mp4: Path, output_gif: Path, fps: int = 2) -> None:
    """Compile generated frame images into an MP4 video and animated GIF."""
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        print("[Ilādṛṣṭi Demo] No frame images found to compile into video.")
        return

    print(f"[Ilādṛṣṭi Demo] Compiling {len(frame_files)} frames into video demo ({fps} FPS)...")

    # 1. Read first frame to determine dimensions
    sample_img = cv2.imread(str(frame_files[0]))
    h, w, _ = sample_img.shape

    # 2. Write MP4 video using OpenCV
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_mp4), fourcc, fps, (w, h))

    pil_frames = []
    for frame_path in frame_files:
        bgr_frame = cv2.imread(str(frame_path))
        if bgr_frame is not None:
            # OpenCV write
            video_writer.write(bgr_frame)
            # PIL conversion for GIF
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(rgb_frame))

    video_writer.release()
    print(f"[Ilādṛṣṭi Demo] MP4 video demo saved: {output_mp4}")

    # 3. Write Animated GIF using PIL (for instant markdown preview)
    if pil_frames:
        duration_ms = int(1000 / fps)
        pil_frames[0].save(
            output_gif,
            save_all=True,
            append_images=pil_frames[1:],
            optimize=True,
            duration=duration_ms,
            loop=0,
        )
        print(f"[Ilādṛṣṭi Demo] Animated GIF demo saved: {output_gif}")


def main():
    args = parse_args()
    set_seed(args.seed)

    # Setup device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        # Fallback to full model if fp16 not present
        alt_path = Path("checkpoints/best_baseline_model.pt")
        if alt_path.exists():
            checkpoint_path = alt_path
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[Ilādṛṣṭi Demo] Loading model checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    in_channels = args.in_channels or checkpoint.get("in_channels", 4)
    print(f"[Ilādṛṣṭi Demo] Configured input channels: {in_channels}")

    # Instantiate Model
    model = WildfireBaselineUNet(
        in_channels=in_channels,
        num_classes=1,
        pretrained=False,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load Dataset
    dataset = WildfireDataset(
        data_dir=args.data_dir,
        split=args.split,
        in_channels=in_channels,
    )
    total_samples = len(dataset)
    num_to_render = total_samples if args.num_samples in (-1, None) else min(args.num_samples, total_samples)

    # Setup Output Paths
    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[Ilādṛṣṭi Demo] Generating {num_to_render} visual dashboards from '{args.split}' split..."
    )

    all_dice = []
    all_iou = []

    with torch.no_grad():
        for idx in tqdm(range(num_to_render), desc="[Rendering Dashboards]", dynamic_ncols=True):
            img_tensor, mask_tensor = dataset[idx]
            img_path, mask_path = dataset.samples[idx]
            sample_name = img_path.name

            # Forward inference
            inputs = img_tensor.unsqueeze(0).to(device)
            logits = model(inputs, return_logits=True)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H, W)

            # Binarize
            pred_mask = (probs >= args.threshold).astype(np.float32)
            gt_mask = mask_tensor.squeeze().cpu().numpy()

            # Quantitative metrics for this sample
            p_flat = pred_mask.reshape(-1)
            g_flat = gt_mask.reshape(-1)

            intersection = np.sum(p_flat * g_flat)
            union = np.sum(p_flat) + np.sum(g_flat) - intersection

            dice = float((2.0 * intersection + 1e-7) / (np.sum(p_flat) + np.sum(g_flat) + 1e-7))
            iou = float((intersection + 1e-7) / (union + 1e-7))

            tp = float(intersection)
            fp = float(np.sum((p_flat == 1.0) & (g_flat == 0.0)))
            fn = float(np.sum((p_flat == 0.0) & (g_flat == 1.0)))

            precision = float((tp + 1e-7) / (tp + fp + 1e-7))
            recall = float((tp + 1e-7) / (tp + fn + 1e-7))

            all_dice.append(dice)
            all_iou.append(iou)

            frame_filename = f"frame_{idx + 1:03d}_{sample_name.replace('.tif', '')}.png"
            frame_path = frames_dir / frame_filename

            render_dashboard(
                sample_name=sample_name,
                img_tensor=img_tensor,
                gt_mask=gt_mask,
                pred_mask=pred_mask,
                pred_prob=probs,
                dice=dice,
                iou=iou,
                precision=precision,
                recall=recall,
                in_channels=in_channels,
                output_path=frame_path,
            )

    mean_dice = float(np.mean(all_dice))
    mean_iou = float(np.mean(all_iou))
    print(f"\n[Ilādṛṣṭi Demo] Completed rendering {num_to_render} dashboards.")
    print(f"                 Mean Sample Dice: {mean_dice:.4f} | Mean Sample IoU: {mean_iou:.4f}")
    print(f"                 Dashboard frames saved to: {frames_dir}")

    # Compile video demo
    if args.save_video:
        output_mp4 = output_dir / "iladristi_wildfire_demo.mp4"
        output_gif = output_dir / "iladristi_wildfire_demo.gif"
        compile_video(frames_dir, output_mp4, output_gif, fps=args.fps)

    print(f"\n[Ilādṛṣṭi Demo] All visual demonstration assets ready in: {output_dir}\n")


if __name__ == "__main__":
    main()
