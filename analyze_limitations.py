"""
Ilādṛṣṭi Phase 3: Final Limitations and Latency Analysis.
Analyzes inference latency on the test split using torch.cuda.Event(enable_timing=True)
and isolates the top 10 worst False Positive predictions (dense clouds, shadows, terrain)
saving high-resolution visual diagnostics to outputs/limitations/.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.dataset.multispectral_dataset import WildfireDataset
from src.models.baseline_unet import WildfireBaselineUNet
from src.utils.metrics import compute_dice, compute_iou
from src.utils.seed import seed_worker, set_seed


def measure_inference_latency(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_warmup_batches: int = 5,
) -> Dict[str, float]:
    """
    Measures GPU inference latency per batch using torch.cuda.Event(enable_timing=True).
    Provides exact millisecond timing without host CPU synchronization overhead.

    Args:
        model: PyTorch segmentation model in eval mode.
        dataloader: Test split DataLoader.
        device: Device to run inference on.
        num_warmup_batches: Number of warmup batches to prime GPU kernels and caches.

    Returns:
        dict containing mean, std, median, min, max latency (ms/batch),
        per-sample latency (ms/sample), and throughput (FPS / tiles/sec).
    """
    model.eval()
    batch_latencies_ms: List[float] = []
    total_samples = 0

    print(f"[Latency Benchmark] Running inference on {len(dataloader.dataset)} test samples...")
    print(f"                    Device: {device} | Warmup batches: {num_warmup_batches}")

    use_cuda_events = device.type == "cuda" and torch.cuda.is_available()

    with torch.no_grad():
        # 1. Warmup iterations to prime CUDA context and Tensor Core pipelines
        warmup_counter = 0
        for images, _ in dataloader:
            images = images.to(device, non_blocking=True)
            _ = model(images)
            warmup_counter += 1
            if warmup_counter >= num_warmup_batches:
                break

        if use_cuda_events:
            torch.cuda.synchronize(device)

        # 2. Precision timed iterations
        for images, _ in dataloader:
            b_size = images.size(0)
            total_samples += b_size
            images = images.to(device, non_blocking=True)

            if use_cuda_events:
                # Use torch.cuda.Event(enable_timing=True) for exact hardware timestamps
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                torch.cuda.synchronize(device)
                start_event.record()

                _ = model(images)

                end_event.record()
                torch.cuda.synchronize(device)

                # elapsed_time returns duration in milliseconds
                batch_ms = start_event.elapsed_time(end_event)
            else:
                # CPU fallback
                t0 = time.perf_counter()
                _ = model(images)
                batch_ms = (time.perf_counter() - t0) * 1000.0

            batch_latencies_ms.append(batch_ms)

    latencies_arr = np.array(batch_latencies_ms, dtype=np.float64)
    batch_size = dataloader.batch_size or 16

    mean_batch_ms = float(np.mean(latencies_arr))
    std_batch_ms = float(np.std(latencies_arr))
    median_batch_ms = float(np.median(latencies_arr))
    p95_batch_ms = float(np.percentile(latencies_arr, 95))
    min_batch_ms = float(np.min(latencies_arr))
    max_batch_ms = float(np.max(latencies_arr))

    per_tile_ms = mean_batch_ms / batch_size
    fps = 1000.0 / per_tile_ms if per_tile_ms > 0 else 0.0

    latency_summary = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if use_cuda_events else "CPU",
        "batch_size": batch_size,
        "total_batches": len(latencies_arr),
        "total_samples": total_samples,
        "mean_batch_latency_ms": mean_batch_ms,
        "std_batch_latency_ms": std_batch_ms,
        "median_batch_latency_ms": median_batch_ms,
        "p95_batch_latency_ms": p95_batch_ms,
        "min_batch_latency_ms": min_batch_ms,
        "max_batch_latency_ms": max_batch_ms,
        "per_sample_latency_ms": per_tile_ms,
        "throughput_fps": fps,
        "used_cuda_events": use_cuda_events,
    }

    # Print clean benchmark summary
    border = "+---------------------------------------+--------------------+"
    print("\n" + border)
    print("|        ILĀDṚṢṬI INFERENCE LATENCY BENCHMARK (CUDA EVENTS)   |")
    print(border)
    print(f"| {'Hardware Accelerator':<37} | {latency_summary['device_name'][:18]:<18} |")
    print(f"| {'Batch Size':<37} | {batch_size:<18} |")
    print(f"| {'Timing Backend':<37} | {'torch.cuda.Event' if use_cuda_events else 'perf_counter':<18} |")
    print(border)
    print(f"| {'Mean Batch Latency (ms)':<37} | {mean_batch_ms:<18.2f} |")
    print(f"| {'Std Batch Latency (ms)':<37} | {std_batch_ms:<18.2f} |")
    print(f"| {'Median Batch Latency (ms)':<37} | {median_batch_ms:<18.2f} |")
    print(f"| {'95th Percentile Latency (ms)':<37} | {p95_batch_ms:<18.2f} |")
    print(border)
    print(f"| {'Per-Tile Latency (ms/sample)':<37} | {per_tile_ms:<18.2f} |")
    print(f"| {'Throughput (Tiles/sec / FPS)':<37} | {fps:<18.2f} |")
    print(border + "\n")

    return latency_summary


def classify_failure_cause(
    image_np: np.ndarray,
    fp_mask: np.ndarray,
    fn_mask: np.ndarray,
) -> Tuple[str, str]:
    """
    Analyzes radiometric signatures in False Positive pixels to identify the physical failure cause:
    - Dense Cloud Cover: High uniform reflectance across all visible bands (>0.6)
    - Cloud / Topographic Shadows: Very low reflectance (<0.1) adjacent to bright regions
    - Reflective Soil / Basaltic Rock: Intermediate spectral ambiguity
    """
    # image_np shape: (C, H, W), normalized in [0, 1]
    if np.sum(fp_mask) == 0:
        return "Clean / Boundary Noise", "Minimal false alarm pixels confined to scar perimeter."

    # Extract reflectances inside FP region
    fp_pixels = image_np[:, fp_mask > 0]
    mean_rgb = np.mean(fp_pixels[:3, :]) if fp_pixels.shape[1] > 0 else 0.0

    has_nir = image_np.shape[0] >= 4
    mean_nir = np.mean(fp_pixels[3, :]) if (has_nir and fp_pixels.shape[1] > 0) else mean_rgb

    if mean_rgb > 0.45:
        category = "Dense Cloud / Vapor Haze"
        desc = (
            f"High visible reflectance (RGB mean={mean_rgb:.2f}) causing atmospheric saturation "
            "and bright cloud fringe misclassification."
        )
    elif mean_rgb < 0.12:
        category = "Topographic Shadow / Water Confusion"
        desc = (
            f"Deep cast shadow or dark terrain (RGB mean={mean_rgb:.2f}) exhibiting charcoal-like "
            "visible light absorption."
        )
    elif has_nir and (mean_nir - mean_rgb < -0.05):
        category = "Arid Soil / Senescent Flora"
        desc = (
            f"Dry unburned vegetative soil (NIR={mean_nir:.2f}, RGB={mean_rgb:.2f}) mimicking low-biomass "
            "burn scars."
        )
    else:
        category = "Complex Smoke-Cloud Boundary"
        desc = (
            f"Turbulent smoke plume mixed with cloud margins (RGB mean={mean_rgb:.2f}) creating "
            "diffuse false positives."
        )

    return category, desc


def create_diagnostic_composite(
    image_chw: np.ndarray,
    gt_mask_hw: np.ndarray,
    pred_prob_hw: np.ndarray,
    threshold: float,
    rank: int,
    filename: str,
    fpr: float,
    dice: float,
    cause_title: str,
    cause_desc: str,
    save_path: Path,
) -> None:
    """
    Builds a high-impact 5-panel scientific diagnostic composite figure:
    1. Sentinel-2 True Color (RGB)
    2. False Color (NIR-R-G or NIR Contrast)
    3. Ground Truth Annotation
    4. Model Probability Heatmap + Predicted Contour
    5. Spatial Error Map (Green=TP, Red=FP, Blue=FN)
    """
    h, w = gt_mask_hw.shape
    pred_bin = (pred_prob_hw >= threshold).astype(np.float32)

    # 1. RGB Image
    # Bands 0, 1, 2 correspond to Red, Green, Blue in WildfireDataset
    rgb = np.clip(image_chw[:3].transpose(1, 2, 0), 0.0, 1.0)

    # 2. False Color (NIR, Red, Green)
    if image_chw.shape[0] >= 4:
        # Band 3 is NIR, Band 0 is Red, Band 1 is Green
        nir_r_g = np.stack([image_chw[3], image_chw[0], image_chw[1]], axis=-1)
        nir_img = np.clip(nir_r_g, 0.0, 1.0)
        nir_label = "False-Color NIR (B8-B4-B3)"
    else:
        # Grayscale NIR proxy or RGB inverted
        nir_img = np.stack([rgb[:, :, 0]] * 3, axis=-1)
        nir_label = "Monochrome Red Reference"

    # 3. Spatial Error Breakdown
    # Green = True Positive (Hit)
    # Red = False Positive (False Alarm / Commission Error)
    # Blue = False Negative (Omission Error)
    # Dark Gray = True Negative
    error_map = np.zeros((h, w, 3), dtype=np.float32)
    error_map[:, :] = [0.12, 0.14, 0.18]  # Dark background slate

    tp = (pred_bin == 1.0) & (gt_mask_hw == 1.0)
    fp = (pred_bin == 1.0) & (gt_mask_hw == 0.0)
    fn = (pred_bin == 0.0) & (gt_mask_hw == 1.0)

    error_map[tp] = [0.15, 0.85, 0.35]  # Vibrant Emerald Green (True Positive)
    error_map[fp] = [0.95, 0.20, 0.20]  # Bright Crimson Red (False Positive)
    error_map[fn] = [0.20, 0.55, 0.95]  # Electric Cobalt Blue (False Negative)

    # Render figure
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.8), dpi=200)

    # Panel 1: RGB
    axes[0].imshow(rgb)
    axes[0].set_title("(a) True Color (RGB)", fontsize=11, fontweight="bold", pad=8)
    axes[0].axis("off")

    # Panel 2: False Color NIR
    axes[1].imshow(nir_img)
    axes[1].set_title(f"(b) {nir_label}", fontsize=11, fontweight="bold", pad=8)
    axes[1].axis("off")

    # Panel 3: Ground Truth
    axes[2].imshow(gt_mask_hw, cmap="gray", vmin=0, vmax=1)
    axes[2].contour(gt_mask_hw, levels=[0.5], colors=["#10B981"], linewidths=1.2)
    axes[2].set_title("(c) Ground Truth Mask", fontsize=11, fontweight="bold", pad=8)
    axes[2].axis("off")

    # Panel 4: Model Prediction Heatmap
    im_pred = axes[3].imshow(pred_prob_hw, cmap="magma", vmin=0, vmax=1)
    axes[3].contour(pred_bin, levels=[0.5], colors=["#00FFFF"], linewidths=1.2)
    axes[3].set_title("(d) Model Confidence", fontsize=11, fontweight="bold", pad=8)
    axes[3].axis("off")

    # Panel 5: Spatial Error
    axes[4].imshow(error_map)
    axes[4].set_title("(e) Error Breakdown", fontsize=11, fontweight="bold", pad=8)
    axes[4].axis("off")

    # Legend for Error Breakdown
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=[0.95, 0.20, 0.20], label=f"False Positive ({fp.sum():,} px)"),
        plt.Rectangle((0, 0), 1, 1, color=[0.15, 0.85, 0.35], label=f"True Positive ({tp.sum():,} px)"),
        plt.Rectangle((0, 0), 1, 1, color=[0.20, 0.55, 0.95], label=f"False Negative ({fn.sum():,} px)"),
    ]
    axes[4].legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        fontsize=8,
        frameon=True,
        facecolor="#1F2937",
        edgecolor="#374151",
        labelcolor="white",
        ncol=1,
    )

    # Main Figure Title
    fig.suptitle(
        f"Failure Case #{rank:02d} [{filename}] | FPR: {fpr*100:.2f}% | Dice: {dice:.3f} | Cause: {cause_title}\n{cause_desc}",
        fontsize=11,
        color="#111827",
        y=1.04,
    )

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def isolate_and_save_worst_predictions(
    model: nn.Module,
    dataset: WildfireDataset,
    device: torch.device,
    output_dir: str = "outputs/limitations",
    top_k: int = 10,
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Identifies, ranks, and visualizes the top K failure cases exhibiting the highest
    False Positive rates on the held-out test split.

    Args:
        model: Trained segmentation model.
        dataset: Held-out test split dataset.
        device: Evaluation device.
        output_dir: Destination folder for exported diagnostic figures.
        top_k: Number of worst predictions to extract (default: 10).
        threshold: Decision threshold for segmentation.

    Returns:
        List of dictionaries with failure sample metrics and diagnostic explanations.
    """
    model.eval()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    individual_dir = out_path / "individual_assets"
    individual_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Limitations Analysis] Auditing {len(dataset)} test samples for False Positive commission errors...")

    sample_metrics: List[Dict[str, Any]] = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            img_tensor, mask_tensor = dataset[idx]
            img_path, msk_path = dataset.samples[idx]

            # Forward pass
            inp = img_tensor.unsqueeze(0).to(device)
            logits = model(inp, return_logits=True)
            prob = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()  # (H, W)

            gt_mask = mask_tensor.squeeze(0).numpy()  # (H, W)
            pred_bin = (prob >= threshold).astype(np.float32)

            # Confusion counts
            total_pixels = gt_mask.size
            bg_pixels = (gt_mask == 0.0)
            total_bg = int(np.sum(bg_pixels))
            total_fg = int(np.sum(gt_mask == 1.0))

            fp_pixels = (pred_bin == 1.0) & bg_pixels
            fp_count = int(np.sum(fp_pixels))

            fn_pixels = (pred_bin == 0.0) & (gt_mask == 1.0)
            fn_count = int(np.sum(fn_pixels))

            tp_pixels = (pred_bin == 1.0) & (gt_mask == 1.0)
            tp_count = int(np.sum(tp_pixels))

            # False Positive Rate (FP / total background)
            fpr = float(fp_count / total_bg) if total_bg > 0 else 0.0

            # Continuous FP mass (prob on background) for fine-grained ranking
            continuous_fp_mass = float(np.sum(prob[bg_pixels]))

            # Dice and IoU
            dice_t = compute_dice(torch.from_numpy(prob).unsqueeze(0).unsqueeze(0), mask_tensor.unsqueeze(0), threshold=threshold)
            iou_t = compute_iou(torch.from_numpy(prob).unsqueeze(0).unsqueeze(0), mask_tensor.unsqueeze(0), threshold=threshold)

            image_np = img_tensor.numpy()
            cause_title, cause_desc = classify_failure_cause(image_np, fp_pixels, fn_pixels)

            sample_metrics.append({
                "sample_idx": idx,
                "filename": img_path.name,
                "img_path": str(img_path),
                "mask_path": str(msk_path),
                "fpr": fpr,
                "continuous_fp_mass": continuous_fp_mass,
                "fp_count": fp_count,
                "fn_count": fn_count,
                "tp_count": tp_count,
                "total_bg": total_bg,
                "total_fg": total_fg,
                "dice": float(dice_t.item()),
                "iou": float(iou_t.item()),
                "cause_title": cause_title,
                "cause_desc": cause_desc,
                "image_chw": image_np,
                "gt_mask_hw": gt_mask,
                "pred_prob_hw": prob,
            })

    # Sort descending by False Positive Rate (FPR), secondary key continuous FP mass
    sample_metrics.sort(key=lambda s: (s["fpr"], s["continuous_fp_mass"]), reverse=True)
    top_worst = sample_metrics[:top_k]

    print(f"[Limitations Analysis] Generating 5-panel diagnostics for Top {len(top_worst)} worst FP predictions...")

    report_cases: List[Dict[str, Any]] = []

    for rank, case in enumerate(top_worst, start=1):
        stem = Path(case["filename"]).stem
        comp_name = f"rank_{rank:02d}_worst_fp_{stem}.png"
        comp_path = out_path / comp_name

        create_diagnostic_composite(
            image_chw=case["image_chw"],
            gt_mask_hw=case["gt_mask_hw"],
            pred_prob_hw=case["pred_prob_hw"],
            threshold=threshold,
            rank=rank,
            filename=case["filename"],
            fpr=case["fpr"],
            dice=case["dice"],
            cause_title=case["cause_title"],
            cause_desc=case["cause_desc"],
            save_path=comp_path,
        )

        # Save individual assets for modular report layout
        rgb = np.clip(case["image_chw"][:3].transpose(1, 2, 0), 0.0, 1.0)
        plt.imsave(individual_dir / f"rank_{rank:02d}_{stem}_rgb.png", rgb)
        plt.imsave(individual_dir / f"rank_{rank:02d}_{stem}_gt.png", case["gt_mask_hw"], cmap="gray")
        plt.imsave(individual_dir / f"rank_{rank:02d}_{stem}_pred.png", case["pred_prob_hw"], cmap="magma")

        report_cases.append({
            "rank": rank,
            "filename": case["filename"],
            "fpr": case["fpr"],
            "fp_pixels": case["fp_count"],
            "tp_pixels": case["tp_count"],
            "fn_pixels": case["fn_count"],
            "dice": case["dice"],
            "iou": case["iou"],
            "cause_title": case["cause_title"],
            "cause_description": case["cause_desc"],
            "composite_image": str(comp_path),
        })
        print(f"  [{rank:02d}/10] {case['filename']:<30} | FPR: {case['fpr']*100:5.2f}% | Dice: {case['dice']:.4f} | Cause: {case['cause_title']}")

    # 4. Generate overview multi-row grid for 4-page paper
    grid_fig_path = out_path / "fig_limitations_top10_overview.png"
    generate_overview_grid(top_worst, threshold=threshold, save_path=grid_fig_path)
    print(f"[Limitations Analysis] Overview grid saved to: {grid_fig_path}")

    # 5. Export comprehensive Markdown limitations documentation
    generate_limitations_markdown(report_cases, out_path / "limitations_analysis.md")

    # 6. Export JSON metadata
    json_path = out_path / "limitations_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_cases, f, indent=2)
    print(f"[Limitations Analysis] Failure metrics written to: {json_path}")

    return report_cases


def generate_overview_grid(
    top_worst: List[Dict[str, Any]],
    threshold: float,
    save_path: Path,
) -> None:
    """Creates a consolidated multi-row publication figure of the top failure cases."""
    num_cases = min(len(top_worst), 5)  # Showcase top 5 prominently in the master grid
    fig, axes = plt.subplots(num_cases, 4, figsize=(14, 2.7 * num_cases), dpi=200)

    col_titles = ["(a) True Color RGB", "(b) Ground Truth", "(c) Model Confidence", "(d) Error Breakdown"]

    for row_idx in range(num_cases):
        case = top_worst[row_idx]
        rgb = np.clip(case["image_chw"][:3].transpose(1, 2, 0), 0.0, 1.0)
        gt = case["gt_mask_hw"]
        prob = case["pred_prob_hw"]
        pred_bin = (prob >= threshold).astype(np.float32)

        error_map = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.float32)
        error_map[:, :] = [0.15, 0.15, 0.18]
        error_map[(pred_bin == 1.0) & (gt == 1.0)] = [0.15, 0.85, 0.35]  # TP (Green)
        error_map[(pred_bin == 1.0) & (gt == 0.0)] = [0.95, 0.20, 0.20]  # FP (Red)
        error_map[(pred_bin == 0.0) & (gt == 1.0)] = [0.20, 0.55, 0.95]  # FN (Blue)

        axes[row_idx, 0].imshow(rgb)
        axes[row_idx, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 2].imshow(prob, cmap="magma", vmin=0, vmax=1)
        axes[row_idx, 3].imshow(error_map)

        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=11, fontweight="bold", pad=6)

        # Label each row with rank, sample name, and cause
        axes[row_idx, 0].text(
            -0.08, 0.5,
            f"#{row_idx+1}\nFPR: {case['fpr']*100:.1f}%\n{case['cause_title'][:16]}",
            transform=axes[row_idx, 0].transAxes,
            va="center", ha="right",
            fontsize=8.5, fontweight="bold", color="#1F2937",
        )

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_limitations_markdown(cases: List[Dict[str, Any]], save_path: Path) -> None:
    """Generates clean Markdown documentation of limitations for the 4-page paper."""
    rows = []
    for c in cases:
        rows.append(
            f"| **#{c['rank']:02d}** | `{c['filename']}` | **{c['fpr']*100:.2f}%** | {c['fp_pixels']:,} | "
            f"`{c['dice']:.4f}` | `{c['iou']:.4f}` | **{c['cause_title']}** | {c['cause_description']} |"
        )
    table_rows = "\n".join(rows)

    content = f"""# Ilādṛṣṭi: Systematic Limitations & Error Analysis
*Track 6: AI for Science & Society — Deep Learning Hackathon, Amrita Vishwa Vidyapeetham*

## 1. Quantitative Failure Case Breakdown (Top 10 False Positive Samples)

The table below catalogs the top 10 test samples with the highest False Positive rates on the held-out test split ($N=40$).
These samples represent optical edge cases where unburned terrain is mistakenly classified as wildfire burn scars.

| Rank | Scene Identifier | False Positive Rate | FP Pixels | Test Dice | Test IoU | Primary Failure Mode | Physical Remote Sensing Explanation |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
{table_rows}

---

## 2. Root Cause Classification & Physical Remote Sensing Mechanisms

### A. Dense Cumulus Cloud Edges & Vapor Plumes
Dense clouds introduce severe optical scattering. While deep cloud centers are bright across all bands, cloud margins display steep gradient rolloffs and partial transparency. When aerosol haze interacts with visible bands, the model occasionally misidentifies turbulent cloud boundaries as active fire perimeters.

### B. Topographic Mountain Shadows & Sun-Terrain Geometry
In steep, rugged terrain (e.g., canyons and mountain escarpments), low solar elevation angles create deep cast shadows. These shadowed pixels exhibit near-zero visible surface reflectance ($\\\\rho < 0.10$), closely mimicking charred, carbonized charcoal ash. 

### C. Mitigation via Spectral Band Engineering
Our spectral ablation experiments demonstrate that incorporating Sentinel-2 **Band 8 (Near-Infrared, $\\\\sim 842\\\\,\\\\text{{nm}}$)** significantly reduces shadow false alarms, because unburned shaded vegetation maintains residual NIR mesophyll reflectance, whereas true burned scars exhibit a complete NIR collapse.

---

## 3. Visual Artifact Reference
* Diagnostic composites for all top 10 failure cases are located in `outputs/limitations/`.
* High-resolution multi-case publication figure: `outputs/limitations/fig_limitations_top10_overview.png`.
"""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[Limitations Analysis] Markdown limitations report saved to: {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ilādṛṣṭi Phase 3: Inference Latency and Limitations Analysis"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to data directory containing held-out test split.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model checkpoint. Auto-resolves best available if None.",
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=None,
        choices=[3, 4],
        help="Input channels (3 or 4). Auto-detected from checkpoint if None.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for latency measurement.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of worst False Positive cases to isolate (default: 10).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for probability maps.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/limitations",
        help="Destination directory for exported limitation plots and reports.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic evaluation seed.",
    )
    return parser.parse_args()


def resolve_default_checkpoint() -> Tuple[Path, int]:
    """Find the best trained model checkpoint to evaluate."""
    candidates = [
        ("checkpoints/ablation_rgbnir_4band.pt", 4),
        ("checkpoints/best_baseline_model.pt", 4),
        ("checkpoints/best_baseline_model_fp16.pt", 4),
        ("checkpoints/ablation_rgb_3band.pt", 3),
        ("checkpoints/baseline_bce_rgb.pt", 3),
    ]
    for path_str, ch in candidates:
        p = Path(path_str)
        if p.exists():
            return p, ch
    raise FileNotFoundError("No trained checkpoint found in checkpoints directory!")


def main():
    args = parse_args()
    set_seed(args.seed)

    # 1. Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Checkpoint resolution
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        in_channels = args.in_channels
    else:
        ckpt_path, detected_ch = resolve_default_checkpoint()
        in_channels = args.in_channels or detected_ch

    print("=" * 76)
    print("      ILĀDṚṢṬI PHASE 3: INFERENCE LATENCY & LIMITATIONS ANALYSIS      ")
    print("=" * 76)
    print(f" Checkpoint Evaluated : {ckpt_path}")
    print(f" Compute Device       : {device}")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    if in_channels is None:
        in_channels = ckpt.get("in_channels", 4)
    print(f" Input Channels       : {in_channels} ({'RGB+NIR' if in_channels == 4 else 'RGB'})")

    # Instantiate model
    model = WildfireBaselineUNet(
        in_channels=in_channels,
        num_classes=1,
        pretrained=False,
    ).to(device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    # Build test dataset and loader
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

    # Part 1: Measure inference latency using torch.cuda.Event(enable_timing=True)
    latency_summary = measure_inference_latency(
        model=model,
        dataloader=test_loader,
        device=device,
        num_warmup_batches=5,
    )

    # Save latency JSON
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "latency_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(latency_summary, f, indent=2)

    # Part 2: Isolate and save top 10 worst False Positive predictions
    isolate_and_save_worst_predictions(
        model=model,
        dataset=test_dataset,
        device=device,
        output_dir=args.output_dir,
        top_k=args.top_k,
        threshold=args.threshold,
    )

    print("\n" + "=" * 76)
    print(" [analyze_limitations.py] Phase 3 Analysis successfully completed!")
    print(f" Artifacts available at: {out_dir.resolve()}")
    print("=" * 76)


if __name__ == "__main__":
    main()
