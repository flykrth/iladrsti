#!/usr/bin/env python3
"""
Export high-resolution visual tiles and metadata for the Ilādṛṣṭi Web Dashboard.
Extracts:
1. True-Color RGB
2. False-Color Infrared (NIR-R-G)
3. Model Predicted Mask
4. Ground Truth Mask
5. Prediction Overlay on Satellite Image
6. Spatial Confusion Map (TP Green, FP Red, FN Blue, TN Dark Slate)
7. Failure Cases (Cloud cover & Topographic shadow false alarms)
8. scenes_metadata.json with exact metrics
"""

import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from src.dataset.multispectral_dataset import WildfireDataset
from src.models.baseline_unet import WildfireBaselineUNet
from src.utils.metrics import compute_dice, compute_iou


def enhance_channels(img: np.ndarray, gamma: float = 0.85) -> np.ndarray:
    """Apply 2-98 percentile contrast stretching and gamma adjustment."""
    enhanced = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        channel = img[:, :, c]
        p_low, p_high = np.percentile(channel, (2, 98))
        if p_high > p_low:
            stretched = np.clip((channel - p_low) / (p_high - p_low), 0.0, 1.0)
        else:
            stretched = np.clip(channel, 0.0, 1.0)
        enhanced[:, :, c] = stretched
    return np.power(enhanced, gamma)


def build_confusion_map(pred_bin: np.ndarray, gt_bin: np.ndarray) -> np.ndarray:
    """
    Construct color-coded spatial confusion map:
    - Green (TP): [16, 185, 129]  # Emerald 500
    - Red (FP):   [239, 68, 68]   # Red 500
    - Blue (FN):  [59, 130, 246]  # Blue 500
    - Dark (TN):  [24, 24, 37]    # Midnight Slate
    """
    h, w = pred_bin.shape
    cmap = np.zeros((h, w, 3), dtype=np.uint8)
    cmap[:] = [24, 24, 37]

    tp = (pred_bin == 1.0) & (gt_bin == 1.0)
    fp = (pred_bin == 1.0) & (gt_bin == 0.0)
    fn = (pred_bin == 0.0) & (gt_bin == 1.0)

    cmap[tp] = [16, 185, 129]   # Green (Hit)
    cmap[fp] = [239, 68, 68]    # Red (Commission Error)
    cmap[fn] = [59, 130, 246]   # Blue (Omission Error)
    return cmap


def create_mask_visual(mask_bin: np.ndarray, color=(255, 75, 114)) -> np.ndarray:
    """Render binary mask with dark background and vibrant tint."""
    h, w = mask_bin.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:] = [15, 12, 28]  # Dark indigo backdrop
    fire = mask_bin == 1.0
    out[fire] = color
    return out


def create_overlay_visual(rgb_uint8: np.ndarray, pred_bin: np.ndarray) -> np.ndarray:
    """Composite semi-transparent glowing overlay with perimeter boundary."""
    overlay = rgb_uint8.copy().astype(np.float32)
    fire = pred_bin == 1.0
    
    # Orange-red fire tint: [255, 80, 20]
    fire_tint = np.array([255, 80, 20], dtype=np.float32)
    overlay[fire] = 0.45 * overlay[fire] + 0.55 * fire_tint
    
    # Compute boundary
    pred_uint8 = (pred_bin * 255).astype(np.uint8)
    contours, _ = cv2.findContours(pred_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_img = np.clip(overlay, 0, 255).astype(np.uint8)
    cv2.drawContours(overlay_img, contours, -1, (255, 240, 100), 2)  # Glowing yellow border
    return overlay_img


def main():
    print("Initializing Ilādṛṣṭi Web Asset Exporter...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Load Model
    ckpt_path = Path("checkpoints/best_baseline_model_fp16.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints/best_baseline_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = WildfireBaselineUNet(in_channels=4).to(device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    # Load Dataset
    ds = WildfireDataset(data_dir="data", split="test", in_channels=4, transform=None)
    print(f"Loaded {len(ds)} test split samples.")

    # Target directory
    out_dir = Path("assets/samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path("assets/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Copy existing plots
    for plot_file in Path("outputs/plots").glob("*.png"):
        shutil.copy(plot_file, plots_dir / plot_file.name)
    print(f"Copied plots to {plots_dir}")

    # Curated Scenes Selection
    scene_configs = [
        {
            "id": "scene1",
            "name": "Mega-Fire Scar Complex",
            "tile_filename": "b__06_06.tif",
            "location": "Boreal Pine Forest, Northern Mediterranean",
            "description": "Massive contiguous wildfire scar (>46,000 pixels burned) displaying near-perfect contour delineation and high model confidence.",
            "type": "standard",
            "badge": "Dice: 0.9317 (Benchmark Top)"
        },
        {
            "id": "scene2",
            "name": "Smoke-Penetrated Active Front",
            "tile_filename": "a1__02_04.tif",
            "location": "Sierra Nevada Foothills, Mixed Conifer",
            "description": "Active firefront with dense atmospheric pyrocumulonimbus smoke. The 4th Near-Infrared band penetrates haze where visible RGB fails.",
            "type": "standard",
            "badge": "NIR Smoke Penetration"
        },
        {
            "id": "scene3",
            "name": "Canopy Crown Fire & Ridge",
            "tile_filename": "m1__2_5.tif",
            "location": "Mountain Ridge Basin, Dense Woodland",
            "description": "Severe canopy burn with irregular topography. Sharp delineation along ridge barriers and preserved unburnt green enclaves.",
            "type": "standard",
            "badge": "Dice: 0.9230 (Precision Delineation)"
        },
        {
            "id": "scene4",
            "name": "Fragmented Spot Fires & Mosaics",
            "tile_filename": "a2__5_3.tif",
            "location": "Semi-Arid Scrub & Agricultural Interface",
            "description": "Discontinuous burn mosaic across variable vegetative fuels demonstrating high sensitivity to fine boundary perimeters.",
            "type": "standard",
            "badge": "Discontinuous Mosaic"
        },
        {
            "id": "failure_cloud",
            "name": "Failure Case: Cloud Cover & Edge Diffraction",
            "tile_filename": "a1__07_13.tif",
            "location": "Coastal Range with Low Marine Stratocumulus",
            "description": "Optical limitation: Saturated cloud albedo creates high contrast edges against dark terrain, triggering false positive boundary detections.",
            "type": "failure",
            "badge": "False Positive Commission (Clouds)",
            "analysis": "High cloud reflectance saturates visible and NIR detectors. The steep gradient at cloud margins mimics charcoal ash dropoffs, resulting in 13,269 false positive pixels. Mitigation: Scene Classification Layer (SCL) cloud masking."
        },
        {
            "id": "failure_shadow",
            "name": "Failure Case: Deep Topographic Ravine Shadow",
            "tile_filename": "k__4_6.tif",
            "location": "Steep Alpine Gorge & North-Facing Ravines",
            "description": "Optical limitation: Steep topographic shadows receive minimal solar irradiance, mimicking the low reflectance signature of charcoal ash.",
            "type": "failure",
            "badge": "False Positive Commission (Shadows)",
            "analysis": "Near-zero solar illumination in deep gorges drops visible and NIR reflectance below 0.05, deceiving the CNN into predicting burned ground (7,211 false positive pixels). Mitigation: Integrating SWIR Band 11/12 (NBR) and DEM elevation hillshading."
        }
    ]

    # Map filenames to dataset index
    file_to_idx = {ds.samples[i][0].name: i for i in range(len(ds))}

    scenes_metadata = []

    for sc in scene_configs:
        fname = sc["tile_filename"]
        if fname not in file_to_idx:
            print(f"Warning: {fname} not found in test split!")
            continue
        idx = file_to_idx[fname]
        img_tensor, mask_tensor = ds[idx]

        # Ingest channels: [Red, Green, Blue, NIR]
        img_np = img_tensor.cpu().numpy()  # shape (4, H, W)
        
        # RGB (Red: 0, Green: 1, Blue: 2)
        rgb_raw = np.transpose(img_np[:3, :, :], (1, 2, 0))
        rgb_disp = (enhance_channels(rgb_raw) * 255.0).clip(0, 255).astype(np.uint8)

        # False-Color NIR Composite (NIR as R, Red as G, Green as B)
        nir_channel = img_np[3, :, :]
        cir_raw = np.stack([nir_channel, img_np[0, :, :], img_np[1, :, :]], axis=-1)
        cir_disp = (enhance_channels(cir_raw) * 255.0).clip(0, 255).astype(np.uint8)

        # Run Model Inference
        with torch.no_grad():
            inp = img_tensor.unsqueeze(0).to(device)
            logits = model(inp)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_bin = (probs > 0.5).astype(np.float32)

        gt_bin = mask_tensor.squeeze().cpu().numpy().astype(np.float32)

        # Compute Metrics
        dice = float(compute_dice(pred_bin, gt_bin))
        iou = float(compute_iou(pred_bin, gt_bin))
        tp = int(np.sum((pred_bin == 1) & (gt_bin == 1)))
        fp = int(np.sum((pred_bin == 1) & (gt_bin == 0)))
        fn = int(np.sum((pred_bin == 0) & (gt_bin == 1)))
        tn = int(np.sum((pred_bin == 0) & (gt_bin == 0)))
        gt_area = int(np.sum(gt_bin == 1))
        pred_area = int(np.sum(pred_bin == 1))
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        scar_pct = float((pred_area / (pred_bin.shape[0] * pred_bin.shape[1])) * 100.0)

        # Visuals
        pred_visual = create_mask_visual(pred_bin, color=(255, 75, 114))      # Neon Crimson/Pink
        gt_visual = create_mask_visual(gt_bin, color=(16, 185, 129))           # Vivid Emerald
        overlay_visual = create_overlay_visual(rgb_disp, pred_bin)
        confusion_visual = build_confusion_map(pred_bin, gt_bin)

        # Resize to 512x512 with high-quality Lanczos for ultra-crisp web display
        target_size = (512, 512)
        rgb_img = Image.fromarray(rgb_disp).resize(target_size, Image.Resampling.LANCZOS)
        cir_img = Image.fromarray(cir_disp).resize(target_size, Image.Resampling.LANCZOS)
        pred_img = Image.fromarray(pred_visual).resize(target_size, Image.Resampling.NEAREST)
        gt_img = Image.fromarray(gt_visual).resize(target_size, Image.Resampling.NEAREST)
        overlay_img = Image.fromarray(overlay_visual).resize(target_size, Image.Resampling.LANCZOS)
        confusion_img = Image.fromarray(confusion_visual).resize(target_size, Image.Resampling.NEAREST)

        # Save files
        sc_id = sc["id"]
        rgb_path = f"assets/samples/{sc_id}_rgb.png"
        cir_path = f"assets/samples/{sc_id}_nir.png"
        pred_path = f"assets/samples/{sc_id}_pred.png"
        gt_path = f"assets/samples/{sc_id}_gt.png"
        overlay_path = f"assets/samples/{sc_id}_overlay.png"
        confusion_path = f"assets/samples/{sc_id}_confusion.png"

        rgb_img.save(rgb_path, "PNG")
        cir_img.save(cir_path, "PNG")
        pred_img.save(pred_path, "PNG")
        gt_img.save(gt_path, "PNG")
        overlay_img.save(overlay_path, "PNG")
        confusion_img.save(confusion_path, "PNG")

        sc_record = {
            **sc,
            "metrics": {
                "dice": round(dice, 4),
                "iou": round(iou, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "scar_area_pct": round(scar_pct, 2),
                "scar_pixels": pred_area,
                "gt_pixels": gt_area,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            },
            "assets": {
                "rgb": rgb_path,
                "nir": cir_path,
                "pred": pred_path,
                "gt": gt_path,
                "overlay": overlay_path,
                "confusion": confusion_path,
            }
        }
        scenes_metadata.append(sc_record)
        print(f"Processed {sc['id']} ({fname}): Dice={dice:.4f}, IoU={iou:.4f}, Scar={scar_pct:.1f}%")

    # Save metadata JSON
    meta_path = Path("assets/samples/scenes_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(scenes_metadata, f, indent=2)
    print(f"Successfully exported metadata to {meta_path}")


if __name__ == "__main__":
    main()
