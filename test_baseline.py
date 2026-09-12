"""
Unit and Integration Test Suite for Ilādṛṣṭi Mandatory Baseline Pipeline.
Verifies reproducibility, dataset loading, model forward pass, weight adaptation, and metrics.
"""

import sys
import torch
import numpy as np
from pathlib import Path

from src.utils.seed import set_seed, seed_worker
from src.utils.metrics import compute_dice, compute_iou, MetricTracker
from src.dataset.multispectral_dataset import WildfireDataset, get_dataloaders
from src.models.baseline_unet import WildfireBaselineUNet


def test_seed_determinism():
    print("[Test 1/5] Testing deterministic seed locking...")
    set_seed(42)
    t1 = torch.randn(5, 5)
    np1 = np.random.rand(5)

    set_seed(42)
    t2 = torch.randn(5, 5)
    np2 = np.random.rand(5)

    assert torch.equal(t1, t2), "PyTorch tensors are not identical under seed 42!"
    assert np.array_equal(np1, np2), "NumPy arrays are not identical under seed 42!"
    print("  -> Passed: Seed locking is strictly reproducible.")


def test_metrics():
    print("[Test 2/5] Testing tensor metrics (Dice & IoU)...")
    # Case 1: Perfect overlap
    targets = torch.tensor([[1.0, 1.0], [0.0, 0.0]]).unsqueeze(0).unsqueeze(0)
    preds = torch.tensor([[1.0, 1.0], [0.0, 0.0]]).unsqueeze(0).unsqueeze(0)
    dice = compute_dice(preds, targets)
    iou = compute_iou(preds, targets)
    assert abs(dice.item() - 1.0) < 1e-4, f"Expected Dice 1.0, got {dice.item()}"
    assert abs(iou.item() - 1.0) < 1e-4, f"Expected IoU 1.0, got {iou.item()}"

    # Case 2: Zero overlap
    preds_zero = torch.tensor([[0.0, 0.0], [1.0, 1.0]]).unsqueeze(0).unsqueeze(0)
    dice_zero = compute_dice(preds_zero, targets)
    iou_zero = compute_iou(preds_zero, targets)
    assert abs(dice_zero.item() - 0.0) < 1e-4, f"Expected Dice 0.0, got {dice_zero.item()}"
    assert abs(iou_zero.item() - 0.0) < 1e-4, f"Expected IoU 0.0, got {iou_zero.item()}"

    # Case 3: MetricTracker
    tracker = MetricTracker()
    tracker.update(loss=0.5, dice=dice, iou=iou, batch_size=2)
    summary = tracker.compute()
    assert abs(summary["loss"] - 0.5) < 1e-5
    assert abs(summary["dice"] - 1.0) < 1e-5
    print("  -> Passed: Metrics compute strictly correct Dice and IoU.")


def test_dataset_loading():
    print("[Test 3/5] Testing WildfireDataset with 4-band and 3-band modes...")
    data_dir = "data"
    if not Path(data_dir).exists():
        print("  -> Skipping: 'data' folder not found.")
        return

    # 4-band test
    ds4 = WildfireDataset(data_dir=data_dir, split="train", in_channels=4)
    img4, msk4 = ds4[0]
    assert img4.shape == (4, 256, 256), f"Expected img shape (4, 256, 256), got {img4.shape}"
    assert msk4.shape == (1, 256, 256), f"Expected mask shape (1, 256, 256), got {msk4.shape}"
    assert img4.min() >= 0.0 and img4.max() <= 1.0, f"Image reflectances out of range [0, 1]: min={img4.min()}, max={img4.max()}"
    unique_mask_vals = torch.unique(msk4).tolist()
    for v in unique_mask_vals:
        assert v in (0.0, 1.0), f"Mask contains non-binary value: {v}"

    # 3-band test
    ds3 = WildfireDataset(data_dir=data_dir, split="val", in_channels=3)
    img3, msk3 = ds3[0]
    assert img3.shape == (3, 256, 256), f"Expected img shape (3, 256, 256), got {img3.shape}"
    print(f"  -> Passed: Dataset correctly reads 4-band ({img4.shape}) and 3-band ({img3.shape}) GeoTIFFs.")


def test_model_forward():
    print("[Test 4/5] Testing WildfireBaselineUNet architecture & 4-channel adaptation...")
    # 4-band model
    model4 = WildfireBaselineUNet(in_channels=4, num_classes=1, pretrained=True)
    dummy_input4 = torch.randn(2, 4, 256, 256)
    logits4 = model4(dummy_input4, return_logits=True)
    probs4 = model4.predict(dummy_input4)
    assert logits4.shape == (2, 1, 256, 256), f"Expected logits (2, 1, 256, 256), got {logits4.shape}"
    assert probs4.shape == (2, 1, 256, 256), f"Expected probs (2, 1, 256, 256), got {probs4.shape}"
    assert probs4.min() >= 0.0 and probs4.max() <= 1.0, "Probabilities outside [0, 1]"

    # Check 4th channel initialization (should equal mean of RGB channels in conv1)
    old_conv1_mean = model4.init_conv[0].weight[:, :3, :, :].mean(dim=1, keepdim=True)
    nir_conv1 = model4.init_conv[0].weight[:, 3:4, :, :]
    diff = torch.max(torch.abs(old_conv1_mean - nir_conv1)).item()
    assert diff < 1e-5, f"NIR channel not initialized from RGB mean! Diff={diff}"

    # 3-band model
    model3 = WildfireBaselineUNet(in_channels=3, num_classes=1, pretrained=True)
    dummy_input3 = torch.randn(2, 3, 256, 256)
    logits3 = model3(dummy_input3)
    assert logits3.shape == (2, 1, 256, 256)
    print("  -> Passed: Model forward pass, shapes, and 4-channel transfer weights verified.")


def test_dataloaders():
    print("[Test 5/5] Testing reproducible DataLoaders...")
    data_dir = "data"
    if not Path(data_dir).exists():
        return
    loaders = get_dataloaders(data_dir=data_dir, batch_size=4, in_channels=4, num_workers=0)
    batch_img, batch_msk = next(iter(loaders["train"]))
    assert batch_img.shape == (4, 4, 256, 256)
    assert batch_msk.shape == (4, 1, 256, 256)
    print(f"  -> Passed: DataLoaders produce batch shape {batch_img.shape} and {batch_msk.shape}.")


if __name__ == "__main__":
    print("==================================================")
    print(" Running Ilādṛṣṭi Baseline Test Suite")
    print("==================================================")
    test_seed_determinism()
    test_metrics()
    test_dataset_loading()
    test_model_forward()
    test_dataloaders()
    print("==================================================")
    print(" ALL TESTS PASSED SUCCESSFULLY! ")
    print("==================================================")
