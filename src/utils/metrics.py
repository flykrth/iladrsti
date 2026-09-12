"""
Evaluation metrics for binary wildfire segmentation in PyTorch.
Implements tensor-based Dice Coefficient and Intersection over Union (IoU).
"""

from typing import Union
import torch


def _prepare_tensors(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Ensure predictions and targets are binarized float tensors with compatible shapes.

    Args:
        preds (torch.Tensor): Model predictions (logits or probabilities).
        targets (torch.Tensor): Ground truth masks (binary or {0, 1}).
        threshold (float): Classification threshold for probabilities.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Binarized (preds_bin, targets_bin) of shape (B, -1).
    """
    if not isinstance(preds, torch.Tensor):
        preds = torch.as_tensor(preds)
    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets, device=preds.device)

    # If predictions are logits (contain values outside [0, 1]), apply sigmoid
    if preds.min() < 0.0 or preds.max() > 1.0:
        preds = torch.sigmoid(preds)

    # Threshold predictions
    preds_bin = (preds >= threshold).float()
    targets_bin = (targets >= 0.5).float()

    # Flatten spatial dimensions: (B, C, H, W) -> (B, C * H * W)
    if preds_bin.dim() > 1:
        preds_flat = preds_bin.reshape(preds_bin.size(0), -1)
        targets_flat = targets_bin.reshape(targets_bin.size(0), -1)
    else:
        preds_flat = preds_bin.unsqueeze(0)
        targets_flat = targets_bin.unsqueeze(0)

    return preds_flat, targets_flat


def compute_dice(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Compute Dice Coefficient (F1-score) for binary segmentation.
    Formula: (2 * |X ∩ Y| + eps) / (|X| + |Y| + eps)

    Args:
        preds (torch.Tensor): Predicted tensor (logits, probabilities, or binary).
        targets (torch.Tensor): Ground truth tensor (binary {0, 1}).
        threshold (float): Classification threshold for probabilities. Default: 0.5.
        eps (float): Epsilon for numerical stability. Default: 1e-7.

    Returns:
        torch.Tensor: Mean Dice coefficient across the batch (scalar tensor).
    """
    p, t = _prepare_tensors(preds, targets, threshold=threshold)
    intersection = (p * t).sum(dim=1)
    cardinality = p.sum(dim=1) + t.sum(dim=1)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return dice.mean()


def compute_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Compute Intersection over Union (IoU / Jaccard Index) for binary segmentation.
    Formula: (|X ∩ Y| + eps) / (|X ∪ Y| + eps)

    Args:
        preds (torch.Tensor): Predicted tensor (logits, probabilities, or binary).
        targets (torch.Tensor): Ground truth tensor (binary {0, 1}).
        threshold (float): Classification threshold for probabilities. Default: 0.5.
        eps (float): Epsilon for numerical stability. Default: 1e-7.

    Returns:
        torch.Tensor: Mean IoU score across the batch (scalar tensor).
    """
    p, t = _prepare_tensors(preds, targets, threshold=threshold)
    intersection = (p * t).sum(dim=1)
    union = p.sum(dim=1) + t.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()


class MetricTracker:
    """
    Tracks running metric totals over batches for accurate epoch-level reporting.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.total_loss = 0.0
        self.total_dice = 0.0
        self.total_iou = 0.0
        self.count = 0

    def update(
        self,
        loss: float,
        dice: Union[float, torch.Tensor],
        iou: Union[float, torch.Tensor],
        batch_size: int = 1,
    ) -> None:
        dice_val = float(dice.item()) if isinstance(dice, torch.Tensor) else float(dice)
        iou_val = float(iou.item()) if isinstance(iou, torch.Tensor) else float(iou)
        self.total_loss += loss * batch_size
        self.total_dice += dice_val * batch_size
        self.total_iou += iou_val * batch_size
        self.count += batch_size

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            return {"loss": 0.0, "dice": 0.0, "iou": 0.0}
        return {
            "loss": self.total_loss / self.count,
            "dice": self.total_dice / self.count,
            "iou": self.total_iou / self.count,
        }
