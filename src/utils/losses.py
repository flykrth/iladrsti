"""
Loss functions for binary wildfire segmentation in PyTorch.
Implements the Focal-Tversky Loss, Tversky Loss, Dice Loss, and factory helpers.

Mathematical Formulation:
Let TP = True Positives, FN = False Negatives, FP = False Positives.
Tversky Index (TI):
    TI = (TP + eps) / (TP + alpha * FN + beta * FP + eps)

Focal-Tversky Loss (FTL):
    FTL = (1 - TI) ** gamma

Parameters:
    alpha: Weight penalizing False Negatives (missed fire scars; higher boosts recall).
    beta: Weight penalizing False Positives (false alarms; higher boosts precision).
    gamma: Focusing exponent penalizing hard, ambiguous boundary pixels (gamma >= 1.0).
"""

from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
    """
    Focal-Tversky Loss for addressing extreme foreground-background class imbalance
    and penalizing hard-to-segment lesions/scars.

    References:
        Abraham & Khan (2019): "A Novel Focal Tversky Loss Function With Improved
        Attention U-Net for Lesion Segmentation". IEEE ISBI.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 1.333,
        smooth: float = 1e-6,
        from_logits: bool = True,
        reduction: str = "mean",
    ) -> None:
        """
        Args:
            alpha (float): Multiplier for False Negatives (missed fire regions). Default: 0.7.
            beta (float): Multiplier for False Positives (false alarms). Default: 0.3.
            gamma (float): Focal parameter to focus on hard examples. Default: 1.333 (4/3).
            smooth (float): Numerical stability smoothing epsilon. Default: 1e-6.
            from_logits (bool): If True, applies sigmoid to raw model output. Default: True.
            reduction (str): 'mean', 'sum', or 'none'. Default: 'mean'.
        """
        super().__init__()
        if alpha < 0.0 or beta < 0.0:
            raise ValueError(f"alpha and beta must be non-negative, got alpha={alpha}, beta={beta}")
        if gamma <= 0.0:
            raise ValueError(f"gamma must be strictly positive, got gamma={gamma}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction: {reduction}. Choose 'mean', 'sum', or 'none'.")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.smooth = float(smooth)
        self.from_logits = from_logits
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal-Tversky Loss.

        Args:
            inputs (torch.Tensor): Predicted logits or probabilities of shape (B, 1, H, W) or (B, H, W).
            targets (torch.Tensor): Ground truth binary mask {0, 1} of shape (B, 1, H, W) or (B, H, W).

        Returns:
            torch.Tensor: Computed Focal-Tversky Loss.
        """
        if self.from_logits:
            probs = torch.sigmoid(inputs)
        else:
            probs = inputs

        # Ensure target matches tensor dtype and device
        targets = targets.to(dtype=probs.dtype, device=probs.device)

        # Standardize shapes to (B, -1) for per-sample spatial aggregation
        if probs.dim() > 1:
            batch_size = probs.size(0)
            probs_flat = probs.reshape(batch_size, -1)
            targets_flat = targets.reshape(batch_size, -1)
        else:
            probs_flat = probs.unsqueeze(0)
            targets_flat = targets.unsqueeze(0)

        # Calculate True Positives, False Negatives, and False Positives per sample
        tp = (probs_flat * targets_flat).sum(dim=1)
        fn = ((1.0 - probs_flat) * targets_flat).sum(dim=1)
        fp = (probs_flat * (1.0 - targets_flat)).sum(dim=1)

        # Tversky Index
        numerator = tp + self.smooth
        denominator = tp + self.alpha * fn + self.beta * fp + self.smooth
        ti = numerator / denominator

        # Clamp (1 - TI) to prevent floating-point underflow / negative values before power
        one_minus_ti = torch.clamp(1.0 - ti, min=1e-7, max=1.0)
        ftl = torch.pow(one_minus_ti, self.gamma)

        if self.reduction == "mean":
            return ftl.mean()
        elif self.reduction == "sum":
            return ftl.sum()
        else:
            return ftl

    def extra_repr(self) -> str:
        return (
            f"alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}, "
            f"smooth={self.smooth}, from_logits={self.from_logits}, reduction='{self.reduction}'"
        )


class TverskyLoss(FocalTverskyLoss):
    """
    Standard Tversky Loss (Focal-Tversky with gamma=1.0).
    Loss = 1 - TI
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        smooth: float = 1e-6,
        from_logits: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__(
            alpha=alpha,
            beta=beta,
            gamma=1.0,
            smooth=smooth,
            from_logits=from_logits,
            reduction=reduction,
        )


class DiceLoss(FocalTverskyLoss):
    """
    Sørensen-Dice Loss.
    Corresponds to Tversky Loss with alpha=0.5, beta=0.5, gamma=1.0.
    """

    def __init__(
        self,
        smooth: float = 1e-6,
        from_logits: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__(
            alpha=0.5,
            beta=0.5,
            gamma=1.0,
            smooth=smooth,
            from_logits=from_logits,
            reduction=reduction,
        )


class BCEDiceLoss(nn.Module):
    """
    Compound loss combining BCEWithLogitsLoss and DiceLoss.
    Useful for balancing pixel-level cross-entropy with region-level overlap.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth, from_logits=True)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def get_loss(
    loss_name: str,
    alpha: float = 0.7,
    beta: float = 0.3,
    gamma: float = 1.333,
    smooth: float = 1e-6,
    from_logits: bool = True,
) -> nn.Module:
    """
    Loss factory helper function.

    Args:
        loss_name (str): One of 'focal_tversky', 'tversky', 'bce', 'dice', 'bce_dice'.
        alpha (float): False negative penalty for (Focal) Tversky.
        beta (float): False positive penalty for (Focal) Tversky.
        gamma (float): Focusing exponent for Focal Tversky.
        smooth (float): Smoothing term for stability.
        from_logits (bool): Whether input contains unnormalized logits.

    Returns:
        nn.Module: Instantiated PyTorch loss function.
    """
    name = loss_name.lower().strip()
    if name in ("focal_tversky", "ftl"):
        return FocalTverskyLoss(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            smooth=smooth,
            from_logits=from_logits,
        )
    elif name in ("tversky", "tl"):
        return TverskyLoss(
            alpha=alpha,
            beta=beta,
            smooth=smooth,
            from_logits=from_logits,
        )
    elif name in ("bce", "bce_with_logits", "bcewithlogitsloss"):
        return nn.BCEWithLogitsLoss()
    elif name in ("dice", "diceloss"):
        return DiceLoss(smooth=smooth, from_logits=from_logits)
    elif name in ("bce_dice", "combo", "bcedice"):
        return BCEDiceLoss(smooth=smooth)
    else:
        raise ValueError(
            f"Unsupported loss '{loss_name}'. Available: 'focal_tversky', 'tversky', 'bce', 'dice', 'bce_dice'"
        )
