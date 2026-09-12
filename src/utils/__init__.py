from src.utils.seed import set_seed, seed_worker
from src.utils.metrics import compute_dice, compute_iou, MetricTracker
from src.utils.logger import log_benchmark_result
from src.utils.losses import (
    FocalTverskyLoss,
    TverskyLoss,
    DiceLoss,
    BCEDiceLoss,
    get_loss,
)

__all__ = [
    "set_seed",
    "seed_worker",
    "compute_dice",
    "compute_iou",
    "MetricTracker",
    "log_benchmark_result",
    "FocalTverskyLoss",
    "TverskyLoss",
    "DiceLoss",
    "BCEDiceLoss",
    "get_loss",
    "run_evaluation_suite",
    "MODEL_SPECS",
]


def __getattr__(name: str):
    if name in ("run_evaluation_suite", "MODEL_SPECS"):
        from src.utils.evaluation_suite import MODEL_SPECS as _SPECS
        from src.utils.evaluation_suite import run_evaluation_suite as _run
        if name == "run_evaluation_suite":
            return _run
        return _SPECS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

