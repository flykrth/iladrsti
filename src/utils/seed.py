"""
Seed locking and reproducibility utilities for Ilādṛṣṭi Wildfire Segmentation.
Strictly ensures deterministic behavior across Python, NumPy, PyTorch, and CuDNN.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set deterministic seeds across all random number generators.

    Args:
        seed (int): The master seed to lock. Default is 42.
    """
    # 1. Python built-in random and environment hash seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # 2. NumPy random generator
    np.random.seed(seed)

    # 3. PyTorch CPU & CUDA
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 4. CuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 5. PyTorch deterministic operations (if supported by active ops)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


def seed_worker(worker_id: int) -> None:
    """
    Worker initialization function for PyTorch DataLoader to ensure
    multi-process data loading determinism.

    Args:
        worker_id (int): Worker process index.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
