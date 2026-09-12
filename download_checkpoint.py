"""
Download Utility for Full FP32 Baseline Checkpoint (502 MB).
Retrieves the complete training checkpoint (with optimizer states)
from the official GitHub Release asset.
"""

import os
import urllib.request
from pathlib import Path

RELEASE_URL = "https://github.com/flykrth/iladrsti/releases/download/v1.0.0-baseline/best_baseline_model.pt"
DEST_PATH = Path("checkpoints/best_baseline_model.pt")


def download_checkpoint():
    if DEST_PATH.exists() and DEST_PATH.stat().st_size > 400 * 1024 * 1024:
        print(f"[Ilādṛṣṭi] Checkpoint already exists at: {DEST_PATH}")
        return

    DEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Ilādṛṣṭi] Downloading full FP32 baseline checkpoint from:")
    print(f"            {RELEASE_URL}")

    def progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rDownloading: {percent}% ({count * block_size / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB)", end="")

    try:
        urllib.request.urlretrieve(RELEASE_URL, DEST_PATH, reporthook=progress)
        print(f"\n[Ilādṛṣṭi] Download complete! Saved to: {DEST_PATH}")
    except Exception as e:
        print(f"\n[Warning] Could not download from release URL: {e}")
        print("Note: The repository already includes checkpoints/best_baseline_model_fp16.pt for immediate inference.")


if __name__ == "__main__":
    download_checkpoint()
