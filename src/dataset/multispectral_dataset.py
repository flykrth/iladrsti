"""
Multispectral Wildfire Dataset Module for Ilādṛṣṭi.
Provides PyTorch Dataset and DataLoader loaders reading GeoTIFF imagery using Rasterio
and applying synchronized Albumentations augmentations.
"""

import os
from pathlib import Path
from typing import Optional, Sequence, Union

import albumentations as A
import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.seed import seed_worker


class WildfireDataset(Dataset):
    """
    Multispectral Wildfire & Burned Area Dataset.

    Reads multispectral GeoTIFF (.tif) satellite imagery using Rasterio.
    Supports toggling between 3-band (RGB) and 4-band (RGB + NIR) inputs.
    Applies spatial augmentations to train split only.
    """

    # Default Sentinel-2 L2A band indices (0-indexed):
    # Band 4 (Red) -> index 3
    # Band 3 (Green) -> index 2
    # Band 2 (Blue) -> index 1
    # Band 8 (NIR) -> index 7
    S2_RGB_INDICES = [3, 2, 1]
    S2_RGBNIR_INDICES = [3, 2, 1, 7]

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        in_channels: int = 4,
        band_indices: Optional[Sequence[int]] = None,
        transform: Optional[A.Compose] = None,
        normalize: bool = True,
        scale_factor: float = 10000.0,
    ) -> None:
        """
        Args:
            data_dir (str or Path): Path to base data directory or split directory.
            split (str): One of 'train', 'val', or 'test'.
            in_channels (int): 3 for RGB, 4 for RGB + NIR.
            band_indices (list of int, optional): Explicit band indices (0-indexed).
            transform (A.Compose, optional): Custom albumentations transforms.
            normalize (bool): Whether to normalize reflectance values.
            scale_factor (float): Divisor for Sentinel-2 DN to surface reflectance [0, 1].
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split.lower().strip()
        self.in_channels = in_channels
        self.normalize = normalize
        self.scale_factor = scale_factor
        self.user_band_indices = band_indices

        if self.in_channels not in (3, 4):
            raise ValueError(f"in_channels must be 3 or 4, got {in_channels}")

        # Resolve images and masks directories
        if (self.data_dir / self.split / "images").exists():
            self.img_dir = self.data_dir / self.split / "images"
            self.msk_dir = self.data_dir / self.split / "masks"
        elif (self.data_dir / "images").exists():
            self.img_dir = self.data_dir / "images"
            self.msk_dir = self.data_dir / "masks"
        else:
            raise FileNotFoundError(
                f"Could not locate images and masks in {self.data_dir} for split '{self.split}'"
            )

        # Collect and pair files
        self.samples = self._load_samples()
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid image/mask pairs found in {self.img_dir}")

        # Setup augmentations
        if transform is not None:
            self.transform = transform
        elif self.split == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                ]
            )
        else:
            self.transform = None

    def _load_samples(self) -> list[tuple[Path, Path]]:
        """Pair image files with corresponding mask files."""
        valid_extensions = (".tif", ".tiff", ".TIF", ".TIFF")
        all_imgs = sorted([
            f for f in self.img_dir.iterdir() if f.suffix in valid_extensions
        ])

        pairs = []
        for img_path in all_imgs:
            # Check identical name first
            mask_path = self.msk_dir / img_path.name
            if not mask_path.exists():
                # Check single underscore alternative
                alt_name = img_path.name.replace("__", "_")
                mask_path = self.msk_dir / alt_name

            if not mask_path.exists():
                # Check double underscore alternative
                alt_name2 = img_path.name.replace("_", "__")
                mask_path = self.msk_dir / alt_name2

            if mask_path.exists():
                pairs.append((img_path, mask_path))
            else:
                print(f"[Warning] Mask missing for image {img_path.name}, skipping.")

        return pairs

    def _determine_band_indices(self, total_bands: int) -> list[int]:
        """Determine which bands to extract based on input channels and file band count."""
        if self.user_band_indices is not None:
            if len(self.user_band_indices) != self.in_channels:
                raise ValueError(
                    f"User band_indices length ({len(self.user_band_indices)}) "
                    f"does not match in_channels ({self.in_channels})"
                )
            return list(self.user_band_indices)

        # Multi-band Sentinel-2 file (>= 12 bands)
        if total_bands >= 12:
            return self.S2_RGBNIR_INDICES if self.in_channels == 4 else self.S2_RGB_INDICES

        # Exactly 4 bands in file: Assume [Red, Green, Blue, NIR]
        if total_bands == 4:
            return [0, 1, 2, 3] if self.in_channels == 4 else [0, 1, 2]

        # Exactly 3 bands in file (RGB)
        if total_bands == 3:
            if self.in_channels == 4:
                raise ValueError(
                    f"File has only 3 bands, cannot extract 4 channels without synthetic NIR."
                )
            return [0, 1, 2]

        # Generic fallback: take first N channels
        return list(range(self.in_channels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load single sample.

        Returns:
            image (torch.Tensor): Shape (C, H, W), float32 normalized reflectances.
            mask (torch.Tensor): Shape (1, H, W), float32 binary mask {0.0, 1.0}.
        """
        img_path, msk_path = self.samples[idx]

        # 1. Read multispectral image via Rasterio
        with rasterio.open(img_path) as src:
            total_bands = src.count
            band_idxs = self._determine_band_indices(total_bands)
            # Rasterio indexes bands 1-based
            rasterio_1based_idxs = [b + 1 for b in band_idxs]
            image = src.read(rasterio_1based_idxs).astype(np.float32)  # (C, H, W)

        # 2. Read mask via Rasterio
        with rasterio.open(msk_path) as src:
            mask = src.read(1).astype(np.float32)  # (H, W)

        # Clean NaNs / Infs
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)

        # 3. Normalization (Sentinel-2 DN / scale_factor clipped to [0, 1])
        if self.normalize:
            if np.nanmax(image) > 1.5:  # Raw digital numbers (typically 0 - 10000)
                image = np.clip(image / self.scale_factor, 0.0, 1.0)
            else:
                image = np.clip(image, 0.0, 1.0)

        # 4. Mask binarization: strictly 0.0 or 1.0
        mask = (mask > 0.5).astype(np.float32)

        # 5. Spatial Augmentations (Albumentations requires H, W, C)
        image_hwc = np.transpose(image, (1, 2, 0))  # (H, W, C)

        if self.transform is not None:
            augmented = self.transform(image=image_hwc, mask=mask)
            image_hwc = augmented["image"]
            mask = augmented["mask"]

        # Back to (C, H, W)
        image_chw = np.transpose(image_hwc, (2, 0, 1))

        # Convert to PyTorch tensors
        img_tensor = torch.from_numpy(np.ascontiguousarray(image_chw)).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0).float()

        return img_tensor, mask_tensor


def get_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 16,
    in_channels: int = 4,
    num_workers: int = 2,
    pin_memory: bool = True,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """
    Construct reproducible DataLoaders for train, val, and test splits.

    Args:
        data_dir (str or Path): Root data folder.
        batch_size (int): Mini-batch size.
        in_channels (int): 3 or 4 channels.
        num_workers (int): Number of worker subprocesses.
        pin_memory (bool): Pin memory for faster GPU transfer.
        seed (int): Reproducibility seed.

    Returns:
        dict[str, DataLoader]: Dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    g = torch.Generator()
    g.manual_seed(seed)

    splits = ["train", "val", "test"]
    dataloaders = {}

    for split in splits:
        is_train = split == "train"
        dataset = WildfireDataset(
            data_dir=data_dir,
            split=split,
            in_channels=in_channels,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=g if is_train else None,
            drop_last=is_train,
        )
        dataloaders[split] = loader

    return dataloaders
