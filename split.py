import os
import shutil
import random
from pathlib import Path

# Lock the seed for hackathon reproducibility requirements
random.seed(42)

# Define source directories
raw_img_dir = Path('data/raw/images')
raw_msk_dir = Path('data/raw/masks')

# Ensure directories exist
if not raw_img_dir.exists() or not raw_msk_dir.exists():
    raise FileNotFoundError("Raw images or masks not found in data/raw/")

# 1. Filter for only valid Image/Mask pairs
valid_image_files = []
all_images = sorted([f for f in os.listdir(raw_img_dir) if f.endswith('.tif')])

for filename in all_images:
    # Map double underscore to single underscore for checking
    mask_filename = filename.replace('__', '_')
    
    # Only keep the image if the mask actually exists
    if (raw_msk_dir / mask_filename).exists():
        valid_image_files.append(filename)

print(f"Found {len(valid_image_files)} perfect image/mask pairs out of {len(all_images)} total images.")

# 2. Shuffle the clean list
random.shuffle(valid_image_files)

# 3. Calculate accurate split indices (80/10/10) on the clean data
total = len(valid_image_files)
train_idx = int(total * 0.8)
val_idx = int(total * 0.9)

splits = {
    'train': valid_image_files[:train_idx],
    'val': valid_image_files[train_idx:val_idx],
    'test': valid_image_files[val_idx:]
}

# 4. Copy files to their new destinations
for split_name, files in splits.items():
    print(f"Copying {len(files)} files to {split_name}...")
    for filename in files:
        dest_img_dir = Path(f'data/{split_name}/images')
        dest_msk_dir = Path(f'data/{split_name}/masks')
        
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_msk_dir.mkdir(parents=True, exist_ok=True)
        
        mask_filename = filename.replace('__', '_')
        
        # Copy image and mask, enforcing identical naming in the final directories
        shutil.copy2(raw_img_dir / filename, dest_img_dir / filename)
        shutil.copy2(raw_msk_dir / mask_filename, dest_msk_dir / filename)

print("Split completed successfully!")
