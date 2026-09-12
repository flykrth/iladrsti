# Ilādṛṣṭi (इलादृष्टि): Multispectral Wildfire & Burned Area Segmentation

[![Track](https://img.shields.io/badge/Track-6%3A%20AI%20for%20Science%20%26%20Society-orange.svg)](#)
[![Competition](https://img.shields.io/badge/Hackathon-Deep%20Learning%20Hackathon%20%40%20Amrita-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Seed](https://img.shields.io/badge/Seed-42%20Locked-purple.svg)](#reproducibility--seed-locking)

> **"Ilādṛṣṭi" (Sanskrit: इला / Earth + दृष्टि / Vision)** — An Earth Observation Deep Learning system engineered for high-precision autonomous wildfire detection and burned area delineation from multispectral Sentinel-2 satellite imagery.

Developed for **Track 6 (AI for Science & Society)** of the Deep Learning Hackathon, Amrita Vishwa Vidyapeetham.

---

## Table of Contents
1. [Benchmark Evaluation Results](#1-benchmark-evaluation-results)
2. [Mandatory Baseline Architecture](#2-mandatory-baseline-architecture)
3. [Multispectral Dataset & Band Engineering](#3-multispectral-dataset--band-engineering)
4. [Strict Split Isolation & Anti-Leakage Audit](#4-strict-split-isolation--anti-leakage-audit)
5. [Reproducibility & Deterministic Controls](#5-reproducibility--deterministic-controls)
6. [Single-Command Execution (`run.sh`)](#6-single-command-execution-runsh)
7. [Repository Structure](#7-repository-structure)
8. [Step-by-Step Reproduction Guide](#8-step-by-step-reproduction-guide)
9. [Ablation Study Roadmap](#9-ablation-study-roadmap)

---

## 1. Benchmark Evaluation Results

The **Mandatory Baseline** was trained on the training split ($N=317$), checkpointed strictly against validation Dice score, and evaluated exclusively on the isolated held-out test split ($N=40$).

### Held-Out Test Set Performance Table

| Metric / Parameter | Mandatory Baseline Value | Evaluation Details |
| :--- | :---: | :--- |
| **Test Dice Coefficient ($F_1$)** | **0.7835** | Harmonic mean of precision and recall on unseen test fires |
| **Test Mean IoU (Jaccard Index)** | **0.7046** | Strict spatial intersection over union on fire scars |
| **Test Loss (BCEWithLogits)** | **0.3781** | Mandatory comparison floor loss |
| **Best Training Epoch** | **Epoch 5 / 15** | Peak validation Dice: $0.6924$, IoU: $0.6100$ |
| **Inference Threshold** | **$0.50$** | Standard sigmoid probability threshold |
| **Seed Locked** | **$42$** | Fully deterministic across Python, NumPy, PyTorch, CuDNN |

*Persistent metric artifacts are tracked in [`outputs/benchmark_results.md`](outputs/benchmark_results.md), [`outputs/benchmark_results.csv`](outputs/benchmark_results.csv), and [`outputs/benchmark_results.json`](outputs/benchmark_results.json).*

### Validation & Training Progression

```
+---------+--------------+--------------+--------------+--------------+--------------+-----------------------+
|  Epoch  |  Train Loss  |  Train Dice  |   Val Loss   |   Val Dice   |   Val IoU    | Checkpoint Status     |
+---------+--------------+--------------+--------------+--------------+--------------+-----------------------+
|  01/15  |    0.5488    |    0.4698    |    0.6584    |    0.2157    |    0.1421    | ⭐ Best Model Saved   |
|  02/15  |    0.4466    |    0.6321    |    0.4880    |    0.3783    |    0.2987    | ⭐ Best Model Saved   |
|  03/15  |    0.4281    |    0.5896    |    0.4282    |    0.5177    |    0.4340    | ⭐ Best Model Saved   |
|  04/15  |    0.3944    |    0.6375    |    0.3941    |    0.6749    |    0.5943    | ⭐ Best Model Saved   |
|  05/15  |    0.3895    |    0.6366    |    0.3598    |  ⭐ 0.6924   |  ⭐ 0.6100   | ⭐ Best Model Saved   |
|  06/15  |    0.3707    |    0.6848    |    0.3604    |    0.6894    |    0.6091    |                       |
|  07/15  |    0.3567    |    0.6617    |    0.4002    |    0.6004    |    0.5201    |                       |
|  08/15  |    0.3437    |    0.7393    |    0.3468    |    0.6878    |    0.6113    |                       |
|  09/15  |    0.3434    |    0.7189    |    0.3440    |    0.6613    |    0.5837    |                       |
|  10/15  |    0.3294    |    0.7120    |    0.3383    |    0.6829    |    0.6061    |                       |
|  11/15  |    0.3275    |    0.7227    |    0.3297    |    0.6593    |    0.5859    |                       |
|  12/15  |    0.3231    |    0.7254    |    0.3310    |    0.6516    |    0.5772    |                       |
|  13/15  |    0.3155    |    0.7346    |    0.3308    |    0.6624    |    0.5891    |                       |
|  14/15  |    0.3112    |    0.7410    |    0.3312    |    0.6610    |    0.5878    |                       |
|  15/15  |    0.3203    |    0.7282    |    0.3315    |    0.6595    |    0.5862    |                       |
+---------+--------------+--------------+--------------+--------------+--------------+-----------------------+
```

---

## 2. Mandatory Baseline Architecture

The baseline pipeline adheres strictly to the competition constraints:

* **Backbone Encoder**: ResNet-50 initialized with pretrained ImageNet weights (`ResNet50_Weights.DEFAULT`).
* **4-Band Weight Adaptation**: Standard CNN encoders expect 3-channel RGB. To ingest the 4th multispectral band (**Near-Infrared / NIR**), the initial $7\times 7$ convolution layer (`conv1`) is adapted:
  $$\mathbf{W}_{\text{conv1}}[:, 0:3, :, :] \leftarrow \mathbf{W}_{\text{ImageNet}}$$
  $$\mathbf{W}_{\text{conv1}}[:, 3:4, :, :] \leftarrow \frac{1}{3} \sum_{c=0}^{2} \mathbf{W}_{\text{ImageNet}}[:, c:c+1, :, :]$$
  Initializing the NIR filter bank with the channel-wise mean of pretrained RGB filters preserves the activation magnitude and gradient flow.
* **Decoder**: Symmetrical 5-stage U-Net decoder with bilinear upsampling, residual skip connections from ResNet stages ($C_1, C_2, C_3, C_4$), double $3\times 3$ convolutions with Batch Normalization and ReLU, terminating in a $1\times 1$ convolution.
* **Loss Function**: Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss`), numerically stable and serving as the mandatory comparison floor.

---

## 3. Multispectral Dataset & Band Engineering

Imagery consists of European Space Agency (ESA) Sentinel-2 L2A surface reflectance tiles ($256 \times 256$ pixels).

### Band Mapping
Sentinel-2 12-band GeoTIFFs are ingested via `rasterio` and selectively filtered:
* **Band 4 (Red, $\sim 665\,\text{nm}$)**: Index 3
* **Band 3 (Green, $\sim 560\,\text{nm}$)**: Index 2
* **Band 2 (Blue, $\sim 490\,\text{nm}$)**: Index 1
* **Band 8 (Broadband NIR, $\sim 842\,\text{nm}$)**: Index 7

### Reflectance Scaling & Augmentation
* **Radiometric Calibration**: Raw Digital Numbers (DN) are normalized using Sentinel-2 L2A surface reflectance scaling:
  $$\text{Reflectance} = \text{clip}\left(\frac{\text{DN}}{10000.0}, 0.0, 1.0\right)$$
* **Synchronized Augmentations**: Albumentations spatial transforms (`HorizontalFlip(p=0.5)`, `VerticalFlip(p=0.5)`, `RandomRotate90(p=0.5)`) strictly applied simultaneously to images and masks for the `train` split only.

---

## 4. Strict Split Isolation & Anti-Leakage Audit

To prevent data contamination, the dataset was split into an 80/10/10 partition. A standalone verification tool [`verify_splits.py`](verify_splits.py) audits the data directories prior to every training run:

```
===========================================================================
      ILĀDṚṢṬI DATASET SPLIT INTEGRITY & DATA LEAKAGE AUDIT
===========================================================================
 Split 'train': 317 images | 317 masks (79.8%)
 Split 'val  ':  40 images |  40 masks (10.1%)
 Split 'test ':  40 images |  40 masks (10.1%)
---------------------------------------------------------------------------
 Total Dataset Size: 397 images | 397 masks (100%)
---------------------------------------------------------------------------
[Check 1/4] Filename Exclusivity Check:
  - Train ∩ Val  Overlap : 0 files
  - Train ∩ Test Overlap : 0 files
  - Val   ∩ Test Overlap : 0 files
  -> PASSED: All splits are strictly disjoint by filename.

[Check 2/4] SHA-256 Binary Content Collision Check:
  - Train ∩ Val  Hash Collisions : 0
  - Train ∩ Test Hash Collisions : 0
  - Val   ∩ Test Hash Collisions : 0
  -> PASSED: Zero identical image data duplicated across splits.

[Check 3/4] 1-to-1 Image-Mask Alignment & Geometry:
  -> PASSED: All 397 pairs strictly matched (256x256 resolution, 12 imagery bands, 1 mask band).

[Check 4/4] Leakage Protection Status:
  -> PASSED: Held-out test split is strictly isolated. Safe for ablation studies.
===========================================================================
```

---

## 5. Reproducibility & Deterministic Controls

Every random number generator in the pipeline is locked to seed **`42`** via [`src/utils/seed.py`](src/utils/seed.py):
* `random.seed(42)`
* `os.environ["PYTHONHASHSEED"] = "42"`
* `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"`
* `np.random.seed(42)`
* `torch.manual_seed(42)`
* `torch.cuda.manual_seed_all(42)`
* `torch.backends.cudnn.deterministic = True`
* `torch.backends.cudnn.benchmark = False`
* `worker_init_fn = seed_worker` ensures deterministic multi-process DataLoader batching.

---

## 6. Single-Command Execution (`run.sh`)

To reproduce the entire baseline from scratch in a single terminal command:

```bash
chmod +x run.sh
./run.sh
```

The script automatically:
1. Locks all deterministic environment variables (`PYTHONHASHSEED=42`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`).
2. Installs dependencies from `requirements.txt`.
3. Runs the strict split isolation audit (`verify_splits.py`).
4. Executes baseline training (`train.py`) saving `checkpoints/best_baseline_model.pt`.
5. Executes decoupled test evaluation (`evaluate.py`) on held-out test data.
6. Formats and outputs the final metrics table.

---

## 7. Repository Structure

```
iladrsti/
├── requirements.txt                   # Dependency specifications
├── run.sh                             # Single-command runner (Setup -> Audit -> Train -> Eval)
├── split.py                           # 80/10/10 reproducible data partitioner
├── train.py                           # Baseline training loop with BCE & val checkpointing
├── evaluate.py                        # Decoupled held-out test inference with formatted table
├── test_baseline.py                   # Automated unit & integration verification test suite
├── verify_splits.py                   # Automated split exclusivity and data leakage auditor
├── README.md                          # Comprehensive project documentation
├── .gitignore                         # Git exclusion rules
├── configs/
│   └── baseline_config.yaml           # Centralized configuration parameters
├── checkpoints/
│   ├── best_baseline_model_fp16.pt    # Lightweight FP16 checkpoint (83.9MB, repo-hosted)
│   └── best_baseline_model.pt         # Full FP32 checkpoint with optimizer state (Release asset)
├── outputs/
│   ├── benchmark_results.md           # Markdown results registry for reports
│   ├── benchmark_results.csv          # Structured CSV results for automated comparison
│   ├── benchmark_results.json         # JSON metrics registry
│   └── training_history.json          # Epoch-by-epoch loss, dice, and learning rate curves
└── src/
    ├── __init__.py
    ├── dataset/
    │   ├── __init__.py
    │   └── multispectral_dataset.py   # Rasterio GeoTIFF reader, Sentinel-2 scaling & Albumentations
    ├── models/
    │   ├── __init__.py
    │   └── baseline_unet.py           # ResNet-50 U-Net with 3-to-4 channel transfer learning
    └── utils/
        ├── __init__.py
        ├── seed.py                    # Deterministic seed locker (seed=42)
        ├── metrics.py                 # Pure tensor Dice and IoU metric engine
        └── logger.py                  # Synchronized benchmark table writer
```

---

## 8. Step-by-Step Reproduction Guide

### Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Verification Test Suite
```bash
python test_baseline.py
```
*Validates determinism, tensor metrics, band loading, transfer weights, and dataloaders.*

### Run Data Leakage Audit
```bash
python verify_splits.py
```

### Train Mandatory Baseline
```bash
python train.py --data_dir data --epochs 15 --batch_size 16 --lr 2e-4 --in_channels 4
```

### Run Decoupled Test Evaluation
```bash
python evaluate.py --data_dir data --checkpoint checkpoints/best_baseline_model_fp16.pt --in_channels 4
```

---

## 9. Ablation Study Roadmap

With the mandatory baseline established and verified, the following ablation experiments are prepared:

1. **Spectral Ablation (3-Band RGB vs 4-Band RGB+NIR)**:
   Quantify the empirical contribution of Near-Infrared surface reflectance in penetrating smoke plumes and differentiating active fire scars.
   ```bash
   python train.py --in_channels 3 --epochs 15
   ```
2. **Loss Function Ablation**:
   Benchmark BCE against Dice Loss, Focal Loss, and Compound Combo Loss ($\mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}}$) to address severe foreground/background spatial imbalance.
3. **Architectural Backbone Ablation**:
   Compare ResNet-50 against EfficientNet-B4 and SegFormer/MiT-B2 encoders.

---

## Citation & Acknowledgments
* Deep Learning Hackathon — Track 6: AI for Science & Society, Amrita Vishwa Vidyapeetham.
* European Space Agency (ESA) Copernicus Sentinel-2 Open Access Data.
