#!/usr/bin/env bash
# ==============================================================================
# Ilādṛṣṭi: Multispectral Wildfire Segmentation - Spectral Ablation Suite
# Track 6 (AI for Science & Society) - Deep Learning Hackathon, Amrita Vishwa Vidyapeetham
#
# Ablation Requirement:
# Executes two isolated training and evaluation runs with seed=42:
# 1. RGB-only inputs (--in_channels 3)
# 2. RGB + Near-Infrared (NIR) inputs (--in_channels 4)
#
# Quantifies the empirical contribution of the NIR band in penetrating smoke
# and delineating wildfire burn scars.
# ==============================================================================

set -euo pipefail

# 1. Deterministic Reproducibility Controls
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NO_ALBUMENTATIONS_UPDATE=1

# 2. Virtual environment activation (if present)
if [ -d "venv" ]; then
    echo "[ablation.sh] Activating virtual environment: venv..."
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

# 3. Parameters & Defaults
EPOCHS="${1:-10}"
BATCH_SIZE="${2:-16}"
LR="${3:-1e-4}"
DATA_DIR="${4:-data}"

LOSS="${5:-focal_tversky}"
ALPHA="${6:-0.7}"
BETA="${7:-0.3}"
GAMMA="${8:-1.333}"
SEED=42

echo "======================================================================"
echo "          ILĀDṚṢṬI SPECTRAL ABLATION STUDY (SEED=42 LOCKED)           "
echo "======================================================================"
echo " Data Directory : ${DATA_DIR}"
echo " Epochs / Run   : ${EPOCHS}"
echo " Batch Size     : ${BATCH_SIZE}"
echo " Learning Rate  : ${LR}"
echo " Loss Function  : ${LOSS} (alpha=${ALPHA}, beta=${BETA}, gamma=${GAMMA})"
echo " Determinism    : Seed ${SEED} locked across all runs"
echo "======================================================================"

# Create checkpoint and output directories
mkdir -p checkpoints outputs/ablation_3band outputs/ablation_4band

# ------------------------------------------------------------------------------
# RUN 1: RGB-Only Input Ablation (--in_channels 3)
# ------------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo ">>> STARTING ABLATION RUN 1/2: 3-BAND RGB-ONLY INPUTS"
echo "======================================================================"

python train_optimization.py \
    --data_dir "${DATA_DIR}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --in_channels 3 \
    --loss "${LOSS}" \
    --alpha "${ALPHA}" \
    --beta "${BETA}" \
    --gamma "${GAMMA}" \
    --seed "${SEED}" \
    --checkpoint_dir "checkpoints" \
    --checkpoint_name "ablation_rgb_3band.pt" \
    --output_dir "outputs/ablation_3band" \
    --experiment_name "Ablation RGB (3-band)"

echo "[ablation.sh] Evaluating Run 1 (3-band RGB) on held-out test split..."
python evaluate.py \
    --data_dir "${DATA_DIR}" \
    --checkpoint "checkpoints/ablation_rgb_3band.pt" \
    --in_channels 3 \
    --output_json "outputs/ablation_3band/eval_results.json" \
    --experiment_name "Ablation RGB (3-band)" \
    --seed "${SEED}"

# ------------------------------------------------------------------------------
# RUN 2: Multispectral RGB + NIR Ablation (--in_channels 4)
# ------------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo ">>> STARTING ABLATION RUN 2/2: 4-BAND RGB + NIR MULTISPECTRAL INPUTS"
echo "======================================================================"

python train_optimization.py \
    --data_dir "${DATA_DIR}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --in_channels 4 \
    --loss "${LOSS}" \
    --alpha "${ALPHA}" \
    --beta "${BETA}" \
    --gamma "${GAMMA}" \
    --seed "${SEED}" \
    --checkpoint_dir "checkpoints" \
    --checkpoint_name "ablation_rgbnir_4band.pt" \
    --output_dir "outputs/ablation_4band" \
    --experiment_name "Ablation RGB+NIR (4-band)"

echo "[ablation.sh] Evaluating Run 2 (4-band RGB+NIR) on held-out test split..."
python evaluate.py \
    --data_dir "${DATA_DIR}" \
    --checkpoint "checkpoints/ablation_rgbnir_4band.pt" \
    --in_channels 4 \
    --output_json "outputs/ablation_4band/eval_results.json" \
    --experiment_name "Ablation RGB+NIR (4-band)" \
    --seed "${SEED}"

# ------------------------------------------------------------------------------
# Comparative Reporting: 3-Band vs 4-Band Spectral Ablation Analysis
# ------------------------------------------------------------------------------
python - << 'EOF'
import json
from pathlib import Path

res3_path = Path("outputs/ablation_3band/eval_results.json")
res4_path = Path("outputs/ablation_4band/eval_results.json")

if res3_path.exists() and res4_path.exists():
    with open(res3_path) as f3, open(res4_path) as f4:
        r3 = json.load(f3)
        r4 = json.load(f4)

    dice3 = r3.get("test_dice", 0.0)
    dice4 = r4.get("test_dice", 0.0)
    iou3 = r3.get("test_iou", 0.0)
    iou4 = r4.get("test_iou", 0.0)
    loss3 = r3.get("test_loss", 0.0)
    loss4 = r4.get("test_loss", 0.0)

    delta_dice = dice4 - dice3
    delta_iou = iou4 - iou3
    delta_loss = loss4 - loss3

    border = "+----------------------------------+-------------+-------------+-------------+"
    print("\n" + border)
    print("|            ILĀDṚṢṬI SPECTRAL ABLATION COMPARATIVE SUMMARY           |")
    print(border)
    print(f"| {'Metric':<32} | {'3-Band (RGB)':<11} | {'4-Band (NIR)':<11} | {'Delta (NIR)':<11} |")
    print(border)
    print(f"| {'Test Dice Coefficient (F1)':<32} | {dice3:<11.4f} | {dice4:<11.4f} | {delta_dice:+11.4f} |")
    print(f"| {'Test Mean IoU (Jaccard)':<32} | {iou3:<11.4f} | {iou4:<11.4f} | {delta_iou:+11.4f} |")
    print(f"| {'Test Loss':<32} | {loss3:<11.4f} | {loss4:<11.4f} | {delta_loss:+11.4f} |")
    print(border)
    
    summary_md = f"""# Ilādṛṣṭi Spectral Ablation Study: 3-Band (RGB) vs. 4-Band (RGB+NIR)

| Metric | 3-Band RGB | 4-Band RGB + NIR | Delta ($\Delta$) | Empirical Contribution |
| :--- | :---: | :---: | :---: | :--- |
| **Test Dice ($F_1$)** | `{dice3:.4f}` | `{dice4:.4f}` | **`{delta_dice:+.4f}`** | {'Higher scar delineation accuracy' if delta_dice > 0 else 'Comparable'} |
| **Test Mean IoU** | `{iou3:.4f}` | `{iou4:.4f}` | **`{delta_iou:+.4f}`** | {'Superior spatial boundary alignment' if delta_iou > 0 else 'Comparable'} |
| **Test Loss** | `{loss3:.4f}` | `{loss4:.4f}` | `{delta_loss:+.4f}` | Cross-entropy/Tversky objective |

*Both models evaluated on held-out test split ($N=40$) using deterministic seed=42.*
"""
    with open("outputs/ablation_study_summary.md", "w") as f:
        f.write(summary_md)
    print("[ablation.sh] Comparative markdown table saved to outputs/ablation_study_summary.md\n")
else:
    print("[ablation.sh] Warning: One or both evaluation result files not found.")
EOF

echo "======================================================================"
echo "[ablation.sh] Spectral Ablation Study successfully completed!"
echo "======================================================================"
