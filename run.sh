#!/usr/bin/env bash
# ==============================================================================
# Ilādṛṣṭi: Multispectral Wildfire Segmentation Baseline Pipeline
# Track 6 (AI for Science & Society) - Deep Learning Hackathon, Amrita Vishwa Vidyapeetham
#
# Single-command end-to-end execution:
# 1. Sets deterministic environment variables (Seed 42 & CuBLAS workspace)
# 2. Installs required dependencies
# 3. Trains baseline ResNet-50 U-Net with BCEWithLogitsLoss
# 4. Evaluates decoupled metrics (Test Loss, Dice, IoU) on held-out test split
# ==============================================================================

set -euo pipefail

# 1. Lock environment variables for strict determinism & CuDNN reproducibility
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NO_ALBUMENTATIONS_UPDATE=1

# 2. Virtual environment activation (if present)
if [ -d "venv" ]; then
    echo "[run.sh] Activating virtual environment: venv..."
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

# 3. Configuration & Parameter Defaults
DATA_DIR="${1:-data}"
EPOCHS="${2:-20}"
BATCH_SIZE="${3:-16}"
LR="${4:-1e-4}"
IN_CHANNELS="${5:-4}"

echo "======================================================================"
echo "                 ILĀDṚṢṬI BASELINE PIPELINE RUNNER                    "
echo "======================================================================"
echo " Data Directory : ${DATA_DIR}"
echo " Epochs         : ${EPOCHS}"
echo " Batch Size     : ${BATCH_SIZE}"
echo " Learning Rate  : ${LR}"
echo " Input Channels : ${IN_CHANNELS} (Bands: Red, Green, Blue, NIR)"
echo " Seed Locked    : 42"
echo "======================================================================"

# 4. Install dependencies
echo "[run.sh] Installing dependencies from requirements.txt..."
python -m pip install --quiet -r requirements.txt

# 5. Verify Dataset Split Integrity & Prevent Data Leakage
echo "[run.sh] Running strict split separation audit..."
python verify_splits.py

# 6. Execute Training Pipeline
echo "[run.sh] Launching training pipeline (BCEWithLogitsLoss)..."
python train.py \
    --data_dir "${DATA_DIR}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --in_channels "${IN_CHANNELS}"

# 6. Immediately Execute Decoupled Evaluation Pipeline
echo "[run.sh] Launching decoupled evaluation on held-out test split..."
python evaluate.py \
    --data_dir "${DATA_DIR}" \
    --checkpoint "checkpoints/best_baseline_model.pt" \
    --in_channels "${IN_CHANNELS}"

echo "======================================================================"
echo "[run.sh] Ilādṛṣṭi Baseline Pipeline completed successfully!"
echo "======================================================================"
