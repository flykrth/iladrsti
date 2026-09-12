#!/usr/bin/env python3
"""
Ilādṛṣṭi Pipeline Updater: update_run_sh.py
Phase 3: Final Evaluation and Limitations Analysis.

Modifies the existing `run.sh` so that it seamlessly chains:
  1. Pre-flight verification (anti-leakage audit)
  2. Baseline model training (Model 1: Baseline BCE RGB and Baseline BCE RGB+NIR)
  3. Optimization & Spectral ablation runs (Model 2: Focal-Tversky RGB & Model 3: Focal-Tversky RGB+NIR)
  4. Final evaluation suite (src/utils/evaluation_suite.py) evaluating all 3 models on held-out test data
  5. Limitations & Latency analysis (analyze_limitations.py) measuring latency via CUDA events and
     isolating the top 10 worst False Positive predictions
into a single, robust, end-to-end command.
"""

import os
import shutil
import stat
import sys
from pathlib import Path

RUN_SH_PATH = Path("run.sh")
RUN_SH_BAK_PATH = Path("run.sh.bak")

UPDATED_RUN_SH_CONTENT = """#!/usr/bin/env bash
# ==============================================================================
# Ilādṛṣṭi: End-to-End Multispectral Wildfire Deep Learning Pipeline
# Track 6 (AI for Science & Society) - Deep Learning Hackathon, Amrita Vishwa Vidyapeetham
#
# Single-Command Seamless Pipeline Execution:
#   1. Strict Determinism Locking (Seed 42, CuBLAS workspace, CuDNN flags)
#   2. Pre-flight Leakage Verification (verify_splits.py)
#   3. Baseline Model Training:
#      - Model 1: Baseline BCE (RGB, 3-band) -> checkpoints/baseline_bce_rgb.pt
#      - Mandatory Baseline: BCE (RGB + NIR, 4-band) -> checkpoints/best_baseline_model.pt
#   4. Optimization & Spectral Ablation Runs:
#      - Model 2: Focal-Tversky (RGB, 3-band) -> checkpoints/ablation_rgb_3band.pt
#      - Model 3: Focal-Tversky (RGB + NIR, 4-band) -> checkpoints/ablation_rgbnir_4band.pt
#   5. Decoupled Final Evaluation Suite (src/utils/evaluation_suite.py):
#      - Strictly evaluates all 3 models on held-out test split (N=40)
#      - Outputs formatted markdown comparison table (Test Dice & Test IoU)
#   6. Limitations & Latency Analysis (analyze_limitations.py):
#      - Measures batch inference latency with torch.cuda.Event(enable_timing=True)
#      - Isolates top 10 worst False Positive predictions into outputs/limitations/
#   7. Automated Publication Plotting (plot_ablation_results.py)
# ==============================================================================

set -euo pipefail

# 1. Deterministic Environment Variables & Reproducibility Locks
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NO_ALBUMENTATIONS_UPDATE=1

# 2. Virtual environment activation (if present)
if [ -d "venv" ]; then
    echo "[run.sh] Activating virtual environment: venv..."
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

# 3. CLI Options and Configurable Defaults
DATA_DIR="data"
EPOCHS=15
BATCH_SIZE=16
LR="1e-4"
SKIP_TRAIN=false

# Parse options and positional arguments robustly
POSITIONAL_ARGS=()
for arg in "$@"; do
    case $arg in
        --skip-train|--eval-only)
            SKIP_TRAIN=true
            ;;
        --quick)
            EPOCHS=3
            ;;
        --epochs=*)
            EPOCHS="${arg#*=}"
            ;;
        --batch-size=*)
            BATCH_SIZE="${arg#*=}"
            ;;
        --lr=*)
            LR="${arg#*=}"
            ;;
        --data-dir=*)
            DATA_DIR="${arg#*=}"
            ;;
        --help|-h)
            echo "Usage: ./run.sh [DATA_DIR] [EPOCHS] [BATCH_SIZE] [LR] [--skip-train] [--quick]"
            exit 0
            ;;
        -*)
            echo "[run.sh] Warning: Unrecognized option: $arg"
            ;;
        *)
            POSITIONAL_ARGS+=("$arg")
            ;;
    esac
done

# Map positional parameters if provided
if [ ${#POSITIONAL_ARGS[@]} -gt 0 ]; then
    DATA_DIR="${POSITIONAL_ARGS[0]}"
fi
if [ ${#POSITIONAL_ARGS[@]} -gt 1 ]; then
    EPOCHS="${POSITIONAL_ARGS[1]}"
fi
if [ ${#POSITIONAL_ARGS[@]} -gt 2 ]; then
    BATCH_SIZE="${POSITIONAL_ARGS[2]}"
fi
if [ ${#POSITIONAL_ARGS[@]} -gt 3 ]; then
    LR="${POSITIONAL_ARGS[3]}"
fi

echo "======================================================================"
echo "                 ILĀDṚṢṬI END-TO-END MASTER PIPELINE                  "
echo "======================================================================"
echo " Data Directory    : ${DATA_DIR}"
echo " Epochs per Run    : ${EPOCHS}"
echo " Batch Size        : ${BATCH_SIZE}"
echo " Learning Rate     : ${LR}"
echo " Deterministic Seed: 42 (Locked across all training & evaluation)"
echo " Skip Training     : ${SKIP_TRAIN}"
echo "======================================================================"

# 4. Dependency installation check
if [ -f "requirements.txt" ]; then
    echo "[run.sh] Verifying installed dependencies..."
    python -m pip install --quiet -r requirements.txt
fi

# 5. Step 1: Strict Anti-Leakage & Split Integrity Audit
echo ""
echo "======================================================================"
echo ">>> STEP 1/6: PRE-FLIGHT DATA INTEGRITY & ANTI-LEAKAGE AUDIT"
echo "======================================================================"
python verify_splits.py

# Create required output directories
mkdir -p checkpoints outputs outputs/limitations outputs/ablation_3band outputs/ablation_4band

if [ "${SKIP_TRAIN}" = "false" ]; then
    # --------------------------------------------------------------------------
    # Step 2: Baseline Model Training
    # --------------------------------------------------------------------------
    echo ""
    echo "======================================================================"
    echo ">>> STEP 2/6: BASELINE MODEL TRAINING (BCE OBJECTIVE)"
    echo "======================================================================"

    echo "[run.sh] Training Model 1: Baseline BCE (RGB, 3-band)..."
    python train_optimization.py \\
        --data_dir "${DATA_DIR}" \\
        --epochs "${EPOCHS}" \\
        --batch_size "${BATCH_SIZE}" \\
        --lr "${LR}" \\
        --in_channels 3 \\
        --loss "bce" \\
        --seed 42 \\
        --checkpoint_dir "checkpoints" \\
        --checkpoint_name "baseline_bce_rgb.pt" \\
        --output_dir "outputs/baseline_bce_rgb" \\
        --experiment_name "Model 1: Baseline BCE (RGB)"

    echo "[run.sh] Training Mandatory Baseline: BCE (RGB + NIR, 4-band)..."
    python train.py \\
        --data_dir "${DATA_DIR}" \\
        --epochs "${EPOCHS}" \\
        --batch_size "${BATCH_SIZE}" \\
        --lr "${LR}" \\
        --in_channels 4 \\
        --checkpoint_dir "checkpoints" \\
        --output_dir "outputs"

    # --------------------------------------------------------------------------
    # Step 3: Optimization & Spectral Ablation Runs
    # --------------------------------------------------------------------------
    echo ""
    echo "======================================================================"
    echo ">>> STEP 3/6: OPTIMIZATION TRAINING & SPECTRAL ABLATION RUNS"
    echo "======================================================================"

    echo "[run.sh] Training Model 2: Focal-Tversky Loss (RGB, 3-band)..."
    python train_optimization.py \\
        --data_dir "${DATA_DIR}" \\
        --epochs "${EPOCHS}" \\
        --batch_size "${BATCH_SIZE}" \\
        --lr "${LR}" \\
        --in_channels 3 \\
        --loss "focal_tversky" \\
        --alpha 0.7 \\
        --beta 0.3 \\
        --gamma 1.333 \\
        --seed 42 \\
        --checkpoint_dir "checkpoints" \\
        --checkpoint_name "ablation_rgb_3band.pt" \\
        --output_dir "outputs/ablation_3band" \\
        --experiment_name "Model 2: Focal-Tversky (RGB)"

    echo "[run.sh] Training Model 3: Focal-Tversky Loss (RGB + NIR, 4-band)..."
    python train_optimization.py \\
        --data_dir "${DATA_DIR}" \\
        --epochs "${EPOCHS}" \\
        --batch_size "${BATCH_SIZE}" \\
        --lr "${LR}" \\
        --in_channels 4 \\
        --loss "focal_tversky" \\
        --alpha 0.7 \\
        --beta 0.3 \\
        --gamma 1.333 \\
        --seed 42 \\
        --checkpoint_dir "checkpoints" \\
        --checkpoint_name "ablation_rgbnir_4band.pt" \\
        --output_dir "outputs/ablation_4band" \\
        --experiment_name "Model 3: Focal-Tversky (RGB+NIR)"
else
    echo ""
    echo "[run.sh] Skipping training stages (--skip-train enabled). Using existing checkpoints."
fi

# ------------------------------------------------------------------------------
# Step 4: Comprehensive Final Evaluation Suite (Phase 3)
# ------------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo ">>> STEP 4/6: DECOUPLED FINAL EVALUATION SUITE ON HELD-OUT TEST DATA"
echo "======================================================================"
python -m src.utils.evaluation_suite \\
    --data_dir "${DATA_DIR}" \\
    --batch_size "${BATCH_SIZE}" \\
    --threshold 0.5 \\
    --seed 42 \\
    --output_md "outputs/evaluation_suite_summary.md" \\
    --output_json "outputs/evaluation_suite_results.json"

# ------------------------------------------------------------------------------
# Step 5: Limitations & Inference Latency Benchmark
# ------------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo ">>> STEP 5/6: LATENCY BENCHMARK & SYSTEMATIC LIMITATIONS AUDIT"
echo "======================================================================"
python analyze_limitations.py \\
    --data_dir "${DATA_DIR}" \\
    --batch_size "${BATCH_SIZE}" \\
    --top_k 10 \\
    --threshold 0.5 \\
    --seed 42 \\
    --output_dir "outputs/limitations"

# ------------------------------------------------------------------------------
# Step 6: Automated Plotting & Visual Asset Compilation
# ------------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo ">>> STEP 6/6: GENERATING PUBLICATION PLOTS & REPORT ASSETS"
echo "======================================================================"
if [ -f "plot_ablation_results.py" ]; then
    python plot_ablation_results.py || true
fi

echo ""
echo "======================================================================"
echo " [run.sh] Ilādṛṣṭi Complete Pipeline Finished Successfully!"
echo " Results Summary Table : outputs/evaluation_suite_summary.md"
echo " Limitations Artifacts : outputs/limitations/"
echo "======================================================================"
"""


def update_run_sh() -> None:
    """Modifies run.sh to chain baseline, optimization, ablation, and evaluation."""
    print("==================================================================")
    print("       ILĀDṚṢṬI PIPELINE UPDATER: update_run_sh.py               ")
    print("==================================================================")

    # 1. Backup original run.sh if exists
    if RUN_SH_PATH.exists():
        if not RUN_SH_BAK_PATH.exists():
            shutil.copyfile(RUN_SH_PATH, RUN_SH_BAK_PATH)
            print(f"[+] Backed up original run.sh to: {RUN_SH_BAK_PATH}")
        else:
            print(f"[*] Existing backup found at: {RUN_SH_BAK_PATH}")

    # 2. Write updated run.sh
    with open(RUN_SH_PATH, "w", encoding="utf-8") as f:
        f.write(UPDATED_RUN_SH_CONTENT.strip() + "\n")
    print(f"[+] Successfully wrote updated master pipeline to: {RUN_SH_PATH}")

    # 3. Grant executable permissions
    st = os.stat(RUN_SH_PATH)
    os.chmod(RUN_SH_PATH, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    print(f"[+] Granted executable permissions (chmod +x {RUN_SH_PATH})")

    print("\nChained Pipeline Stages:")
    print("  1. Environment & Determinism Setup (Seed 42, CuBLAS, venv)")
    print("  2. Pre-flight Leakage Verification (verify_splits.py)")
    print("  3. Baseline Training (Model 1: Baseline BCE RGB & Baseline BCE RGB+NIR)")
    print("  4. Optimization & Ablation (Model 2: Focal-Tversky RGB & Model 3: Focal-Tversky RGB+NIR)")
    print("  5. Final Evaluation Suite (src/utils/evaluation_suite.py)")
    print("  6. Limitations & Latency Analysis (analyze_limitations.py)")
    print("  7. Publication Plot Generation (plot_ablation_results.py)")
    print("==================================================================")


if __name__ == "__main__":
    update_run_sh()
