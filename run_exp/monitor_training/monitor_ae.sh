#!/bin/bash
# monitor_ae.sh
# PGD adversarial training of ResNet18 on CIFAR-10 while monitoring the
# adversarial-example (AE) generation path.
#
# The inner PGD attack produces a perturbation trajectory
#   Delta_e(x) = [delta_1, ..., delta_T]   (delta_t = x_adv_t - x, T = NUM_STEPS)
# and --track_path records it every epoch for a fixed set of images, logging how
# much the path drifts from the previous epoch (Delta_e vs Delta_{e-1}).
#
# Config: resnet18 / cifar10 / adv_pgd, PGD with NO random start, 10 PGD steps,
#         100 epochs.

set -euo pipefail

# Resolve the project root (two levels up from this script) and add it to
# PYTHONPATH so top-level packages (arch/, src/, utils/) are importable. We also
# cd to the project root so relative paths like ./dataset and scripts/... work
# regardless of where this script is invoked from.
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# Configurable hyper-parameters
# ---------------------------------------------------------------------------
GPU_ID=0                     # GPU device ID to use
export CUDA_VISIBLE_DEVICES=${GPU_ID}

DATASET="cifar10"
ARCH="resnet18"
TRAINING_TYPE="adv_pgd"      # standard | adv_pgd

DATA_ROOT="./dataset"
EPOCHS=100
BATCH_SIZE=512
LR=0.01
WEIGHT_DECAY=5e-4
SEED=42

# PGD attack budget (no random start -> --random_start flag omitted below)
NORM="linf"
EPSILON=0.03137              # 8/255
ALPHA=0.00784                # 2/255
NUM_STEPS=10

# PGD attack-path tracking
PATH_TRACK_N=16              # number of fixed images whose AE path is tracked

# Save directory
SAVE_DIR="./ckp/monitor_training/${DATASET}/${ARCH}/${TRAINING_TYPE}"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "======================================================"
echo "  Using GPU: ${GPU_ID}"
echo "  arch=${ARCH}  dataset=${DATASET}  training_type=${TRAINING_TYPE}"
echo "  PGD: eps=${EPSILON} alpha=${ALPHA} steps=${NUM_STEPS} random_start=OFF"
echo "  epochs=${EPOCHS}  path-tracking on ${PATH_TRACK_N} images"
echo "  save_dir=${SAVE_DIR}"
echo "======================================================"

python scripts/monitor_training.py \
    --dataset        "${DATASET}"       \
    --data_root      "${DATA_ROOT}"     \
    --arch           "${ARCH}"          \
    --training_type  "${TRAINING_TYPE}" \
    --epochs         "${EPOCHS}"        \
    --batch_size     "${BATCH_SIZE}"    \
    --lr             "${LR}"            \
    --weight_decay   "${WEIGHT_DECAY}"  \
    --seed           "${SEED}"          \
    --norm           "${NORM}"          \
    --epsilon        "${EPSILON}"       \
    --alpha          "${ALPHA}"         \
    --num_steps      "${NUM_STEPS}"     \
    --track_path --path_track_n "${PATH_TRACK_N}" \
    --eval_pgd --pgd_steps 10 --pgd_norm linf \
    --save_dir       "${SAVE_DIR}"

echo ""
echo "monitor_ae run completed. Path-drift metrics are in ${SAVE_DIR}/${ARCH}_${DATASET}_${TRAINING_TYPE}.log"
echo "and the path_drift_* columns of ${SAVE_DIR}/${ARCH}_${DATASET}_${TRAINING_TYPE}_training_info.csv"
