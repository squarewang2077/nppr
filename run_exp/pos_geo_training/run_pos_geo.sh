#!/bin/bash
# run_pos_geo.sh
# Position-geometry training: sweep the five weighting strategies on one
# (arch, dataset) pair.
#
# All five runs share the same level-set solver settings, so the only thing
# that differs is *which* positions on the level set the training signal comes
# from. That is the comparison this script exists to produce.
#
#   uniform    every valid position counts equally
#   sharp      favour large ||grad_delta CE||_*  — steep spots
#   flat       favour small ||grad_delta CE||_*  — wide-valley spots
#   min_norm   all weight on the position closest to the clean image
#   max_norm   all weight on the position furthest from it
#
# Outputs per run, renamed so the sweep does not clobber itself:
#   ckp/pos_geo_training/pos_geo_<mode>_t<T>_N<NUM_STARTS>.{pth,log}
#   results/pos_geo_training/pos_geo_<mode>_t<T>_N<NUM_STARTS>_training_info.csv
#   results/pos_geo_training/pos_geo_<mode>_t<T>_N<NUM_STARTS>_probe_ep{N}.npz

set -euo pipefail

# Resolve the project root (two levels up) and put it on PYTHONPATH so the
# top-level packages (arch/, src/, utils/) import. cd so ./dataset resolves.
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# Configurable hyper-parameters
# ---------------------------------------------------------------------------
GPU_ID=0
export CUDA_VISIBLE_DEVICES=${GPU_ID}

DATA_ROOT="./dataset"
ARCH="resnet18"
DATASET="cifar10"

EPOCHS=100
BATCH_SIZE=128
LR=0.01
WEIGHT_DECAY=5e-4
SEED=42

# Perturbation budget
NORM="linf"
EPSILON=0.03137              # 8/255

# Level-set solver
T_LEVEL=0.0                  # target margin level
NUM_STARTS=8                 # N positions per image
NUM_STEPS=50
STEP_SIZE=1e-2
ANCHOR_LAMBDA=0.02           # keep small, or perturbations miss the level
PSI_ALPHA=10.0
TOL=0.05                     # |m - t| <= TOL counts as valid

# Weighting sweep — the experiment
WEIGHT_MODES=("uniform" "sharp" "flat" "min_norm" "max_norm")
TAU=1.0                      # softmax temperature for sharp / flat

# Per-delta geometry probe
PROBE_N=256

SAVE_DIR="${PROJECT_ROOT}/ckp/pos_geo_training"
RESULTS_DIR="${PROJECT_ROOT}/results/pos_geo_training"
mkdir -p "${SAVE_DIR}" "${RESULTS_DIR}"

echo "======================================================"
echo "  Position-geometry training"
echo "  GPU:            ${GPU_ID}"
echo "  Arch / dataset: ${ARCH} / ${DATASET}"
echo "  Level:          t=${T_LEVEL}  eps=${EPSILON}  N=${NUM_STARTS}"
echo "  Weight sweep:   ${WEIGHT_MODES[*]}"
echo "======================================================"
echo ""

for WEIGHT_MODE in "${WEIGHT_MODES[@]}"; do
    RUN_NAME="pos_geo_${WEIGHT_MODE}_t${T_LEVEL}_N${NUM_STARTS}"
    echo "======================================================"
    echo "  weight_mode=${WEIGHT_MODE}   run=${RUN_NAME}"
    echo "======================================================"

    python scripts/pr_training/pos_geo_training.py \
        --dataset       "${DATASET}"       \
        --data_root     "${DATA_ROOT}"     \
        --arch          "${ARCH}"          \
        --training_type level              \
        --epochs        "${EPOCHS}"        \
        --batch_size    "${BATCH_SIZE}"    \
        --lr            "${LR}"            \
        --weight_decay  "${WEIGHT_DECAY}"  \
        --seed          "${SEED}"          \
        --norm          "${NORM}"          \
        --epsilon       "${EPSILON}"       \
        --t             "${T_LEVEL}"       \
        --num_starts    "${NUM_STARTS}"    \
        --num_steps     "${NUM_STEPS}"     \
        --step_size     "${STEP_SIZE}"     \
        --anchor_lambda "${ANCHOR_LAMBDA}" \
        --psi_alpha     "${PSI_ALPHA}"     \
        --tol           "${TOL}"           \
        --weight_mode   "${WEIGHT_MODE}"   \
        --tau           "${TAU}"           \
        --probe_n       "${PROBE_N}"       \
        --eval_pgd --pgd_steps 10 --pgd_norm linf \
        --save_dir      "${SAVE_DIR}"      \
        --results_dir   "${RESULTS_DIR}"

    # Rename so each sweep entry keeps its own files. Without --augment the
    # trainer writes <arch>_<dataset>_level.{pth,log} and, into RESULTS_DIR,
    # <arch>_<dataset>_level_training_info.csv plus _probe_ep*.npz.
    ARCH_LOWER="${ARCH,,}"
    DATASET_LOWER="${DATASET,,}"
    SRC="${ARCH_LOWER}_${DATASET_LOWER}_level"

    mv "${SAVE_DIR}/${SRC}.pth" "${SAVE_DIR}/${RUN_NAME}.pth"
    mv "${SAVE_DIR}/${SRC}.log" "${SAVE_DIR}/${RUN_NAME}.log"
    mv "${RESULTS_DIR}/${SRC}_training_info.csv" \
       "${RESULTS_DIR}/${RUN_NAME}_training_info.csv"
    for f in "${RESULTS_DIR}/${SRC}"_probe_ep*.npz; do
        [ -e "$f" ] || continue
        mv "$f" "${RESULTS_DIR}/${RUN_NAME}_probe_ep${f##*_probe_ep}"
    done

    echo "  -> renamed outputs to ${RUN_NAME}.*"
    echo ""
done

echo "All ${#WEIGHT_MODES[@]} weighting runs completed."
echo "  checkpoints: ${SAVE_DIR}"
echo "  results:     ${RESULTS_DIR}"
