#!/bin/bash
# run_pos_geo_sweep.sh
# Full grid for position-geometry training: arch x dataset x weight_mode.
#
# Same experiment as run_pos_geo.sh, widened to every architecture and dataset.
# Each run's outputs are tagged with all three loop variables so nothing in the
# grid collides:
#   pos_geo_<arch>_<dataset>_<mode>_t<T>_N<NUM_STARTS>.{pth,log,csv,npz}

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# Configurable hyper-parameters
# ---------------------------------------------------------------------------
GPU_ID=0
export CUDA_VISIBLE_DEVICES=${GPU_ID}

DATA_ROOT="./dataset"
ARCHS=("resnet18" "resnet50")
DATASETS=("cifar10" "cifar100")
WEIGHT_MODES=("uniform" "sharp" "flat" "min_norm" "max_norm")

EPOCHS=100
BATCH_SIZE=128
LR=0.01
WEIGHT_DECAY=5e-4
SEED=42

NORM="linf"
EPSILON=0.03137              # 8/255

T_LEVEL=0.0
NUM_STARTS=8
NUM_STEPS=50
STEP_SIZE=1e-2
ANCHOR_LAMBDA=0.02
PSI_ALPHA=10.0
TOL=0.05
TAU=1.0
PROBE_N=256

SAVE_DIR="${PROJECT_ROOT}/ckp/pos_geo_training"
RESULTS_DIR="${PROJECT_ROOT}/results/pos_geo_training"
mkdir -p "${SAVE_DIR}" "${RESULTS_DIR}"

TOTAL=$(( ${#ARCHS[@]} * ${#DATASETS[@]} * ${#WEIGHT_MODES[@]} ))
echo "======================================================"
echo "  Position-geometry sweep: ${TOTAL} runs"
echo "  archs:    ${ARCHS[*]}"
echo "  datasets: ${DATASETS[*]}"
echo "  modes:    ${WEIGHT_MODES[*]}"
echo "======================================================"
echo ""

RUN_IDX=0
for ARCH in "${ARCHS[@]}"; do
for DATASET in "${DATASETS[@]}"; do
for WEIGHT_MODE in "${WEIGHT_MODES[@]}"; do
    RUN_IDX=$(( RUN_IDX + 1 ))
    RUN_NAME="pos_geo_${ARCH}_${DATASET}_${WEIGHT_MODE}_t${T_LEVEL}_N${NUM_STARTS}"
    echo "------------------------------------------------------"
    echo "  [${RUN_IDX}/${TOTAL}] ${RUN_NAME}"
    echo "------------------------------------------------------"

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

    echo "  -> ${RUN_NAME}.*"
    echo ""
done
done
done

echo "All ${TOTAL} runs completed."
echo "  checkpoints: ${SAVE_DIR}"
echo "  results:     ${RESULTS_DIR}"
