#!/bin/bash
# run_level_attack.sh
# Local-entropy attack evaluation of a single checkpoint over a GAMMA sweep.
#
# For each GAMMA value, runs scripts/eval_prob_perturbations(LocEnt).py and
# writes its summary CSV / log under SAVE_DIR with a per-gamma tag, so that
# all sweep runs accumulate into the same directory without clobbering.

set -euo pipefail

# Resolve the project root (two levels up from this script) and add it to
# PYTHONPATH so that top-level packages (arch/, src/, utils/, configs/) are
# importable. cd ensures ./dataset and scripts/... resolve correctly no
# matter where this script is invoked from.
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# Configurable hyper-parameters
# ---------------------------------------------------------------------------
GPU_ID=0                     # GPU device ID to use
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Checkpoint + dataset
CKPT="./ckp/nppr_eval/standard/resnet/resnet18_cifar10.pth"
DATASET="cifar10"
ARCH="resnet18"
DATA_ROOT="./dataset"
BATCH_SIZE=256
SEED=42

# Attack: perturbation budget
NORM="linf"
EPSILON=0.03137              # 8/255
NUM_STARTS=8

# Attack: Langevin dynamics (typically stronger than training defaults)
NUM_STEPS=50
STEP_SIZE=1e-2

# Energy function
PSI_ALPHA=10.0

# Level-set solver
T_LEVEL=0.0
TOL=0.05

# ANCHOR_LAMBDA sweep — L2 pull toward each random start.
# Keep these small: large values stop perturbations reaching the level set.
ANCHOR_LAMBDAS=(0.0 0.01 0.02 0.05 0.1)

# Save root for attack summaries / logs
SAVE_DIR="./results/nppr_training/level_attack/${DATASET}/${ARCH}"
mkdir -p "${SAVE_DIR}"

# ---------------------------------------------------------------------------
# Sweep loop
# ---------------------------------------------------------------------------
echo "======================================================"
echo "  Using GPU:       ${GPU_ID}"
echo "  Checkpoint:      ${CKPT}"
echo "  Dataset / arch:  ${DATASET} / ${ARCH}"
echo "  ANCHOR_LAMBDA sweep: ${ANCHOR_LAMBDAS[*]}"
echo "  Output dir:      ${SAVE_DIR}"
echo "======================================================"
echo ""

for ANCHOR_LAMBDA in "${ANCHOR_LAMBDAS[@]}"; do
    TAG="t${T_LEVEL}_S${NUM_STEPS}_A${ANCHOR_LAMBDA}"
    echo "======================================================"
    echo "  anchor_lambda=${ANCHOR_LAMBDA}   tag=${TAG}"
    echo "======================================================"

    python "scripts/eval_prob_perturbations(LocEnt).py" \
        --ckpt           "${CKPT}"           \
        --dataset        "${DATASET}"        \
        --data_root      "${DATA_ROOT}"      \
        --arch           "${ARCH}"           \
        --batch_size     "${BATCH_SIZE}"     \
        --seed           "${SEED}"           \
        --norm           "${NORM}"           \
        --epsilon        "${EPSILON}"        \
        --num_starts     "${NUM_STARTS}"      \
        --num_steps      "${NUM_STEPS}"       \
        --step_size      "${STEP_SIZE}"       \
        --t              "${T_LEVEL}"         \
        --anchor_lambda  "${ANCHOR_LAMBDA}"   \
        --psi_alpha      "${PSI_ALPHA}"       \
        --tol            "${TOL}"             \
        --save_dir       "${SAVE_DIR}"       \
        --tag            "${TAG}"
done

echo ""
echo "All level-set attack runs completed."
echo "Results in: ${SAVE_DIR}"
