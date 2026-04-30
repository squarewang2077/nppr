#!/bin/bash
# run_locent_attack.sh
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
NUM_PARTICLES=8
INIT_METHOD="uniform"        # zero | gaussian | uniform

# Attack: Langevin dynamics (typically stronger than training defaults)
LANGEVIN_STEPS=10
STEP_SIZE=1e-2
LANGEVIN_BETA=100
NOISE_SCALE=1.0

# Energy function
PSI_TYPE="softplus"          # softplus | hinge
PSI_ALPHA=10.0

# Threshold strategy
THRESHOLD_MODE="fixed"       # fixed | adaptive
T0=-0.05
T_FLOOR=0.0

# Scope strategy
SCOPE_MODE="fixed"           # fixed | dynamic

# GAMMA sweep
GAMMAS=(50 100 200 500 1000 5000)

# Save root for attack summaries / logs
SAVE_DIR="./results/nppr_training/locent_attack/${DATASET}/${ARCH}"
mkdir -p "${SAVE_DIR}"

# ---------------------------------------------------------------------------
# Sweep loop
# ---------------------------------------------------------------------------
echo "======================================================"
echo "  Using GPU:       ${GPU_ID}"
echo "  Checkpoint:      ${CKPT}"
echo "  Dataset / arch:  ${DATASET} / ${ARCH}"
echo "  GAMMA sweep:     ${GAMMAS[*]}"
echo "  Output dir:      ${SAVE_DIR}"
echo "======================================================"
echo ""

for GAMMA in "${GAMMAS[@]}"; do
    TAG="eps${LANGEVIN_STEPS}_L${LANGEVIN_BETA}_G${GAMMA}"
    echo "======================================================"
    echo "  gamma=${GAMMA}   tag=${TAG}"
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
        --num_particles  "${NUM_PARTICLES}"  \
        --init_method    "${INIT_METHOD}"    \
        --langevin_steps "${LANGEVIN_STEPS}" \
        --step_size      "${STEP_SIZE}"      \
        --langevin_beta  "${LANGEVIN_BETA}"  \
        --noise_scale    "${NOISE_SCALE}"    \
        --psi_type       "${PSI_TYPE}"       \
        --psi_alpha      "${PSI_ALPHA}"      \
        --threshold_mode "${THRESHOLD_MODE}" \
        --t0             "${T0}"             \
        --t_floor        "${T_FLOOR}"        \
        --scope_mode     "${SCOPE_MODE}"     \
        --gamma          "${GAMMA}"          \
        --save_dir       "${SAVE_DIR}"       \
        --tag            "${TAG}"
done

echo ""
echo "All local-entropy attack runs completed."
echo "Results in: ${SAVE_DIR}"
