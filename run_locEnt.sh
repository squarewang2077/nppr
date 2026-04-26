#!/bin/bash
# run_locEnt.sh
# Local-Entropy Probabilistic Robustness (PR) training of
# ResNet18 / ResNet50 / WideResNet50-2 / VGG16 on
# CIFAR-10, CIFAR-100, and TinyImageNet.

set -euo pipefail

# Resolve the project root and add it to PYTHONPATH so that top-level
# packages (arch/, src/, utils/, configs/) are importable.
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Configurable hyper-parameters
# ---------------------------------------------------------------------------
GPU_ID=1                     # GPU device ID to use (change this to 0, 1, 3, etc.)
export CUDA_VISIBLE_DEVICES=${GPU_ID}

DATA_ROOT="./dataset"
EPOCHS=50
BATCH_SIZE=512
LR=0.01
WEIGHT_DECAY=5e-4
SEED=42

ARCHS=("resnet18" "resnet50" "wide_resnet50_2" "vgg16")
# ARCHS=("resnet18")
DATASETS=("cifar10" "cifar100" "tinyimagenet")
TRAINING_TYPE="loc_entropy"  # standard | loc_entropy

# Perturbation budget
NORM="linf"
EPSILON=0.03137              # 8/255

# Particle settings
NUM_PARTICLES=8
INIT_METHOD="uniform"        # zero | gaussian | uniform

# Langevin dynamics
LANGEVIN_STEPS=5
STEP_SIZE=1e-2
LANGEVIN_BETA=100.0
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
GAMMA=0.05

# Outer TRADES-style loss
BETA_OUTER=12.0

# Save root
SAVE_ROOT="./ckp/dignoise/locEnt_training"

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
echo "======================================================"
echo "  Using GPU: ${GPU_ID}"
echo "  Training Type: ${TRAINING_TYPE}"
echo "======================================================"
echo ""

for DATASET in "${DATASETS[@]}"; do
    for ARCH in "${ARCHS[@]}"; do
        SAVE_DIR="${SAVE_ROOT}/${DATASET}"
        echo "======================================================"
        echo "  arch=${ARCH}  dataset=${DATASET}"
        echo "  training_type=${TRAINING_TYPE}"
        echo "  save_dir=${SAVE_DIR}"
        echo "======================================================"

        python scripts/train_classifiers_pr.py \
            --dataset        "${DATASET}"        \
            --data_root      "${DATA_ROOT}"      \
            --arch           "${ARCH}"           \
            --training_type  "${TRAINING_TYPE}"  \
            --epochs         "${EPOCHS}"         \
            --batch_size     "${BATCH_SIZE}"     \
            --lr             "${LR}"             \
            --weight_decay   "${WEIGHT_DECAY}"   \
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
            --beta_outer     "${BETA_OUTER}"     \
            --save_dir       "${SAVE_DIR}"
    done
done

echo ""
echo "All ${TRAINING_TYPE} training runs completed."
