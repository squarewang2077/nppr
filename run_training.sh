#!/bin/bash
# run_training.sh
# Standard/Adversarial/PR training of ResNet18 / ResNet50 / WideResNet50-2 / VGG16
# on CIFAR-10, CIFAR-100, and TinyImageNet.

set -euo pipefail

# Resolve the project root and add it to PYTHONPATH so that top-level
# packages (arch/, src/, utils/, configs/) are importable.
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Configurable hyper-parameters
# ---------------------------------------------------------------------------
DATA_ROOT="./dataset"
EPOCHS=200
BATCH_SIZE=4096
LR=0.01
WEIGHT_DECAY=5e-4
SEED=42

# ARCHS=("resnet18" "resnet50" "wide_resnet50_2" "vgg16")
ARCHS=("vgg16")
DATASETS=("cifar10" "cifar100" "tinyimagenet")
TRAINING_TYPE="adv_pgd"     # standard | adv_pgd | trades | pr

# Adversarial / PR attack budget (used by adv_pgd, trades, and pr)
NORM="linf"
EPSILON=0.03137             # 8/255
ALPHA=0.00784               # 2/255
NUM_STEPS=3

# Save root derived from training type
SAVE_ROOT="./ckp/${TRAINING_TYPE}"

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
for DATASET in "${DATASETS[@]}"; do
    for ARCH in "${ARCHS[@]}"; do
        SAVE_DIR="${SAVE_ROOT}/${DATASET}"
        echo "======================================================"
        echo "  arch=${ARCH}  dataset=${DATASET}"
        echo "  training_type=${TRAINING_TYPE}"
        echo "  save_dir=${SAVE_DIR}"
        echo "======================================================"

        python scripts/train_classifiers.py \
            --dataset        "${DATASET}"       \
            --data_root      "${DATA_ROOT}"      \
            --arch           "${ARCH}"           \
            --training_type  "${TRAINING_TYPE}"  \
            --norm           "${NORM}"           \
            --epsilon        "${EPSILON}"        \
            --alpha          "${ALPHA}"          \
            --num_steps      "${NUM_STEPS}"      \
            --epochs         "${EPOCHS}"         \
            --batch_size     "${BATCH_SIZE}"     \
            --lr             "${LR}"             \
            --weight_decay   "${WEIGHT_DECAY}"   \
            --seed           "${SEED}"           \
            --save_dir       "${SAVE_DIR}"
    done
done

echo ""
echo "All ${TRAINING_TYPE} training runs completed."
