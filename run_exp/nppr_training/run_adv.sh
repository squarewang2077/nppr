#!/bin/bash
# run_adv.sh
# PGD-AT / TRADES adversarial training (WITH data augmentation) of
# ResNet18 / ResNet50 / WideResNet50-2 / VGG16 on
# CIFAR-10, CIFAR-100, and TinyImageNet.
#
# Identical to run_adv_noAug.sh except the train loader uses augmentation
# (RandomCrop + HorizontalFlip + RandAugment + RandomErasing). Output
# filenames are tagged with '_Aug' by the trainer so they do not collide
# with the no-aug checkpoints.

set -euo pipefail

# Resolve the project root (two levels up from this script) and add it to
# PYTHONPATH so that top-level packages (arch/, src/, utils/, configs/) are
# importable. We also cd to the project root so relative paths like
# ./dataset and scripts/... work regardless of where this script is invoked from.
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# Configurable hyper-parameters
# ---------------------------------------------------------------------------
GPU_ID=0                     # GPU device ID to use (change this to 0, 1, 3, etc.)
export CUDA_VISIBLE_DEVICES=${GPU_ID}

DATA_ROOT="./dataset"
EPOCHS=100
BATCH_SIZE=512
LR=0.01
WEIGHT_DECAY=5e-4
SEED=42

ARCHS=("resnet50")
# ARCHS=("resnet18" "resnet50" "wide_resnet50_2" "vgg16")
# DATASETS=("cifar10")
DATASETS=("cifar10" "cifar100" "tinyimagenet")
TRAINING_TYPE="adv_pgd"       # standard | adv_pgd | trades

# Adversarial attack budget (used by adv_pgd and trades)
NORM="linf"
EPSILON=0.03137              # 8/255
ALPHA=0.00784                # 2/255
NUM_STEPS=10

# TRADES-only KL regularization weight
BETA=12.0

# Save root
SAVE_ROOT="./ckp/nppr_training/adv_training"
# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
echo "======================================================"
echo "  Using GPU: ${GPU_ID}"
echo "  Training Type: ${TRAINING_TYPE}  (with data augmentation)"
echo "======================================================"
echo ""

for DATASET in "${DATASETS[@]}"; do
    for ARCH in "${ARCHS[@]}"; do
        SAVE_DIR="${SAVE_ROOT}/${DATASET}/${ARCH}/${TRAINING_TYPE}"
        echo "======================================================"
        echo "  arch=${ARCH}  dataset=${DATASET}"
        echo "  training_type=${TRAINING_TYPE}  (augmentation=ON)"
        echo "  save_dir=${SAVE_DIR}"
        echo "======================================================"

        python scripts/train_classifiers_adv.py \
            --dataset        "${DATASET}"       \
            --data_root      "${DATA_ROOT}"     \
            --arch           "${ARCH}"          \
            --training_type  "${TRAINING_TYPE}" \
            --augment                            \
            --epochs         "${EPOCHS}"        \
            --batch_size     "${BATCH_SIZE}"    \
            --lr             "${LR}"            \
            --weight_decay   "${WEIGHT_DECAY}"  \
            --seed           "${SEED}"          \
            --norm           "${NORM}"          \
            --epsilon        "${EPSILON}"       \
            --alpha           "${ALPHA}"        \
            --num_steps      "${NUM_STEPS}"     \
            --beta           "${BETA}"          \
            --eval_pgd --pgd_steps 10 --pgd_norm linf \
            --eval_locent --locent_n 8 --locent_steps 10 --locent_norm linf \
            --eval_random --random_n 8 --random_norm linf \
            --random_dist gaussian uniform laplace \
            --eval_corruptions --corruption_severities 1 \
            --save_dir       "${SAVE_DIR}"
    done
done

echo ""
echo "All ${TRAINING_TYPE} (aug) training runs completed."
