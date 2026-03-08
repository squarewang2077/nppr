#!/bin/bash
# run_std_train.sh
# Standard training of ResNet18 / ResNet50 / WideResNet50-2 / VGG16
# on CIFAR-10, CIFAR-100, and TinyImageNet.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configurable hyper-parameters
# ---------------------------------------------------------------------------
DATA_ROOT="./dataset"
SAVE_ROOT="./ckp/pr_training"
EPOCHS=50
BATCH_SIZE=512
LR=0.1
WEIGHT_DECAY=5e-4
SEED=42

ARCHS=("resnet18" "resnet50" "wide_resnet50_2" "vgg16")
DATASETS=("cifar10" "cifar100" "tinyimagenet")
Training_Type="pr"
# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
for DATASET in "${DATASETS[@]}"; do
    for ARCH in "${ARCHS[@]}"; do
        SAVE_DIR="${SAVE_ROOT}/${DATASET}"
        LOG_FILE="${SAVE_DIR}/${ARCH}_${DATASET}_${Training_Type}.log"
        echo "======================================================"
        echo "  arch=${ARCH}  dataset=${DATASET}"
        echo "  save_dir=${SAVE_DIR}"
        echo "  log_file=${LOG_FILE}"
        echo "======================================================"

        python fit_classifiers.py \
            --dataset        "${DATASET}"  \
            --data_root      "${DATA_ROOT}" \
            --arch           "${ARCH}"      \
            --training_type  "${Training_Type}"       \
            --epochs         "${EPOCHS}"    \
            --batch_size     "${BATCH_SIZE}" \
            --lr             "${LR}"         \
            --weight_decay   "${WEIGHT_DECAY}" \
            --seed           "${SEED}"       \
            --save_dir       "${SAVE_DIR}"
    done
done

echo ""
echo "All standard training runs completed."
