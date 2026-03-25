#!/bin/bash
# run_ablation.sh
# Ablation Study of PR training on ResNet18
# on CIFAR-10

set -euo pipefail

# ---------------------------------------------------------------------------
# Configurable hyper-parameters
# ---------------------------------------------------------------------------
DATA_ROOT="./dataset"
SAVE_ROOT="./ckp/ablation"
EPOCHS=50
BATCH_SIZE=1024
LR=0.1
WEIGHT_DECAY=5e-4
SEED=42

ARCH="resnet18"
DATASET="cifar10"

# ---------------------------------------------------------------------------
# Tuning parameters for PR training
# ---------------------------------------------------------------------------
TAU_VALUES=(1e0 1e-2 1e-4 1e-8)
KAPPA_VALUES=(1 0.2 0.02)

TRAINING_TYPE="pr"
# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
for kappa in "${KAPPA_VALUES[@]}"; do
    for tau in "${TAU_VALUES[@]}"; do
        SAVE_DIR="${SAVE_ROOT}/${DATASET}/kappa${kappa}_tau${tau}"
        LOG_FILE="${SAVE_DIR}/${ARCH}_${DATASET}_${TRAINING_TYPE}.log"
        echo "======================================================"
        echo "  arch=${ARCH}  dataset=${DATASET}"
        echo "  kappa=${kappa}  tau=${tau}"
        echo "  save_dir=${SAVE_DIR}"
        echo "  log_file=${LOG_FILE}"
        echo "======================================================"

        python fit_classifiers.py \
            --data_root      "${DATA_ROOT}" \
            --save_dir       "${SAVE_DIR}"  \
            --epochs         "${EPOCHS}"    \
            --batch_size     "${BATCH_SIZE}" \
            --lr             "${LR}"         \
            --weight_decay   "${WEIGHT_DECAY}" \
            --seed           "${SEED}"       \
            --arch           "${ARCH}"      \
            --training_type  "${TRAINING_TYPE}" \
            --dataset        "${DATASET}"   \
            --kappa          "${kappa}"     \
            --tau            "${tau}"
    done
done

echo ""
echo "All ablation study runs completed."
