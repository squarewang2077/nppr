#!/bin/bash

# ------------------------------------------------------------------
# run_train_mixture.sh — Train Mixture Model on ResNet-18 / CIFAR-10
#
# Components : 1 Gaussian + 1 Laplace + 1 Salt-and-Pepper  (K=3)
# Classifier : ./ckp/standard/resnet/resnet18_cifar10.pth
# Output     : ./ckp/mixture_fitting4std_models/
# ------------------------------------------------------------------

# Resolve the project root and add to PYTHONPATH
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Usage: bash run_train_mixture.sh [GPU_ID]
GPU_ID=0
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Configuration
ARCH="resnet18"
DATASET="cifar10"
EPOCHS=50
K=3
CLF_CKPT="./ckp/standard/resnet/${ARCH}_${DATASET}.pth"
CKP_DIR="./ckp/mixture_fitting4std_models"

echo "========================================"
echo "  Mixture Model Training"
echo "  Model      : ${ARCH} on ${DATASET}"
echo "  Components : 1 Gaussian + 1 Laplace + 1 Salt-and-Pepper"
echo "  K          : ${K}"
echo "  Epochs     : ${EPOCHS}"
echo "  GPU        : ${GPU_ID}"
echo "  Output     : ${CKP_DIR}"
echo "========================================"

if [ ! -f "${CLF_CKPT}" ]; then
    echo "ERROR: Classifier checkpoint not found: ${CLF_CKPT}"
    exit 1
fi

python scripts/train_mixture.py \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    --clf_ckpt "${CLF_CKPT}" \
    --ckp_dir "${CKP_DIR}" \
    --epochs ${EPOCHS} \
    --K ${K} \
    --component_types "gaussian:1,laplace:1,salt_and_pepper:1"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Training complete. Model saved to ${CKP_DIR}/"
else
    echo ""
    echo "✗ Training failed."
    exit 1
fi
