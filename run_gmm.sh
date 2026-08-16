#!/bin/bash

# ------------------------------------------------------------------
# run_gmm.sh — Train GMM4PR on adversarially trained models
#
# This script trains GMM models on top of adversarially trained
# classifiers from ./ckp/adv_pgd directory.
#
# Datasets: cifar10, cifar100, tinyimagenet
# Models: resnet18, resnet50, wide_resnet50_2
# Training: 50 epochs, K=3 components
# Output: ./ckp/gmm_fitting4adv_models/
# ------------------------------------------------------------------

# Resolve the project root and add to PYTHONPATH
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Configuration
EPOCHS=50
K=3
BASE_CKP_DIR="./ckp/gmm_fitting4adv"
ADV_MODEL_DIR="./ckp/adv_pgd"

# Arrays for datasets and models
DATASETS=("cifar10" "cifar100" "tinyimagenet")
MODELS=("resnet18" "resnet50" "wide_resnet50_2")
# DATASETS=("cifar10")
# MODELS=("resnet18")

echo "========================================"
echo "Training GMM on Adversarially Trained Models"
echo "========================================"
echo "Epochs: ${EPOCHS}"
echo "K components: ${K}"
echo "Output directory: ${BASE_CKP_DIR}"
echo "========================================"

# Loop through all combinations
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo ""
        echo "----------------------------------------"
        echo "Dataset: ${dataset} | Model: ${model}"
        echo "----------------------------------------"

        # Determine checkpoint path based on dataset naming convention
        # CIFAR10 models don't have _adv_pgd suffix, but CIFAR100 and TinyImageNet do
        if [ "${dataset}" = "cifar10" ]; then
            CLF_CKPT="${ADV_MODEL_DIR}/${dataset}/${model}_${dataset}.pth"
        else
            CLF_CKPT="${ADV_MODEL_DIR}/${dataset}/${model}_${dataset}_adv_pgd.pth"
        fi

        # Check if model file exists
        if [ ! -f "${CLF_CKPT}" ]; then
            echo "WARNING: Model file not found: ${CLF_CKPT}"
            echo "Skipping..."
            continue
        fi

        # Set output directory
        CKP_DIR="${BASE_CKP_DIR}"

        # Run training
        python scripts/train_gmm.py \
            --dataset "${dataset}" \
            --arch "${model}" \
            --clf_ckpt "${CLF_CKPT}" \
            --ckp_dir "${CKP_DIR}" \
            --epochs ${EPOCHS} \
            --K ${K}

        if [ $? -eq 0 ]; then
            echo "✓ Successfully trained GMM for ${model} on ${dataset}"
        else
            echo "✗ Failed to train GMM for ${model} on ${dataset}"
        fi
    done
done

echo ""
echo "========================================"
echo "All GMM training jobs completed!"
echo "========================================"
