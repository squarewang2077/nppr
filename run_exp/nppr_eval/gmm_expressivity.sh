#!/bin/bash

# Resolve the project root (one level above this script) and add it to
# PYTHONPATH so that top-level packages (arch/, src/, utils/, configs/)
# are importable regardless of where the script is invoked from.
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# GPU to use (change this to 0, 1, 2, ... or "0,1" for multi-GPU visibility)
export CUDA_VISIBLE_DEVICES=1

# ------------------------------------------------------------------
# GMM Expressivity experiment
#
# Model   : ResNet-18, CIFAR-10, L-inf
# Mixture : K, all Gaussian (diag), xy-conditioning
# Budget  : 500 epochs, 20/40 batches/epoch max
# Output  : ./results/gmm_expressivity/
# ------------------------------------------------------------------
for num_modes in 1 3 5 7; do
    python scripts/train_mixture.py \
        --arch resnet18 \
        --dataset cifar10 \
        --clf_ckpt ./ckp/standard/resnet/resnet18_cifar10.pth \
        --ckp_dir ./results/gmm_expressivity/max_batch_40 \
        --K ${num_modes} \
        --component_types "gaussian:${num_modes}" \
        --cond_mode xy \
        --cov_type diag \
        --epochs 500 \
        --batch_index_max 40 \
        --batch_size 256 \
        --num_samples 32 \
        --latent_dim 128
done 
