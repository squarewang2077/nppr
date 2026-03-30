#!/bin/bash

# Resolve the project root (one level above this script) and add it to
# PYTHONPATH so that top-level packages (arch/, src/, utils/, configs/)
# are importable regardless of where the script is invoked from.
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ------------------------------------------------------------------
# train_mixture.py  fit a Mixture Model (mixture4pr) on top of each
# of the three trained classifiers (standard, adversarial, PR-trained).
#
# Model   : ResNet-18, CIFAR-10, L-inf
# Overrides for quick testing: 30 epochs, K=3
# Output  : ./tests/{standard,adv,pr}_training/by_mixture/
# ------------------------------------------------------------------

# 1. Mixture Model trained on top of the standard-trained classifier
python scripts/train_mixture.py \
    --arch resnet18 \
    --dataset cifar10 \
    --clf_ckpt ./tests/standard_training/resnet18_cifar10_standard.pth \
    --ckp_dir ./tests/standard_training/by_mixture \
    --K 3 \
    --latent_dim 128 \
    --component_types "gaussian:3" \
    --cond_mode xy \
    --cov_type diag \
    --epochs 5 \
    --batch_size 256 \
    --num_samples 32

# # 2. Mixture Model trained on top of the PGD adversarially-trained classifier
# python scripts/train_mixture.py \
#     --arch resnet18 \
#     --dataset cifar10 \
#     --clf_ckpt ./tests/adv_training/resnet18_cifar10_adv_pgd.pth \
#     --ckp_dir ./tests/adv_training/by_mixture \
#     --K 3 \
#     --latent_dim 128 \
#     --component_types "gaussian:3" \
#     --cond_mode xy \
#     --cov_type diag \
#     --norm linf \
#     --epsilon 0.0314 \
#     --epochs 5 \
#     --batch_size 256 \
#     --num_samples 32

# # 3. Mixture Model trained on top of the PR-trained classifier
# python scripts/train_mixture.py \
#     --arch resnet18 \
#     --dataset cifar10 \
#     --clf_ckpt ./tests/pr_training/resnet18_cifar10_pr.pth \
#     --ckp_dir ./tests/pr_training/by_mixture \
#     --K 3 \
#     --latent_dim 128 \
#     --component_types "gaussian:3" \
#     --cond_mode xy \
#     --cov_type diag \
#     --norm linf \
#     --epsilon 0.0314 \
#     --epochs 5 \
#     --batch_size 256 \
#     --num_samples 32
