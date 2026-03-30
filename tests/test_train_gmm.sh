#!/bin/bash

# Resolve the project root (one level above this script) and add it to
# PYTHONPATH so that top-level packages (arch/, src/, utils/, configs/)
# are importable regardless of where the script is invoked from.
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ------------------------------------------------------------------
# train_gmm.py — fit a GMM4PR on top of each of the three trained
# classifiers (standard, adversarial, PR-trained).
#
# Config  : resnet18_on_cifar10_linf  (ResNet-18, CIFAR-10, L-inf)
# Overrides for quick testing: 30 epochs, K=3
# Output  : ./ckp/gmm_fitting/resnet18_on_cifar10/
# ------------------------------------------------------------------

# 1. GMM trained on top of the standard-trained classifier
python scripts/train_gmm.py \
    --config resnet18_on_cifar10_linf \
    --clf_ckpt ./tests/standard_training/resnet18_cifar10_standard.pth \
    --ckp_dir ./tests/standard_training/by_gmm \
    --epochs 5 --K 3

# # 2. GMM trained on top of the PGD adversarially-trained classifier
# python scripts/train_gmm.py \
#     --config resnet18_on_cifar10_linf \
#     --clf_ckpt ./tests/adv_training/resnet18_cifar10_adv_pgd.pth \
#     --ckp_dir ./tests/adv_training/by_gmm \
#     --epochs 5 --K 3

# # 3. GMM trained on top of the PR-trained classifier
# python scripts/train_gmm.py \
#     --config resnet18_on_cifar10_linf \
#     --clf_ckpt ./tests/pr_training/resnet18_cifar10_pr.pth \
#     --ckp_dir ./tests/pr_training/by_gmm \
#     --epochs 5 --K 3
