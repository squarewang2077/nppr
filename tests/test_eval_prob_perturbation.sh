#!/bin/bash

# Resolve the project root (one level above this script) and add it to
# PYTHONPATH so that top-level packages (arch/, src/, utils/, configs/)
# are importable regardless of where the script is invoked from.
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ------------------------------------------------------------------
# eval_prob_perturbation.py — probabilistic robustness (PR, PR-G,
# PR-U, PR-L) for all three checkpoints.
#
# Attack budget: L-inf, ε=8/255≈0.03137, N=32 samples, K=3
# (matches the training budget used for adv_training / pr_training).
# Random-noise baselines (Gaussian, Uniform, Laplace) are included.
# ------------------------------------------------------------------

# 1. Standard-trained model
python scripts/eval_prob_perturbation.py \
    --arch resnet18 --dataset cifar10 \
    --ckp_path ./tests/standard_training/resnet18_cifar10_standard.pth \
    --norm linf --epsilon 0.03137 --num_samples 32 --K 3 \
    --gmm_path "./tests/standard_training/by_gmm/resnet18_on_cifar10/gmm_K7_cond(xy)_decoder(trainable_128)_linf(16)_reg(none).pt" \
    --save_csv ./tests/standard_training/eval_prob_perturbation_results2.csv

# 2. PGD adversarially-trained model
python scripts/eval_prob_perturbation.py \
    --arch resnet18 --dataset cifar10 \
    --ckp_path ./tests/adv_training/resnet18_cifar10_adv_pgd.pth \
    --norm linf --epsilon 0.03137 --num_samples 32 --K 3 \
    --gmm_path "./tests/adv_training/by_gmm/resnet18_on_cifar10/gmm_K7_cond(xy)_decoder(trainable_128)_linf(16)_reg(none).pt" \
    --save_csv ./tests/adv_training/eval_prob_perturbation_results2.csv

# 3. PR-trained model
python scripts/eval_prob_perturbation.py \
    --arch resnet18 --dataset cifar10 \
    --ckp_path ./tests/pr_training/resnet18_cifar10_pr.pth \
    --norm linf --epsilon 0.03137 --num_samples 32 --K 3 \
    --gmm_path "./tests/pr_training/by_gmm/resnet18_on_cifar10/gmm_K7_cond(xy)_decoder(trainable_128)_linf(16)_reg(none).pt" \
    --save_csv ./tests/pr_training/eval_prob_perturbation_results2.csv
