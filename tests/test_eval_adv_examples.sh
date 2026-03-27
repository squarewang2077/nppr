#!/bin/bash

# Resolve the project root (one level above this script) and add it to
# PYTHONPATH so that top-level packages (arch/, src/, utils/, configs/)
# are importable regardless of where the script is invoked from.
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ------------------------------------------------------------------
# eval_adv_examples.py — PGD / CW adversarial accuracy for all three
# checkpoints.  AutoAttack is skipped (--no_aa) as it is too slow
# for a routine test run; remove the flag to enable it.
#
# Attack budget: L-inf, ε=8/255≈0.03137, α=2/255≈0.00784, PGD-20
# (matches the training budget used for adv_training / pr_training).
# ------------------------------------------------------------------

# 1. Standard-trained model
python scripts/eval_adv_examples.py \
    --arch resnet18 --dataset cifar10 \
    --ckp_path ./tests/standard_training/resnet18_cifar10_standard.pth \
    --norm linf --epsilon 0.03137 --alpha 0.00784 --pgd_steps 10 \
    --cw_steps 10 \
    --no_aa \
    --save_csv ./tests/standard_training/eval_adv_results.csv

# 2. PGD adversarially-trained model
python scripts/eval_adv_examples.py \
    --arch resnet18 --dataset cifar10 \
    --ckp_path ./tests/adv_training/resnet18_cifar10_adv_pgd.pth \
    --norm linf --epsilon 0.03137 --alpha 0.00784 --pgd_steps 10 \
    --cw_steps 10 \
    --no_aa \
    --save_csv ./tests/adv_training/eval_adv_results.csv

# 3. PR-trained model
python scripts/eval_adv_examples.py \
    --arch resnet18 --dataset cifar10 \
    --ckp_path ./tests/pr_training/resnet18_cifar10_pr.pth \
    --norm linf --epsilon 0.03137 --alpha 0.00784 --pgd_steps 10 \
    --cw_steps 10 \
    --no_aa \
    --save_csv ./tests/pr_training/eval_adv_results.csv
