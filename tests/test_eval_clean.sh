#!/bin/bash

# Resolve the project root (one level above this script) and add it to
# PYTHONPATH so that top-level packages (arch/, src/, utils/, configs/)
# are importable regardless of where the script is invoked from.
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ------------------------------------------------------------------
# eval_clean.py — standard (clean) accuracy for all three checkpoints
# ------------------------------------------------------------------

# 1. Standard-trained model
python scripts/eval_clean.py \
    --arch resnet18 --dataset cifar10 \
    --ckp_path ./tests/standard_training/resnet18_cifar10_standard.pth \
    --save_csv ./tests/standard_training/eval_clean_results.csv

# 2. PGD adversarially-trained model
python scripts/eval_clean.py \
    --arch resnet18 --dataset cifar10 \
    --ckp_path ./tests/adv_training/resnet18_cifar10_adv_pgd.pth \
    --save_csv ./tests/adv_training/eval_clean_results.csv

# 3. PR-trained model
python scripts/eval_clean.py \
    --arch resnet18 --dataset cifar10 \
    --ckp_path ./tests/pr_training/resnet18_cifar10_pr.pth \
    --save_csv ./tests/pr_training/eval_clean_results.csv
