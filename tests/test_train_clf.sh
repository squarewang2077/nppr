#!/bin/bash

# Resolve the project root (one level above this script) and add it to
# PYTHONPATH so that top-level packages (arch/, src/, utils/, configs/)
# are importable regardless of where the script is invoked from.
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

  # Standard training on CIFAR-10 with ResNet-18
#   python scripts/train_classifiers.py \
#       --dataset cifar10 --arch resnet18 \
#       --training_type standard --epochs 50 \
#       --save_dir ./tests/standard_training

  # PGD adversarial training on CIFAR-10 with ResNet-18
  python scripts/train_classifiers.py \
      --dataset cifar10 --arch resnet18 \
      --training_type adv_pgd \
      --epsilon 0.03137 --alpha 0.00784 --num_steps 3 --epochs 50 \
      --save_dir ./tests/adv_training

  # PR training on CIFAR-10 with ResNet-18
  python scripts/train_classifiers.py \
      --dataset cifar10 --arch resnet18 \
      --training_type pr \
      --K 3 --num_samples 32 --epochs 50 \
      --save_dir ./tests/pr_training