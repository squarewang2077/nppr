#!/bin/bash

# Resolve the project root (one level above this script) and add it to
# PYTHONPATH so that top-level packages (arch/, src/, utils/, configs/)
# are importable regardless of where the script is invoked from.
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Configurable hyper-parameters
# ---------------------------------------------------------------------------
GPU_ID=0                    # GPU device ID to use (change this to 0, 1, 3, etc.)
export CUDA_VISIBLE_DEVICES=${GPU_ID}



ARCHS=("vgg16")
DATASETS=("cifar10" "cifar100" "tinyimagenet")

  # Standard training on CIFAR-10 with ResNet-18
for DATASET in "${DATASETS[@]}"; do
    for ARCH in "${ARCHS[@]}"; do
        python scripts/train_classifiers.py \
            --dataset "${DATASET}" --arch "${ARCH}" \
            --training_type standard --epochs 75 \
            --save_dir ./tests/standard_training \
            --pretrained 
    done
done


  # # PGD adversarial training on CIFAR-10 with ResNet-18
  # python scripts/train_classifiers.py \
  #     --dataset cifar10 --arch resnet18 \
  #     --training_type adv_pgd \
  #     --epsilon 0.03137 --alpha 0.00784 --num_steps 3 --epochs 50 \
  #     --save_dir ./tests/adv_training

  # # PR training on CIFAR-10 with ResNet-18
  # python scripts/train_classifiers.py \
  #     --dataset cifar10 --arch resnet18 \
  #     --training_type pr \
  #     --K 3 --num_samples 32 --epochs 50 \
  #     --save_dir ./tests/pr_training