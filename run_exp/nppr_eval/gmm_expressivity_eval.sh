#!/bin/bash

# Resolve the project root and add it to PYTHONPATH.
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# GPU to use
export CUDA_VISIBLE_DEVICES=1

# ------------------------------------------------------------------
# GMM Expressivity — evaluation
#
# Loads K=1 / K=3 / K=7 mixture models trained by gmm_expressivity.sh
# from ./results/gmm_expressivity/resnet18_on_cifar10/ and evaluates
# each one against the standard-trained ResNet-18 classifier.
#
# Outputs one CSV per K value in the same directory.
# ------------------------------------------------------------------

CLF_CKPT="./tests/standard_training/resnet18_cifar10_standard.pth"
GMM_DIR="./results/gmm_expressivity/resnet18_on_cifar10"

for K in 5; do

    # Locate the checkpoint for this K — match mixture_K{K}_*.pt
    GMM_CKPT=$(ls "${GMM_DIR}/mixture_K${K}_"*.pt 2>/dev/null | head -1)

    if [ -z "${GMM_CKPT}" ]; then
        echo "[skip] No mixture checkpoint found for K=${K} in ${GMM_DIR}"
        continue
    fi

    echo "============================================================"
    echo " K=${K}  →  ${GMM_CKPT}"
    echo "============================================================"

    python scripts/eval_prob_pert.py \
        --arch          resnet18 \
        --dataset       cifar10 \
        --ckp_path      "${CLF_CKPT}" \
        --mixture_path  "${GMM_CKPT}" \
        --norm          linf \
        --epsilon       0.06274 \
        --num_samples   32 \
        --save_csv      "${GMM_DIR}/eval_K${K}.csv"

done

echo "All evaluations complete. Results in ${GMM_DIR}/"
