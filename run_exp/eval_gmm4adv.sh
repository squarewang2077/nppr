#!/bin/bash

# Usage: bash eval_gmm4adv.sh [GPU_ID]
# GPU_ID defaults to 1 if not specified.
GPU_ID=0
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Resolve the project root and add it to PYTHONPATH.
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ------------------------------------------------------------------
# eval_gmm4adv.sh — Probabilistic robustness evaluation of PGD
# adversarially-trained models using GMMs that were fitted on
# adversarial models.
#
# Models  : resnet18, resnet50, wide_resnet50_2
# Datasets: cifar10, cifar100, tinyimagenet
#   (GMMs currently available for cifar10 and cifar100 only;
#    tinyimagenet combos are skipped if the GMM is not found)
#
# Classifier checkpoints : ./ckp/adv_pgd/
#   cifar10      → {model}_{dataset}.pth
#   cifar100     → {model}_{dataset}_adv_pgd.pth
#   tinyimagenet → {model}_{dataset}_adv_pgd.pth
#
# GMM checkpoints : ./ckp/gmm_fitting4adv/{model}_on_{dataset}/
#   gmm_K3_cond(xy)_decoder(bicubic_trainable)_linf(0.063)_epochs50.pt
#
# Evaluation:
#   - Probabilistic robustness (eval_prob_perturbation.py)
#     GMM feature extractor + classifier: both adversarially trained
#
# Note: Clean accuracy, adversarial robustness, and corruption
#       robustness are evaluated in eval_adv_models.sh
#
# Results: ./results/eval_gmm4adv/eval_prob_perturbation_{model}_{dataset}.csv
# ------------------------------------------------------------------

ADV_CKPT_DIR="${PROJECT_ROOT}/ckp/adv_pgd"
GMM_CKPT_DIR="${PROJECT_ROOT}/ckp/gmm_fitting4adv"
RESULTS_DIR="${PROJECT_ROOT}/results/eval_gmm4adv"
GMM_NAME="gmm_K3_cond(xy)_decoder(bicubic_trainable)_linf(0.063)_epochs50.pt"

DATASETS=("cifar10" "cifar100" "tinyimagenet")
MODELS=("resnet18" "resnet50" "wide_resnet50_2")

echo "=========================================="
echo "  Probabilistic Robustness Evaluation"
echo "  Adv Models + Adv GMMs"
echo "  GPU: ${GPU_ID}"
echo "  Results: ${RESULTS_DIR}"
echo "=========================================="

mkdir -p "${RESULTS_DIR}"

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo ""
        echo "------------------------------------------"
        echo "  Dataset: ${dataset} | Model: ${model}"
        echo "------------------------------------------"

        # Adversarial classifier checkpoint (cifar10 has no _adv_pgd suffix)
        if [ "${dataset}" = "cifar10" ]; then
            ADV_CKPT="${ADV_CKPT_DIR}/${dataset}/${model}_${dataset}.pth"
        else
            ADV_CKPT="${ADV_CKPT_DIR}/${dataset}/${model}_${dataset}_adv_pgd.pth"
        fi

        GMM_CKPT="${GMM_CKPT_DIR}/${model}_on_${dataset}/${GMM_NAME}"

        if [ ! -f "${ADV_CKPT}" ]; then
            echo "WARNING: Adversarial checkpoint not found: ${ADV_CKPT}, skipping."
            continue
        fi

        # ── Probabilistic robustness (adv GMM + adv classifier) ──
        echo "  Evaluating probabilistic robustness (adv GMM)..."
        if [ ! -f "${GMM_CKPT}" ]; then
            echo "  WARNING: Adv GMM not found: ${GMM_CKPT}, skipping."
        else
            python scripts/eval_prob_perturbation.py \
                --arch "${model}" \
                --dataset "${dataset}" \
                --ckp_path "${ADV_CKPT}" \
                --norm linf --epsilon 0.03137 \
                --num_samples 32 --K 3 \
                --gmm_path "${GMM_CKPT}" \
                --save_csv "${RESULTS_DIR}/eval_prob_perturbation_${model}_${dataset}.csv"
        fi

        echo "  Done: ${RESULTS_DIR}/eval_prob_perturbation_${model}_${dataset}.csv"
    done
done

echo ""
echo "=========================================="
echo "  Probabilistic robustness evaluation complete."
echo "  Results: ${RESULTS_DIR}"
echo "=========================================="
