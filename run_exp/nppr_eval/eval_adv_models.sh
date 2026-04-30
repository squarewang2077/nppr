#!/bin/bash

# Usage: bash eval_adv_models.sh [GPU_ID]
# GPU_ID defaults to 1 if not specified.
GPU_ID=1
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Resolve the project root and add it to PYTHONPATH.
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ------------------------------------------------------------------
# eval_adv_models.sh — Full evaluation of PGD adversarially-trained
# models (resnet18, resnet50, wide_resnet50_2) on cifar10/100 and
# tinyimagenet.  No vgg16 for adversarial models.
#
# Evaluations:
#   1. Clean accuracy          (eval_clean.py)
#   2. Adversarial robustness  (eval_adv_examples.py)
#   3. Corruption robustness   (eval_corruptions.py)
#   4. Probabilistic robustness (eval_prob_perturbation.py)
#      GMM feature extractor : standard-trained model
#                              (./ckp/gmm_fitting/{subdir}/...)
#      Evaluated classifier  : adversarial-trained model
#
# Checkpoint naming convention in ./ckp/adv_pgd/:
#   cifar10      : {model}_{dataset}.pth
#   cifar100     : {model}_{dataset}_adv_pgd.pth
#   tinyimagenet : {model}_{dataset}_adv_pgd.pth
#
# Results: ./results/eval_adv_models/eval_{type}_{model}_{dataset}.csv
# ------------------------------------------------------------------

ADV_CKPT_DIR="${PROJECT_ROOT}/ckp/adv_pgd"
GMM_CKPT_DIR="${PROJECT_ROOT}/ckp/gmm_fitting"
RESULTS_DIR="${PROJECT_ROOT}/results/eval_adv_models"
GMM_NAME="gmm_K7_cond(xy)_decoder(trainable_128)_linf(16)_reg(none).pt"

DATASETS=("cifar10" "cifar100" "tinyimagenet")
MODELS=("resnet18" "resnet50" "wide_resnet50_2")

echo "=========================================="
echo "  Full Evaluation — PGD Adversarial Models"
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

        # GMM subdir (standard-fitted GMM)
        case "${model}" in
            resnet18|resnet50) SUBDIR="resnet" ;;
            wide_resnet50_2)   SUBDIR="wrn"    ;;
            *)
                echo "ERROR: Unknown model: ${model}, skipping."
                continue
                ;;
        esac

        # Adversarial checkpoint path — cifar10 has no _adv_pgd suffix
        if [ "${dataset}" = "cifar10" ]; then
            ADV_CKPT="${ADV_CKPT_DIR}/${dataset}/${model}_${dataset}.pth"
        else
            ADV_CKPT="${ADV_CKPT_DIR}/${dataset}/${model}_${dataset}_adv_pgd.pth"
        fi

        GMM_CKPT="${GMM_CKPT_DIR}/${SUBDIR}/${model}_on_${dataset}/${GMM_NAME}"

        if [ ! -f "${ADV_CKPT}" ]; then
            echo "WARNING: Adversarial checkpoint not found: ${ADV_CKPT}, skipping."
            continue
        fi

        # ── 1. Clean accuracy ────────────────────────────────────────
        echo "  [1/4] Clean accuracy..."
        python scripts/eval_clean.py \
            --arch "${model}" \
            --dataset "${dataset}" \
            --ckp_path "${ADV_CKPT}" \
            --save_csv "${RESULTS_DIR}/eval_clean_${model}_${dataset}.csv"

        # ── 2. Adversarial robustness (PGD / CW, no AutoAttack) ─────
        echo "  [2/4] Adversarial robustness..."
        python scripts/eval_adv_examples.py \
            --arch "${model}" \
            --dataset "${dataset}" \
            --ckp_path "${ADV_CKPT}" \
            --norm linf --epsilon 0.03137 --alpha 0.00784 \
            --pgd_steps 10 --cw_steps 10 \
            --save_csv "${RESULTS_DIR}/eval_adv_${model}_${dataset}.csv"

        # ── 3. Corruption robustness ─────────────────────────────────
        echo "  [3/4] Corruption robustness..."
        python scripts/eval_corruptions.py \
            --arch "${model}" \
            --dataset "${dataset}" \
            --ckp_path "${ADV_CKPT}" \
            --save_csv "${RESULTS_DIR}/eval_corruptions_${model}_${dataset}.csv"

        # ── 4. Probabilistic robustness ──────────────────────────────
        # GMM uses standard-model feature extractor; classifier is adv-trained.
        echo "  [4/4] Probabilistic robustness (GMM from standard model)..."
        if [ ! -f "${GMM_CKPT}" ]; then
            echo "  WARNING: GMM checkpoint not found: ${GMM_CKPT}, skipping."
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

        echo "  Done: ${RESULTS_DIR}/eval_*_${model}_${dataset}.csv"
    done
done

echo ""
echo "=========================================="
echo "  All evaluations complete."
echo "  Results: ${RESULTS_DIR}"
echo "=========================================="
