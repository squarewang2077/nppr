#!/bin/bash

# Usage: bash eval_all.sh [GPU_ID]
# GPU_ID defaults to 1 if not specified.
GPU_ID=0
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Resolve the project root (one level above this script) and add it to
# PYTHONPATH so that top-level packages (arch/, src/, utils/, configs/)
# are importable regardless of where the script is invoked from.
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ------------------------------------------------------------------
# eval_all.sh — Full evaluation of standard-trained models
#
# Models  : resnet18, resnet50, wide_resnet50_2, vgg16
# Datasets: cifar10, cifar100, tinyimagenet
# Sources : ./ckp/standard/{resnet,wrn,vgg}/
# Results : ./results/eval_std_models/{dataset}/{model}/
#
# Evaluations:
#   1. Clean accuracy          (eval_clean.py)
#   2. Adversarial robustness  (eval_adv_examples.py)
#   3. Corruption robustness   (eval_corruptions.py)
#   4. Probabilistic robustness (eval_prob_perturbation.py)
# ------------------------------------------------------------------

STD_CKPT_DIR="${PROJECT_ROOT}/ckp/standard"
GMM_CKPT_DIR="${PROJECT_ROOT}/ckp/gmm_fitting"
RESULTS_DIR="${PROJECT_ROOT}/results/eval_std_models"
GMM_NAME="gmm_K7_cond(xy)_decoder(trainable_128)_linf(16)_reg(none).pt"

DATASETS=("cifar10" "cifar100" "tinyimagenet")
MODELS=("resnet18" "resnet50" "wide_resnet50_2" "vgg16")

echo "=========================================="
echo "  Full Evaluation — Standard Training"
echo "  GPU: ${GPU_ID}"
echo "  Results: ${RESULTS_DIR}"
echo "=========================================="

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo ""
        echo "------------------------------------------"
        echo "  Dataset: ${dataset} | Model: ${model}"
        echo "------------------------------------------"

        # Resolve subdir and checkpoint paths
        case "${model}" in
            resnet18|resnet50)
                SUBDIR="resnet"
                ;;
            wide_resnet50_2)
                SUBDIR="wrn"
                ;;
            vgg16)
                SUBDIR="vgg"
                ;;
            *)
                echo "ERROR: Unknown model: ${model}, skipping."
                continue
                ;;
        esac

        CLF_CKPT="${STD_CKPT_DIR}/${SUBDIR}/${model}_${dataset}.pth"
        GMM_CKPT="${GMM_CKPT_DIR}/${SUBDIR}/${model}_on_${dataset}/${GMM_NAME}"

        # Skip if classifier checkpoint is missing
        if [ ! -f "${CLF_CKPT}" ]; then
            echo "WARNING: Classifier checkpoint not found: ${CLF_CKPT}"
            echo "Skipping."
            continue
        fi

        mkdir -p "${RESULTS_DIR}"

        # ── 1. Clean accuracy ────────────────────────────────────────
        echo "  [1/4] Clean accuracy..."
        python scripts/eval_clean.py \
            --arch "${model}" \
            --dataset "${dataset}" \
            --ckp_path "${CLF_CKPT}" \
            --save_csv "${RESULTS_DIR}/eval_clean_${model}_${dataset}.csv"

        # ── 2. Adversarial robustness (PGD / CW, no AutoAttack) ─────
        echo "  [2/4] Adversarial robustness..."
        python scripts/eval_adv_examples.py \
            --arch "${model}" \
            --dataset "${dataset}" \
            --ckp_path "${CLF_CKPT}" \
            --norm linf --epsilon 0.03137 --alpha 0.00784 \
            --pgd_steps 10 --cw_steps 10 \
            --save_csv "${RESULTS_DIR}/eval_adv_${model}_${dataset}.csv"

        # ── 3. Corruption robustness ─────────────────────────────────
        echo "  [3/4] Corruption robustness..."
        python scripts/eval_corruptions.py \
            --arch "${model}" \
            --dataset "${dataset}" \
            --ckp_path "${CLF_CKPT}" \
            --save_csv "${RESULTS_DIR}/eval_corruptions_${model}_${dataset}.csv"

        # ── 4. Probabilistic robustness ──────────────────────────────
        echo "  [4/4] Probabilistic robustness..."
        if [ ! -f "${GMM_CKPT}" ]; then
            echo "  WARNING: GMM checkpoint not found: ${GMM_CKPT}"
            echo "  Skipping probabilistic robustness for this combo."
        else
            python scripts/eval_prob_perturbation.py \
                --arch "${model}" \
                --dataset "${dataset}" \
                --ckp_path "${CLF_CKPT}" \
                --norm linf --epsilon 0.03137 \
                --num_samples 32 --K 3 \
                --gmm_path "${GMM_CKPT}" \
                --save_csv "${RESULTS_DIR}/eval_prob_perturbation_${model}_${dataset}.csv"
        fi

        echo "  Done: results saved to ${RESULTS_DIR}/"
    done
done

echo ""
echo "=========================================="
echo "  All evaluations complete."
echo "  Results: ${RESULTS_DIR}"
echo "=========================================="
