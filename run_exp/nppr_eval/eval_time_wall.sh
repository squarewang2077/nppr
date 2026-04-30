#!/bin/bash

# Usage: bash eval_time_wall.sh [GPU_ID]
# GPU_ID defaults to 1 if not specified.
GPU_ID=${1:-1}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Resolve the project root (one level above this script) and add it to
# PYTHONPATH so that top-level packages (arch/, src/, utils/, configs/)
# are importable regardless of where the script is invoked from.
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ------------------------------------------------------------------
# Wall-time analysis for all four evaluation types.
# Mirrors eval_all.sh: same models, datasets, checkpoints, args.
# Reports per-eval-type and per-combo timing, plus a grand total.
# ------------------------------------------------------------------

STD_CKPT_DIR="${PROJECT_ROOT}/ckp/standard"
GMM_CKPT_DIR="${PROJECT_ROOT}/ckp/gmm_fitting"
RESULTS_DIR="${PROJECT_ROOT}/results/extend_comparison"
GMM_NAME="gmm_K7_cond(xy)_decoder(trainable_128)_linf(16)_reg(none).pt"

DATASETS=("cifar10" "cifar100" "tinyimagenet")
MODELS=("resnet18" "resnet50" "wide_resnet50_2" "vgg16")

echo "=========================================="
echo "  Evaluation Wall-time Analysis"
echo "  GPU: ${GPU_ID}"
echo "=========================================="

# Helper: elapsed seconds between two ns timestamps
elapsed() { echo "scale=3; ($2 - $1) / 1000000000" | bc; }

GRAND_START=$(date +%s%N)

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo ""
        echo "------------------------------------------"
        echo "  Dataset: ${dataset} | Model: ${model}"
        echo "------------------------------------------"

        case "${model}" in
            resnet18|resnet50) SUBDIR="resnet" ;;
            wide_resnet50_2)   SUBDIR="wrn"    ;;
            vgg16)             SUBDIR="vgg"    ;;
            *)
                echo "ERROR: Unknown model: ${model}, skipping."
                continue
                ;;
        esac

        CLF_CKPT="${STD_CKPT_DIR}/${SUBDIR}/${model}_${dataset}.pth"
        GMM_CKPT="${GMM_CKPT_DIR}/${SUBDIR}/${model}_on_${dataset}/${GMM_NAME}"

        if [ ! -f "${CLF_CKPT}" ]; then
            echo "WARNING: Classifier checkpoint not found: ${CLF_CKPT}, skipping."
            continue
        fi

        mkdir -p "${RESULTS_DIR}"
        COMBO_START=$(date +%s%N)

        # ── 1. Clean accuracy ────────────────────────────────────────
        T0=$(date +%s%N)
        python scripts/eval_clean.py \
            --arch "${model}" \
            --dataset "${dataset}" \
            --ckp_path "${CLF_CKPT}" \
            --save_csv "${RESULTS_DIR}/eval_clean_${model}_${dataset}.csv"
        T1=$(date +%s%N)
        echo "  [1/4] clean          : $(elapsed $T0 $T1) s"

        # ── 2. Adversarial robustness ────────────────────────────────
        T0=$(date +%s%N)
        python scripts/eval_adv_examples.py \
            --arch "${model}" \
            --dataset "${dataset}" \
            --ckp_path "${CLF_CKPT}" \
            --norm linf --epsilon 0.03137 --alpha 0.00784 \
            --pgd_steps 10 --cw_steps 10 \
            --no_aa \
            --save_csv "${RESULTS_DIR}/eval_adv_${model}_${dataset}.csv"
        T1=$(date +%s%N)
        echo "  [2/4] adversarial    : $(elapsed $T0 $T1) s"

        # ── 3. Corruption robustness ─────────────────────────────────
        T0=$(date +%s%N)
        python scripts/eval_corruptions.py \
            --arch "${model}" \
            --dataset "${dataset}" \
            --ckp_path "${CLF_CKPT}" \
            --save_csv "${RESULTS_DIR}/eval_corruptions_${model}_${dataset}.csv"
        T1=$(date +%s%N)
        echo "  [3/4] corruptions    : $(elapsed $T0 $T1) s"

        # ── 4. Probabilistic robustness ──────────────────────────────
        if [ ! -f "${GMM_CKPT}" ]; then
            echo "  [4/4] prob_pert      : SKIPPED (GMM checkpoint not found)"
        else
            T0=$(date +%s%N)
            python scripts/eval_prob_perturbation.py \
                --arch "${model}" \
                --dataset "${dataset}" \
                --ckp_path "${CLF_CKPT}" \
                --norm linf --epsilon 0.03137 \
                --num_samples 32 --K 3 \
                --gmm_path "${GMM_CKPT}" \
                --save_csv "${RESULTS_DIR}/eval_prob_perturbation_${model}_${dataset}.csv"
            T1=$(date +%s%N)
            echo "  [4/4] prob_pert      : $(elapsed $T0 $T1) s"
        fi

        COMBO_END=$(date +%s%N)
        echo "  >>> combo total      : $(elapsed $COMBO_START $COMBO_END) s"
    done
done

GRAND_END=$(date +%s%N)
echo ""
echo "=========================================="
echo "  Grand total wall-time : $(elapsed $GRAND_START $GRAND_END) s"
echo "  GPU                   : ${GPU_ID}"
echo "=========================================="
