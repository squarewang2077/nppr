#!/bin/bash

# Usage: bash run_eval_rnd_on_adv_examples.sh [GPU_ID]
# GPU_ID defaults to 0 if not specified.
GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Resolve the project root and add it to PYTHONPATH.
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ------------------------------------------------------------------
# run_eval_rnd_on_adv_examples.sh — Evaluation of standard-trained
# models using 1-step PGD + Gaussian sampling
#
# Models  : resnet18, resnet50, wide_resnet50_2, vgg16
# Datasets: cifar10, cifar100, tinyimagenet
#
# Checkpoint naming convention in ./ckp/standard/:
#   ./ckp/standard/{subdir}/{model}_{dataset}.pth
#   where subdir is: resnet/ for resnet18/50, wrn/ for wide_resnet50_2, vgg/ for vgg16
#
# Results: ./results/eval_rnd_on_std_models/eval_rnd_std_{model}_{dataset}.csv
# ------------------------------------------------------------------

STD_CKPT_DIR="${PROJECT_ROOT}/ckp/standard"
RESULTS_DIR="${PROJECT_ROOT}/results/eval_rnd_on_std_models"

# Attack and sampling parameters
NORM="linf"
EPSILON=0.0627 # 16/255 ≈ 0.0627
ALPHA=0.00784
NUM_SAMPLES=32
VARIANCE=0.001

DATASETS=("cifar10" "cifar100" "tinyimagenet")
MODELS=("resnet18" "resnet50" "wide_resnet50_2" "vgg16")

echo "=========================================="
echo "  1-step PGD + Gaussian Sampling Evaluation"
echo "  Standard-Trained Models"
echo "  GPU: ${GPU_ID}"
echo "  Results: ${RESULTS_DIR}"
echo "=========================================="
echo "  Parameters:"
echo "    norm=${NORM}, epsilon=${EPSILON}, alpha=${ALPHA}"
echo "    num_samples=${NUM_SAMPLES}, variance=${VARIANCE}"
echo "=========================================="

mkdir -p "${RESULTS_DIR}"

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo ""
        echo "------------------------------------------"
        echo "  Dataset: ${dataset} | Model: ${model}"
        echo "------------------------------------------"

        # Determine checkpoint subdirectory based on model
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

        STD_CKPT="${STD_CKPT_DIR}/${SUBDIR}/${model}_${dataset}.pth"

        if [ ! -f "${STD_CKPT}" ]; then
            echo "WARNING: Checkpoint not found: ${STD_CKPT}, skipping."
            continue
        fi

        # Run evaluation
        echo "  Evaluating 1-step PGD + Gaussian sampling..."
        python scripts/eval_rnd_on_adv_examples.py \
            --arch "${model}" \
            --dataset "${dataset}" \
            --ckp_path "${STD_CKPT}" \
            --norm "${NORM}" \
            --epsilon "${EPSILON}" \
            --alpha "${ALPHA}" \
            --num_samples "${NUM_SAMPLES}" \
            --variance "${VARIANCE}" \
            --save_csv "${RESULTS_DIR}/eval_rnd_std_${model}_${dataset}.csv"

        echo "  Done: ${RESULTS_DIR}/eval_rnd_std_${model}_${dataset}.csv"
    done
done

echo ""
echo "=========================================="
echo "  All evaluations complete."
echo "  Results: ${RESULTS_DIR}"
echo "=========================================="
