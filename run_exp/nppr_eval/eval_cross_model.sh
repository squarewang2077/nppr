#!/bin/bash

# Usage: bash eval_same_feat_ext_on_diff_models.sh [GPU_ID]
# GPU_ID defaults to 0 if not specified.
GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Resolve the project root and add it to PYTHONPATH.
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ------------------------------------------------------------------
# Cross-model GMM PR evaluation.
#
# Fixes one GMM (trained on a specific feature extractor / dataset /
# radius) and evaluates it on classifiers with DIFFERENT architectures
# trained on the same dataset.
#
# GMM : K=7, cond(xy), trainable_128 decoder, linf(16)
#        feat_arch = resnet18, dataset = cifar10
#
# Target classifiers (same dataset, different archs):
#   resnet18, resnet50, wide_resnet50_2, vgg16
#
# Eval radius matches GMM training radius (16/255 ≈ 0.06275).
#
# Results: ./results/gmm_cross_model/
#   eval_prob_pert_gmmFeat{feat_arch}_{dataset}_clf{clf_arch}.csv
# ------------------------------------------------------------------

# ---- Fixed GMM settings ----
GMM_FEAT_ARCH="vgg16"
DATASET="cifar10"
GMM_NORM="linf"
GMM_R=16
GMM_EPS=0.06275
GMM_CKPT="${PROJECT_ROOT}/ckp/gmm_fitting/vgg/${GMM_FEAT_ARCH}_on_${DATASET}/gmm_K7_cond(xy)_decoder(trainable_128)_${GMM_NORM}(${GMM_R})_reg(none).pt"

# ---- Target classifier architectures ----
# Each entry is: "arch  ckp_subdir"
CLF_ARCHS=(
    "resnet18        resnet"
    "resnet50        resnet"
    "wide_resnet50_2 wrn"
    "vgg16           vgg"
)

RESULTS_DIR="${PROJECT_ROOT}/results/gmm_cross_model"
mkdir -p "${RESULTS_DIR}"

echo "=========================================="
echo "  GMM Cross-Model Evaluation"
echo "  GMM feat arch : ${GMM_FEAT_ARCH}"
echo "  Dataset       : ${DATASET}"
echo "  GMM checkpoint: ${GMM_CKPT}"
echo "  Eval budget   : ${GMM_NORM}(${GMM_R}) = ${GMM_EPS}"
echo "  GPU           : ${GPU_ID}"
echo "  Output        : ${RESULTS_DIR}"
echo "=========================================="

if [ ! -f "${GMM_CKPT}" ]; then
    echo "ERROR: GMM checkpoint not found: ${GMM_CKPT}"
    exit 1
fi

for entry in "${CLF_ARCHS[@]}"; do
    CLF_ARCH=$(echo "${entry}" | awk '{print $1}')
    CLF_SUBDIR=$(echo "${entry}" | awk '{print $2}')
    CLF_CKPT="${PROJECT_ROOT}/ckp/standard/${CLF_SUBDIR}/${CLF_ARCH}_${DATASET}.pth"
    OUT_CSV="${RESULTS_DIR}/eval_prob_pert_gmmFeat_${GMM_FEAT_ARCH}_${DATASET}_clf${CLF_ARCH}.csv"

    echo ""
    echo "------------------------------------------"
    echo "  Classifier arch: ${CLF_ARCH}"
    echo "------------------------------------------"

    if [ ! -f "${CLF_CKPT}" ]; then
        echo "WARNING: Classifier checkpoint not found: ${CLF_CKPT}, skipping."
        continue
    fi

    python scripts/eval_prob_perturbation.py \
        --arch         "${CLF_ARCH}" \
        --dataset      "${DATASET}" \
        --ckp_path     "${CLF_CKPT}" \
        --gmm_path     "${GMM_CKPT}" \
        --gmm_epsilon  "${GMM_EPS}" \
        --gmm_norm     "${GMM_NORM}" \
        --num_samples  32 \
        --save_csv     "${OUT_CSV}"

    echo "  -> saved: ${OUT_CSV}"
done

echo ""
echo "=========================================="
echo "  Done. Results in: ${RESULTS_DIR}"
echo "=========================================="
