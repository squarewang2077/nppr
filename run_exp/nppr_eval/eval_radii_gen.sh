#!/bin/bash

# Usage: bash eval_radii_gen.sh [GPU_ID]
# GPU_ID defaults to 1 if not specified.
GPU_ID=0
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Resolve the project root and add it to PYTHONPATH.
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ------------------------------------------------------------------
# Radii generalisation evaluation for resnet18 on cifar100.
#
# GMM : K=7, cond=xy, trainable_128 decoder, trained at linf radii
#        4, 8, 16  (ckp/gmm_fitting/resnet/resnet18_on_cifar100/)
#
# For each loaded GMM radius, evaluate PR at all four eval radii
# (4, 8, 16, 32) → 12 combinations total.
#
# Epsilon mapping (integer label → float):
#   4  → 4/255  ≈ 0.01569
#   8  → 8/255  ≈ 0.03137
#  16  → 16/255 ≈ 0.06275
#  32  → 32/255 ≈ 0.12549
#
# Results: ./results/gmm_radii_gen/
#   eval_prob_pert_vgg16_cifar10_gmmR{gmm_r}_evalR{eval_r}.csv
# ------------------------------------------------------------------

ARCH="vgg16"
DATASET="tinyimagenet"
CLF_CKPT="${PROJECT_ROOT}/ckp/standard/vgg/${ARCH}_${DATASET}.pth"
GMM_BASE="${PROJECT_ROOT}/ckp/gmm_fitting/vgg/${ARCH}_on_${DATASET}"
RESULTS_DIR="${PROJECT_ROOT}/results/gmm_radii_gen"

# Integer radius labels and their float equivalents (paired by index)
RADII_INT=(4       8       16      32     )
RADII_FLT=(0.01569 0.03137 0.06275 0.12549)

mkdir -p "${RESULTS_DIR}"

echo "=========================================="
echo "  GMM Radii Generalisation Evaluation"
echo "  Model  : ${ARCH} on ${DATASET}"
echo "  GPU    : ${GPU_ID}"
echo "  Output : ${RESULTS_DIR}"
echo "=========================================="

if [ ! -f "${CLF_CKPT}" ]; then
    echo "ERROR: Classifier checkpoint not found: ${CLF_CKPT}"
    exit 1
fi

for i in "${!RADII_INT[@]}"; do
    GMM_R="${RADII_INT[$i]}"
    GMM_CKPT="${GMM_BASE}/gmm_K7_cond(xy)_decoder(trainable_128)_linf(${GMM_R})_reg(none).pt"

    echo ""
    echo "------------------------------------------"
    echo "  Loaded GMM radius: linf(${GMM_R})"
    echo "------------------------------------------"

    if [ ! -f "${GMM_CKPT}" ]; then
        echo "WARNING: GMM checkpoint not found: ${GMM_CKPT}, skipping."
        continue
    fi

    for j in "${!RADII_INT[@]}"; do
        EVAL_R="${RADII_INT[$j]}"
        EVAL_EPS="${RADII_FLT[$j]}"
        OUT_CSV="${RESULTS_DIR}/eval_prob_pert_${ARCH}_${DATASET}_gmmR${GMM_R}_evalR${EVAL_R}.csv"

        echo "  eval radius: linf(${EVAL_R}) = ${EVAL_EPS} ..."
        python scripts/eval_prob_perturbation.py \
            --arch "${ARCH}" \
            --dataset "${DATASET}" \
            --ckp_path "${CLF_CKPT}" \
            --gmm_path "${GMM_CKPT}" \
            --gmm_epsilon "${EVAL_EPS}" \
            --gmm_norm linf \
            --num_samples 32 \
            --save_csv "${OUT_CSV}"

        echo "  -> saved: ${OUT_CSV}"
    done
done

echo ""
echo "=========================================="
echo "  Done. Results in: ${RESULTS_DIR}"
echo "=========================================="
