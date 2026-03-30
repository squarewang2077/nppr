#!/bin/bash

# Usage: bash gmm_time_wall.sh [GPU_ID]
# GPU_ID defaults to 1 if not specified.
GPU_ID=1
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Resolve the project root (one level above this script) and add it to
# PYTHONPATH so that top-level packages (arch/, src/, utils/, configs/)
# are importable regardless of where the script is invoked from.
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ------------------------------------------------------------------
# Wall-time analysis for Mixture Model training.
# Mirrors run_mixture.sh: same datasets, models, K, checkpoint paths.
# Runs 5 epochs per combo and reports per-combo and grand-total time.
# ------------------------------------------------------------------

EPOCHS=1
K=7
STD_MODEL_DIR="${PROJECT_ROOT}/ckp/standard"
CKPT_DIR="${PROJECT_ROOT}/ckp/mixture_fitting4std_models"

DATASETS=("tinyimagenet")
MODELS=("resnet50" "wide_resnet50_2" "vgg16")

echo "=========================================="
echo "  Mixture Model Wall-time Analysis"
echo "  GPU: ${GPU_ID} | Epochs per run: ${EPOCHS} | K: ${K}"
echo "=========================================="

# Accumulate per-combo timing lines for the final summary
SUMMARY=()

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo ""
        echo "------------------------------------------"
        echo "  Dataset: ${dataset} | Model: ${model}"
        echo "------------------------------------------"

        case "${model}" in
            resnet18|resnet50)
                CLF_CKPT="${STD_MODEL_DIR}/resnet/${model}_${dataset}.pth"
                ;;
            wide_resnet50_2)
                CLF_CKPT="${STD_MODEL_DIR}/wrn/${model}_${dataset}.pth"
                ;;
            vgg16)
                CLF_CKPT="${STD_MODEL_DIR}/vgg/${model}_${dataset}.pth"
                ;;
            *)
                echo "ERROR: Unknown model type: ${model}, skipping."
                SUMMARY+=("  ${dataset} / ${model} : SKIPPED (unknown model)")
                continue
                ;;
        esac

        if [ ! -f "${CLF_CKPT}" ]; then
            echo "WARNING: Checkpoint not found: ${CLF_CKPT}, skipping."
            SUMMARY+=("  ${dataset} / ${model} : SKIPPED (checkpoint not found)")
            continue
        fi

        COMBO_START=$(date +%s%N)

        python scripts/train_mixture.py \
            --dataset "${dataset}" \
            --arch "${model}" \
            --clf_ckpt "${CLF_CKPT}" \
            --ckp_dir "${CKPT_DIR}" \
            --epochs ${EPOCHS} \
            --K ${K} \
            --component_types "gaussian:7" \
            --batch_size 64 

        COMBO_END=$(date +%s%N)
        COMBO_S=$(echo "scale=3; (${COMBO_END} - ${COMBO_START}) / 1000000000" | bc)
        AVG_S=$(echo "scale=3; ${COMBO_S} / ${EPOCHS}" | bc)
        SUMMARY+=("  ${dataset} / ${model} : total=${COMBO_S}s | avg/epoch=${AVG_S}s")
    done
done

echo ""
echo "=========================================="
echo "  Wall-time Summary (GPU ${GPU_ID})"
echo "=========================================="
for line in "${SUMMARY[@]}"; do
    echo "${line}"
done
echo "=========================================="
