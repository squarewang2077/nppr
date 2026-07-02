#!/bin/bash
# evalution_gmm4pr.sh
# Drives scripts/evaluation_gmm4pr.py across the full grid of
#   archs   = {resnet18, resnet50, wide_resnet50_2, vgg16}
#   datasets= {cifar10, cifar100, tinyimagenet}
# evaluating each classifier under:
#   * Random PR    (Gaussian / Uniform / Laplace)
#   * PGD adversarial accuracy
#   * CW adversarial accuracy
#   * AutoAttack adversarial accuracy   (off by default — slow)
#   * GMM4PR probabilistic robustness
#
# For every (arch, dataset) pair the GMM checkpoint is the one trained on the
# SAME arch and dataset — i.e. the GMM's feature-extractor matches the
# classifier under evaluation. The arch-mismatch case is rejected up front so
# you cannot accidentally pair, say, a resnet18 classifier with a resnet50 GMM.
#
# Classifier checkpoints : ./ckp/nppr_eval/standard/{sub}/{arch}_{dataset}.pth
# GMM checkpoints        : ./ckp/nppr_eval/gmm_fitting/{sub}/{arch}_on_{dataset}/<GMM_NAME>
#   where sub = resnet | wrn | vgg (resolved from arch)
#
# Results: ./results/nppr_eval/gmm_std_models/<ckpt_basename>_eval.csv (one row per combo).

set -euo pipefail

# ---------------------------------------------------------------------------
# Project root / PYTHONPATH (scripts/evaluation_gmm4pr.py imports arch, utils, src)
# ---------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# GPU / global eval knobs
# ---------------------------------------------------------------------------
GPU_ID=0
export CUDA_VISIBLE_DEVICES=${GPU_ID}

BATCH_SIZE=128 
NUM_WORKERS=5
# List of random seeds. The experiment is repeated once per seed; the result
# CSV is additive (scripts/evaluation_gmm4pr.py appends one row per run), so
# all seed rows accumulate in <ckpt_basename>_eval.csv.
SEEDS=(42 43 44 45 56)

# Shared budget (used by random-PR, PGD, CW, AA)
NORM="linf"
EPSILON=0.06274              # 16/255
ALPHA=0.00784                # 2/255

# Random-PR settings
RANDOM_N=32
RANDOM_DISTS=("gaussian" "uniform" "laplace")

# PGD / CW / AA settings
PGD_STEPS=3
CW_STEPS=3
AA_VERSION="standard"

# GMM4PR settings (epsilon/norm default to the GMM's training budget)
GMM_N=32

# ---------------------------------------------------------------------------
# Which evaluation methods to run. Flip 0/1 to enable/disable.
# AA is disabled by default — it can take 30+ min per combo on TinyImageNet.
# ---------------------------------------------------------------------------
DO_RANDOM=1
DO_PGD=1
DO_CW=1
DO_AA=1
DO_GMM=1

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------
ARCHS=("vgg16")
# ARCHS=("resnet18" "resnet50" "wide_resnet50_2" "vgg16")
DATASETS=("tinyimagenet")
# DATASETS=("cifar10" "cifar100" "tinyimagenet")

# Map arch -> subdirectory used in both the classifier-ckpt and GMM trees.
arch_subdir() {
    case "$1" in
        resnet18|resnet50) echo "resnet" ;;
        wide_resnet50_2)   echo "wrn" ;;
        vgg16)             echo "vgg" ;;
        *) echo "UNKNOWN" ;;
    esac
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CKPT_ROOT="${PROJECT_ROOT}/ckp/nppr_eval/standard"
GMM_ROOT="${PROJECT_ROOT}/ckp/nppr_eval/gmm_fitting"
GMM_NAME="gmm_K7_cond(xy)_decoder(trainable_128)_linf(16)_reg(none).pt"
SAVE_DIR="${PROJECT_ROOT}/results/nppr_eval/gmm_std_models"

mkdir -p "${SAVE_DIR}"

# ---------------------------------------------------------------------------
# Tee all stdout/stderr to a timestamped log alongside the CSVs. The exec line
# replaces this shell's stdout/stderr with a process-substituted `tee`, so
# every subsequent echo and every python child's output is written to the log
# *and* still shown on the terminal.
# ---------------------------------------------------------------------------
LOG_FILE="${SAVE_DIR}/evalution_gmm4pr_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[log] writing terminal output to: ${LOG_FILE}"

# ---------------------------------------------------------------------------
# Build the eval-method flag array once (combo-independent)
# ---------------------------------------------------------------------------
EVAL_FLAGS=()
[ "${DO_RANDOM}" -eq 1 ] && EVAL_FLAGS+=(--eval_random)
[ "${DO_PGD}"    -eq 1 ] && EVAL_FLAGS+=(--eval_pgd)
[ "${DO_CW}"     -eq 1 ] && EVAL_FLAGS+=(--eval_cw)
[ "${DO_AA}"     -eq 1 ] && EVAL_FLAGS+=(--eval_aa)

if [ ${#EVAL_FLAGS[@]} -eq 0 ] && [ "${DO_GMM}" -ne 1 ]; then
    echo "ERROR: no evaluation methods enabled. Set at least one of "\
         "DO_RANDOM/DO_PGD/DO_CW/DO_AA/DO_GMM to 1." >&2
    exit 1
fi

echo "============================================================"
echo "  evaluation_gmm4pr — sweep over (arch, dataset)"
echo "  GPU         : ${GPU_ID}"
echo "  Archs       : ${ARCHS[*]}"
echo "  Datasets    : ${DATASETS[*]}"
echo "  Seeds       : ${SEEDS[*]}"
echo "  Eval methods:"
echo "    random=${DO_RANDOM}  pgd=${DO_PGD}  cw=${DO_CW}  aa=${DO_AA}  gmm=${DO_GMM}"
echo "  GMM file    : ${GMM_NAME}"
echo "  Save dir    : ${SAVE_DIR}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
for DATASET in "${DATASETS[@]}"; do
    for ARCH in "${ARCHS[@]}"; do
        SUB="$(arch_subdir "${ARCH}")"
        CKPT="${CKPT_ROOT}/${SUB}/${ARCH}_${DATASET}.pth"
        GMM_PATH="${GMM_ROOT}/${SUB}/${ARCH}_on_${DATASET}/${GMM_NAME}"

        echo ""
        echo "------------------------------------------------------------"
        echo "  arch=${ARCH}  dataset=${DATASET}"
        echo "  ckpt: ${CKPT}"
        echo "  gmm : ${GMM_PATH}"
        echo "------------------------------------------------------------"

        if [ ! -f "${CKPT}" ]; then
            echo "  WARNING: classifier ckpt not found — skipping combo." >&2
            continue
        fi

        # Per-combo GMM args. If --eval_gmm is enabled but the matching GMM
        # is missing, we keep the rest of the eval methods running and just
        # warn.
        GMM_ARGS=()
        if [ "${DO_GMM}" -eq 1 ]; then
            if [ -f "${GMM_PATH}" ]; then
                GMM_ARGS+=(--eval_gmm --gmm_path "${GMM_PATH}" --gmm_num_samples "${GMM_N}")
            else
                echo "  WARNING: GMM not found for this combo — skipping --eval_gmm." >&2
            fi
        fi

        # If neither GMM_ARGS nor any other EVAL_FLAGS exist, there's nothing
        # to run for this combo.
        if [ ${#EVAL_FLAGS[@]} -eq 0 ] && [ ${#GMM_ARGS[@]} -eq 0 ]; then
            echo "  Nothing to evaluate for this combo, skipping." >&2
            continue
        fi

        for SEED in "${SEEDS[@]}"; do
            echo ""
            echo "  >>> seed=${SEED}"
            python scripts/evaluation_gmm4pr.py \
                --arch                "${ARCH}"             \
                --dataset             "${DATASET}"          \
                --ckp_path            "${CKPT}"             \
                --batch_size          "${BATCH_SIZE}"       \
                --num_workers         "${NUM_WORKERS}"      \
                --seed                "${SEED}"             \
                --norm                "${NORM}"             \
                --epsilon             "${EPSILON}"          \
                --alpha               "${ALPHA}"            \
                --random_dist         "${RANDOM_DISTS[@]}"  \
                --num_samples_random  "${RANDOM_N}"         \
                --pgd_steps           "${PGD_STEPS}"        \
                --cw_steps            "${CW_STEPS}"         \
                --aa_version          "${AA_VERSION}"       \
                "${EVAL_FLAGS[@]}"                          \
                "${GMM_ARGS[@]}"                            \
                --save_dir            "${SAVE_DIR}"
        done
    done
done

echo ""
echo "============================================================"
echo "  All evaluations completed."
echo "  Results: ${SAVE_DIR}"
echo "  Log    : ${LOG_FILE}"
echo "============================================================"
