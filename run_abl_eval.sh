#!/bin/bash
# run_abl_eval.sh
# Ablation evaluation of PR-trained ResNet18 on CIFAR-10.
# Mirrors the kappa × tau grid defined in run_ablation.sh and evaluates
# each checkpoint with standard, PGD, and PR attacks.
#
# Expected checkpoint layout (matches run_ablation.sh save structure):
#   ./ckp/ablation/<dataset>/kappa<kappa>_tau<tau>/<arch>_<dataset>.pth
#
# Outputs one CSV per run to:
#   ./results/ablation/<dataset>/kappa<kappa>_tau<tau>/<arch>_<dataset>.csv

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT="./dataset"
CKP_ROOT="./ckp/tmp"
RESULTS_ROOT="./results"

# ---------------------------------------------------------------------------
# Model / dataset
# ---------------------------------------------------------------------------
ARCH="resnet18"
DATASET="cifar10"
SEED=42
BATCH_SIZE=256

# ---------------------------------------------------------------------------
# Attack settings  (match the threat model used during training)
# ---------------------------------------------------------------------------
NORM="linf"
EPSILON=0.03137          # 8/255
ALPHA=0.00784            # 2/255
PGD_STEPS=20

# PR evaluation settings
NUM_SAMPLES=32
K=3
SIGMA_DIST_TYPE="geometric"

# ---------------------------------------------------------------------------
# Ablation grid  (must match run_ablation.sh)
# ---------------------------------------------------------------------------
TAU_VALUES=(1e0 1e-2 1e-4 1e-8)
KAPPA_VALUES=(1 0.2 0.02)

# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
for kappa in "${KAPPA_VALUES[@]}"; do
    for tau in "${TAU_VALUES[@]}"; do
        CKP_PATH="${CKP_ROOT}/${DATASET}/${ARCH}_${DATASET}.pth"
        CSV_DIR="${RESULTS_ROOT}/${ARCH}_${DATASET}"
        CSV_PATH="${CSV_DIR}/kappa${kappa}_tau${tau}.csv"

        echo "======================================================"
        echo "  arch=${ARCH}  dataset=${DATASET}"
        echo "  kappa=${kappa}  tau=${tau}"
        echo "  checkpoint=${CKP_PATH}"
        echo "  csv=${CSV_PATH}"
        echo "======================================================"

        if [[ ! -f "${CKP_PATH}" ]]; then
            echo "  [SKIP] checkpoint not found: ${CKP_PATH}"
            continue
        fi

        mkdir -p "${CSV_DIR}"

        python eval_classifier.py \
            --ckp_path        "${CKP_PATH}"       \
            --dataset         "${DATASET}"         \
            --arch            "${ARCH}"            \
            --data_root       "${DATA_ROOT}"       \
            --batch_size      "${BATCH_SIZE}"      \
            --norm            "${NORM}"            \
            --epsilon         "${EPSILON}"         \
            --alpha           "${ALPHA}"           \
            --pgd_steps       "${PGD_STEPS}"       \
            --num_samples     "${NUM_SAMPLES}"     \
            --K               "${K}"               \
            --sigma_dist_type "${SIGMA_DIST_TYPE}" \
            --kappa           "${kappa}"           \
            --tau             "${tau}"             \
            --seed            "${SEED}"            \
            --save_csv        "${CSV_PATH}"

    done
done

echo ""
echo "All ablation evaluation runs completed."
echo "Results written to: ${RESULTS_ROOT}/${DATASET}/"
