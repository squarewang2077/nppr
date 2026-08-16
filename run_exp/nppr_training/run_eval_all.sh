#!/bin/bash
# run_eval_all.sh
# Evaluate one or more trained classifier checkpoints with scripts/eval_all.py.
# Reports clean test accuracy, PGD-10 robust accuracy, random-PR for
# Gaussian / Uniform / Laplace noise, and accuracy under all 4 corruptions
# (salt_pepper, motion_blur, brightness, jpeg) at the chosen severity levels.
#
# arch and dataset are auto-detected from each checkpoint, so the same
# script works for any model trained by run_pr*.sh / run_adv*.sh.
#
# Outputs:
#   stdout: a formatted summary block per checkpoint
#   csv   : <SAVE_CSV_DIR>/<ckpt_basename>_eval.csv (one row per ckpt)

set -euo pipefail

# Resolve the project root (two levels up from this script) and add it to
# PYTHONPATH so that top-level packages (arch/, src/, utils/, configs/) are
# importable. cd ensures ./dataset and scripts/... resolve correctly no
# matter where this script is invoked from.
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# Configurable hyper-parameters
# ---------------------------------------------------------------------------
GPU_ID=0                     # GPU device ID to use
export CUDA_VISIBLE_DEVICES=${GPU_ID}

DATA_ROOT="./dataset"
BATCH_SIZE=256
NUM_WORKERS=4
SEED=42

# Perturbation budget (used by both PGD and random-PR)
NORM="linf"
EPSILON=0.03137              # 8/255

# PGD eval
PGD_STEPS=10

# Random-PR eval
RANDOM_N=10
RANDOM_DISTS=("gaussian" "uniform" "laplace")

# Corruption eval
CORRUPTION_NAMES=("salt_pepper" "motion_blur" "brightness" "jpeg")
CORRUPTION_SEVERITIES=(1 3 5)

# Where to write the per-ckpt CSV summary. Set to "" to disable CSV output.
SAVE_CSV_DIR="./results/nppr_training/eval_all"

# ---------------------------------------------------------------------------
# Checkpoints to evaluate
# ---------------------------------------------------------------------------
# Either set CKPT_LIST explicitly (one or more .pth files), or pass
# checkpoint paths as arguments to this script. CLI args take precedence.
#
# Examples:
#   ./run_exp/nppr_training/run_eval_all.sh path/to/a.pth path/to/b.pth
#   (or edit CKPT_LIST below and run with no args)

CKPT_LIST=(
    # "./ckp/nppr_training/pr_training/cifar10/resnet18/loc_entropy/loc_ent_eps10_L100_G0.05_N1_Aug.pth"
    # "./ckp/nppr_training/pr_training/cifar10/resnet50/loc_entropy/loc_ent_eps10_L100_G0.05_N1_Aug.pth"
    # "./ckp/nppr_training/pr_training/cifar10/wide_resnet50_2/loc_entropy/loc_ent_eps10_L100_G0.05_N1_Aug.pth"
    # "./ckp/nppr_training/pr_training/cifar100/resnet18/loc_entropy/loc_ent_eps10_L100_G0.05_N1_Aug.pth"
    # "./ckp/nppr_training/pr_training/cifar100/resnet50/loc_entropy/loc_ent_eps10_L100_G0.05_N1_Aug.pth"
    # "./ckp/nppr_training/pr_training/cifar100/wide_resnet50_2/loc_entropy/loc_ent_eps10_L100_G0.05_N1_Aug.pth"

    # "./ckp/nppr_training/adv_training/cifar10/resnet18/pgd10_Aug/resnet18_cifar10_adv_pgd_Aug.pth"
    # "./ckp/nppr_training/adv_training/cifar10/resnet50/pgd10_Aug/resnet50_cifar10_adv_pgd_Aug.pth"
    # "./ckp/nppr_training/adv_training/cifar10/wide_resnet50_2/pgd10_Aug/wide_resnet50_2_cifar10_adv_pgd_Aug.pth"
    # "./ckp/nppr_training/adv_training/cifar100/resnet18/pgd10_Aug/resnet18_cifar100_adv_pgd_Aug.pth"
    # "./ckp/nppr_training/adv_training/cifar100/resnet50/pgd10_Aug/resnet50_cifar100_adv_pgd_Aug.pth"
    # "./ckp/nppr_training/adv_training/cifar100/wide_resnet50_2/pgd10_Aug/wide_resnet50_2_cifar100_adv_pgd_Aug.pth"

    # "./ckp/nppr_eval/adv_pgd/cifar10/resnet18_cifar10.pth"
    # "./ckp/nppr_eval/adv_pgd/cifar10/resnet50_cifar10.pth"
    # "./ckp/nppr_eval/adv_pgd/cifar10/wide_resnet50_2_cifar10.pth"
    # "./ckp/nppr_eval/adv_pgd/cifar100/resnet18_cifar100_adv_pgd.pth"
    # "./ckp/nppr_eval/adv_pgd/cifar100/resnet50_cifar100_adv_pgd.pth"
    # "./ckp/nppr_eval/adv_pgd/cifar100/wide_resnet50_2_cifar100_adv_pgd.pth"

    # "./ckp/nppr_eval/standard/resnet/resnet18_cifar10.pth"
    # "./ckp/nppr_eval/standard/resnet/resnet18_cifar100.pth"
    # "./ckp/nppr_eval/standard/resnet/resnet50_cifar10.pth"
    # "./ckp/nppr_eval/standard/resnet/resnet50_cifar100.pth"
    # "./ckp/nppr_eval/standard/wrn/wide_resnet50_2_cifar10.pth"
    # "./ckp/nppr_eval/standard/wrn/wide_resnet50_2_cifar100.pth"

    # "./ckp/nppr_training/adv_training/cifar10/resnet18/trades_Aug/resnet18_cifar10_trades_Aug.pth"
    "./ckp/nppr_training/adv_training/cifar100/resnet18/trades_Aug/resnet18_cifar100_trades_Aug.pth"



    # add more checkpoint paths here
)

if [[ $# -gt 0 ]]; then
    CKPT_LIST=("$@")
fi

if [[ ${#CKPT_LIST[@]} -eq 0 ]]; then
    echo "ERROR: no checkpoints to evaluate. Edit CKPT_LIST in this script "\
         "or pass paths as arguments." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Run eval
# ---------------------------------------------------------------------------
echo "======================================================"
echo "  Using GPU: ${GPU_ID}"
echo "  Checkpoints: ${#CKPT_LIST[@]}"
echo "  Random dists: ${RANDOM_DISTS[*]}"
echo "  Corruptions:  ${CORRUPTION_NAMES[*]}"
echo "  Severities:   ${CORRUPTION_SEVERITIES[*]}"
[[ -n "${SAVE_CSV_DIR}" ]] && echo "  CSV output:   ${SAVE_CSV_DIR}"
echo "======================================================"
echo ""

if [[ -n "${SAVE_CSV_DIR}" ]]; then
    mkdir -p "${SAVE_CSV_DIR}"
fi

for CKPT in "${CKPT_LIST[@]}"; do
    if [[ ! -f "${CKPT}" ]]; then
        echo "WARNING: skipping missing checkpoint: ${CKPT}" >&2
        continue
    fi

    echo "------------------------------------------------------"
    echo "  ckpt: ${CKPT}"
    echo "------------------------------------------------------"

    CSV_ARG=()
    if [[ -n "${SAVE_CSV_DIR}" ]]; then
        BASE="$(basename "${CKPT}" .pth)"
        CSV_ARG=(--save_csv "${SAVE_CSV_DIR}/${BASE}_eval.csv")
    fi

    python scripts/eval_all.py \
        --ckpt           "${CKPT}"            \
        --data_root      "${DATA_ROOT}"       \
        --batch_size     "${BATCH_SIZE}"      \
        --num_workers    "${NUM_WORKERS}"     \
        --seed           "${SEED}"            \
        --epsilon        "${EPSILON}"         \
        --norm           "${NORM}"            \
        --pgd_steps      "${PGD_STEPS}"       \
        --random_n       "${RANDOM_N}"        \
        --random_dist    "${RANDOM_DISTS[@]}" \
        --corruption_names "${CORRUPTION_NAMES[@]}"   \
        --corruption_severities "${CORRUPTION_SEVERITIES[@]}" \
        "${CSV_ARG[@]}"
done

echo ""
echo "All evaluations completed."
