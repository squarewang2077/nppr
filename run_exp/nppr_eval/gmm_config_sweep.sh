#!/bin/bash
# evalution_gmm4pr_configs.sh
# Like evalution_gmm4pr.sh, but instead of sweeping (arch, dataset) it FIXES
# the classifier to ResNet18 on CIFAR-10 and sweeps the GMM4PR configuration:
#
#   * number of modes        K       ∈ {3, 7, 12}
#   * dependency structure   cond    ∈ {none, x, y, xy}   (4 structures)
#   * upsampler / decoder    decoder ∈ {none, nontrainable, trainable_128}
#
# => 3 x 4 x 3 = 36 GMM configurations. All GMMs are pre-trained and stored in
#    ckp/nppr_eval/gmm_fitting/resnet/resnet18_on_cifar10/ with filename
#      gmm_K{K}_cond({cond})_decoder({decoder})_linf(16)_reg(none).pt
#
# Evaluated metrics (all opt-in via DO_* flags below — defaults: all on):
#   * clean accuracy (always)
#   * Random-PR (Gaussian / Uniform / Laplace)
#   * PGD / CW / AutoAttack adversarial accuracy
#   * GMM4PR probabilistic robustness
#
# In addition to sweeping the GMM configuration, the perturbation budget is
# also swept: EPSILON_LIST = (4/255, 8/255, 16/255). The same epsilon drives
# the shared budget (--epsilon, used by random-PR / PGD / CW / AA) and the
# GMM eval (--gmm_epsilon, which overrides the GMM's training radius — the
# GMMs themselves are all trained at linf(16)). PGD/CW step size alpha is
# tied to epsilon/4 via the parallel ALPHA_LIST.
#
# Note: random-PR / PGD / CW / AA depend only on (classifier, eps, seed), not
# on the GMM config — so within a given epsilon band their columns will be
# identical across all (K, cond, decoder) rows. We accept that redundancy so
# every row in the CSV is self-contained.
#
# Every (K, cond, decoder, eps, seed) appends one row to a single additive
# CSV:
#   ./results/nppr_eval/gmm_config_sweep(resnet18_on_cifar10)/resnet18_cifar10_eval.csv
# The gmm_path + epsilon columns identify K / cond / decoder / eps for each row.

set -euo pipefail

# ---------------------------------------------------------------------------
# Project root / PYTHONPATH
# ---------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# GPU / global eval knobs
# ---------------------------------------------------------------------------
GPU_ID=1
export CUDA_VISIBLE_DEVICES=${GPU_ID}

BATCH_SIZE=256
NUM_WORKERS=5
# List of random seeds. The experiment is repeated once per seed; the result
# CSV is additive (scripts/evaluation_gmm4pr.py appends one row per run), so
# all seed rows accumulate in the per-classifier CSV.
SEEDS=(42)

# GMM4PR draws per input (epsilon/norm default to each GMM's training budget
# unless overridden via --gmm_epsilon / --gmm_norm — both are overridden below
# so the GMM is evaluated at the swept epsilon rather than its trained 16/255).
GMM_N=32

# GMM diagnostics (check_mode_collapse): π-distribution stats per config.
# Set DO_GMM_DIAG=0 to skip; GMM_DIAG_BATCHES controls how many loader
# batches check_mode_collapse samples.
DO_GMM_DIAG=1
GMM_DIAG_BATCHES=10

# ---------------------------------------------------------------------------
# Perturbation-budget sweep (shared by random-PR / PGD / CW / AA / GMM)
#   EPSILON_LIST[i] : eps_i  (linf radius, used as --epsilon and --gmm_epsilon)
#   ALPHA_LIST[i]   : alpha_i = eps_i / 4   (PGD / CW step size)
# ---------------------------------------------------------------------------
NORM="linf"
EPSILON_LIST=(0.01569 0.03137 0.06274)   # 4/255, 8/255, 16/255
ALPHA_LIST=(  0.00392 0.00784 0.01569)   # 1/255, 2/255, 4/255  (= eps/4)

# Random-PR settings
RANDOM_N=32
RANDOM_DISTS=("gaussian" "uniform" "laplace")

# PGD / CW / AA settings
PGD_STEPS=3
CW_STEPS=3
AA_VERSION="standard"

# ---------------------------------------------------------------------------
# Which evaluation methods to run. Flip 0/1 to enable/disable.
# AA can be slow — set DO_AA=0 to skip it.
# ---------------------------------------------------------------------------
DO_RANDOM=1
DO_PGD=1
DO_CW=0
DO_AA=0
DO_GMM=1

# ---------------------------------------------------------------------------
# Fixed classifier: ResNet18 on CIFAR-10
# ---------------------------------------------------------------------------
ARCH="resnet18"
DATASET="cifar10"
CKPT="${PROJECT_ROOT}/ckp/nppr_eval/standard/resnet/${ARCH}_${DATASET}.pth"

# ---------------------------------------------------------------------------
# GMM configuration sweep
# ---------------------------------------------------------------------------
GMM_DIR="${PROJECT_ROOT}/ckp/nppr_eval/gmm_fitting/resnet/${ARCH}_on_${DATASET}"
GMM_BUDGET="linf(16)_reg(none)"   # fixed budget/reg suffix for every config

K_LIST=(3 7 12)                                 # number of modes
COND_LIST=("none" "x" "y" "xy")                 # 4 dependency structures
DEC_LIST=("none" "nontrainable" "trainable_128")  # upsampler variants

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
SAVE_DIR="${PROJECT_ROOT}/results/nppr_eval/gmm_config_sweep(${ARCH}_on_${DATASET})"
mkdir -p "${SAVE_DIR}"

# Tee all stdout/stderr to a timestamped log alongside the CSV.
LOG_FILE="${SAVE_DIR}/evalution_gmm4pr_configs_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[log] writing terminal output to: ${LOG_FILE}"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Build the eval-method flag array once (combo/eps-independent)
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

if [ "${#EPSILON_LIST[@]}" -ne "${#ALPHA_LIST[@]}" ]; then
    echo "ERROR: EPSILON_LIST and ALPHA_LIST must have the same length." >&2
    exit 1
fi

echo "============================================================"
echo "  evaluation_gmm4pr — GMM-config sweep (fixed classifier)"
echo "  GPU         : ${GPU_ID}"
echo "  Classifier  : ${ARCH} / ${DATASET}"
echo "  ckpt        : ${CKPT}"
echo "  K modes     : ${K_LIST[*]}"
echo "  cond struct : ${COND_LIST[*]}"
echo "  decoders    : ${DEC_LIST[*]}"
echo "  Epsilons    : ${EPSILON_LIST[*]}   (norm=${NORM})"
echo "  Alphas      : ${ALPHA_LIST[*]}     (= eps/4)"
echo "  Seeds       : ${SEEDS[*]}"
echo "  Eval methods: random=${DO_RANDOM}  pgd=${DO_PGD}  cw=${DO_CW}  aa=${DO_AA}  gmm=${DO_GMM}"
echo "  GMM diag    : ${DO_GMM_DIAG}  (num_batches=${GMM_DIAG_BATCHES})"
echo "  Save dir    : ${SAVE_DIR}"
echo "============================================================"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: classifier ckpt not found: ${CKPT}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Sweep: K x cond x decoder x epsilon x seed
# ---------------------------------------------------------------------------
N_DONE=0
N_SKIP=0
for K in "${K_LIST[@]}"; do
    for COND in "${COND_LIST[@]}"; do
        for DEC in "${DEC_LIST[@]}"; do
            GMM_PATH="${GMM_DIR}/gmm_K${K}_cond(${COND})_decoder(${DEC})_${GMM_BUDGET}.pt"

            echo ""
            echo "------------------------------------------------------------"
            echo "  K=${K}  cond=${COND}  decoder=${DEC}"
            echo "  gmm : ${GMM_PATH}"
            echo "------------------------------------------------------------"

            if [ ! -f "${GMM_PATH}" ]; then
                echo "  WARNING: GMM not found — skipping this config." >&2
                N_SKIP=$((N_SKIP + 1))
                continue
            fi

            # Per-combo GMM args (epsilon/norm overrides are added per-eps below).
            GMM_BASE_ARGS=()
            if [ "${DO_GMM}" -eq 1 ]; then
                GMM_BASE_ARGS+=(--eval_gmm
                                --gmm_path "${GMM_PATH}"
                                --gmm_num_samples "${GMM_N}")
            fi

            DIAG_ARGS=()
            if [ "${DO_GMM_DIAG}" -eq 1 ]; then
                DIAG_ARGS+=(--gmm_diagnostics
                            --gmm_diag_num_batches "${GMM_DIAG_BATCHES}")
            fi

            for i in "${!EPSILON_LIST[@]}"; do
                EPSILON="${EPSILON_LIST[$i]}"
                ALPHA="${ALPHA_LIST[$i]}"

                # Add the per-epsilon GMM override to a fresh copy of GMM_ARGS.
                GMM_ARGS=("${GMM_BASE_ARGS[@]}")
                if [ "${DO_GMM}" -eq 1 ]; then
                    GMM_ARGS+=(--gmm_epsilon "${EPSILON}" --gmm_norm "${NORM}")
                fi

                echo ""
                echo "  --- epsilon=${EPSILON}  alpha=${ALPHA}  norm=${NORM} ---"

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
                        "${DIAG_ARGS[@]}"                           \
                        --save_dir            "${SAVE_DIR}"
                    N_DONE=$((N_DONE + 1))
                done
            done
        done
    done
done

echo ""
echo "============================================================"
echo "  GMM-config sweep completed."
echo "  Runs done: ${N_DONE}   configs skipped (missing GMM): ${N_SKIP}"
echo "  Results: ${SAVE_DIR}/${ARCH}_${DATASET}_eval.csv"
echo "  Log    : ${LOG_FILE}"
echo "============================================================"
