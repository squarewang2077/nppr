#!/bin/bash
# run_resnet18_on_cifar10.sh
# Ablation study for level-set position-geometry training, ResNet-18 / CIFAR-10.
#
# See run_resnet18_on_cifar10.md for what each axis is testing, what would
# falsify it, and the confirmed-effective configuration once it is known.
#
# This is one-factor-at-a-time, not a grid: every run differs from the
# reference configuration in exactly one thing, so any difference in the
# result is attributable. A full grid over the same axes would be 120 runs at
# ~13 h each; the point here is to answer five questions, not to fill a table.
#
# Three stages, selected by the first argument:
#
#   screen    cheap settings, all 14 ablation runs. Finds which configurations
#             are worth the expensive confirmation.
#   baseline  standard / PGD-AT / FGSM-RS at the same epochs and optimiser, so
#             "is our method effective" has something to be effective against.
#   confirm   full settings, only the tags listed in CONFIRM_TAGS below.
#
# Two GPUs: shard rather than queue. Run the same command twice with different
# SHARD values and each process takes every NUM_SHARDS-th run.
#
#   GPU_ID=0 SHARD=0 NUM_SHARDS=2 bash run_exp/pos_geo_training/run_resnet18_on_cifar10.sh screen
#   GPU_ID=1 SHARD=1 NUM_SHARDS=2 bash run_exp/pos_geo_training/run_resnet18_on_cifar10.sh screen
#
# Outputs (no checkpoints — this is an ablation, the numbers are the product):
#   results/pos_geo_training/resnet18_cifar10/<stage>/<tag>.log
#   results/pos_geo_training/resnet18_cifar10/<stage>/<tag>_training_info.csv
#
# Then:
#   python scripts/pr_training/summarize_ablation.py \
#       results/pos_geo_training/resnet18_cifar10/screen \
#       --out run_exp/pos_geo_training/run_resnet18_on_cifar10.md

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

STAGE="${1:-}"
if [[ ! "${STAGE}" =~ ^(screen|baseline|confirm)$ ]]; then
    echo "usage: $0 {screen|baseline|confirm}" >&2
    exit 2
fi

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES=${GPU_ID}
SHARD="${SHARD:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"

# ---------------------------------------------------------------------------
# Held fixed across every run — this is what makes the comparison a comparison
# ---------------------------------------------------------------------------
DATA_ROOT="./dataset"
ARCH="resnet18"
DATASET="cifar10"

BATCH_SIZE=1024
LR=0.01
WEIGHT_DECAY=5e-4
SEED=42
NORM="linf"                  # epsilon left unset -> 8/255 derived for linf

# Evaluation, identical on both sides of the comparison: clean / PGD-10 /
# FGSM / Laplace-PR, on the train subset and the test set, every 5 epochs.
EVAL_EVERY=5
PGD_STEPS=10
PR_SAMPLES=32

# ---------------------------------------------------------------------------
# Reference configuration for the level-set runs
#
# Every choice here has a measurement behind it; the numbers are in the .md.
# In short:
#   solver=newton        scale free, no psi_alpha to retune as the margin
#                        scale moves. 0.22 -> 0.62 valid rate on a trained
#                        model at ten steps, same cost per step.
#   anchor_lambda auto   resolves to 0 for newton. A Newton step shrinks to
#                        nothing near the level set while a fixed anchor does
#                        not, so any anchor eventually drags the solver back
#                        off it: valid 0.228 vs 0.898, |m-t| 0.20 vs 0.003.
#   step_size auto       1.0 for newton, 0.2 for energy. Not interchangeable.
#   t_mode=reachable     t as a fraction of the margin floor the ball can
#                        actually reach, per sample. Training raises that
#                        floor (-1.54 -> -0.20 in six epochs), so a fixed t
#                        walks out of the ball: 0.96 -> 0.15 valid, against
#                        0.88-1.00 held by t_frac=0.5.
#   geometry_mode=coarea ||grad m||_2, not the dual norm. The coarea formula
#                        is a Euclidean identity and its surface-measure
#                        factor is the L2 norm whatever the threat model is;
#                        the L-inf geometry already enters through the ball
#                        and through delta_norm. `dual` answers a different
#                        question and is ablated below rather than assumed.
# ---------------------------------------------------------------------------
REF_FLAGS=(
    --solver         newton
    --weight_mode    uniform
    --geometry_mode  coarea
    --t_mode         reachable
    --t_frac         0.5
    --tol_mode       absolute
)
MIN_VALID_RATE=0.3           # warn below this — the core failure mode

# ---------------------------------------------------------------------------
# Stage settings
# ---------------------------------------------------------------------------
# Solver cost is ~proportional to num_starts * num_steps. Measured: 48 s/epoch
# at 4 x 10 on this box, so screening (4 x 20) is ~96 s/epoch and confirmation
# (8 x 50) is ~480 s/epoch. Override any of these from the environment.
case "${STAGE}" in
    screen)
        EPOCHS="${EPOCHS:-30}"
        NUM_STARTS="${NUM_STARTS:-4}"
        NUM_STEPS="${NUM_STEPS:-20}"
        ;;
    confirm)
        EPOCHS="${EPOCHS:-100}"
        NUM_STARTS="${NUM_STARTS:-8}"
        NUM_STEPS="${NUM_STEPS:-50}"
        ;;
    baseline)
        EPOCHS="${EPOCHS:-30}"       # match whichever stage you are comparing to
        NUM_STARTS=0
        NUM_STEPS=0
        ;;
esac

# Which tags the confirm stage re-runs at full settings. Left empty on purpose:
# pick them from the screen summary rather than guessing.
CONFIRM_TAGS=()

# Baselines are the reference models a later AutoAttack comparison would need,
# and there are only three of them. Set to 1 to keep their checkpoints.
KEEP_BASELINE_CKPT=0

# ---------------------------------------------------------------------------
# The ablation runs: "tag|flags overriding the reference configuration"
#
# num_starts is a stage setting rather than a per-run flag, so the two runs
# that ablate it pass --num_starts explicitly and override it.
# ---------------------------------------------------------------------------
RUNS=(
    # reference
    "ref|"

    # Does it matter WHERE on the level set the signal comes from?
    # This is the question the whole method exists to answer.
    "w_sharp|--weight_mode sharp"
    "w_flat|--weight_mode flat"
    "w_min_norm|--weight_mode min_norm"
    "w_max_norm|--weight_mode max_norm"

    # Does newton's convergence advantage become robustness, or does it only
    # make the valid rate look good?
    "solver_energy|--solver energy"

    # Training strength, and whether reachable really beats a fixed level.
    "tfrac025|--t_frac 0.25"
    "tfrac075|--t_frac 0.75"
    "tfix05|--t_mode fixed --t -0.5"
    "tfix10|--t_mode fixed --t -1.0"

    # Do multiple starts do anything at all? With num_starts=1 every
    # weight_mode collapses to the same thing, so if this ties with ref the
    # weighting axis above is answering a question that does not exist.
    "starts1|--num_starts 1"
    "starts8|--num_starts 8"

    # L2 vs dual-norm scoring. geometry_mode is only read by sharp/flat, so
    # these pair against w_sharp / w_flat; running it against ref (uniform)
    # would change nothing and waste a run.
    "w_sharp_dual|--weight_mode sharp --geometry_mode dual"
    "w_flat_dual|--weight_mode flat --geometry_mode dual"
)

BASELINES=("standard" "adv_pgd" "adv_fgsm")

# RESULTS_ROOT is overridable so a dry run (EPOCHS=1 NUM_STEPS=3 ...) can be
# pointed somewhere disposable instead of landing in the real results tree.
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/pos_geo_training}"
OUT_DIR="${RESULTS_ROOT}/${ARCH}_${DATASET}/${STAGE}"
CKP_DIR="${PROJECT_ROOT}/ckp/pos_geo_training/${ARCH}_${DATASET}/baseline"

# ---------------------------------------------------------------------------
# Assemble the job list for this stage
#
# Before mkdir, so a stage that bails out (confirm with no CONFIRM_TAGS) does
# not leave an empty directory behind for the summariser to trip over.
# ---------------------------------------------------------------------------
JOBS=()
case "${STAGE}" in
    screen)   JOBS=("${RUNS[@]}") ;;
    baseline) for b in "${BASELINES[@]}"; do JOBS+=("bl_${b}|${b}"); done ;;
    confirm)
        if [ ${#CONFIRM_TAGS[@]} -eq 0 ]; then
            echo "CONFIRM_TAGS is empty. Run the screen stage first, then:" >&2
            echo "  python scripts/pr_training/summarize_ablation.py \\" >&2
            echo "      results/pos_geo_training/${ARCH}_${DATASET}/screen" >&2
            echo "and put the winners in CONFIRM_TAGS near the top of this file." >&2
            exit 3
        fi
        for want in "${CONFIRM_TAGS[@]}"; do
            found=0
            for entry in "${RUNS[@]}"; do
                [ "${entry%%|*}" = "${want}" ] && { JOBS+=("${entry}"); found=1; break; }
            done
            [ ${found} -eq 1 ] || { echo "unknown tag in CONFIRM_TAGS: ${want}" >&2; exit 3; }
        done
        ;;
esac

mkdir -p "${OUT_DIR}"

TOTAL=${#JOBS[@]}
MINE=0
for (( i = 0; i < TOTAL; i++ )); do
    [ $(( i % NUM_SHARDS )) -eq ${SHARD} ] && MINE=$(( MINE + 1 ))
done

echo "======================================================"
echo "  stage=${STAGE}  ${ARCH}/${DATASET}  epochs=${EPOCHS}"
if [ "${STAGE}" != "baseline" ]; then
    echo "  num_starts=${NUM_STARTS} num_steps=${NUM_STEPS}"
    # ~48 s/epoch measured at num_starts*num_steps = 40; cost is roughly linear.
    PER_RUN_MIN=$(( EPOCHS * NUM_STARTS * NUM_STEPS * 48 / 40 / 60 ))
    echo "  estimated ~${PER_RUN_MIN} min/run, ${MINE} runs on this shard"
    echo "  (rough: linear in num_starts*num_steps from a 48 s/epoch measurement)"
fi
echo "  shard ${SHARD}/${NUM_SHARDS}, GPU ${GPU_ID}, ${MINE} of ${TOTAL} runs"
echo "  -> ${OUT_DIR}"
echo "======================================================"
echo ""

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
RUN=0
for (( i = 0; i < TOTAL; i++ )); do
    [ $(( i % NUM_SHARDS )) -eq ${SHARD} ] || continue
    RUN=$(( RUN + 1 ))

    entry="${JOBS[$i]}"
    TAG="${entry%%|*}"
    REST="${entry#*|}"

    echo "------------------------------------------------------"
    echo "  [${RUN}/${MINE}] ${TAG}  ${REST:-(reference config)}"
    echo "------------------------------------------------------"

    if [ "${STAGE}" = "baseline" ]; then
        # REST is the training_type. --alpha and --num_steps are read by
        # adv_pgd only; standard ignores both and adv_fgsm pins its own step.
        BL_CKPT_FLAGS=(--no_save_ckpt)
        [ "${KEEP_BASELINE_CKPT}" = "1" ] && { mkdir -p "${CKP_DIR}"; BL_CKPT_FLAGS=(); }

        python scripts/train_classifiers.py \
            --dataset        "${DATASET}"      \
            --data_root      "${DATA_ROOT}"    \
            --arch           "${ARCH}"         \
            --training_type  "${REST}"         \
            --epochs         "${EPOCHS}"       \
            --batch_size     "${BATCH_SIZE}"   \
            --lr             "${LR}"           \
            --weight_decay   "${WEIGHT_DECAY}" \
            --seed           "${SEED}"         \
            --norm           "${NORM}"         \
            --alpha          0.00784           \
            --num_steps      10                \
            --eval_pgd --pgd_steps "${PGD_STEPS}" --pgd_norm "${NORM}" \
            --eval_fgsm                        \
            --eval_random --random_dist laplace --random_n "${PR_SAMPLES}" \
            --random_norm    "${NORM}"         \
            --save_dir       "${CKP_DIR}"      \
            --results_dir    "${OUT_DIR}"      \
            "${BL_CKPT_FLAGS[@]}"
    else
        # REST is unquoted on purpose: it is a flag list that must word-split.
        # It goes LAST so that a run's own flag beats the reference value for
        # the same flag — argparse keeps the final occurrence. That is how
        # w_sharp overrides --weight_mode and starts1 overrides --num_starts
        # without needing a separate reference list per run.
        # shellcheck disable=SC2086
        python scripts/pr_training/pos_geo_training.py \
            --dataset        "${DATASET}"      \
            --data_root      "${DATA_ROOT}"    \
            --arch           "${ARCH}"         \
            --epochs         "${EPOCHS}"       \
            --batch_size     "${BATCH_SIZE}"   \
            --lr             "${LR}"           \
            --weight_decay   "${WEIGHT_DECAY}" \
            --seed           "${SEED}"         \
            --norm           "${NORM}"         \
            "${REF_FLAGS[@]}"                  \
            --num_starts     "${NUM_STARTS}"   \
            --num_steps      "${NUM_STEPS}"    \
            --min_valid_rate "${MIN_VALID_RATE}" \
            --eval_every     "${EVAL_EVERY}"   \
            --pgd_steps      "${PGD_STEPS}"    \
            --pr_samples     "${PR_SAMPLES}"   \
            --tag            "${TAG}"          \
            --no_save_ckpt                     \
            --results_dir    "${OUT_DIR}"      \
            ${REST}
    fi

    echo ""
done

echo "Shard ${SHARD}/${NUM_SHARDS} finished ${MINE} runs -> ${OUT_DIR}"
echo ""
echo "Summarise with:"
echo "  python scripts/pr_training/summarize_ablation.py ${OUT_DIR} \\"
echo "      --out run_exp/pos_geo_training/run_resnet18_on_cifar10.md"
