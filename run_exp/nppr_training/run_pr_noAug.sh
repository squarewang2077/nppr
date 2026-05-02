#!/bin/bash
# run_pr_noAug.sh
# Local-Entropy Probabilistic Robustness (PR) training of
# ResNet18 / ResNet50 / WideResNet50-2 / VGG16 on
# CIFAR-10, CIFAR-100, and TinyImageNet.
#
# Sweeps over a list of GAMMA values; each run's outputs are renamed to
#   loc_ent_eps<LANGEVIN_STEPS>_L<LANGEVIN_BETA>_G<GAMMA>.{pth,log,csv}
# so that the gamma sweep does not clobber prior runs.

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
GPU_ID=1                     # GPU device ID to use (change this to 0, 1, 3, etc.)
export CUDA_VISIBLE_DEVICES=${GPU_ID}

DATA_ROOT="./dataset"
EPOCHS=100
BATCH_SIZE=512
LR=0.01
WEIGHT_DECAY=5e-4
SEED=42

ARCHS=("resnet18")
# ARCHS=("resnet18" "resnet50" "wide_resnet50_2" "vgg16")
DATASETS=("cifar10")
# DATASETS=("cifar10" "cifar100" "tinyimagenet")
TRAINING_TYPE="loc_entropy"  # standard | loc_entropy

# Perturbation budget
NORM="linf"
EPSILON=0.03137              # 8/255

# Particle settings
NUM_PARTICLES_LIST=(1)
INIT_METHOD="uniform"        # zero | gaussian | uniform

# Langevin dynamics
LANGEVIN_STEPS_LIST=(10)
STEP_SIZE=1e-2
LANGEVIN_BETA=100
NOISE_SCALE=1.0

# Energy function
PSI_TYPE="softplus"          # softplus | hinge
PSI_ALPHA=10.0

# Threshold strategy
THRESHOLD_MODE="fixed"       # fixed | adaptive
T0_LIST=(-0.05)
T_FLOOR=0.0

# Scope strategy
SCOPE_MODE="fixed"           # fixed | dynamic

# GAMMA sweep
GAMMAS=(-5000 -1000 -50 -1 -0.05 -0.0001 0 0.0001)

# Outer TRADES-style loss
BETA_OUTER=12.0

# Save root
SAVE_ROOT="./ckp/nppr_training/pr_training"

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
echo "======================================================"
echo "  Using GPU: ${GPU_ID}"
echo "  Training Type: ${TRAINING_TYPE}"
echo "  GAMMA sweep: ${GAMMAS[*]}"
echo "  NUM_PARTICLES sweep: ${NUM_PARTICLES_LIST[*]}"
echo "  LANGEVIN_STEPS sweep: ${LANGEVIN_STEPS_LIST[*]}"
echo "  T0 sweep: ${T0_LIST[*]}"
echo "======================================================"
echo ""

for DATASET in "${DATASETS[@]}"; do
    for ARCH in "${ARCHS[@]}"; do
        SAVE_DIR="${SAVE_ROOT}/${DATASET}/${ARCH}/${TRAINING_TYPE}"
        mkdir -p "${SAVE_DIR}"

        for NUM_PARTICLES in "${NUM_PARTICLES_LIST[@]}"; do
        for LANGEVIN_STEPS in "${LANGEVIN_STEPS_LIST[@]}"; do
        for GAMMA in "${GAMMAS[@]}"; do
        for T0 in "${T0_LIST[@]}"; do
            RUN_NAME="loc_ent_eps${LANGEVIN_STEPS}_L${LANGEVIN_BETA}_G${GAMMA}_N${NUM_PARTICLES}_T${T0}"
            echo "======================================================"
            echo "  arch=${ARCH}  dataset=${DATASET}  gamma=${GAMMA}  num_particles=${NUM_PARTICLES}  langevin_steps=${LANGEVIN_STEPS}  t0=${T0}"
            echo "  run=${RUN_NAME}"
            echo "  save_dir=${SAVE_DIR}"
            echo "======================================================"

            python scripts/train_classifiers_pr.py \
                --dataset        "${DATASET}"        \
                --data_root      "${DATA_ROOT}"      \
                --arch           "${ARCH}"           \
                --training_type  "${TRAINING_TYPE}"  \
                --epochs         "${EPOCHS}"         \
                --batch_size     "${BATCH_SIZE}"     \
                --lr             "${LR}"             \
                --weight_decay   "${WEIGHT_DECAY}"   \
                --seed           "${SEED}"           \
                --norm           "${NORM}"           \
                --epsilon        "${EPSILON}"        \
                --num_particles  "${NUM_PARTICLES}"  \
                --init_method    "${INIT_METHOD}"    \
                --langevin_steps "${LANGEVIN_STEPS}" \
                --step_size      "${STEP_SIZE}"      \
                --langevin_beta  "${LANGEVIN_BETA}"  \
                --noise_scale    "${NOISE_SCALE}"    \
                --psi_type       "${PSI_TYPE}"       \
                --psi_alpha      "${PSI_ALPHA}"      \
                --threshold_mode "${THRESHOLD_MODE}" \
                --t0             "${T0}"             \
                --t_floor        "${T_FLOOR}"        \
                --scope_mode     "${SCOPE_MODE}"     \
                --gamma          "${GAMMA}"          \
                --beta_outer     "${BETA_OUTER}"     \
                --eval_pgd --pgd_steps 10 --pgd_norm linf \
                --eval_locent --locent_n 8 --locent_steps 10 --locent_norm linf \
                --eval_random --random_n 8 --random_norm linf \
                --random_dist gaussian uniform laplace \
                --eval_corruptions --corruption_severities 1 \
                --save_dir       "${SAVE_DIR}"

            # Rename outputs so each gamma run keeps its own files.
            # The training script writes:
            #   <save_dir>/<arch_lower>_<dataset_lower>_<training_type>.{pth,log}
            #   <save_dir>/<arch_lower>_<dataset_lower>_<training_type>_training_info.csv
            ARCH_LOWER="${ARCH,,}"
            DATASET_LOWER="${DATASET,,}"
            SRC_BASE="${SAVE_DIR}/${ARCH_LOWER}_${DATASET_LOWER}_${TRAINING_TYPE}"
            DST_BASE="${SAVE_DIR}/${RUN_NAME}"

            mv "${SRC_BASE}.pth"                  "${DST_BASE}.pth"
            mv "${SRC_BASE}.log"                  "${DST_BASE}.log"
            mv "${SRC_BASE}_training_info.csv"    "${DST_BASE}_training_info.csv"

            echo "  -> renamed outputs to ${RUN_NAME}.{pth,log,csv}"
        done
        done
        done
        done
    done
done

echo ""
echo "All ${TRAINING_TYPE} training runs completed."
