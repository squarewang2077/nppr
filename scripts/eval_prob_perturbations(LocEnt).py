# eval_prob_perturbations(LocEnt).py
#   Evaluate a trained classifier under a Local-Entropy probabilistic attack.
#
# Description:
#   Given a checkpoint (.pth) produced by scripts/train_classifiers.py,
#   scripts/train_classifiers_pr.py, or scripts/train_classifiers_adv.py,
#   this script loads the model and runs the local-entropy Langevin attack
#   on the test set. It DOES NOT update model weights.
#
#   For every test batch, the attack draws N adversarial particles per input
#   using local-entropy Langevin dynamics. The script reports:
#
#     clean_acc        - top-1 accuracy on clean test inputs.
#     mean_pr          - mean fraction of N particles that the model still
#                        classifies correctly per sample, averaged over the
#                        test set. This is the natural probabilistic-robustness
#                        score for the attack distribution defined by the
#                        Langevin sampler.
#     worst_acc        - fraction of samples where ALL N particles are still
#                        correctly classified (worst-case-over-particles, the
#                        pessimistic bound).
#     attack_success   - 1 - worst_acc. Fraction of samples where at least one
#                        particle fooled the model.
#
#   Outputs:
#     <save_dir>/<arch>_<dataset>_locent_attack.log         text log
#     <save_dir>/<arch>_<dataset>_locent_attack_summary.csv summary metrics
#
# Usage:
#   python scripts/eval_prob_perturbations\(LocEnt\).py \
#       --ckpt ./ckp/.../resnet18_cifar10_loc_entropy.pth \
#       --dataset cifar10 --arch resnet18 \
#       --epsilon 0.03137 --num_particles 8 --langevin_steps 20 --gamma 0.05

import os
import csv
import logging
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from arch import build_model
from utils.preprocess_data import get_dataset, get_img_size
from src.local_entropy4pr import (
    ParticleState,
    EnergyConfig,
    LangevinConfig,
    compute_margins,
    fixed_threshold_update,
    adaptive_threshold_update,
    fixed_scope,
    dynamic_scope,
    langevin_update_local_entropy,
)


def setup_logger(log_path: str) -> logging.Logger:
    """Return a logger that writes to both stdout and *log_path*."""
    logger = logging.getLogger("eval_locent_attack")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def set_seed(seed: int = 42):
    """Seed Python/NumPy/Torch RNGs. Disables cuDNN auto-tuning to keep the
    attack deterministic across runs of the same checkpoint."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


# ------------------------------------------------------------------
#                Local-entropy adversarial generator
# ------------------------------------------------------------------

def local_entropy_generator(model, x, y, **kwargs):
    """
    Run one local-entropy attack on a clean batch.

    Returns:
        x_adv: adversarial particles, shape (B, N, C, H, W).
    """
    norm = kwargs.get("norm", "linf")
    epsilon = kwargs.get("epsilon", 8 / 255)
    num_particles = kwargs.get("num_particles", 8)
    langevin_steps = kwargs.get("langevin_steps", 5)
    step_size = kwargs.get("step_size", 1e-2)
    langevin_beta = kwargs.get("langevin_beta", 100.0)
    noise_scale = kwargs.get("noise_scale", 1.0)
    gamma = kwargs.get("gamma", 1.0)
    psi_type = kwargs.get("psi_type", "softplus")
    psi_alpha = kwargs.get("psi_alpha", 10.0)
    threshold_mode = kwargs.get("threshold_mode", "fixed")
    t0 = kwargs.get("t0", 0.0)
    t_floor = kwargs.get("t_floor", 0.0)
    scope_mode = kwargs.get("scope_mode", "fixed")
    init_method = kwargs.get("init_method", "uniform")

    particle_state = ParticleState(epsilon=epsilon, norm=norm, num_particles=num_particles)

    energy_cfg = EnergyConfig(psi_type=psi_type, psi_alpha=psi_alpha)
    langevin_cfg = LangevinConfig(
        steps=langevin_steps,
        step_size=step_size,
        beta=langevin_beta,
        noise_scale=noise_scale,
    )

    particle_state.init_particles(x, method=init_method, warm_start=False)

    if threshold_mode == "fixed":
        B, N = x.shape[0], num_particles
        margins = torch.zeros((B, N), device=x.device, dtype=x.dtype)
        t_curr = fixed_threshold_update(margins=margins, state=particle_state, t=t0)
    elif threshold_mode == "adaptive":
        margins = compute_margins(model=model, x=x, y=y, state=particle_state)
        t_curr = adaptive_threshold_update(
            margins=margins, state=particle_state, t0=t0, t_floor=t_floor
        )
    else:
        B, N = x.shape[0], num_particles
        margins = torch.zeros((B, N), device=x.device, dtype=x.dtype)
        t_curr = fixed_threshold_update(margins=margins, state=particle_state, t=t0)

    if scope_mode == "fixed":
        gamma_curr = fixed_scope(t_curr=t_curr, gamma=gamma)
    elif scope_mode == "dynamic":
        gamma_curr = dynamic_scope(t_curr=t_curr, t0=t0, t_floor=t_floor)
    else:
        gamma_curr = fixed_scope(t_curr=t_curr, gamma=gamma)

    langevin_update_local_entropy(
        state=particle_state,
        model=model,
        x=x,
        y=y,
        t_curr=t_curr,
        gamma_curr=gamma_curr,
        energy_cfg=energy_cfg,
        cfg=langevin_cfg,
    )

    return particle_state.x_adv


# ------------------------------------------------------------------
#                       Per-batch attack
# ------------------------------------------------------------------

def attack_batch(model, x, y, attack_cfg):
    """
    Returns:
        clean_correct        : (B,)   bool  — clean prediction correct.
        per_particle_correct : (B, N) bool  — adv prediction correct, per particle.
    """
    # Clean prediction (no_grad — no need for autograd here).
    with torch.no_grad():
        clean_pred = model(x).argmax(dim=1)
        clean_correct = (clean_pred == y)

    # Generate adversarial particles. The Langevin update needs autograd
    # through the model w.r.t. inputs (NOT parameters), so we leave the
    # outer torch.no_grad() — autograd.grad inside the sampler still works.
    x_adv = local_entropy_generator(model, x, y, **attack_cfg)
    B, N, C, H, W = x_adv.shape

    with torch.no_grad():
        adv_pred = model(x_adv.reshape(B * N, C, H, W)).argmax(dim=1).view(B, N)

    y_rep = y.unsqueeze(1).expand(B, N)
    per_particle_correct = (adv_pred == y_rep)

    return clean_correct, per_particle_correct


# ------------------------------------------------------------------
#                          Main Function
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    # ============================================================
    # Checkpoint & dataset
    # ============================================================
    ap.add_argument("--ckpt", type=str,
                    default="./ckp/nppr_eval/standard/resnet/resnet18_cifar10.pth",
                    help="Path to the trained model checkpoint (.pth).")
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "tinyimagenet"], default="cifar10")
    ap.add_argument("--data_root", type=str, default="./dataset")
    ap.add_argument(
        "--arch",
        choices=[
            "resnet18", "resnet50", "wide_resnet50_2",
            "vgg16", "densenet121", "mobilenet_v3_large", "efficientnet_b0",
            "vit_b_16",
        ],
        default="resnet18",
    )
    ap.add_argument("--img_size", type=int, default=32,
                    help="Input image size. If None, use the dataset default.")
    ap.add_argument("--batch_size", type=int, default=256)

    # ============================================================
    # Attack: perturbation budget
    # ============================================================
    ap.add_argument("--epsilon", type=float, default=8 / 255,
                    help="Perturbation budget. For linf on CIFAR, 8/255 is standard.")
    ap.add_argument("--norm", choices=["linf", "l2"], default="linf")
    ap.add_argument("--num_particles", type=int, default=8,
                    help="Number of adversarial particles per input.")
    ap.add_argument("--init_method", type=str, default="uniform",
                    choices=["zero", "gaussian", "uniform"])

    # ============================================================
    # Attack: Langevin dynamics
    # ============================================================
    ap.add_argument("--langevin_steps", type=int, default=10,
                    help="Number of Langevin steps (typically larger than during training).")
    ap.add_argument("--step_size", type=float, default=1e-2)
    ap.add_argument("--langevin_beta", type=float, default=100.0)
    ap.add_argument("--noise_scale", type=float, default=1.0)

    # ============================================================
    # Attack: energy + threshold + scope
    # ============================================================
    ap.add_argument("--psi_type", type=str, default="softplus", choices=["softplus", "hinge"])
    ap.add_argument("--psi_alpha", type=float, default=10.0)
    ap.add_argument("--threshold_mode", type=str, default="fixed", choices=["fixed", "adaptive"])
    ap.add_argument("--t0", type=float, default=-0.05)
    ap.add_argument("--t_floor", type=float, default=0.0)
    ap.add_argument("--scope_mode", type=str, default="fixed", choices=["fixed", "dynamic"])
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--gamma_min", type=float, default=0.1)
    ap.add_argument("--gamma_max", type=float, default=10.0)

    # ============================================================
    # Misc
    # ============================================================
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_dir", type=str, default="./results/locent_attack",
                    help="Directory to write log and summary CSV.")
    ap.add_argument("--tag", type=str, default=None,
                    help="Optional run tag appended to output filenames "
                         "(e.g. the gamma sweep value).")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    img_size = get_img_size(args.dataset, args.img_size)

    # Output paths
    os.makedirs(args.save_dir, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    base = f"{args.arch.lower()}_{args.dataset.lower()}_locent_attack{suffix}"
    log_path = os.path.join(args.save_dir, f"{base}.log")
    csv_path = os.path.join(args.save_dir, f"{base}_summary.csv")
    logger = setup_logger(log_path)

    # Log config
    logger.info(f"[config] ckpt={args.ckpt}")
    logger.info(f"[config] dataset={args.dataset}, arch={args.arch}, img_size={img_size}")
    logger.info(f"[config] epsilon={args.epsilon:.4f}, norm={args.norm}, "
                f"num_particles={args.num_particles}, langevin_steps={args.langevin_steps}")
    logger.info(f"[config] step_size={args.step_size}, langevin_beta={args.langevin_beta}, "
                f"noise_scale={args.noise_scale}")
    logger.info(f"[config] psi_type={args.psi_type}, psi_alpha={args.psi_alpha}, "
                f"threshold_mode={args.threshold_mode}, t0={args.t0}, t_floor={args.t_floor}")
    logger.info(f"[config] scope_mode={args.scope_mode}, gamma={args.gamma}, "
                f"init_method={args.init_method}")

    # Test set (no augmentation)
    test_set, num_classes = get_dataset(args.dataset, args.data_root, False, img_size, augment=False)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Build model + load weights
    model = build_model(args.arch, num_classes, args.dataset, pretrained=False)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt.get("model_state", ckpt)  # tolerate raw state_dict checkpoints
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info(f"[load] checkpoint loaded from {args.ckpt}")
    if isinstance(ckpt, dict):
        for k in ("arch", "dataset", "epoch", "training_type"):
            if k in ckpt:
                logger.info(f"[load] ckpt.{k} = {ckpt[k]}")

    # Attack config (forwarded to local_entropy_generator)
    attack_cfg = {
        "norm": args.norm,
        "epsilon": args.epsilon,
        "num_particles": args.num_particles,
        "langevin_steps": args.langevin_steps,
        "step_size": args.step_size,
        "langevin_beta": args.langevin_beta,
        "noise_scale": args.noise_scale,
        "gamma": args.gamma,
        "psi_type": args.psi_type,
        "psi_alpha": args.psi_alpha,
        "threshold_mode": args.threshold_mode,
        "t0": args.t0,
        "t_floor": args.t_floor,
        "scope_mode": args.scope_mode,
        "init_method": args.init_method,
    }

    # ============================================================
    # Attack loop
    # ============================================================
    n_total = 0
    n_clean_correct = 0
    n_worst_correct = 0     # all N particles correct on this sample
    n_any_fooled = 0        # at least one particle fools the model
    sum_per_particle = 0.0  # for mean PR (mean fraction correct)
    sum_per_particle_correct = 0  # raw count of correct (sample, particle) pairs

    pbar = tqdm(test_loader, desc="Attack", leave=False)
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        clean_correct, pp_correct = attack_batch(model, x, y, attack_cfg)
        # pp_correct: (B, N) bool

        B, N = pp_correct.shape

        n_total += B
        n_clean_correct += clean_correct.sum().item()

        worst = pp_correct.all(dim=1)              # (B,) all particles correct
        any_wrong = (~pp_correct).any(dim=1)       # (B,) at least one fooled
        n_worst_correct += worst.sum().item()
        n_any_fooled += any_wrong.sum().item()

        # Probabilistic robustness: per-sample mean over particles, then mean over batch
        sum_per_particle += pp_correct.float().mean(dim=1).sum().item()
        sum_per_particle_correct += pp_correct.sum().item()

        pbar.set_postfix(
            clean=f"{n_clean_correct / n_total:.4f}",
            worst=f"{n_worst_correct / n_total:.4f}",
            mean_pr=f"{sum_per_particle / n_total:.4f}",
        )

    clean_acc = n_clean_correct / n_total
    worst_acc = n_worst_correct / n_total
    mean_pr = sum_per_particle / n_total
    attack_success = n_any_fooled / n_total
    raw_pp_acc = sum_per_particle_correct / (n_total * args.num_particles)  # equals mean_pr

    # ============================================================
    # Report
    # ============================================================
    logger.info("=" * 60)
    logger.info(f"[result] clean_acc       = {clean_acc * 100:.2f}%   ({n_clean_correct}/{n_total})")
    logger.info(f"[result] worst_acc       = {worst_acc * 100:.2f}%   "
                f"(fraction of samples where ALL {args.num_particles} particles still correct)")
    logger.info(f"[result] mean_pr         = {mean_pr * 100:.2f}%   "
                f"(mean fraction of correct particles per sample)")
    logger.info(f"[result] attack_success  = {attack_success * 100:.2f}%   "
                f"(fraction of samples fooled by at least one particle)")
    logger.info(f"[result] per_pair_acc    = {raw_pp_acc * 100:.2f}%   "
                f"(raw count over all sample-particle pairs)")
    logger.info("=" * 60)

    # Summary CSV (one row)
    summary = {
        "ckpt": args.ckpt,
        "arch": args.arch,
        "dataset": args.dataset,
        "img_size": img_size,
        "norm": args.norm,
        "epsilon": args.epsilon,
        "num_particles": args.num_particles,
        "langevin_steps": args.langevin_steps,
        "step_size": args.step_size,
        "langevin_beta": args.langevin_beta,
        "noise_scale": args.noise_scale,
        "gamma": args.gamma,
        "psi_type": args.psi_type,
        "psi_alpha": args.psi_alpha,
        "threshold_mode": args.threshold_mode,
        "t0": args.t0,
        "scope_mode": args.scope_mode,
        "init_method": args.init_method,
        "n_total": n_total,
        "clean_acc": clean_acc,
        "worst_acc": worst_acc,
        "mean_pr": mean_pr,
        "attack_success": attack_success,
    }
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    logger.info(f"[save] summary -> {csv_path}")
    logger.info(f"[save] log     -> {log_path}")


if __name__ == "__main__":
    main()
