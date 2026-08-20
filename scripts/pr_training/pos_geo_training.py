# pos_geo_training.py - Train on margin level-set perturbations, weighted by
#                       where on the level set each perturbation landed.
#
# Description:
#   Each image gets N random starts inside the epsilon-ball. Every start is
#   driven onto the level set m(x + delta, y) = t, and the model is trained on
#   all N with a weighted PGD-style objective:
#
#       loss = sum_r w_r * CE(f(x + delta_r), y)
#
#   The comparison knobs are the experiment:
#     --solver         energy | newton      how each solver step is taken
#     --weight_mode    uniform | sharp | flat | min_norm | max_norm
#     --geometry_mode  coarea | dual | ce   which gradient sharp/flat rank by
#     --t_mode         fixed | adaptive | reachable   how the level is chosen
#     --tol_mode       absolute | relative  how the validity tolerance scales
#
#   Perturbations that never reached the level set (|m - t| > tol) get zero
#   weight in every mode: the premise is "different positions at the *same*
#   level". A low valid rate is therefore this experiment's main failure mode —
#   every geometry statistic then describes points that are not on the level
#   set. --min_valid_rate warns when it happens; the levers, in order of
#   measured effect, are:
#
#     --solver newton   scale free, needs no retuning as training moves the
#                       margin scale; measured 0.22 -> 0.62 valid rate on a
#                       trained model at ten steps, at the same cost per step.
#     --step_size       for --solver energy, 1e-2 -> 0.2 took 0.24 -> 0.84.
#     --t_mode adaptive a fixed t stays put while CE pushes the margin
#                       distribution away from it, so the valid rate decays.
#     --tol_mode relative
#                       tol as a fraction of the clean margin IQR instead of a
#                       fixed margin value. An absolute tol silently tightens as
#                       training widens the margins: measured on a newton run
#                       the IQR went 0.42 -> 24.46 in six epochs, so tol=0.05
#                       went from 8% of the spread to 0.2% and the valid rate
#                       fell 0.98 -> 0.21 for that reason alone.
#     --t_mode reachable
#                       t as a fraction of the margin floor the epsilon-ball
#                       can actually reach, per sample. Training raises that
#                       floor — measured -1.54 -> -0.20 over six epochs — so a
#                       fixed t walks out of the ball and the valid rate went
#                       0.96 -> 0.15. With --t_frac 0.5 the same runs held
#                       0.88-1.00. Costs five extra backward passes per batch.
#     --warmup_epochs   plain CE first, so the margin distribution settles
#                       before a static t has to sit inside it.
#
#   Not --epsilon: 4x epsilon moved the valid rate only 0.20 -> 0.24. The ball
#   was never the binding constraint, the solver was.
#
# Outputs (one CSV, one log, one checkpoint — nothing else):
#   ckp/pos_geo_training/<name_tag>.pth                      model checkpoint
#   ckp/pos_geo_training/<name_tag>.log                      training log
#   results/pos_geo_training/<name_tag>_training_info.csv    one row per epoch
#
#   <name_tag> defaults to <arch>_<dataset>_level_<solver>_<weight_mode>, which
#   does NOT carry --t, --t_frac, --num_starts or --t_mode — an ablation over
#   any of those needs --tag to keep its runs from overwriting each other.
#   --no_save_ckpt drops the checkpoint and moves the log to --results_dir, so
#   an ablation writes nothing under ckp/.
#
#   Every row carries the training loss/accuracy, the wall-clock training time,
#   and the level-set diagnostics averaged over the epoch's batches. Rows on
#   evaluation epochs (every --eval_every) additionally carry clean / PGD /
#   FGSM accuracy and the Laplace PR estimate, on both the train subset and the
#   test set. Non-evaluation rows leave those columns empty, so every row has
#   the same schema.
#
# Requirements:
#   torch >= 2.0, torchvision >= 0.15, numpy, tqdm, pandas
#
# Dependencies:
#   arch/                       model registry and NormalizedModel wrapper
#   utils/preprocess_data.py    dataset loading and preprocessing
#   utils/epoch_eval.py         single-pass clean + PGD + FGSM + PR evaluation
#   src/pos_geo_loss.py         level-set solver, weighting, geometry
#
# Examples:
#   # Compare the two solvers, everything else held fixed
#   python scripts/pr_training/pos_geo_training.py --solver energy --t_mode adaptive
#   python scripts/pr_training/pos_geo_training.py --solver newton --t_mode adaptive
#
#   # Fixed level, with a warmup so the margin distribution settles first
#   python scripts/pr_training/pos_geo_training.py --t_mode fixed --t 0.0 \
#       --warmup_epochs 5 --weight_mode flat --geometry_mode coarea

import argparse
import logging
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from arch import build_model
from src.pos_geo_loss import (
    GEOMETRY_MODES,
    SOLVERS,
    TOL_MODES,
    T_MODES,
    WEIGHT_MODES,
    dispersion,
    level_geometry,
    matched_l2_epsilon,
    pos_geo_loss,
)
from utils.epoch_eval import evaluate_per_epoch
from utils.preprocess_data import get_dataset, get_img_size

DEFAULT_EPSILON_LINF = 8 / 255

# Diagnostics kept per epoch, batch-averaged. Deliberately short: these either
# gate the experiment's validity or are its actual readout.
DIAG_KEYS = [
    "valid_position_rate", "valid_sample_rate", "ae_rate", "ae_rate_valid",
    "t_used", "tol_used",
    "margin1", "margin_gap", "delta_norm",
    "grad_margin_l2", "grad_ce_dual",
    "eff_rank", "active_count", "weight_entropy",
    "loss_batch_mean", "loss_valid_only",
]

EVAL_KEYS = [f"{split}_{metric}"
             for split in ("trainS", "test")
             for metric in ("clean", "pgd", "fgsm", "pr_laplace")]


def setup_logger(log_path):
    """Logger writing to both stdout and *log_path*."""
    logger = logging.getLogger("pos_geo_training")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(), logging.FileHandler(log_path, mode="a")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def set_seed(seed=42):
    """Seed the RNGs and enable cuDNN auto-tuning.

    benchmark=True picks the fastest kernel per input shape, which matters here
    because the (B*N, C, H, W) perturbation tensor is large. The cost is
    bit-exact reproducibility across hardware; run-to-run seeding is unaffected.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = False
    cudnn.benchmark = True


# ------------------------------------------------------------------
#                            Training
# ------------------------------------------------------------------

def train_one_epoch_ce(model, loader, optimizer, device, criterion, epoch, total):
    """Plain cross-entropy epoch. Used only for --warmup_epochs."""
    model.train()
    loss_sum = 0.0
    correct = seen = 0
    pbar = tqdm(loader, desc=f"Warmup [{epoch}/{total}]", leave=False)
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * y.size(0)
        seen += y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        pbar.set_postfix(loss=f"{loss_sum/seen:.4f}", acc=f"{correct/seen:.4f}")

    return loss_sum / seen, correct / seen, {}


def _batch_diagnostics(delta, info):
    """The DIAG_KEYS for one batch, each already reduced to a scalar."""
    # Geometry is restricted to candidates actually on the level set: an
    # off-level perturbation has no place in a "same level" statistic.
    geom = level_geometry(delta, info["margin1"], valid_mask=info["valid_mask"])
    disp = dispersion(geom["pair_cos"], geom["active_mask"])

    w = info["weights"]
    # Shannon entropy of each weight row: 0 when one position takes everything,
    # log(N) when they share equally. Catches sharp/flat collapsing onto one.
    ent = -(w.clamp_min(1e-12).log() * w).sum(dim=1)

    return {
        "valid_position_rate": info["valid_mask"].float().mean().item(),
        "valid_sample_rate":   info["has_valid"].float().mean().item(),
        "ae_rate":             info["ae_rate"].item(),
        "ae_rate_valid":       info["ae_rate_valid"].item(),
        "t_used":              float(info["t_used"]),
        "tol_used":            float(info["tol_used"]),
        "margin1":             info["margin1"].mean().item(),
        "margin_gap":          (info["margin2"] - info["margin1"]).mean().item(),
        "delta_norm":          info["delta_norm"].mean().item(),
        "grad_margin_l2":      info["grad_margin_l2"].mean().item(),
        "grad_ce_dual":        info["grad_ce_dual"].mean().item(),
        "eff_rank":            disp["eff_rank"].mean().item(),
        "active_count":        disp["active_count"].mean().item(),
        "weight_entropy":      ent.mean().item(),
        "loss_batch_mean":     info["loss_batch_mean"].item(),
        "loss_valid_only":     info["loss_valid_only"].item(),
    }


def train_one_epoch_level(model, loader, optimizer, device, criterion,
                          cfg, epoch, total):
    """One epoch of level-set training. Returns (loss, acc, diagnostics)."""
    model.train()
    loss_sum = 0.0
    correct = seen = n_batches = 0
    N = cfg["num_starts"]
    diag_sums = {k: 0.0 for k in DIAG_KEYS}

    pbar = tqdm(loader, desc=f"Level [{epoch}/{total}]", leave=False)
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # pos_geo_loss runs its solver and outer forward under one eval-mode
        # model, so neither touches BatchNorm running statistics. Update them
        # here instead, once, on the clean batch.
        with torch.no_grad():
            model.train()
            model(x)
            model.eval()

        optimizer.zero_grad(set_to_none=True)
        loss, x_adv, info = pos_geo_loss(
            model, x, y, criterion,
            t=cfg["t"], epsilon=cfg["epsilon"],
            t_mode=cfg["t_mode"], t_quantile=cfg["t_quantile"],
            t_frac=cfg["t_frac"],
            tol_mode=cfg["tol_mode"],
            weight_mode=cfg["weight_mode"], geometry_mode=cfg["geometry_mode"],
            tau=cfg["tau"],
            num_starts=N, num_steps=cfg["num_steps"],
            step_size=cfg["step_size"], alpha=cfg["psi_alpha"], norm=cfg["norm"],
            # None means "let the solver pick"; passing it through explicitly
            # would override that with None and crash.
            **({} if cfg["anchor_lambda"] is None
               else {"anchor_lambda": cfg["anchor_lambda"]}),
            **({} if cfg["tol"] is None else {"tol": cfg["tol"]}),
            solver=cfg["solver"],
        )
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * y.size(0)
        seen += y.size(0)
        n_batches += 1
        with torch.no_grad():
            preds = info["logits_adv"].argmax(dim=1)
            correct += (preds == y.repeat_interleave(N)).sum().item()
            for k, v in _batch_diagnostics(x_adv - x.unsqueeze(1), info).items():
                diag_sums[k] += v

        pbar.set_postfix(loss=f"{loss_sum/seen:.4f}",
                         acc=f"{correct/(seen*N):.4f}",
                         valid=f"{diag_sums['valid_position_rate']/n_batches:.2f}")

    diagnostics = {k: v / max(n_batches, 1) for k, v in diag_sums.items()}
    return loss_sum / seen, correct / (seen * N), diagnostics


# ------------------------------------------------------------------
#                           Evaluation
# ------------------------------------------------------------------

def evaluate_both_splits(model, subtrain_loader, test_loader, device, criterion,
                         args, epoch):
    """Clean / PGD / FGSM / Laplace-PR on the train subset and the test set.

    One pass per split gets all four, so the loaders are walked twice rather
    than eight times.
    """
    pgd_cfg = {"epsilon": args.epsilon, "alpha": args.epsilon / 4.0,
               "num_steps": args.pgd_steps, "norm": args.norm}
    fgsm_cfg = {"epsilon": args.epsilon, "alpha": 1.25 * args.epsilon}
    random_cfgs = [{"noise_dist": "laplace", "epsilon": args.epsilon,
                    "norm": args.norm, "num_samples": args.pr_samples}]

    out = {}
    for split, loader in (("trainS", subtrain_loader), ("test", test_loader)):
        m = evaluate_per_epoch(
            model, loader, device, criterion,
            pgd_cfg=pgd_cfg, fgsm_cfg=fgsm_cfg, random_cfgs=random_cfgs,
            eval_name=f"eval-{split} [{epoch}/{args.epochs}]",
        )
        out[f"{split}_clean"] = m["clean_acc"]
        out[f"{split}_pgd"] = m["pgd_acc"]
        out[f"{split}_fgsm"] = m["fgsm_acc"]
        out[f"{split}_pr_laplace"] = m["random_pr_breakdown"]["laplace"]
    return out


# ------------------------------------------------------------------
#                              Main
# ------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Level-set position-geometry adversarial training.")

    # Data / optimisation
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "tinyimagenet", "svhn"],
                    default="cifar10")
    ap.add_argument("--data_root", type=str, default="./dataset")
    ap.add_argument("--arch", type=str, default="resnet18")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--augment", action="store_true",
                    help="Enable train-time augmentation.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_dir", type=str, default="./ckp/pos_geo_training")
    ap.add_argument("--results_dir", type=str, default="./results/pos_geo_training")
    ap.add_argument("--tag", type=str, default=None,
                    help="Stem for every output file, replacing the derived "
                         "<arch>_<dataset>_level_<solver>_<weight_mode>. That "
                         "name does not carry --t, --t_frac, --num_starts or "
                         "--t_mode, so an ablation varying any of them would "
                         "have its runs overwrite each other. The config is "
                         "still recorded in the log's [config] lines and in the "
                         "CSV's own columns.")
    ap.add_argument("--no_save_ckpt", action="store_true",
                    help="Do not write a checkpoint, and put the log in "
                         "--results_dir instead of --save_dir, so an ablation "
                         "leaves nothing under ckp/ at all. The CSV and the "
                         "per-epoch diagnostics are unaffected.")

    # Level set — the experiment's independent variables
    ap.add_argument("--solver", choices=list(SOLVERS), default="energy",
                    help="How each solver step is taken. 'newton' is scale free "
                         "and needs no --psi_alpha; 'energy' is the original.")
    ap.add_argument("--weight_mode", choices=list(WEIGHT_MODES), default="uniform")
    ap.add_argument("--geometry_mode", choices=list(GEOMETRY_MODES), default="coarea",
                    help="Which gradient sharp/flat rank by. Unused by the "
                         "other weight modes.")
    ap.add_argument("--t_mode", choices=list(T_MODES), default="fixed",
                    help="'fixed' uses --t every batch. 'adaptive' uses the "
                         "--t_quantile quantile of each batch's clean margins. "
                         "'reachable' uses --t_frac times each sample's "
                         "reachable margin floor, which is the only mode that "
                         "tracks the ball shrinking as the model hardens.")
    ap.add_argument("--t", type=float, default=-0.5,
                    help="Target margin level for --t_mode fixed, and the knob "
                         "that sets training strength. It should be NEGATIVE: "
                         "m = f_y - max_j!=y f_j is below zero exactly when the "
                         "point is misclassified, so only t < 0 makes the "
                         "landings adversarial examples. At t=0 the tolerance "
                         "straddles the boundary and roughly half of them are "
                         "not (measured 0.46-0.56). |t| is how deep into the "
                         "wrong side they sit. Lower bound: early in training "
                         "the epsilon-ball only reaches about -1 (measured -1.04 "
                         "at init), so -3 is unreachable until the margins have "
                         "widened; watch ae_rate_valid and valid_position_rate.")
    ap.add_argument("--t_quantile", type=float, default=0.5,
                    help="Clean-margin quantile for --t_mode adaptive.")
    ap.add_argument("--t_frac", type=float, default=0.5,
                    help="For --t_mode reachable: t is this fraction of the "
                         "lowest margin the epsilon-ball can reach, estimated "
                         "per sample. Keeps the level inside the ball as "
                         "training raises the floor — measured, the floor rose "
                         "-1.54 -> -0.20 over six epochs and a fixed t=-0.5 "
                         "went 96%% -> 15%% valid, while t_frac=0.5 held "
                         "0.88-1.00. Negative by construction, so the landings "
                         "stay AEs.")
    ap.add_argument("--tau", type=float, default=1.0,
                    help="Temperature for --weight_mode sharp / flat.")
    ap.add_argument("--epsilon", type=float, default=None,
                    help="Perturbation budget. Omit to get 8/255 for linf and "
                         "the matched radius eps*sqrt(d/3) for l2.")
    ap.add_argument("--norm", choices=["linf", "l2"], default="linf")
    ap.add_argument("--num_starts", type=int, default=8)
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--step_size", type=float, default=None,
                    help="Omit to let --solver choose: 1.0 for newton (the full "
                         "Newton step), 0.2 for energy. Not interchangeable.")
    ap.add_argument("--anchor_lambda", type=float, default=None,
                    help="Pull toward each random start. Omit to let --solver "
                         "choose: 0.01 for energy, 0 for newton. Not "
                         "interchangeable — a Newton step shrinks to nothing "
                         "near the level set while a fixed anchor does not, so "
                         "any anchor eventually drags it back off. Measured on "
                         "a trained model: |m-t| median 0.20 vs 0.003, valid "
                         "rate 0.228 vs 0.898.")
    ap.add_argument("--psi_alpha", type=float, default=10.0,
                    help="Sharpness of the symmetric penalty. --solver energy "
                         "only; newton has no such parameter.")
    ap.add_argument("--tol_mode", choices=list(TOL_MODES), default="absolute",
                    help="'absolute': --tol is a margin value, fixed for the "
                         "run. 'relative': --tol is a fraction of each batch's "
                         "clean margin IQR, so the tolerance tracks a widening "
                         "margin distribution instead of silently tightening.")
    ap.add_argument("--tol", type=float, default=None,
                    help="Validity tolerance, |m - t| <= tol. Omit to get the "
                         "mode's default: 0.05 absolute, 0.02 relative. The two "
                         "are on different scales and do not transfer.")
    ap.add_argument("--warmup_epochs", type=int, default=0,
                    help="Plain CE epochs before level-set training starts.")
    ap.add_argument("--min_valid_rate", type=float, default=0.3,
                    help="Warn when the valid rate falls below this.")

    # Evaluation
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--pgd_steps", type=int, default=10)
    ap.add_argument("--pr_samples", type=int, default=32,
                    help="Laplace noise draws per input for the PR estimate.")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    img_size = get_img_size(args.dataset, args.img_size)

    os.makedirs(args.results_dir, exist_ok=True)
    # Without a checkpoint there is nothing to put in save_dir, so do not even
    # create it — the log goes to results_dir and ckp/ stays untouched.
    if not args.no_save_ckpt:
        os.makedirs(args.save_dir, exist_ok=True)
    aug_suffix = "_Aug" if args.augment else ""
    name_tag = args.tag or (f"{args.arch.lower()}_{args.dataset.lower()}_level_"
                            f"{args.solver}_{args.weight_mode}{aug_suffix}")
    log_dir = args.results_dir if args.no_save_ckpt else args.save_dir
    logger = setup_logger(os.path.join(log_dir, f"{name_tag}.log"))
    csv_path = os.path.join(args.results_dir, f"{name_tag}_training_info.csv")

    # Resolve the budget once img_size is known; an explicit --epsilon wins.
    if args.epsilon is None:
        args.epsilon = (DEFAULT_EPSILON_LINF if args.norm == "linf"
                        else matched_l2_epsilon(DEFAULT_EPSILON_LINF, img_size))
        logger.info(f"[config] epsilon={args.epsilon:.4f} (derived for {args.norm})")

    logger.info(f"[config] {args.arch}/{args.dataset}, img_size={img_size}, "
                f"augment={args.augment}, epochs={args.epochs}, bs={args.batch_size}")
    logger.info(f"[config] solver={args.solver}, weight_mode={args.weight_mode}, "
                f"geometry_mode={args.geometry_mode}, t_mode={args.t_mode}"
                + (f", t_quantile={args.t_quantile}" if args.t_mode == "adaptive"
                   else f", t_frac={args.t_frac}" if args.t_mode == "reachable"
                   else f", t={args.t}"))
    logger.info(f"[config] anchor_lambda="
                f"{args.anchor_lambda if args.anchor_lambda is not None else 'auto'}")
    logger.info(f"[config] tol_mode={args.tol_mode}, "
                f"tol={args.tol if args.tol is not None else 'auto'}")
    logger.info(f"[config] epsilon={args.epsilon:.4f}, norm={args.norm}, "
                f"num_starts={args.num_starts}, num_steps={args.num_steps}, "
                f"step_size={args.step_size if args.step_size is not None else 'auto'}, "
                + (f", psi_alpha={args.psi_alpha}" if args.solver == "energy" else ""))
    # One-shot check at startup: under a fixed t we know both numbers up front,
    # so say it once here rather than every batch. Under t_mode=adaptive t moves
    # per batch and there is nothing to check yet — ae_rate_valid covers it.
    # reachable derives a negative t from the floor, so the AE premise holds by
    # construction and there is nothing to check up front.
    if args.t_mode == "fixed":
        startup_tol = args.tol if args.tol is not None else 0.05
        if args.t >= 0:
            logger.warning(
                f"t={args.t:+.3f} is not negative: m < 0 is what makes a point "
                f"misclassified, so a non-negative level puts the landings on "
                f"the correctly-classified side and they will not be "
                f"adversarial examples. Use t < 0; |t| is the training "
                f"strength. Watch ae_rate_valid in the CSV.")
        elif args.tol_mode == "absolute" and args.t + startup_tol >= 0:
            logger.warning(
                f"t={args.t:+.3f} with tol={startup_tol}: t + tol >= 0, so the "
                f"tolerance band crosses the decision boundary. Landings cluster "
                f"near t rather than filling the band, so this often still gives "
                f"only AEs — but it is no longer guaranteed. Watch "
                f"ae_rate_valid; t <= -tol removes the doubt.")

    if args.warmup_epochs:
        logger.info(f"[config] warmup_epochs={args.warmup_epochs} (plain CE first)")
    logger.info(f"[save] {csv_path}")

    # Data
    train_set, num_classes = get_dataset(args.dataset, args.data_root, True,
                                         img_size, augment=args.augment)
    test_set, _ = get_dataset(args.dataset, args.data_root, False, img_size,
                              augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=max(256, args.batch_size),
                             shuffle=False, num_workers=4, pin_memory=True)

    # Fixed, un-augmented train subset the size of the test set, so trainS and
    # test numbers are directly comparable.
    train_noaug, _ = get_dataset(args.dataset, args.data_root, True, img_size,
                                 augment=False)
    rng = np.random.default_rng(seed=args.seed)
    idx = rng.choice(len(train_noaug), len(test_set), replace=False)
    subtrain_loader = DataLoader(Subset(train_noaug, idx), batch_size=args.batch_size,
                                 shuffle=False, num_workers=4, pin_memory=True)
    logger.info(f"[eval] trainS subset: {len(idx)}/{len(train_noaug)} samples")

    # Model / optimiser
    model = build_model(args.arch, num_classes, args.dataset).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    level_cfg = {k: getattr(args, k) for k in
                 ("t", "t_mode", "t_quantile", "t_frac", "tol_mode", "tol", "epsilon",
                  "norm", "num_starts", "num_steps", "step_size",
                  "anchor_lambda", "psi_alpha", "weight_mode", "geometry_mode",
                  "tau", "solver")}

    history, epoch_times = [], []
    for ep in range(1, args.epochs + 1):
        in_warmup = ep <= args.warmup_epochs
        start = time.time()
        if in_warmup:
            train_loss, train_acc, diagnostics = train_one_epoch_ce(
                model, train_loader, optimizer, device, criterion, ep, args.epochs)
        else:
            train_loss, train_acc, diagnostics = train_one_epoch_level(
                model, train_loader, optimizer, device, criterion, level_cfg,
                ep, args.epochs)
        scheduler.step()
        # Training only — evaluation happens after this point.
        train_time = time.time() - start
        epoch_times.append(train_time)

        # The config columns are what lets a summariser tell one ablation run
        # from another: --tag names the file but says nothing about what was
        # varied, and t_used/tol_used are outcomes rather than settings.
        row = {
            "tag": name_tag,
            "arch": args.arch, "dataset": args.dataset, "epoch": ep,
            "lr": scheduler.get_last_lr()[0],
            "solver": args.solver, "weight_mode": args.weight_mode,
            "geometry_mode": args.geometry_mode, "t_mode": args.t_mode,
            "t": args.t, "t_frac": args.t_frac,
            "num_starts": args.num_starts, "num_steps": args.num_steps,
            "train_loss": train_loss, "train_acc": train_acc,
            "train_time_s": train_time,
            **{k: diagnostics.get(k) for k in DIAG_KEYS},
            **{k: None for k in EVAL_KEYS},
        }

        msg = (f"[{ep:03d}/{args.epochs}] {'warmup' if in_warmup else 'level '} "
               f"loss={train_loss:.4f} acc={train_acc*100:.2f}% "
               f"time={train_time:.1f}s")
        if not in_warmup:
            msg += (f" | valid={diagnostics['valid_position_rate']:.2f}/"
                    f"{diagnostics['valid_sample_rate']:.2f} "
                    f"m1={diagnostics['margin1']:+.3f} "
                    f"t={diagnostics['t_used']:+.3f} "
                    f"tol={diagnostics['tol_used']:.3f} "
                    f"ae={diagnostics['ae_rate_valid']:.2f} "
                    f"eff_rank={diagnostics['eff_rank']:.2f}")
        logger.info(msg)

        if not in_warmup and diagnostics["valid_position_rate"] < args.min_valid_rate:
            logger.warning(
                f"[{ep:03d}/{args.epochs}] valid_position_rate="
                f"{diagnostics['valid_position_rate']:.3f} < {args.min_valid_rate}: "
                f"most perturbations are NOT on the level set, so the geometry "
                f"columns describe off-level points. Try --solver newton "
                f"(currently {args.solver}), a larger --step_size, more "
                f"--num_steps ({args.num_steps}), --t_mode adaptive, or "
                f"--tol_mode relative (currently {args.tol_mode}) if the margin "
                f"distribution has widened since the run started.")

        # The landings are supposed to be adversarial examples. Validity alone
        # does not give that: the tolerance is two-sided, so a t too close to
        # zero lets half of them land on the correctly-classified side.
        if not in_warmup and diagnostics["ae_rate_valid"] < 0.99:
            logger.warning(
                f"[{ep:03d}/{args.epochs}] ae_rate_valid="
                f"{diagnostics['ae_rate_valid']:.3f}: some weighted positions "
                f"are NOT adversarial examples (m >= 0). Only t < 0 puts the "
                f"level set on the misclassified side, and t + tol must stay "
                f"below zero — currently t={diagnostics['t_used']:+.3f}, "
                f"tol={diagnostics['tol_used']:.3f}.")

        if ep % args.eval_every == 0 or ep == args.epochs:
            row.update(evaluate_both_splits(model, subtrain_loader, test_loader,
                                            device, criterion, args, ep))
            logger.info(
                f"  trainS: clean={row['trainS_clean']*100:.2f}% "
                f"pgd{args.pgd_steps}={row['trainS_pgd']*100:.2f}% "
                f"fgsm={row['trainS_fgsm']*100:.2f}% "
                f"pr_laplace={row['trainS_pr_laplace']*100:.2f}%\n"
                f"  test  : clean={row['test_clean']*100:.2f}% "
                f"pgd{args.pgd_steps}={row['test_pgd']*100:.2f}% "
                f"fgsm={row['test_fgsm']*100:.2f}% "
                f"pr_laplace={row['test_pr_laplace']*100:.2f}%\n"
                f"  train time: mean={np.mean(epoch_times):.1f}s "
                f"over {len(epoch_times)} epochs")
            if not args.no_save_ckpt:
                torch.save({"epoch": ep, "arch": args.arch, "dataset": args.dataset,
                            "model_state": model.state_dict(), "config": vars(args)},
                           os.path.join(args.save_dir, f"{name_tag}.pth"))

        history.append(row)
        pd.DataFrame(history).to_csv(csv_path, index=False)

    mean_time = float(np.mean(epoch_times))
    history[-1]["train_time_mean_s"] = mean_time
    pd.DataFrame(history).to_csv(csv_path, index=False)
    logger.info(f"[done] mean train time {mean_time:.1f}s/epoch "
                f"over {len(epoch_times)} epochs -> {csv_path}")


if __name__ == "__main__":
    main()
