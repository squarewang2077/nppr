import os
import numpy as np
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm


# ------------------------------------------------------------------
#                    PR random baseline generator
# ------------------------------------------------------------------

def pr_random_generator(model, x, y,
                         epsilon=8/255, norm="linf",
                         num_samples=10, noise_dist="gaussian",
                         return_stats=True,
                         **kwargs):
    """
    Random-baseline perturbation generator with the same interface as
    pr_generator in src/langevin4pr.py.

    Instead of computing a gradient-guided posterior, it samples N deltas
    per input i.i.d. from a chosen distribution, projects each onto the
    epsilon-ball, and returns the perturbed inputs.

    Args:
        model : not used — kept for API compatibility with pr_generator.
        x     : clean inputs (B, C, H, W), values in [0, 1].
        y     : not used — kept for API compatibility.
        epsilon     : perturbation budget radius.
        norm        : "linf" or "l2".
        num_samples : number of perturbation draws per input (N).
        noise_dist  : sampling distribution — one of
                      "gaussian"  ~ N(0, epsilon^2 * I)  then projected,
                      "uniform"   ~ Uniform(-epsilon, epsilon)^d then projected,
                      "laplace"   ~ Laplace(0, epsilon) then projected.
        return_stats: whether to return a stats dict (matches pr_generator API).
        **kwargs    : absorbs unused pr_generator parameters (K, sigma_list,
                      beta_mix, kappa, fisher_damping, tau, noise_scale, …)
                      so this function can be called with the same config dict.

    Returns:
        x_adv : (B, N, C, H, W) perturbed inputs, values in [0, 1].
        stats : dict with scalar entries D_proxy, pi_entropy (fixed 0 — no
                mixture model), and string entry dist_type; or None if
                return_stats=False.
    """
    noise_dist = noise_dist.lower()
    valid_dists = ("gaussian", "uniform", "laplace")
    if noise_dist not in valid_dists:
        raise ValueError(f"noise_dist must be one of {valid_dists}, got '{noise_dist}'")

    norm = norm.lower()
    B, C, H, W = x.shape
    N = num_samples
    d = C * H * W
    device = x.device
    x0 = x.detach()

    # ------------------------------------------------------------------
    # 1) Sample raw delta (B, N, d)
    # ------------------------------------------------------------------
    if noise_dist == "gaussian":
        delta_flat = torch.randn(B, N, d, device=device) * epsilon

    elif noise_dist == "uniform":
        delta_flat = torch.empty(B, N, d, device=device).uniform_(-epsilon, epsilon)

    else:  # laplace — inverse-CDF method
        # If U ~ Uniform(0, 1) then -b * sign(U - 0.5) * log(1 - 2|U - 0.5|) ~ Laplace(0, b)
        u = torch.empty(B, N, d, device=device).uniform_(1e-6, 1.0 - 1e-6)
        u_c = u - 0.5
        delta_flat = -epsilon * u_c.sign() * torch.log1p(-2.0 * u_c.abs())

    delta = delta_flat.view(B, N, C, H, W)

    # ------------------------------------------------------------------
    # 2) Project onto epsilon-ball (matches pr_generator step 5)
    # ------------------------------------------------------------------
    if norm in ("linf", "l_inf", "l-infty"):
        delta = delta.clamp(-epsilon, epsilon)

    elif norm in ("l2", "l_2"):
        BN = B * N
        delta_2d = delta.reshape(BN, -1)
        norms = delta_2d.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        factors = torch.minimum(torch.ones_like(norms), epsilon / norms)
        delta = (delta_2d * factors).view(B, N, C, H, W)

    else:
        raise ValueError(f"Unsupported norm: {norm}")

    x_adv = torch.clamp(x0.unsqueeze(1) + delta, 0.0, 1.0).detach()   # (B, N, C, H, W)

    # ------------------------------------------------------------------
    # 3) Stats (optional)
    # ------------------------------------------------------------------
    stats = None
    if return_stats:
        delta_proj = (x_adv - x0.unsqueeze(1)).reshape(B * N, -1)      # (B*N, d)
        D_proxy = delta_proj.norm(p=2, dim=1).mean()
        stats = {
            "D_proxy":    D_proxy.detach(),
            # pi_entropy is 0 for a non-mixture generator; kept so the same
            # stats accumulator and print logic works without branching.
            "pi_entropy": torch.tensor(0.0, device=device),
            "dist_type":  noise_dist,   # string — ignored by StatsAccumulator
        }

    return x_adv, stats


# ------------------------------------------------------------------
#                    GMM-based PR generator
# ------------------------------------------------------------------

def load_gmm_model(gmm_path, dataset, device="cuda",
                   feat_arch=None, feat_num_classes=None):
    """
    Load a trained GMM4PR model from a checkpoint.

    The feature extractor injected into the GMM is built from feat_arch /
    feat_num_classes, which are *independent* of whatever classifier is being
    evaluated.  This matters because the GMM was trained with a specific
    backbone and that backbone must be reproduced here — even if the classifier
    under evaluation uses a completely different architecture.

    When feat_extractor weights ARE already present in the checkpoint's
    state_dict (newer saves) they are loaded automatically and the externally-
    built extractor is only used to reconstruct the module structure.

    Args:
        gmm_path         : Path to the .pt checkpoint produced by GMM4PR.save().
        dataset          : Dataset name (e.g. "cifar10") — used for normalisation
                           stats when building the feat_extractor.
        device           : Target device string ("cuda" or "cpu").
        feat_arch        : Architecture of the feature extractor that the GMM was
                           trained with (e.g. "resnet50").  When None, inferred
                           from the checkpoint's stored config.
        feat_num_classes : Number of classes used when the feat_extractor backbone
                           was built.  Defaults to the value stored in the GMM
                           checkpoint (cfg["num_cls"]) when None.

    Returns:
        gmm : GMM4PR instance in eval mode, all parameters frozen.
              The resolved feat_arch is stored as ``gmm._feat_arch``.
    """
    # Lazy imports to avoid circular dependencies at module load time
    from src.gmm4pr import GMM4PR
    from arch import build_feat_extractor

    if not os.path.isfile(gmm_path):
        raise FileNotFoundError(f"GMM checkpoint not found: {gmm_path}")

    # Load checkpoint config once to resolve any missing arguments
    _ckpt_cfg = torch.load(gmm_path, map_location="cpu", weights_only=False)["config"]
    _nested_cfg = _ckpt_cfg.get("config", {})

    # Resolve feat_arch: try top-level key first (future saves), then
    # the nested training config written by train_gmm.py (existing saves).
    if feat_arch is None:
        feat_arch = _ckpt_cfg.get("feat_arch") or _nested_cfg.get("arch")
        if feat_arch is None:
            raise ValueError(
                "feat_arch could not be inferred from the checkpoint. "
                "Please pass feat_arch explicitly."
            )

    # Resolve feat_num_classes from num_cls stored in the checkpoint
    if feat_num_classes is None:
        feat_num_classes = _ckpt_cfg.get("num_cls")
        if feat_num_classes is None:
            raise ValueError(
                "feat_num_classes could not be inferred from the checkpoint. "
                "Please pass feat_num_classes explicitly."
            )

    feat_extractor = build_feat_extractor(feat_arch, feat_num_classes, dataset).to(device).eval()
    for p in feat_extractor.parameters():
        p.requires_grad_(False)

    # Rebuild the decoder (upsampler) if it was used during training.
    # Non-trainable decoders have no weights in the state_dict so they must
    # be reconstructed from the stored training config.
    up_sampler = None
    if _nested_cfg.get("use_decoder"):
        from src.gmm4pr import build_decoder_from_flag
        from utils.preprocess_data import get_img_size
        _decoder_backend = _nested_cfg["decoder_backend"]
        _resize = _nested_cfg.get("resize") or None
        _img_size = get_img_size(dataset, _resize)
        _out_shape = (3, _img_size, _img_size)
        up_sampler = build_decoder_from_flag(
            _decoder_backend, _ckpt_cfg["latent_dim"], _out_shape, device
        )
        for p in up_sampler.parameters():
            p.requires_grad_(False)

    gmm = GMM4PR.load_from_checkpoint(
        gmm_path,
        feat_extractor=feat_extractor,
        up_sampler=up_sampler,
        map_location=device,
        strict=True,
    )
    gmm = gmm.to(device).eval()
    for p in gmm.parameters():
        p.requires_grad_(False)

    gmm._feat_arch = feat_arch
    return gmm


def pr_gmm_generator(model, x, y, gmm,
                      num_samples=10, return_stats=True,
                      epsilon=None, norm=None,
                      **kwargs):
    """
    GMM-based perturbation generator matching the pr_generator interface.

    Uses a trained GMM4PR model to draw N perturbation samples per input.
    By default the perturbation budget (epsilon, norm) comes from the GMM's
    own training configuration.  Pass epsilon / norm explicitly to evaluate
    under a *different* budget than the one used at training time — the
    generated deltas are then re-projected onto the new budget without
    modifying the GMM.

    Note: the *classifier* being evaluated (model arg) is fully independent
    of the GMM's internal feature extractor.  The GMM conditions on x through
    its own frozen backbone; predictions are made by whatever model is passed
    to the Evaluator.

    Args:
        model       : not used directly — kept for API compatibility.
        x           : clean inputs (B, C, H, W), values in [0, 1].
        y           : labels (B,) — forwarded to the GMM for y/xy conditioning;
                      ignored when the GMM was trained unconditionally.
        gmm         : trained GMM4PR instance (eval mode, gradients frozen).
        num_samples : number of perturbation draws per input (N).
        return_stats: whether to return a stats dict.
        epsilon     : override the perturbation radius at evaluation time.
                      If None the GMM's training epsilon is used unchanged.
        norm        : override the perturbation norm ("linf" or "l2") at
                      evaluation time.  If None the GMM's training norm is used.
        **kwargs    : absorbs unused pr_generator parameters (K, sigma_list,
                      beta_mix, …) for drop-in compatibility.

    Returns:
        x_adv : (B, N, C, H, W) perturbed inputs, values in [0, 1].
        stats : dict with D_proxy (mean L2 norm of projected deltas) and
                pi_entropy (0.0 placeholder); or None if return_stats=False.
    """
    x0 = x.detach()
    B = x0.size(0)
    N = num_samples

    # Resolve effective budget
    train_eps  = float(gmm.budget["eps"])
    train_norm = gmm.budget["norm"].lower()
    eval_eps   = epsilon if epsilon is not None else train_eps
    eval_norm  = norm.lower() if norm is not None else train_norm

    gmm.eval()
    with torch.no_grad():
        out = gmm.sample(x=x0, y=y, num_samples=N)

    # GMM4PR.sample() returns delta shaped (N, B, C, H, W) — transpose to
    # (B, N, C, H, W) to match the expected distributional batch format.
    delta = out["delta"].permute(1, 0, 2, 3, 4).contiguous()   # (B, N, C, H, W)

    # Re-project onto the evaluation budget if it differs from training budget.
    # This does not mutate the GMM — the delta tensor is simply re-clamped /
    # rescaled.  For L-inf the GMM produces eps*tanh(u) ∈ (-train_eps, train_eps),
    # so a clamp to [-eval_eps, eval_eps] is exact.  For L2 the GMM produces
    # L2-normalised deltas scaled to train_eps, so we rescale accordingly.
    if eval_eps != train_eps or eval_norm != train_norm:
        if eval_norm in ("linf", "l_inf"):
            delta = delta.clamp(-eval_eps, eval_eps)
        elif eval_norm in ("l2", "l_2"):
            BN = B * N
            d2 = delta.reshape(BN, -1)
            norms = d2.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            factors = torch.minimum(torch.ones_like(norms), eval_eps / norms)
            delta = (d2 * factors).view(B, N, *x0.shape[1:])
        else:
            raise ValueError(f"Unsupported eval norm: {eval_norm}")

    x_adv = torch.clamp(x0.unsqueeze(1) + delta, 0.0, 1.0).detach()

    stats = None
    if return_stats:
        delta_proj = (x_adv - x0.unsqueeze(1)).reshape(B * N, -1)
        D_proxy = delta_proj.norm(p=2, dim=1).mean()
        stats = {
            "D_proxy":    D_proxy.detach(),
            # pi_entropy could be computed via gmm.forward() but skipped here
            # to avoid an extra forward pass; kept at 0.0 for API consistency.
            "pi_entropy": torch.tensor(0.0, device=x.device),
        }

    return x_adv, stats


# ------------------------------------------------------------------
#                    Diagnostic utilities
# ------------------------------------------------------------------

@torch.no_grad()
def check_mode_collapse(gmm, loader, device, num_batches=10):
    """Check mode collapse by sampling from loader."""
    gmm.eval()
    try:
        pi_distributions = []

        for i, (x, y, _) in enumerate(loader):
            if i >= num_batches:
                break
            x, y = x.to(device), y.to(device)

            out = gmm.forward(x=x, y=y)
            pi_logits = out['cache']['pi_logits']
            pi_probs = F.softmax(pi_logits, dim=-1)
            pi_distributions.append(pi_probs.cpu())

        all_pi = torch.cat(pi_distributions, dim=0)
        mean_pi = all_pi.mean(dim=0)
        max_pi = mean_pi.max().item()
        min_pi = mean_pi.min().item()
        std_pi = mean_pi.std().item()
        entropy = -(mean_pi * torch.log(mean_pi + 1e-8)).sum().item()
        max_entropy = np.log(gmm.K)

        print(f"\n{'='*60}")
        print(f"MODE COLLAPSE CHECK (K={gmm.K})")
        print(f"{'='*60}")
        print(f"Average π per component: {mean_pi.numpy()}")
        print(f"Max π: {max_pi:.4f} | Min π: {min_pi:.4f} | Std: {std_pi:.4f}")
        print(f"Entropy: {entropy:.4f} / {max_entropy:.4f} ({entropy/max_entropy*100:.1f}%)")

        if max_pi > 0.5:
            print(f"WARNING: Potential mode collapse!")
        elif std_pi > 0.15:
            print(f"WARNING: High variance in usage")
        else:
            print(f"Component usage looks balanced")
        print(f"{'='*60}\n")

        return {
            'mean_pi': mean_pi.numpy(),
            'max_pi': max_pi,
            'min_pi': min_pi,
            'std_pi': std_pi,
            'entropy': entropy,
            'entropy_ratio': entropy / max_entropy
        }
    finally:
        # Always restore training mode, even if an exception occurs
        gmm.train()
