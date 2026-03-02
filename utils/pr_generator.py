# pr_generator.py - Probabilistic / Bayesian adversarial training loss
#
# Requirements:
#   torch >= 2.0
#
# Supports:
#   - Beta-Mixed Loss (CE + soft 0-1 margin surrogate)
#   - PR / Bayesian-style perturbation generator (placeholder)

import torch
import torch.nn.functional as F

# Soft 0-1 margin surrogate
def _soft01_margin_surrogate(logits, y, kappa=1.0):
    """
    Soft 0-1 surrogate based on multiclass margin:
        margin = logit_y - max_{j!=y} logit_j
        soft01 = sigmoid(-margin / kappa)
    Args:
        output logits -> logits: (B, C)
        class -> y: (B,)
        margin (CW-like loss) -> kappa: > 0
    Returns:
        soft01: (B,)
        margin: (B,)
    """
    # Basic validation
    if kappa <= 0:
        raise ValueError("kappa must be > 0")

    # logit of true class
    f_y = logits.gather(1, y.view(-1, 1)).squeeze(1)

    # max logit over incorrect classes
    tmp = logits.clone()
    tmp[torch.arange(logits.size(0), device=logits.device), y] = -1e9
    f_other = tmp.max(dim=1).values

    margin = f_y - f_other
    soft01 = torch.sigmoid(-margin / kappa)

    return soft01, margin


def pr_generator(model, x, y, epsilon=8/255, norm="linf", # general args
                beta_mix=0.5, kappa=1.0, # loss args
                K=2, sigma_list=None, fisher_damping=1e-4, tau=1.0, # posterior GMM args
                num_samples=10, noise_scale=1.0, # sampling args
                ):
    """
    Random-sampling PR/Bayesian perturbation generator:
      1) compute gradient: g = ∇_x ℓ'_{β,κ} 
      2) Fisher diag to approximate the Hessian: A = g^2 + damping
      3) MoG posterior: update (μ_k, Σ_k, π_k)
      4) sample k ~ Cat(π_post), then δ ~ N(μ_k, Σ_k)
      5) project to L_inf or L2 ball and clamp to [0,1]
    Args:
        model: classifier
        x: clean inputs (B,C,H,W)
        y: labels (B,)
        epsilon: perturbation budget radius
        norm: "linf" or "l2"

        sigma_list: list of prior std per component (len K). If None, uses a 2-scale default.
        fisher_damping: damping added to Fisher diag for stability
        K: number of mixture components
        
        tau: temperature (>0)
        beta_mix: interpolation in [0,1]
        kappa: softness (>0) for margin surrogate
        
        noise_scale: >=0, scales posterior sampling noise (1.0 default, 0.0 -> deterministic)
    Returns:
        x_adv: perturbed inputs (B,N,C,H,W)
    """
    device = x.device
    B = x.size(0)

    # A default setting for sigma_list when not provided (2-scale prior)
    if sigma_list is None:
        # two-scale prior (good default)
        sigma_list = [epsilon / 2, epsilon]
        K = 2

    # Basic input validation
    if len(sigma_list) != K:
        raise ValueError(f"len(sigma_list) must equal K, got {len(sigma_list)} vs {K}")
    if tau <= 0:
        raise ValueError("tau must be > 0")
    if not (0.0 <= beta_mix <= 1.0):
        raise ValueError("beta_mix must be in [0,1]")
    if kappa <= 0:
        raise ValueError("kappa must be > 0")
    if noise_scale < 0:
        raise ValueError("noise_scale must be >= 0")

    # --------------------------------------------------
    # 1) Compute gradient: g = ∇_x ℓ'_{β,κ}
    # --------------------------------------------------
    model.eval()
    x0 = x.detach() # ensure no gradient history for safety w/o affecting original x

    x_req = x0.clone().detach().requires_grad_(True) # input w/ grad for generator
    logits = model(x_req)

    # CE loss
    ce = F.cross_entropy(logits, y, reduction="none")  # (B,)

    # margin surrogate (reuses helper to avoid duplication)
    soft01, _ = _soft01_margin_surrogate(logits, y, kappa)  # (B,)

    # mixed loss
    mixed = (1.0 - beta_mix) * ce + beta_mix * soft01
    loss = mixed.mean()

    # gradient
    g = torch.autograd.grad(loss, x_req)[0].detach()   # (B,C,H,W)
    
    # --------------------------------------------------
    # 2) Fisher diag A = g^2 + damping
    # --------------------------------------------------
    model.train()
    A = g.pow(2) + fisher_damping                       # (B,C,H,W)

    # flatten
    g_flat = g.view(B, -1)                              # (B,d)
    A_flat = A.view(B, -1)                              # (B,d)
    d = g_flat.size(1)

    # --------------------------------------------------
    # 3) Posterior params for each mode
    #    prior: μ0=0, Σ0 = σ_k^2 I  (diagonal isotropic)
    # --------------------------------------------------
    pi_ref = torch.full((K,), 1.0 / K, device=device)   # (K,)

    mu_all = torch.empty((B, K, d), device=device) # posterior means
    sig_all = torch.empty((B, K, d), device=device) # posterior variances (diagonal)
    logw = torch.empty((B, K), device=device) # log weights for numerical stability

    for k in range(K):
        sigma0 = float(sigma_list[k])
        prec0 = 1.0 / (sigma0 ** 2)                     # scalar

        # Σ_post = 1 / (prec0 + A/tau)
        sig_k = 1.0 / (prec0 + A_flat / tau)            # (B,d)

        # μ_post = Σ_post * (g/tau)
        mu_k = sig_k * (g_flat / tau)                   # (B,d)

        sig_all[:, k, :] = sig_k
        mu_all[:, k, :] = mu_k

        # simplified logZ (fast; can replace with full logdet form later)
        logZ_k = 0.5 * (mu_k * (g_flat / tau)).sum(dim=1)  # (B,)
        logw[:, k] = torch.log(pi_ref[k]) + logZ_k

    pi_post = torch.softmax(logw, dim=1)                # (B,K)

    # --------------------------------------------------
    # 4) Random sampling from mixture (vectorized)
    # --------------------------------------------------

    # (B, N) sample component indices for each sample in the batch
    N = num_samples
    B, K, d = mu_all.shape
    # sample k ~ Cat(π_post) for each of the N samples per input in the batch
    k_idx = torch.multinomial(pi_post, num_samples=N, replacement=True)  # (B, N)

    # gather μ and Σ (diagonal variances) for chosen components
    # mu_all:  (B, K, d)
    # k_idx:   (B, N) -> expand to (B, N, d) to gather along dim=1
    idx_exp = k_idx.unsqueeze(-1).expand(B, N, d)                        # (B, N, d)

    mu_sel  = torch.gather(mu_all,  dim=1, index=idx_exp)                # (B, N, d)
    sig_sel = torch.gather(sig_all, dim=1, index=idx_exp)                # (B, N, d)

    # sample δ ~ N(μ, Σ)
    eps = torch.randn_like(mu_sel)                                       # (B, N, d)
    delta_flat = mu_sel + noise_scale * torch.sqrt(sig_sel) * eps        # (B, N, d)

    # reshape back to image shape
    delta = delta_flat.view(B, N, *x0.shape[1:])                         # (B, N, C, H, W)
    
    # --------------------------------------------------
    # 5) Project onto budget + clamp image
    # --------------------------------------------------
        
    # x0: (B,C,H,W) -> (B,1,C,H,W)
    x0_exp = x0.unsqueeze(1)                                             # (B, 1, C, H, W)

    # project each delta onto the norm ball
    # L_inf: elementwise clamp
    if norm.lower() in ["linf", "l_inf", "l-infty", "l∞"]:
        delta = torch.clamp(delta, -epsilon, epsilon)

    # L2: per-sample scaling on (B*N, -1)
    elif norm.lower() in ["l2", "l_2"]:
        BN = B * N
        delta_2d = delta.reshape(BN, -1)
        norms = torch.norm(delta_2d, p=2, dim=1, keepdim=True).clamp_min(1e-12)
        factors = torch.minimum(torch.ones_like(norms), epsilon / norms)
        delta = (delta_2d * factors).view(B, N, *x0.shape[1:])

    else:
        raise ValueError(f"Unsupported norm: {norm}")

    # build adversarial inputs
    x_adv = torch.clamp(x0_exp + delta, 0.0, 1.0).detach()               # (B, N, C, H, W)
    return x_adv



