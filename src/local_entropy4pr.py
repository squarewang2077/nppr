# region_generator.py - Region-aware / local-Gibbs perturbation generator
#
# Requirements:
#   torch >= 2.0
#
# Supports:
#   - single particle population (empirical perturbation distribution)
#   - adaptive threshold t_k
#   - dynamic scope gamma_k linked to t_k
#   - local-Gibbs-guided projected Langevin dynamics
#
# Notes:
#   1) This file implements only the INNER perturbation generator.
#   2) The OUTER objective (e.g. KL regularization / TRADES-like training)
#      should be handled in the training loop.
#   3) State dict can be used for warm-start across iterations.

import torch
import torch.nn.functional as F


# --------------------------------------------------
# Margin
# --------------------------------------------------

def _margin(logits, y):
    """
    Multiclass margin:
        m(x, y) = h_y(x) - max_{j!=y} h_j(x)

    Args:
        logits: (N, C)
        y:      (N,)

    Returns:
        margin: (N,)
    """
    f_y = logits.gather(1, y.view(-1, 1)).squeeze(1)

    tmp = logits.clone()
    tmp[torch.arange(logits.size(0), device=logits.device), y] = -1e9
    f_other = tmp.max(dim=1).values

    margin = f_y - f_other
    return margin


# --------------------------------------------------
# Thresholded energy E_t(u; x, y) = psi(m(x+u,y)-t)
# --------------------------------------------------

def _threshold_energy(margin, t, psi_type="softplus", psi_alpha=10.0):
    """
    Thresholded energy:
        E_t = psi(margin - t)

    Low energy corresponds to margin < t.

    Args:
        margin:    (...,)
        t:         broadcastable to margin
        psi_type:  "softplus" or "hinge"
        psi_alpha: softness parameter for softplus

    Returns:
        energy: (...,)
    """
    z = margin - t

    if psi_type == "softplus":
        # smooth version, preferred for Langevin
        energy = F.softplus(psi_alpha * z) / psi_alpha
    elif psi_type == "hinge":
        energy = torch.relu(z)
    else:
        raise ValueError(f"Unsupported psi_type: {psi_type}")

    return energy


# --------------------------------------------------
# Projection onto perturbation set
# --------------------------------------------------

def _project_delta(delta, epsilon, norm="linf"):
    """
    Project perturbations onto L_inf or L2 ball.

    Args:
        delta:   (..., C, H, W)
        epsilon: radius
        norm:    "linf" or "l2"

    Returns:
        projected delta
    """
    if norm.lower() in ["linf", "l_inf", "l-infty", "l∞"]:
        return delta.clamp(-epsilon, epsilon)

    elif norm.lower() in ["l2", "l_2"]:
        orig_shape = delta.shape
        flat = delta.reshape(delta.size(0), -1)
        norms = torch.norm(flat, p=2, dim=1, keepdim=True).clamp_min(1e-12)
        factors = torch.minimum(torch.ones_like(norms), epsilon / norms)
        flat = flat * factors
        return flat.view(orig_shape)

    else:
        raise ValueError(f"Unsupported norm: {norm}")


def _clamp_x_adv(x, delta):
    """
    Build adversarial input and clamp to [0,1].
    x:     (B, C, H, W)
    delta: (B, M, C, H, W)

    Returns:
        x_adv: (B, M, C, H, W)
    """
    return torch.clamp(x.unsqueeze(1) + delta, 0.0, 1.0)


# --------------------------------------------------
# Threshold update: minimum-of-three
# --------------------------------------------------

def _update_threshold(
    margins,              # (B, M)
    t_prev,               # (B,)
    t0,                   # scalar or (B,)
    step_idx,
    q=0.4,
    delta_min=0.01,
    t_floor=0.0,
    tau_decay=0.995,
):
    """
    Adaptive threshold update:
        t_candidate = q-quantile of current particle margins
        t_progress  = t_prev - delta_min
        t_schedule  = max(t_floor, t0 * tau_decay^(step_idx+1))
        t_next      = min(t_progress, t_candidate, t_schedule)
        t_next      = max(t_floor, t_next)

    Returns:
        t_next:      (B,)
        t_candidate: (B,)
        t_schedule:  (B,)
    """
    device = margins.device
    B = margins.size(0)

    # PyTorch quantile works on float tensors
    t_candidate = torch.quantile(margins, q=q, dim=1)  # (B,)

    # progress term
    t_progress = t_prev - delta_min

    # schedule term
    if not torch.is_tensor(t0):
        t0_tensor = torch.full((B,), float(t0), device=device, dtype=margins.dtype)
    else:
        t0_tensor = t0.to(device=device, dtype=margins.dtype).view(B)

    t_schedule = torch.maximum(
        torch.full((B,), float(t_floor), device=device, dtype=margins.dtype),
        t0_tensor * (tau_decay ** (step_idx + 1))
    )

    t_next = torch.minimum(torch.minimum(t_progress, t_candidate), t_schedule)
    t_next = torch.maximum(
        torch.full((B,), float(t_floor), device=device, dtype=margins.dtype),
        t_next
    )

    return t_next, t_candidate, t_schedule


# --------------------------------------------------
# Scope update gamma = Gamma(t)
# --------------------------------------------------

def _update_scope(
    t_next,               # (B,)
    t0,
    gamma_min=0.1,
    gamma_max=10.0,
    t_floor=0.0,
    eps=1e-8,
):
    """
    Dynamic scope:
        gamma = gamma_min + (gamma_max - gamma_min) * (t0 - t) / (t0 - t_floor + eps)

    Larger t -> smaller gamma -> broader exploration
    Smaller t -> larger gamma -> more localized refinement

    Returns:
        gamma_next: (B,)
    """
    device = t_next.device
    B = t_next.size(0)

    if not torch.is_tensor(t0):
        t0_tensor = torch.full((B,), float(t0), device=device, dtype=t_next.dtype)
    else:
        t0_tensor = t0.to(device=device, dtype=t_next.dtype).view(B)

    gamma_next = (
        gamma_min
        + (gamma_max - gamma_min) * (t0_tensor - t_next) / (t0_tensor - t_floor + eps)
    )
    gamma_next = gamma_next.clamp(min=gamma_min, max=gamma_max)
    return gamma_next


# --------------------------------------------------
# Particle initialization
# --------------------------------------------------

def _init_particles(x, num_particles, epsilon, norm="linf", init="uniform"):
    """
    Initialize delta particles inside perturbation set B.

    Args:
        x:             (B, C, H, W)
        num_particles: M
        epsilon:       perturbation budget
        norm:          "linf" or "l2"
        init:          "uniform", "zero", or "normal"

    Returns:
        particles: (B, M, C, H, W)
    """
    device = x.device
    B = x.size(0)
    shape = (B, num_particles, *x.shape[1:])

    if init == "zero":
        delta = torch.zeros(shape, device=device, dtype=x.dtype)

    elif init == "uniform":
        if norm.lower() in ["linf", "l_inf", "l-infty", "l∞"]:
            delta = torch.empty(shape, device=device, dtype=x.dtype).uniform_(-epsilon, epsilon)
        elif norm.lower() in ["l2", "l_2"]:
            delta = torch.randn(shape, device=device, dtype=x.dtype)
            delta = _project_delta(delta.view(B * num_particles, *x.shape[1:]), epsilon, norm="l2")
            delta = delta.view(shape)
        else:
            raise ValueError(f"Unsupported norm: {norm}")

    elif init == "normal":
        delta = 0.1 * epsilon * torch.randn(shape, device=device, dtype=x.dtype)
        delta = _project_delta(delta.view(B * num_particles, *x.shape[1:]), epsilon, norm=norm)
        delta = delta.view(shape)

    else:
        raise ValueError(f"Unsupported init mode: {init}")

    return delta


# --------------------------------------------------
# Main region generator
# --------------------------------------------------

def region_generator(
    model,
    x,
    y,
    epsilon=8/255,
    norm="linf",

    # particle population
    num_particles=8,
    inner_steps=3,
    init_mode="uniform",

    # threshold update
    t0=0.5,
    q=0.4,
    delta_min=0.01,
    t_floor=0.0,
    tau_decay=0.995,
    step_idx=0,

    # scope update
    gamma_min=0.1,
    gamma_max=10.0,

    # thresholded energy
    psi_type="softplus",
    psi_alpha=10.0,

    # Langevin
    eta_delta=1e-2,
    beta=1.0,
    noise_scale=1.0,

    # stateful warm start
    state=None,

    # return statistics
    return_stats=True,
):
    """
    Region-aware local-Gibbs perturbation generator.

    Pipeline:
      1) maintain a single particle population {delta_i}
      2) compute margins of current particles
      3) update threshold t_k by minimum-of-three rule
      4) update scope gamma_k using Gamma(t_k)
      5) define local Gibbs potential:
            U(u) = E_t(u; x, y) + gamma * 0.5||u - z_i||^2
         with anchor z_i = delta_i^(k)
      6) run projected Langevin dynamics for each particle
      7) return x_adv = x + updated particles, plus updated state

    Args:
        model: classifier
        x: clean inputs, shape (B, C, H, W)
        y: labels, shape (B,)
        epsilon: perturbation budget
        norm: "linf" or "l2"

        num_particles: number of particles M
        inner_steps: number of Langevin steps L
        init_mode: particle init mode if state is None

        t0: initial threshold
        q: quantile level
        delta_min: minimum decrease in threshold
        t_floor: terminal threshold floor
        tau_decay: global schedule coefficient
        step_idx: outer iteration index for schedule

        gamma_min, gamma_max: scope bounds

        psi_type: "softplus" or "hinge"
        psi_alpha: softplus steepness

        eta_delta: inner step size
        beta: inverse temperature
        noise_scale: scale on Langevin noise

        state: optional dict containing:
            - "particles": (B, M, C, H, W)
            - "t": (B,)
          for warm-start

    Returns:
        x_adv:   (B, M, C, H, W)
        stats:   dict or None
        state:   updated state dict
        p_bar:   (B, C) particle-averaged softmax, for outer KL objective
    """
    device = x.device
    B = x.size(0)

    if noise_scale < 0:
        raise ValueError("noise_scale must be >= 0")
    if inner_steps <= 0:
        raise ValueError("inner_steps must be >= 1")
    if num_particles <= 0:
        raise ValueError("num_particles must be >= 1")

    # --------------------------------------------------
    # 0) Initialize / warm-start particles and threshold
    # --------------------------------------------------
    if state is not None and "particles" in state:
        particles = state["particles"].detach().clone()   # (B, M, C, H, W)
        if particles.shape[:2] != (B, num_particles):
            raise ValueError(
                f"state['particles'] shape mismatch: expected (B={B}, M={num_particles}, ...), "
                f"got {tuple(particles.shape)}"
            )
    else:
        particles = _init_particles(x, num_particles, epsilon, norm=norm, init=init_mode)

    if state is not None and "t" in state:
        t_prev = state["t"].detach().clone().to(device=device, dtype=x.dtype).view(B)
    else:
        t_prev = torch.full((B,), float(t0), device=device, dtype=x.dtype)

    # --------------------------------------------------
    # 1) Compute margins of current particles
    # --------------------------------------------------
    _was_training = model.training
    model.eval()
    with torch.no_grad():
        x_adv_curr = _clamp_x_adv(x, particles)                          # (B, M, C, H, W)
        logits_curr = model(x_adv_curr.view(B * num_particles, *x.shape[1:]))  # (B*M, C)
        y_rep = y.unsqueeze(1).expand(B, num_particles).reshape(-1)     # (B*M,)
        margins_curr = _margin(logits_curr, y_rep).view(B, num_particles)  # (B, M)

    # --------------------------------------------------
    # 2) Threshold update
    # --------------------------------------------------
    t_next, t_candidate, t_schedule = _update_threshold(
        margins=margins_curr,
        t_prev=t_prev,
        t0=t0,
        step_idx=step_idx,
        q=q,
        delta_min=delta_min,
        t_floor=t_floor,
        tau_decay=tau_decay,
    )  # all shape (B,)

    # --------------------------------------------------
    # 3) Scope update
    # --------------------------------------------------
    gamma_next = _update_scope(
        t_next=t_next,
        t0=t0,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        t_floor=t_floor,
        eps=1e-8,
    )  # (B,)

    # --------------------------------------------------
    # 4) Local-Gibbs-guided projected Langevin dynamics
    # --------------------------------------------------
    anchors = particles.detach().clone()  # z_i = delta_i^(k)

    for _ in range(inner_steps):
        particles_req = particles.detach().clone().requires_grad_(True)   # (B,M,C,H,W)

        # x + delta
        x_adv = _clamp_x_adv(x, particles_req)                            # (B,M,C,H,W)
        logits = model(x_adv.view(B * num_particles, *x.shape[1:]))       # (B*M,C)
        margins = _margin(logits, y_rep).view(B, num_particles)           # (B,M)

        # thresholded energy
        t_expand = t_next.view(B, 1).expand(B, num_particles)             # (B,M)
        energy = _threshold_energy(
            margin=margins,
            t=t_expand,
            psi_type=psi_type,
            psi_alpha=psi_alpha,
        )  # (B,M)

        # localization term: 0.5 ||u - z_i||^2
        diff = particles_req - anchors
        local_term = 0.5 * diff.view(B, num_particles, -1).pow(2).sum(dim=2)  # (B,M)

        # local Gibbs potential
        gamma_expand = gamma_next.view(B, 1).expand(B, num_particles)     # (B,M)
        potential = energy + gamma_expand * local_term                    # (B,M)

        # gradient
        loss = potential.sum()
        grad = torch.autograd.grad(loss, particles_req)[0].detach()       # (B,M,C,H,W)

        # Langevin noise
        noise = torch.randn_like(particles_req) * noise_scale
        particles = (
            particles_req.detach()
            - eta_delta * grad
            + (2.0 * eta_delta / beta) ** 0.5 * noise
        )

        # project back to perturbation set
        particles = _project_delta(
            particles.view(B * num_particles, *x.shape[1:]),
            epsilon=epsilon,
            norm=norm,
        ).view(B, num_particles, *x.shape[1:])

    # --------------------------------------------------
    # 5) Build x_adv and empirical predictive summary
    # --------------------------------------------------
    x_adv = _clamp_x_adv(x, particles)                                   # (B,M,C,H,W)

    # Optional predictive summary (not used directly here, but useful for outer KL regularization)
    logits_final = model(x_adv.view(B * num_particles, *x.shape[1:]))    # (B*M,C)
    probs_final = F.softmax(logits_final, dim=1).view(B, num_particles, -1)  # (B,M,C)
    p_bar = probs_final.mean(dim=1)                                      # (B,C)

    # --------------------------------------------------
    # 6) Save updated state
    # --------------------------------------------------
    state_out = {
        "particles": particles.detach(),
        "t": t_next.detach(),
    }

    # --------------------------------------------------
    # 7) Return stats if requested
    # --------------------------------------------------
    stats = None
    if return_stats:
        # final margins after inner dynamics
        final_margins = _margin(logits_final, y_rep).view(B, num_particles)

        # particle spread around anchor / mean
        spread_anchor = (particles - anchors).view(B, num_particles, -1).norm(p=2, dim=2).mean()

        particle_mean = particles.mean(dim=1, keepdim=True)  # (B,1,C,H,W)
        spread_center = (particles - particle_mean).view(B, num_particles, -1).norm(p=2, dim=2).mean()

        stats = {
            # threshold stats
            "t_prev_mean": t_prev.mean().detach(),
            "t_candidate_mean": t_candidate.mean().detach(),
            "t_schedule_mean": t_schedule.mean().detach(),
            "t_next_mean": t_next.mean().detach(),

            # scope stats
            "gamma_mean": gamma_next.mean().detach(),
            "gamma_min_batch": gamma_next.min().detach(),
            "gamma_max_batch": gamma_next.max().detach(),

            # margin stats
            "margin_curr_mean": margins_curr.mean().detach(),
            "margin_final_mean": final_margins.mean().detach(),
            "margin_final_min": final_margins.min().detach(),
            "margin_final_q": torch.quantile(final_margins, q=q).detach(),

            # energy / geometry stats
            "p_bar_entropy": (-p_bar * torch.log(p_bar.clamp_min(1e-12))).sum(dim=1).mean().detach(),
            "spread_anchor": spread_anchor.detach(),
            "spread_center": spread_center.detach(),
        }

    if _was_training:
        model.train()

    return x_adv.detach(), stats, state_out, p_bar