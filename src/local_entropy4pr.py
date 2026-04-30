import torch
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional, Tuple


# ============================================================
# Configs
# ============================================================

@dataclass
class EnergyConfig:
    psi_type: str = "softplus"
    psi_alpha: float = 10.0


@dataclass
class LangevinConfig:
    steps: int = 5
    step_size: float = 1e-2
    beta: float = 100
    noise_scale: float = 1.0


# ============================================================
# Particle State
# ============================================================

class ParticleState:
    """
    Minimal particle state.

    Maintains:
        particles: perturbation particles, shape (B, N, C, H, W)
        x_adv:     adversarial particles, shape (B, N, C, H, W)
        t:         current threshold
        step_idx:  update counter
    """

    def __init__(
            self, epsilon: float, 
            norm: str = "linf",
            num_particles: int = 8
        ):
        self.epsilon = float(epsilon)
        self.norm = norm.lower()
        self.num_particles = int(num_particles)

        if self.norm not in ["linf", "l2"]:
            raise ValueError("norm must be 'linf' or 'l2'.")

        self.particles: Optional[torch.Tensor] = None
        self.x_adv: Optional[torch.Tensor] = None

        self.t: Optional[torch.Tensor] = None
        self.step_idx: int = 0

    def init_particles(
            self, x: torch.Tensor,
            method: str = "uniform",
            scale: Optional[float] = None,
            warm_start: bool = False
            ) -> torch.Tensor:
        """
        Initialize or reuse perturbation particles.
        """
        B, C, H, W = x.shape
        shape = (B, self.num_particles, C, H, W)

        # reuse particles but it is not strictly correct! 
        # neet to refine this in the future.
        # generally not use it before refinement.
        if warm_start and self.particles is not None:
            if tuple(self.particles.shape) == shape:
                self.particles = self.particles.detach().clone().to(
                    device=x.device,
                    dtype=x.dtype,
                )
                self.update_x_adv(x)
                return self.particles

        # if sacle is none, set it to eps
        # however, in practice, we always set a scale
        if scale is None:
            scale = self.epsilon

        if method == "zero":
            particles = torch.zeros(
                shape, 
                device=x.device, 
                dtype=x.dtype
            )

        elif method == "uniform":
            if self.norm == "linf":
                particles = torch.empty(
                    shape, 
                    device=x.device, 
                    dtype=x.dtype
                )
                particles.uniform_(-self.epsilon, self.epsilon)
            else: # this is for L2
                particles = torch.randn(
                    shape, 
                    device=x.device, 
                    dtype=x.dtype
                ) * scale

        elif method == "gaussian":
            particles = torch.randn(
                shape, 
                device=x.device, 
                dtype=x.dtype
                ) * scale

        else:
            raise ValueError("method must be 'zero', 'uniform', or 'gaussian'.")

        self.particles = self.project(particles, x)
        self.update_x_adv(x)

        return self.particles

    def project(
            self, 
            particles: torch.Tensor, 
            x: torch.Tensor
        ) -> torch.Tensor:
        """
        Project particles into:
            ||delta|| <= epsilon
            0 <= x + delta <= 1
        """
        if self.norm == "linf":
            lower = torch.maximum(
                torch.full_like(x, -self.epsilon),
                -x,
            ).unsqueeze(1)

            upper = torch.minimum(
                torch.full_like(x, self.epsilon),
                1.0 - x,
            ).unsqueeze(1)

            particles = torch.max(torch.min(particles, upper), lower)

        elif self.norm == "l2":
            B, N = particles.shape[:2]
            flat = particles.reshape(B, N, -1)

            norms = flat.norm(dim=2, keepdim=True).clamp_min(1e-12)
            scale = (self.epsilon / norms).clamp_max(1.0)

            particles = (flat * scale).reshape_as(particles)
            particles = (x.unsqueeze(1) + particles).clamp(0.0, 1.0) - x.unsqueeze(1)

        return particles

    @torch.no_grad()
    def update_x_adv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Update adversarial particles:
            x_adv = x + particles
        """
        if self.particles is None:
            raise RuntimeError("Particles are not initialized.")

        B, N, C, H, W = self.particles.shape
        x_rep = x.unsqueeze(1).expand(B, N, C, H, W)

        self.x_adv = (x_rep + self.particles).clamp(0.0, 1.0)
        return self.x_adv

    def reset(self) -> None:
        self.particles = None
        self.x_adv = None
        self.t = None
        self.step_idx = 0


# ============================================================
# Basic functions
# ============================================================

def margin(
        logits: torch.Tensor, 
        y: torch.Tensor
    ) -> torch.Tensor:
    """
    Multiclass margin:
        m(x, y) = h_y(x) - max_{j != y} h_j(x)
    """
    f_y = logits.gather(1, y.view(-1, 1)).squeeze(1)

    wrong_logits = logits.clone()
    wrong_logits.scatter_(1, y.view(-1, 1), float("-inf"))
    f_other = wrong_logits.max(dim=1).values

    return f_y - f_other


def threshold_energy(
        margins: torch.Tensor, 
        t: torch.Tensor,
        cfg: EnergyConfig
    ) -> torch.Tensor:
    """
    Thresholded margin energy:
        E_t(m) = psi(m - t)
    """
    z = margins - t

    if cfg.psi_type == "softplus":
        return F.softplus(cfg.psi_alpha * z) / cfg.psi_alpha

    if cfg.psi_type == "hinge":
        return torch.relu(z)

    raise ValueError(f"Unknown psi_type: {cfg.psi_type}")


@torch.no_grad()
def compute_margins(
        *, 
        model, 
        x: torch.Tensor, 
        y: torch.Tensor,
        state: ParticleState
    ) -> torch.Tensor:
    """
    Compute margins for current particles.
    """
    if state.particles is None:
        raise RuntimeError("Particles are not initialized.")

    state.update_x_adv(x)

    B, N, C, H, W = state.x_adv.shape
    y_rep = y.unsqueeze(1).expand(B, N).reshape(-1)

    logits = model(state.x_adv.reshape(B * N, C, H, W))
    margins = margin(logits, y_rep).view(B, N)

    return margins


# ============================================================
# Threshold strategies
# ============================================================

def fixed_threshold_update(
        *, 
        margins: torch.Tensor, 
        state: ParticleState, 
        t: float = 0.0
    ) -> torch.Tensor:
    """
    Fixed threshold:
        t_curr = t
    """
    B = margins.shape[0]

    t_curr = torch.full(
        (B,),
        float(t),
        device=margins.device,
        dtype=margins.dtype,
    )

    state.t = t_curr.detach()
    return t_curr


def adaptive_threshold_update(
        *, 
        margins: torch.Tensor, 
        state: ParticleState,
        t0: float,
        t_floor: float = 0.0,
        q: float = 0.4,
        delta_min: float = 0.01,
        decay: float = 0.995
    ) -> torch.Tensor:
    """
    Adaptive threshold:
        t_next = min(
            t_prev - delta_min, = t_1
            quantile_q(margins), = t_2
            t0 * decay^(step_idx + 1) = t_3
        )
        t_next = max(t_floor, t_next)
    """
    B = margins.shape[0]
    device = margins.device
    dtype = margins.dtype

    if state.t is not None and state.t.shape == (B,):
        t_prev = state.t.to(device=device, dtype=dtype) # t from last step
    else:
        t_prev = torch.full((B,), float(t0), device=device, dtype=dtype) # t0 for first step

    t1 = t_prev - delta_min
    t2 = torch.quantile(margins.detach(), q=q, dim=1) # (B,)

    _t3 = max(float(t_floor), float(t0) * (decay ** (state.step_idx + 1)))
    t3 = torch.full((B,), _t3, device=device, dtype=dtype)

    t_next = torch.minimum(t1, t2)
    t_next = torch.minimum(t_next, t3)

    t_floor_tensor = torch.full((B,), float(t_floor), device=device, dtype=dtype)
    t_next = torch.maximum(t_next, t_floor_tensor)

    state.t = t_next.detach()
    return t_next


# ============================================================
# Scope strategies
# ============================================================

def fixed_scope(
        *,
        t_curr: torch.Tensor,
        gamma: float = 1.0,
    ) -> torch.Tensor:
    """
    Fixed scope:
        gamma_curr = gamma
    """
    return torch.full_like(t_curr, float(gamma))


def dynamic_scope(
        *,
        t_curr: torch.Tensor,
        gamma_min: float = 0.1,
        gamma_max: float = 10.0,
        t0: float = 1.0,
        t_floor: float = 0.0,
    ) -> torch.Tensor:
    """
    Dynamic scope:
        gamma(t) = gamma_min
                 + (gamma_max - gamma_min)
                   * (t0 - t) / (t0 - t_floor + eps)
    """
    gamma_curr = gamma_min + (gamma_max - gamma_min) * (
        float(t0) - t_curr) / ( float(t0) - float(t_floor) + 1e-8)

    return gamma_curr.clamp(min=gamma_min, max=gamma_max)


# ============================================================
# Langevin update
# ============================================================

def langevin_update_local_entropy(
        *,
        state: ParticleState,
        model,
        x: torch.Tensor,
        y: torch.Tensor,
        t_curr: torch.Tensor,
        gamma_curr: torch.Tensor,
        energy_cfg: EnergyConfig,
        cfg: LangevinConfig,
    ) -> ParticleState:
    """
    Explicit local-entropy Langevin update.
    Energy:
        E_t(delta) = psi(m(x + delta, y) - t)
    Localization:
        gamma / (2d) * ||delta - delta_0||^2

    Gradient:
        grad = grad_delta E_t(delta)
               + gamma / d * (delta - delta_0)
    Update:
        delta <- Proj[
            delta - eta * grad + sqrt(2 eta / beta) * noise
        ]
    """
    if state.particles is None:
        raise RuntimeError("Particles are not initialized.")

    particles = state.particles
    anchor = particles.detach().clone()

    B, N, C, H, W = particles.shape
    dim = C * H * W

    x_rep = x.unsqueeze(1).expand(B, N, C, H, W)
    y_rep = y.unsqueeze(1).expand(B, N).reshape(-1)

    noise_std = cfg.noise_scale * (2.0 * cfg.step_size / cfg.beta) ** 0.5

    for _ in range(cfg.steps):
        particles_req = particles.detach().requires_grad_(True)

        # Forward
        x_adv = (x_rep + particles_req).clamp(0.0, 1.0)
        logits = model(x_adv.reshape(B * N, C, H, W))

        margins = margin(logits, y_rep).view(B, N)

        # Energy
        t_expand = t_curr.view(B, 1).expand(B, N)

        energy = threshold_energy(
            margins=margins,
            t=t_expand,
            cfg=energy_cfg,
        )

        # Gradient of energy part
        grad_energy = torch.autograd.grad(
            energy.sum(),
            particles_req,
            only_inputs=True,
        )[0]

        # Explicit localization gradient
        gamma_expand = gamma_curr.view(B, 1, 1, 1, 1)
        grad_local = gamma_expand * (particles_req - anchor)

        grad = grad_energy + grad_local

        # Langevin update
        noise = torch.randn_like(particles_req)

        particles_next = particles_req - cfg.step_size * grad + noise_std * noise
        particles = state.project(particles_next.detach(), x)

    state.particles = particles.detach()
    state.update_x_adv(x)
    state.step_idx += 1

    return state


# ============================================================
# Outer TRADES-style loss
# ============================================================

def local_entropy_trades_loss(
        *,
        model,
        x: torch.Tensor,
        y: torch.Tensor,
        state: ParticleState,
        criterion,
        beta_outer: float = 6.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    TRADES-style outer loss using state.x_adv.

    L = CE(f(x), y)
        + beta_outer * E_delta KL(
              p_clean(. | x) || p_adv(. | x + delta)
          )

    Returns:
        loss:        scalar tensor, the combined outer loss.
        logits_adv:  (B*N, num_classes) logits on adversarial particles.
                     Returned so the training loop can reuse them for
                     accuracy without an extra forward pass.
    """
    if state.x_adv is None:
        raise RuntimeError("state.x_adv is None. Run particle update first.")

    x_adv_particles = state.x_adv
    B, N, C, H, W = x_adv_particles.shape

    logits_clean = model(x)
    loss_natural = criterion(logits_clean, y)

    with torch.no_grad():
        p_clean = F.softmax(logits_clean, dim=1)

    logits_adv = model(x_adv_particles.reshape(B * N, C, H, W))
    log_p_adv = F.log_softmax(logits_adv, dim=1)

    p_clean_rep = p_clean.unsqueeze(1).expand(B, N, -1)
    p_clean_rep = p_clean_rep.reshape(B * N, -1)

    loss_robust = F.kl_div(
        log_p_adv,
        p_clean_rep,
        reduction="batchmean",
    )

    loss = loss_natural + beta_outer * loss_robust

    return loss, logits_adv.detach()