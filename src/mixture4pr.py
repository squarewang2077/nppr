"""
mixture4pr.py — Mixed-Noise Perturbation Framework for Probabilistic Robustness
================================================================================

Overview
--------
This module provides a research-oriented framework for learning adversarial
perturbations as samples from a trainable mixture-of-noise distribution.
A latent vector is drawn from the mixture, decoded into pixel space, and
projected onto an Lp-norm budget ball.  The distribution is trained via a
PR-loss (Probabilistic Robustness loss) that maximises worst-case fooling rate
over the mixture.

Architecture (top-down)
-----------------------
PerturbationModel  (nn.Module, inherits MixedNoiseDistribution)
│   Perturbation-specific logic: budget, up-sampler, save/load.
│   Delegates distribution work to the parent class.
│   Thin wrappers call the module-level pr_loss / evaluate_pr functions.
│
├── MixedNoiseDistribution  (nn.Module)
│   │   Pure distribution class.  Owns the component registry, regularizers,
│   │   and the ConditioningStrategy that wires conditioning signals.
│   │
│   ├── ConditioningStrategy  (ABC)
│   │   Encapsulates how pi / mu heads are built and how inputs are encoded.
│   │   Four concrete implementations:
│   │     UnconditionalStrategy  — free Parameters, no input conditioning
│   │     XStrategy              — image-feature conditioned (feat extractor)
│   │     YStrategy              — label conditioned (y embedding)
│   │     XYStrategy             — conditioned on both image features and label
│   │
│   └── NoiseComponent  (nn.Module, ABC)
│       One component family in the K-component mixture.
│       Four built-in implementations:
│         GaussianComponent       — diagonal / low-rank / full covariance
│         LaplaceComponent        — diagonal independent Laplace
│         UniformComponent        — symmetric uniform half-width
│         SaltAndPepperComponent  — per-dim spike at ±amplitude with prob p
│
├── PerturbationDecoder  (plain Python class, not nn.Module)
│   Decodes a flat latent vector into a shaped delta and projects it
│   onto the Lp budget ball.  Supports linf and l2 norms.
│
└── DistCache  (dataclass)
    Typed snapshot produced by forward(): pi_logits [B,K], mu [B,K,D],
    component_params.  Supports both attribute access (cache.mu) and
    legacy dict-style access (cache["mu"]).

Module-level functions
----------------------
pr_loss(model, x, y, classifier, ...)  → dict with keys loss/main/reg/pr
evaluate_pr(model, x, y, classifier, ...)  → PR scalar tensor
compute_pr(predictions, y, ...)  → PR scalar tensor (pure metric, no model)

Quick start
-----------
    from src.mixture4pr import MixedNoise4PR

    model = MixedNoise4PR(K=4, latent_dim=192, device="cuda")
    model.set_condition(cond_mode=None, num_cls=10)   # unconditional
    model.set_budget(norm="linf", eps=8/255)

    # training step
    result = model.pr_loss(x, y, classifier, num_samples=8)
    result["loss"].backward()

    # evaluation
    pr = model.evaluate_pr(x, y, classifier, num_samples=100)

    # custom component mix (K=2 Gaussian + K=2 Laplace)
    from src.mixture4pr import GaussianComponent, LaplaceComponent
    model = MixedNoise4PR(K=4, latent_dim=48, device="cuda")
    model.set_condition(cond_mode=None, num_cls=10, components=[
        GaussianComponent(K=2, latent_dim=48, device="cuda"),
        LaplaceComponent(K=2,  latent_dim=48, device="cuda"),
    ])

Backward compatibility
----------------------
- ``MixedNoise4PR`` is an alias for ``PerturbationModel``.
- ``set_condition(cond_mode=None/\"x\"/\"y\"/\"xy\", ...)`` still works.
- ``cache[\"pi_logits\"]`` dict-style access is preserved via ``DistCache.__getitem__``.
- State dict key paths are identical to pre-refactor checkpoints (inheritance,
  not composition).
- Checkpoint ``cond_mode`` field is preserved as ``strategy.mode_name``.
"""

import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from torch.distributions import (
    Categorical, Normal, Independent, Laplace,
    LowRankMultivariateNormal, MultivariateNormal,
)

### SECTION 1: Define the Noise Distribution ### 
# --------------------------------------------------
# NoiseComponent abstract base 
# --------------------------------------------------

class NoiseComponent(nn.Module, ABC):
    """
    Abstract base for a group of homogeneous noise distribution components.

    Each subclass handles ``K`` components of one distribution family (e.g.,
    all Gaussian, all Laplace).  A ``MixedNoise4PR`` model is composed of one
    or more ``NoiseComponent`` objects, each covering a contiguous slice of
    the total K mixture components.

    Subclass this to add new distribution families, then pass instances
    directly to ``set_condition(components=[...])``.

    For checkpoint loading support, also set the ``registry_name`` class
    attribute, implement ``get_config()``, and call
    ``register_component(registry_name, YourClass)``.

    Lifecycle
    ---------
    1. Construct: ``comp = MyComponent(K=3, latent_dim=48, device="cpu", ...)``
    2. ``set_condition`` calls one of:
       - ``comp.build_unconditional_params()``  — unconditional / y-only mode
       - ``comp.build_conditional_heads(hidden_dim)``  — x / xy mode
    3. Use: ``get_scale_params``, ``rsample``, ``log_prob``
    """

    registry_name: str = ""  # override in subclasses for checkpoint support

    def __init__(self, K: int, latent_dim: int, device):
        super().__init__()
        self.K = K
        self.latent_dim = latent_dim
        self.device = device

    def get_config(self) -> dict:
        """Return serialisable constructor kwargs (excluding K, latent_dim, device)."""
        return {}

    @abstractmethod
    def build_unconditional_params(self) -> None:
        """Create free ``nn.Parameter`` tensors for unconditional / y-only mode."""
        raise NotImplementedError

    @abstractmethod
    def build_conditional_heads(self, hidden_dim: int) -> None:
        """Create ``nn.Linear`` heads for conditional (x / xy) mode."""
        raise NotImplementedError

    @abstractmethod
    def get_scale_params(self, h: Optional[torch.Tensor], B: int) -> dict:
        """
        Compute scale/covariance params.

        Parameters
        ----------
        h : Tensor[B, hidden_dim] or None
            Shared-trunk output (None in unconditional / y-only mode).
        B : int
            Batch size.

        Returns
        -------
        dict of named tensors consumed by ``rsample`` and ``log_prob``.
        """
        raise NotImplementedError

    @abstractmethod
    def rsample(self, mu: torch.Tensor, scale_params: dict,
                num_samples: int) -> torch.Tensor:
        """
        Reparameterized sample.

        Parameters
        ----------
        mu : Tensor[B, K_type, D]
        scale_params : dict  (output of ``get_scale_params``)
        num_samples : int  (S)

        Returns
        -------
        Tensor[S, B, K_type, D]
        """
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, x: torch.Tensor, mu: torch.Tensor,
                 scale_params: dict) -> torch.Tensor:
        """
        Log-probability of ``x`` under each component.

        Parameters
        ----------
        x : Tensor[B, D]
        mu : Tensor[B, K_type, D]
        scale_params : dict

        Returns
        -------
        Tensor[B, K_type]
        """
        raise NotImplementedError


# --------------------------------------------------
#                   GaussianComponent 
# --------------------------------------------------

class GaussianComponent(NoiseComponent):
    """
    Gaussian components with diagonal, low-rank, or full covariance.

    Parameters
    ----------
    K : int
        Number of Gaussian components.
    latent_dim : int
    device : str or torch.device
    cov_type : "diag" | "lowrank" | "full"
    cov_rank : int
        Rank of low-rank factor (used only when cov_type="lowrank").
    logstd_bounds : (float, float)
        Clamp range for log-standard-deviation.
    """

    registry_name = "gaussian"

    def __init__(self, K: int, latent_dim: int, device,
                 cov_type: str = "diag",
                 cov_rank: int = 0,
                 logstd_bounds: tuple = (-3.0, 1.0)):
        super().__init__(K, latent_dim, device)
        self.cov_type = cov_type
        self.cov_rank = cov_rank
        self.logstd_bounds = logstd_bounds
        self._conditional = False

    def get_config(self) -> dict:
        return {
            "cov_type": self.cov_type,
            "cov_rank": self.cov_rank,
            "logstd_bounds": self.logstd_bounds,
        }

    def build_unconditional_params(self) -> None:
        self._conditional = False
        K, D = self.K, self.latent_dim
        if self.cov_type == "diag":
            self.log_sigma = nn.Parameter(torch.zeros(K, D, device=self.device))
        elif self.cov_type == "lowrank":
            self.log_sigma = nn.Parameter(torch.zeros(K, D, device=self.device))
            self.U = nn.Parameter(torch.zeros(K, D, self.cov_rank, device=self.device))
        elif self.cov_type == "full":
            self.L_raw = nn.Parameter(torch.zeros(K, D, D, device=self.device))
        else:
            raise ValueError(f"cov_type must be diag/lowrank/full, got '{self.cov_type}'")

    def build_conditional_heads(self, hidden_dim: int) -> None:
        self._conditional = True
        K, D = self.K, self.latent_dim
        if self.cov_type == "diag":
            self.logsig = nn.Linear(hidden_dim, K * D).to(self.device)
        elif self.cov_type == "lowrank":
            self.logsig = nn.Linear(hidden_dim, K * D).to(self.device)
            self.U = nn.Linear(hidden_dim, K * D * self.cov_rank).to(self.device)
        elif self.cov_type == "full":
            self.L = nn.Linear(hidden_dim, K * D * D).to(self.device)
        else:
            raise ValueError(f"cov_type must be diag/lowrank/full, got '{self.cov_type}'")

    # ── private helpers ───────────────────────────────────────────────────────

    def _get_log_std(self, h: Optional[torch.Tensor], B: int) -> torch.Tensor:
        K, D = self.K, self.latent_dim
        if self._conditional:
            log_std = self.logsig(h).view(B, K, D)
        else:
            log_std = self.log_sigma.unsqueeze(0).expand(B, K, D)
        lo, hi = self.logstd_bounds
        return torch.clamp(log_std, lo, hi)

    def _get_U_factor(self, h: Optional[torch.Tensor], B: int) -> torch.Tensor:
        K, D, r = self.K, self.latent_dim, self.cov_rank
        if self._conditional:
            return self.U(h).view(B, K, D, r)
        else:
            return self.U.unsqueeze(0).expand(B, K, D, r)

    def _get_cholesky(self, h: Optional[torch.Tensor], B: int) -> torch.Tensor:
        K, D = self.K, self.latent_dim
        if self._conditional:
            L_raw = self.L(h).view(B, K, D, D)
        else:
            L_raw = self.L_raw.unsqueeze(0).expand(B, K, D, D)
        tril_mask = torch.tril(torch.ones(D, D, device=self.device, dtype=torch.bool))
        L = torch.zeros_like(L_raw)
        L[..., tril_mask] = L_raw[..., tril_mask]
        diag_idx = torch.arange(D, device=self.device)
        L[..., diag_idx, diag_idx] = F.softplus(L[..., diag_idx, diag_idx]) + 1e-4
        return L

    # ── public interface ──────────────────────────────────────────────────────

    def get_scale_params(self, h: Optional[torch.Tensor], B: int) -> dict:
        if self.cov_type == "diag":
            return {"log_std": self._get_log_std(h, B)}
        elif self.cov_type == "lowrank":
            return {"log_std": self._get_log_std(h, B), "U": self._get_U_factor(h, B)}
        elif self.cov_type == "full":
            return {"L": self._get_cholesky(h, B)}

    def rsample(self, mu: torch.Tensor, scale_params: dict,
                num_samples: int) -> torch.Tensor:
        """Reparameterized sample. Returns [S, B, K, D]."""
        B, K, D = mu.shape

        if self.cov_type == "diag":
            std = torch.exp(scale_params["log_std"])           # [B, K, D]
            eps = torch.randn(num_samples, B, K, D, device=mu.device)
            return mu.unsqueeze(0) + std.unsqueeze(0) * eps    # [S, B, K, D]

        elif self.cov_type == "lowrank":
            std = torch.exp(scale_params["log_std"])           # [B, K, D]
            U = scale_params["U"]                              # [B, K, D, r]
            r = U.shape[-1]
            eta = torch.randn(num_samples, B, K, r, device=mu.device)
            eps_diag = torch.randn(num_samples, B, K, D, device=mu.device)
            lowrank_term = torch.matmul(U.unsqueeze(0), eta.unsqueeze(-1)).squeeze(-1)
            return mu.unsqueeze(0) + lowrank_term + std.unsqueeze(0) * eps_diag

        elif self.cov_type == "full":
            L = scale_params["L"]                              # [B, K, D, D]
            eps = torch.randn(num_samples, B, K, D, device=mu.device)
            return mu.unsqueeze(0) + torch.matmul(
                L.unsqueeze(0), eps.unsqueeze(-1)
            ).squeeze(-1)

    def log_prob(self, x: torch.Tensor, mu: torch.Tensor,
                 scale_params: dict) -> torch.Tensor:
        """Log probability of x under each component. Returns [B, K]."""
        x_exp = x.unsqueeze(1).expand_as(mu)                  # [B, K, D]
        if self.cov_type == "diag":
            std = torch.exp(scale_params["log_std"])
            return Independent(Normal(mu, std), 1).log_prob(x_exp)
        elif self.cov_type == "lowrank":
            std2 = torch.exp(2.0 * scale_params["log_std"])
            U = scale_params["U"]
            return LowRankMultivariateNormal(
                loc=mu, cov_factor=U, cov_diag=std2
            ).log_prob(x_exp)
        elif self.cov_type == "full":
            return MultivariateNormal(
                loc=mu, scale_tril=scale_params["L"]
            ).log_prob(x_exp)


# --------------------------------------------------
#                   LaplaceComponent 
# --------------------------------------------------

class LaplaceComponent(NoiseComponent):
    """
    Laplace components with independent diagonal scale per dimension.

    Parameters
    ----------
    K : int
        Number of Laplace components.
    latent_dim : int
    device : str or torch.device
    logscale_bounds : (float, float)
        Clamp range for log-scale.
    """

    registry_name = "laplace"

    def __init__(self, K: int, latent_dim: int, device,
                 logscale_bounds: tuple = (-3.0, 1.0)):
        super().__init__(K, latent_dim, device)
        self.logscale_bounds = logscale_bounds
        self._conditional = False

    def get_config(self) -> dict:
        return {"logscale_bounds": self.logscale_bounds}

    def build_unconditional_params(self) -> None:
        self._conditional = False
        self.log_scale = nn.Parameter(
            torch.zeros(self.K, self.latent_dim, device=self.device)
        )

    def build_conditional_heads(self, hidden_dim: int) -> None:
        self._conditional = True
        self.logscale_head = nn.Linear(
            hidden_dim, self.K * self.latent_dim
        ).to(self.device)

    def get_scale_params(self, h: Optional[torch.Tensor], B: int) -> dict:
        K, D = self.K, self.latent_dim
        if self._conditional:
            raw = self.logscale_head(h).view(B, K, D)
        else:
            raw = self.log_scale.unsqueeze(0).expand(B, K, D)
        lo, hi = self.logscale_bounds
        return {"log_scale": torch.clamp(raw, lo, hi)}

    def rsample(self, mu: torch.Tensor, scale_params: dict,
                num_samples: int) -> torch.Tensor:
        """Reparameterized Laplace sample via torch.distributions. Returns [S, B, K, D]."""
        scale = torch.exp(scale_params["log_scale"])           # [B, K, D]
        return Independent(Laplace(mu, scale), 1).rsample((num_samples,))

    def log_prob(self, x: torch.Tensor, mu: torch.Tensor,
                 scale_params: dict) -> torch.Tensor:
        """Log probability of x under each Laplace component. Returns [B, K]."""
        log_scale = scale_params["log_scale"]                  # [B, K, D]
        scale = torch.exp(log_scale)
        x_exp = x.unsqueeze(1).expand_as(mu)                   # [B, K, D]
        log_p = -log_scale - (x_exp - mu).abs() / scale - math.log(2)
        return log_p.sum(dim=-1)                               # [B, K]


# --------------------------------------------------
#                   UniformComponent 
# --------------------------------------------------

class UniformComponent(NoiseComponent):
    """
    Uniform components with independent symmetric half-width per dimension.

    Each component models ``x_d ~ Uniform(mu_d - w_d, mu_d + w_d)`` where
    ``w_d = exp(log_half_width_d)`` is learnable.

    Parameters
    ----------
    K : int
        Number of Uniform components.
    latent_dim : int
    device : str or torch.device
    log_half_width_bounds : (float, float)
        Clamp range for log-half-width.  Defaults keep widths in [exp(-3), exp(1)].
    """

    registry_name = "uniform"

    def __init__(self, K: int, latent_dim: int, device,
                 log_half_width_bounds: tuple = (-3.0, 1.0)):
        super().__init__(K, latent_dim, device)
        self.log_half_width_bounds = log_half_width_bounds
        self._conditional = False

    def get_config(self) -> dict:
        return {"log_half_width_bounds": self.log_half_width_bounds}

    def build_unconditional_params(self) -> None:
        self._conditional = False
        self.log_half_width = nn.Parameter(
            torch.zeros(self.K, self.latent_dim, device=self.device)
        )

    def build_conditional_heads(self, hidden_dim: int) -> None:
        self._conditional = True
        self.log_half_width_head = nn.Linear(
            hidden_dim, self.K * self.latent_dim
        ).to(self.device)

    def get_scale_params(self, h: Optional[torch.Tensor], B: int) -> dict:
        K, D = self.K, self.latent_dim
        if self._conditional:
            raw = self.log_half_width_head(h).view(B, K, D)
        else:
            raw = self.log_half_width.unsqueeze(0).expand(B, K, D)
        lo, hi = self.log_half_width_bounds
        return {"log_half_width": torch.clamp(raw, lo, hi)}

    def rsample(self, mu: torch.Tensor, scale_params: dict,
                num_samples: int) -> torch.Tensor:
        """Reparameterized Uniform sample via torch.distributions. Returns [S, B, K, D]."""
        from torch.distributions import Uniform
        half_width = torch.exp(scale_params["log_half_width"])  # [B, K, D]
        low  = mu - half_width
        high = mu + half_width
        return Independent(Uniform(low, high), 1).rsample((num_samples,))

    def log_prob(self, x: torch.Tensor, mu: torch.Tensor,
                 scale_params: dict) -> torch.Tensor:
        """Log probability of x under each Uniform component. Returns [B, K]."""
        from torch.distributions import Uniform
        half_width = torch.exp(scale_params["log_half_width"])  # [B, K, D]
        low  = mu - half_width
        high = mu + half_width
        x_exp = x.unsqueeze(1).expand_as(mu)                    # [B, K, D]
        log_p = Independent(Uniform(low, high), 1).log_prob(x_exp)
        return log_p                                             # [B, K]


# --------------------------------------------------
#               SaltAndPepperComponent
# --------------------------------------------------

class SaltAndPepperComponent(NoiseComponent):
    """
    Salt-and-pepper noise components.

    Each latent dimension independently takes one of three outcomes centred on
    the component mean ``mu_d``:

      * **salt**   (+amplitude) with probability  p/2
      * **pepper** (-amplitude) with probability  p/2
      * **clean**  (no change)  with probability  1 - p

    Both ``p`` (corruption rate) and ``amplitude`` are learnable, per-component
    and per-dimension, parameterised as::

        p   = sigmoid(logit_p)          ∈ (0, 1)
        amp = exp(log_amplitude)        > 0

    Sampling uses a Gumbel-Softmax relaxation over the three discrete outcomes
    so that gradients flow through the amplitude and probability parameters
    during training.  Temperature controls the sharpness of the relaxation.

    Parameters
    ----------
    K : int
        Number of salt-and-pepper components.
    latent_dim : int
    device : str or torch.device
    logit_p_bounds : (float, float)
        Clamp range for the logit of the corruption probability.
        Default ``(-4.0, 4.0)`` keeps p safely in (0.018, 0.982).
    log_amplitude_bounds : (float, float)
        Clamp range for the log-amplitude.
        Default ``(-3.0, 1.0)`` keeps amplitude in (exp(-3), exp(1)).
    temperature : float
        Gumbel-Softmax temperature for ``rsample``.  Lower values produce
        harder (more spike-like) samples; higher values produce softer
        convex combinations.  Default ``0.5``.
    """

    registry_name = "salt_and_pepper"

    def __init__(self, K: int, latent_dim: int, device,
                 logit_p_bounds: tuple = (-4.0, 4.0),
                 log_amplitude_bounds: tuple = (-3.0, 1.0),
                 temperature: float = 0.5):
        super().__init__(K, latent_dim, device)
        self.logit_p_bounds = logit_p_bounds
        self.log_amplitude_bounds = log_amplitude_bounds
        self.temperature = temperature
        self._conditional = False

    def get_config(self) -> dict:
        return {
            "logit_p_bounds":      self.logit_p_bounds,
            "log_amplitude_bounds": self.log_amplitude_bounds,
            "temperature":         self.temperature,
        }

    def build_unconditional_params(self) -> None:
        self._conditional = False
        K, D = self.K, self.latent_dim
        # Initialise logit_p at 0 → p ≈ 0.5 (moderate corruption rate)
        self.logit_p      = nn.Parameter(torch.zeros(K, D, device=self.device))
        self.log_amplitude = nn.Parameter(torch.zeros(K, D, device=self.device))

    def build_conditional_heads(self, hidden_dim: int) -> None:
        self._conditional = True
        K, D = self.K, self.latent_dim
        self.logit_p_head       = nn.Linear(hidden_dim, K * D).to(self.device)
        self.log_amplitude_head = nn.Linear(hidden_dim, K * D).to(self.device)

    def get_scale_params(self, h: Optional[torch.Tensor], B: int) -> dict:
        K, D = self.K, self.latent_dim
        lo_p, hi_p = self.logit_p_bounds
        lo_a, hi_a = self.log_amplitude_bounds

        if self._conditional:
            logit_p = self.logit_p_head(h).view(B, K, D)
            log_amp = self.log_amplitude_head(h).view(B, K, D)
        else:
            logit_p = self.logit_p.unsqueeze(0).expand(B, K, D)
            log_amp = self.log_amplitude.unsqueeze(0).expand(B, K, D)

        return {
            "p":   torch.sigmoid(torch.clamp(logit_p, lo_p, hi_p)),  # [B, K, D]
            "amp": torch.exp(torch.clamp(log_amp, lo_a, hi_a)),       # [B, K, D]
        }

    def rsample(self, mu: torch.Tensor, scale_params: dict,
                num_samples: int) -> torch.Tensor:
        """
        Differentiable sample via Gumbel-Softmax over {-amp, 0, +amp}.

        Returns
        -------
        Tensor[S, B, K, D]
        """
        p   = scale_params["p"]    # [B, K, D]
        amp = scale_params["amp"]  # [B, K, D]
        B, K, D = mu.shape

        # Build 3-outcome probability vector: [pepper, clean, salt]
        half_p = p / 2.0
        probs  = torch.stack([half_p, 1.0 - p, half_p], dim=-1)          # [B, K, D, 3]
        spikes = torch.stack([-amp, torch.zeros_like(amp), amp], dim=-1)  # [B, K, D, 3]

        # Expand along sample dimension
        probs_exp  = probs.unsqueeze(0).expand(num_samples, B, K, D, 3)   # [S, B, K, D, 3]
        spikes_exp = spikes.unsqueeze(0).expand(num_samples, B, K, D, 3)  # [S, B, K, D, 3]

        # Gumbel-Softmax relaxation
        log_probs = torch.log(probs_exp.clamp(min=1e-20))
        gumbel    = -torch.log(-torch.log(
            torch.rand_like(log_probs).clamp(min=1e-20)
        ))
        weights = F.softmax((log_probs + gumbel) / self.temperature, dim=-1)  # [S, B, K, D, 3]

        noise = (weights * spikes_exp).sum(dim=-1)   # [S, B, K, D]
        return mu.unsqueeze(0) + noise               # [S, B, K, D]

    def log_prob(self, x: torch.Tensor, mu: torch.Tensor,
                 scale_params: dict) -> torch.Tensor:
        """
        Log probability under the per-dimension 3-point discrete distribution,
        approximated with narrow Gaussians centred at each spike.

        Returns
        -------
        Tensor[B, K]
        """
        p   = scale_params["p"]    # [B, K, D]
        amp = scale_params["amp"]  # [B, K, D]
        x_exp = x.unsqueeze(1).expand_as(mu)  # [B, K, D]
        diff  = x_exp - mu                    # [B, K, D]

        # Narrow Gaussian bandwidth to approximate the discrete spikes
        tiny = torch.full_like(amp, 1e-3)
        log_p_salt   = Normal( amp,               tiny).log_prob(diff)
        log_p_pepper = Normal(-amp,               tiny).log_prob(diff)
        log_p_clean  = Normal(torch.zeros_like(amp), tiny).log_prob(diff)

        half_p = p / 2.0
        log_p = torch.log(
            half_p    * torch.exp(log_p_salt)   +
            half_p    * torch.exp(log_p_pepper) +
            (1.0 - p) * torch.exp(log_p_clean)  +
            1e-40
        )
        return log_p.sum(dim=-1)  # [B, K]


# --------------------------------------------------
#    Component Registry (checkpoint loading only)
# --------------------------------------------------

_NOISE_REGISTRY: dict = {
    "gaussian":        GaussianComponent,
    "laplace":         LaplaceComponent,
    "uniform":         UniformComponent,
    "salt_and_pepper": SaltAndPepperComponent,
}


def register_component(name: str, cls: type) -> None:
    """
    Register a ``NoiseComponent`` subclass for checkpoint loading.

    Only needed if you want ``PerturbationModel.load_from_checkpoint`` to
    reconstruct custom component types.  For construction, simply pass
    instances directly to ``set_condition(components=[…])``.

    Parameters
    ----------
    name : str
        Must match the component's ``registry_name`` class attribute.
    cls : type
        A subclass of ``NoiseComponent``.
    """
    if not issubclass(cls, NoiseComponent):
        raise TypeError(f"cls must be a subclass of NoiseComponent, got {cls}")
    _NOISE_REGISTRY[name] = cls


### SECTION 2: Store the Noise Distribution ### 
# --------------------------------------------------
#                   DistCache 
# --------------------------------------------------

@dataclass
class DistCache:
    """
    Typed container for the mixture distribution parameters computed by
    ``MixedNoiseDistribution._build_dist()``.

    Attributes
    ----------
    pi_logits : Tensor[B, K]
        Unnormalised log-weights over the K mixture components.
    mu : Tensor[B, K, D]
        Component means, shared across all distribution types.
    component_params : list[dict]
        One parameter dict per ``NoiseComponent``, in the same order as
        ``_components``.  Contents are distribution-specific (e.g.
        ``{"log_std": …}`` for Gaussian, ``{"log_scale": …}`` for Laplace).

    Notes
    -----
    ``__getitem__`` provides dict-style access so that user-supplied
    regularizer lambdas written as ``lambda c: c["mu"]`` continue to work
    without modification.
    """

    pi_logits: torch.Tensor        # [B, K]
    mu: torch.Tensor               # [B, K, D]
    component_params: list         # list[dict], one per NoiseComponent

    def __getitem__(self, key: str) -> torch.Tensor:
        """Dict-style read access for backward compatibility."""
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


### SECTION 3: Regularizer ### 
# --------------------------------------------------
#               Regularizer Utilities 
# --------------------------------------------------


def reg_pi_entropy(cache: DistCache) -> torch.Tensor:
    """
    Normalised entropy loss on mixture weights.

    Minimising this encourages the model to spread probability evenly across
    all K components (higher entropy = lower loss).

    Usage::

        model.register_regularizer(reg_pi_entropy, coeff=0.01)
    """
    pi_probs = F.softmax(cache.pi_logits, dim=-1)
    entropy = Categorical(probs=pi_probs).entropy()
    K = pi_probs.size(-1)
    return (1.0 - entropy / math.log(K)).mean()


def reg_mean_diversity(cache: DistCache) -> torch.Tensor:
    """
    Log-determinant diversity loss on component means.

    Minimising this pushes component means apart in normalised space
    (maximises the volume of the mean simplex).

    Usage::

        model.register_regularizer(reg_mean_diversity, coeff=0.001)
    """
    mu = cache.mu                                               # [B, K, D]
    K = mu.size(1)
    mu_hat = F.normalize(mu, p=2, dim=-1, eps=1e-8)            # [B, K, D]
    gram = torch.bmm(mu_hat, mu_hat.transpose(1, 2))           # [B, K, K]
    gram = gram + 1e-6 * torch.eye(
        K, device=mu.device, dtype=gram.dtype
    ).unsqueeze(0)
    _, logabsdet = torch.linalg.slogdet(gram)
    return F.relu(math.log(float(K)) - logabsdet).mean()


### SECTION 4: Conditioner ### 
# --------------------------------------------------
#               ConditioningStrategy 
# --------------------------------------------------


class ConditioningStrategy(ABC):
    """
    Encapsulates all logic that varies with the conditioning mode.

    Strategies are pure-logic objects: they carry no learnable parameters and
    hold no references to the model.  ``build()`` returns a dict of
    ``nn.Module`` / ``nn.Parameter`` objects that ``MixedNoiseDistribution``
    installs on itself via ``setattr``.  ``encode()`` and
    ``build_dist_params()`` receive only the specific attributes they need,
    so adding a new conditioning mode requires subclassing this ABC alone —
    no changes to ``MixedNoiseDistribution``.

    Attributes
    ----------
    mode_name : str or None
        Identifier stored in checkpoints.  Must be a key in
        ``_CONDITIONING_REGISTRY`` for save/load to work.
    components_are_unconditional : bool
        Whether ``NoiseComponent.build_unconditional_params()`` should be
        called (True) vs ``build_conditional_heads()`` (False).

    Methods
    -------
    build(K, D, hidden_dim, feat_dim, num_cls, y_dim, device) → dict
        Create and return named heads / parameters to install on the model.
    encode(x, y, *, feat_extractor, y_emb, y_emb_normalize, num_cls, device) → (part_x, part_y)
        Validate inputs and produce conditioning tensors.
    build_dist_params(pi_head, mu_head, shared_trunk, part_x, part_y, B, K, D) → (h_shared, pi_logits, mu)
        Compute distribution parameters from conditioning tensors and heads.
    """

    mode_name = ""  # override in every subclass

    @property
    @abstractmethod
    def components_are_unconditional(self) -> bool:
        """True → build_unconditional_params; False → build_conditional_heads."""

    @abstractmethod
    def build(self, K: int, D: int, hidden_dim: int,
              feat_dim: int, num_cls: int, y_dim: int, device) -> dict:
        """
        Return a dict of named nn.Module / nn.Parameter objects.

        The caller (``MixedNoiseDistribution.set_condition``) will install
        each entry via ``setattr(model, name, obj)``.
        """

    @abstractmethod
    def encode(self, x, y, *,
               feat_extractor, y_emb, y_emb_normalize,
               num_cls: int, device) -> tuple:
        """Validate inputs, extract features. Returns (part_x, part_y)."""

    @abstractmethod
    def build_dist_params(self, pi_head, mu_head, shared_trunk,
                          part_x: torch.Tensor, part_y: torch.Tensor,
                          B: int, K: int, D: int) -> tuple:
        """
        Compute (h_shared, pi_logits [B,K], mu [B,K,D]) from heads and inputs.

        Parameters
        ----------
        pi_head : nn.Parameter or nn.Linear
        mu_head : nn.Parameter or nn.Linear
        shared_trunk : nn.Module or None
        """


def _encode_label(y: torch.Tensor, y_emb, y_emb_normalize: bool,
                  num_cls: int, device) -> torch.Tensor:
    """Module-level helper: encode integer class labels to embedding or one-hot."""
    if y_emb is not None:
        yvec = y_emb(y)
        if y_emb_normalize:
            yvec = F.normalize(yvec, dim=-1)
    else:
        yvec = F.one_hot(y, num_classes=num_cls).float().to(device)
    return yvec


class UnconditionalStrategy(ConditioningStrategy):
    """
    No conditioning.  Pi and mu are free ``nn.Parameter`` tensors.

    Usage::

        model.set_condition(UnconditionalStrategy(), num_cls=10)
        # equivalent to: model.set_condition(cond_mode=None, num_cls=10)
    """

    mode_name = None

    @property
    def components_are_unconditional(self) -> bool:
        return True

    def build(self, K, D, hidden_dim, feat_dim, num_cls, y_dim, device) -> dict:
        return {
            "pi": nn.Parameter(torch.randn(K, device=device) * 0.01),
            "mu": nn.Parameter(torch.zeros(K, D, device=device)),
        }

    def encode(self, x, y, *,
               feat_extractor, y_emb, y_emb_normalize, num_cls, device):
        B = x.size(0) if x is not None else (y.size(0) if y is not None else 1)
        dummy = torch.ones(B, 1, device=device)
        return dummy, dummy

    def build_dist_params(self, pi_head, mu_head, shared_trunk,
                          part_x, part_y, B, K, D):
        h_shared = None
        pi_logits = pi_head.unsqueeze(0).expand(B, -1)
        mu = mu_head.unsqueeze(0).expand(B, K, D)
        return h_shared, pi_logits, mu


class XStrategy(ConditioningStrategy):
    """
    Condition on image features only.  Pi and mu are both outputs of the
    shared trunk applied to ``feat_extractor(x)``.

    Usage::

        model.set_feat_extractor(encoder)
        model.set_condition(XStrategy(), feat_dim=512)
        # equivalent to: model.set_condition(cond_mode="x", feat_dim=512)
    """

    mode_name = "x"

    @property
    def components_are_unconditional(self) -> bool:
        return False

    def build(self, K, D, hidden_dim, feat_dim, num_cls, y_dim, device) -> dict:
        trunk = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        ).to(device)
        return {
            "shared_trunk": trunk,
            "pi": nn.Linear(hidden_dim, K).to(device),
            "mu": nn.Linear(hidden_dim, K * D).to(device),
        }

    def encode(self, x, y, *,
               feat_extractor, y_emb, y_emb_normalize, num_cls, device):
        if feat_extractor is None:
            raise ValueError("XStrategy requires feat_extractor. "
                             "Call set_feat_extractor() first.")
        if x is None:
            raise ValueError("XStrategy requires x input.")
        with torch.no_grad():
            part_x = feat_extractor(x).view(x.size(0), -1)
        part_y = torch.ones(x.size(0), 1, device=device)
        return part_x, part_y

    def build_dist_params(self, pi_head, mu_head, shared_trunk,
                          part_x, part_y, B, K, D):
        h_shared = shared_trunk(part_x)
        pi_logits = pi_head(h_shared)
        mu = mu_head(h_shared).view(B, K, D)
        return h_shared, pi_logits, mu


class YStrategy(ConditioningStrategy):
    """
    Condition pi on class label only; mu is a free ``nn.Parameter``.

    Usage::

        model.set_condition(YStrategy(), num_cls=10)
        # equivalent to: model.set_condition(cond_mode="y", num_cls=10)
    """

    mode_name = "y"

    @property
    def components_are_unconditional(self) -> bool:
        return True

    def build(self, K, D, hidden_dim, feat_dim, num_cls, y_dim, device) -> dict:
        return {
            "pi": nn.Linear(y_dim, K).to(device),
            "mu": nn.Parameter(torch.zeros(K, D, device=device)),
        }

    def encode(self, x, y, *,
               feat_extractor, y_emb, y_emb_normalize, num_cls, device):
        if y is None:
            raise ValueError("YStrategy requires y input.")
        if num_cls is None:
            raise ValueError("num_cls must be set for y-conditioning.")
        yvec = _encode_label(y, y_emb, y_emb_normalize, num_cls, device)
        part_x = torch.ones(y.size(0), 1, device=device)
        return part_x, yvec

    def build_dist_params(self, pi_head, mu_head, shared_trunk,
                          part_x, part_y, B, K, D):
        h_shared = None
        pi_logits = pi_head(part_y)
        mu = mu_head.unsqueeze(0).expand(B, K, D)
        return h_shared, pi_logits, mu


class XYStrategy(ConditioningStrategy):
    """
    Condition pi on class label and mu on image features.

    Usage::

        model.set_feat_extractor(encoder)
        model.set_condition(XYStrategy(), feat_dim=512, num_cls=10)
        # equivalent to: model.set_condition(cond_mode="xy", feat_dim=512, num_cls=10)
    """

    mode_name = "xy"

    @property
    def components_are_unconditional(self) -> bool:
        return False

    def build(self, K, D, hidden_dim, feat_dim, num_cls, y_dim, device) -> dict:
        trunk = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        ).to(device)
        return {
            "shared_trunk": trunk,
            "pi": nn.Linear(y_dim, K).to(device),
            "mu": nn.Linear(hidden_dim, K * D).to(device),
        }

    def encode(self, x, y, *,
               feat_extractor, y_emb, y_emb_normalize, num_cls, device):
        if feat_extractor is None:
            raise ValueError("XYStrategy requires feat_extractor. "
                             "Call set_feat_extractor() first.")
        if x is None or y is None:
            raise ValueError("XYStrategy requires both x and y inputs.")
        if num_cls is None:
            raise ValueError("num_cls must be set for xy-conditioning.")
        with torch.no_grad():
            part_x = feat_extractor(x).view(x.size(0), -1)
        part_y = _encode_label(y, y_emb, y_emb_normalize, num_cls, device)
        return part_x, part_y

    def build_dist_params(self, pi_head, mu_head, shared_trunk,
                          part_x, part_y, B, K, D):
        h_shared = shared_trunk(part_x)
        pi_logits = pi_head(part_y)
        mu = mu_head(h_shared).view(B, K, D)
        return h_shared, pi_logits, mu


# Registry: maps saved cond_mode values back to strategy classes.
# Used only by load_from_checkpoint.
_CONDITIONING_REGISTRY: dict = {
    None:  UnconditionalStrategy,
    "x":   XStrategy,
    "y":   YStrategy,
    "xy":  XYStrategy,
}

# Sentinel for distinguishing "None passed explicitly" from "not passed"
_UNSET = object()


### SECTION 5: MixedNoiseDistribution ### 
# --------------------------------------------------
#               MixedNoiseDistribution 
# --------------------------------------------------


class MixedNoiseDistribution(nn.Module):
    """
    Pure distribution model for a K-component mixture of heterogeneous noise.

    Handles distribution building, reparameterized / hard sampling, and the
    open regularizer registry.  Does **not** know about perturbation budgets,
    decoders, classifiers, or training losses — those belong in
    ``PerturbationModel`` and the standalone training functions.

    Parameters
    ----------
    K : int
        Total number of mixture components.
    latent_dim : int
        Dimensionality of the latent space.
    device : str or torch.device
    logstd_bounds : (float, float)
        Default log-std clamp bounds used when no explicit components list is
        given to ``set_condition``.

    Workflow
    --------
    1. Optionally call ``set_y_embedding`` and/or ``set_feat_extractor``.
    2. Call ``set_condition`` to build pi/mu heads and component parameters.
    3. Call ``forward(x, y)`` to obtain a ``DistCache``.
    4. Use ``_rsample`` or ``_sample_hard`` for sampling.
    """

    def __init__(self, K: int, latent_dim: int, device,
                 logstd_bounds: tuple = (-3.0, 1.0)):
        super().__init__()
        self.K = K
        self.latent_dim = latent_dim
        self.device = device
        self.logstd_bounds = logstd_bounds

        # Regularizer registry: list of (fn, coeff).  Empty by default.
        self._regularizers: list = []

        # Set by set_feat_extractor / set_y_embedding
        self.feat_extractor = None
        self.shared_trunk = None
        self.y_emb = None
        self.y_emb_normalize = True
        self.num_cls = None

        # Set by set_condition (not nn.Module, not in state_dict)
        self._strategy: Optional[ConditioningStrategy] = None
        self._components: Optional[nn.ModuleList] = None
        self._component_slices: Optional[list] = None

    # ── repr and init guard ───────────────────────────────────────────────────

    def __repr__(self) -> str:
        if self._components is None:
            return (f"{type(self).__name__}("
                    f"K={self.K}, latent_dim={self.latent_dim}, uninitialized)")
        comps = ", ".join(
            f"{type(c).__name__}(K={c.K})" for c in self._components
        )
        return (f"{type(self).__name__}("
                f"K={self.K}, latent_dim={self.latent_dim}, "
                f"strategy={type(self._strategy).__name__}, "
                f"components=[{comps}])")

    def _require_initialized(self) -> None:
        """Raise a clear error if set_condition has not been called."""
        if self._components is None:
            raise RuntimeError(
                f"{type(self).__name__} is not initialized. "
                "Call set_condition() before use."
            )

    # ── setup methods ─────────────────────────────────────────────────────────

    def set_y_embedding(self, num_cls: int, y_dim: int,
                        normalize: bool = True) -> None:
        """Install a learnable label embedding table."""
        self.y_emb = nn.Embedding(num_cls, y_dim).to(self.device)
        self.y_emb_normalize = normalize
        self.num_cls = num_cls

    def set_feat_extractor(self, feat_extractor) -> None:
        """Attach a frozen image encoder (used in x and xy modes)."""
        self.feat_extractor = feat_extractor.to(self.device)
        for p in self.feat_extractor.parameters():
            p.requires_grad = False

    def register_regularizer(self, fn, coeff: float = 1.0) -> None:
        """
        Register a regularizer applied during ``pr_loss``.

        Parameters
        ----------
        fn : callable
            ``fn(cache: DistCache) -> Tensor``.  Dict-style access
            ``cache["pi_logits"]`` also works for backward compatibility.
        coeff : float
            Scalar weight multiplied with the returned value.

        Example::

            from src.mixture4pr import reg_pi_entropy, reg_mean_diversity
            model.register_regularizer(reg_pi_entropy,     coeff=0.01)
            model.register_regularizer(reg_mean_diversity, coeff=0.001)
            model.register_regularizer(lambda c: (c["mu"]**2).mean(), coeff=5e-4)
        """
        self._regularizers.append((fn, coeff))

    def compute_regularization(self, cache: DistCache) -> torch.Tensor:
        """
        Compute the total regularization loss.

        Returns a scalar zero tensor when no regularizers are registered.
        """
        if not self._regularizers:
            return torch.tensor(0.0, device=cache.pi_logits.device)
        return sum(coeff * fn(cache) for fn, coeff in self._regularizers)

    def set_condition(self, strategy_or_mode=_UNSET, feat_dim: int = 0,
                      num_cls: int = 0, hidden_dim: int = 128,
                      components: Optional[list] = None,
                      cond_mode=_UNSET) -> None:
        """
        Build all network heads / parameters for the chosen conditioning mode.

        Must be called before any forward pass.

        Parameters
        ----------
        strategy_or_mode : ConditioningStrategy | None | "x" | "y" | "xy"
            Either a ``ConditioningStrategy`` instance (new API) or one of
            the legacy string/None values (automatically converted).
        feat_dim : int
            Feature extractor output dim (required for x / xy modes).
        num_cls : int
            Number of classes (required for y / xy modes).
        hidden_dim : int
            Width of the shared trunk and all conditional heads.
        components : list of NoiseComponent, optional
            Pre-constructed component objects.  Their ``K`` values must sum
            to ``self.K``.  Defaults to a single
            ``GaussianComponent(K=self.K, …)`` with diagonal covariance.

        Examples
        --------
        New strategy API::

            model.set_condition(UnconditionalStrategy(),
                                components=[GaussianComponent(K=3, ...)])

        Legacy string API (backward compatible)::

            model.set_condition(cond_mode=None, num_cls=10)
        """
        # Support legacy keyword: set_condition(cond_mode=None, ...)
        if cond_mode is not _UNSET:
            if strategy_or_mode is not _UNSET:
                raise TypeError(
                    "Pass strategy/mode as the first positional argument "
                    "or as 'cond_mode=', not both."
                )
            strategy_or_mode = cond_mode
        elif strategy_or_mode is _UNSET:
            strategy_or_mode = None  # default: unconditional

        # Convert legacy string/None to strategy instance
        if not isinstance(strategy_or_mode, ConditioningStrategy):
            if strategy_or_mode not in _CONDITIONING_REGISTRY:
                raise ValueError(
                    f"Unknown mode {strategy_or_mode!r}. "
                    "Use None, 'x', 'y', 'xy', or a ConditioningStrategy instance."
                )
            strategy_or_mode = _CONDITIONING_REGISTRY[strategy_or_mode]()

        self._strategy = strategy_or_mode
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim
        self.num_cls = num_cls

        K, D = self.K, self.latent_dim
        y_dim = self.y_emb.embedding_dim if self.y_emb is not None else num_cls

        # Reset optional trunk before strategy installs new heads
        self.shared_trunk = None

        # Delegate head/parameter creation to the strategy; install results
        modules = self._strategy.build(K, D, hidden_dim, feat_dim,
                                       num_cls, y_dim, self.device)
        for name, obj in modules.items():
            setattr(self, name, obj)   # PyTorch __setattr__ registers correctly

        # Build noise components
        if components is None:
            components = [GaussianComponent(
                K=self.K, latent_dim=D, device=self.device,
                logstd_bounds=self.logstd_bounds,
            )]

        total_K = sum(c.K for c in components)
        if total_K != K:
            raise ValueError(
                f"Sum of component K values ({total_K}) != model K ({K}). "
                "Adjust the K values in your components list."
            )

        self._components = nn.ModuleList()
        self._component_slices = []
        idx = 0
        for comp in components:
            if self._strategy.components_are_unconditional:
                comp.build_unconditional_params()
            else:
                comp.build_conditional_heads(hidden_dim)
            self._components.append(comp)
            self._component_slices.append(slice(idx, idx + comp.K))
            idx += comp.K

        # Parameter summary
        total_params = sum(p.numel() for p in self.parameters())
        comp_params = sum(
            p.numel() for c in self._components for p in c.parameters()
        )
        pi_params = (self.pi.numel() if isinstance(self.pi, nn.Parameter)
                     else sum(p.numel() for p in self.pi.parameters()))
        mu_params = (self.mu.numel() if isinstance(self.mu, nn.Parameter)
                     else sum(p.numel() for p in self.mu.parameters()))
        print(
            f"[Params] pi: {pi_params:,} | mu: {mu_params:,} | "
            f"components: {comp_params:,} | Total: {total_params:,}"
        )

    # ── conditioning ──────────────────────────────────────────────────────────

    def _make_condition(self, x=None, y=None) -> tuple:
        """Extract (part_x, part_y) conditioning tensors via the strategy."""
        return self._strategy.encode(
            x, y,
            feat_extractor=self.feat_extractor,
            y_emb=self.y_emb,
            y_emb_normalize=self.y_emb_normalize,
            num_cls=self.num_cls,
            device=self.device,
        )

    def _build_dist(self, part_x: torch.Tensor,
                    part_y: torch.Tensor) -> DistCache:
        """Build the typed distribution parameter cache."""
        B = part_x.size(0)
        K, D = self.K, self.latent_dim

        h_shared, pi_logits, mu = self._strategy.build_dist_params(
            self.pi, self.mu,
            getattr(self, "shared_trunk", None),
            part_x, part_y, B, K, D,
        )

        component_params = [
            comp.get_scale_params(h_shared, B)
            for comp in self._components
        ]
        return DistCache(pi_logits=pi_logits, mu=mu,
                         component_params=component_params)

    def forward(self, x=None, y=None) -> dict:
        """Build and return the distribution cache."""
        self._require_initialized()
        part_x, part_y = self._make_condition(x=x, y=y)
        cache = self._build_dist(part_x, part_y)
        return {"cache": cache}

    # ── sampling ──────────────────────────────────────────────────────────────

    def _rsample(self, cache: DistCache, num_samples: int,
                 gumbel_temperature: float = 1.0) -> torch.Tensor:
        """
        Differentiable reparameterized sampling via Gumbel-Softmax.

        Returns ``[num_samples, B, latent_dim]`` with gradients.

        Steps
        -----
        1. Gumbel-Softmax over pi_logits → soft component weights [S, B, K].
        2. Each ``NoiseComponent`` draws reparameterized samples [S, B, K_type, D].
        3. Concatenate along K, then soft-weighted sum → [S, B, D].
        """
        self._require_initialized()
        pi_logits = cache.pi_logits                            # [B, K]
        mu = cache.mu                                          # [B, K, D]
        B, K, D = mu.shape

        # Step 1: Gumbel-Softmax
        gumbel_noise = -torch.log(-torch.log(
            torch.rand(num_samples, B, K, device=pi_logits.device) + 1e-20
        ) + 1e-20)
        logits_g = (pi_logits.unsqueeze(0) + gumbel_noise) / gumbel_temperature
        soft_weights = F.softmax(logits_g, dim=-1)             # [S, B, K]

        # Step 2: Per-component reparameterized sampling
        parts = []
        for comp, sl, params in zip(self._components,
                                    self._component_slices,
                                    cache.component_params):
            mu_k = mu[:, sl, :]                                # [B, K_type, D]
            parts.append(comp.rsample(mu_k, params, num_samples))

        component_samples = torch.cat(parts, dim=2)            # [S, B, K, D]

        # Step 3: Weighted combination
        weights = soft_weights.unsqueeze(-1)                   # [S, B, K, 1]
        return (component_samples * weights).sum(dim=2)        # [S, B, D]

    def _rsample_from_gmm(self, cache: DistCache, num_samples: int,
                          temperature: float = 1.0) -> torch.Tensor:
        """Backward-compatible alias for ``_rsample``."""
        return self._rsample(cache, num_samples, gumbel_temperature=temperature)

    @torch.no_grad()
    def _sample_hard(self, cache: DistCache, num_samples: int) -> torch.Tensor:
        """
        Non-differentiable hard sampling from the true mixed distribution.

        Samples component indices categorically from pi, then draws from
        the selected component.  Returns ``[num_samples, B, latent_dim]``.
        """
        self._require_initialized()
        pi_logits = cache.pi_logits                            # [B, K]
        mu = cache.mu                                          # [B, K, D]
        B, K, D = mu.shape

        component_idx = Categorical(logits=pi_logits).sample((num_samples,))

        parts = []
        for comp, sl, params in zip(self._components,
                                    self._component_slices,
                                    cache.component_params):
            parts.append(comp.rsample(mu[:, sl, :], params, num_samples))

        all_samples = torch.cat(parts, dim=2)                  # [S, B, K, D]
        idx = component_idx.unsqueeze(-1).unsqueeze(-1).expand(
            num_samples, B, 1, D)
        return all_samples.gather(2, idx).squeeze(2)           # [S, B, D]


### SECTION 6: Perturbation Decoder ### 
# --------------------------------------------------
#          Perturbation Decoder (Upsampler) 
# --------------------------------------------------


class PerturbationDecoder:
    """
    Encapsulates the latent-to-perturbation pipeline: decode then project.

    Not an ``nn.Module`` — holds no learnable parameters.  The optional
    ``up_sampler`` (a learned spatial decoder) is stored separately on
    ``PerturbationModel`` so its weights appear at the expected state-dict
    paths.  Pass it explicitly to ``decode()`` when needed.

    Parameters
    ----------
    norm : "linf" | "l2"
        Which budget constraint to enforce.
    eps : float
        Perturbation radius.

    Usage::

        decoder = PerturbationDecoder(norm="linf", eps=8/255)
        u     = decoder.decode(latent, out_shape=(3, 32, 32), up_sampler=None)
        delta = decoder.project(u)
        # or in one call:
        delta = decoder(latent, out_shape=(3, 32, 32))
    """

    def __init__(self, norm: str = "linf", eps: float = 8 / 255):
        self.norm = norm
        self.eps = float(eps)

    def decode(self, latent: torch.Tensor, out_shape: tuple,
               up_sampler=None) -> torch.Tensor:
        """
        Map latent samples to image-shaped tensor u.

        Parameters
        ----------
        latent : Tensor[B, D] or Tensor[S, B, D]
        out_shape : (C, H, W)
        up_sampler : nn.Module or None
            If None, reshapes directly (requires ``D == prod(out_shape)``).
        """
        if up_sampler is None:
            assert latent.size(-1) == np.prod(out_shape), (
                f"Latent dim {latent.size(-1)} does not match "
                f"out_shape {out_shape}"
            )
            return latent.view(latent.shape[:-1] + tuple(out_shape))
        if latent.dim() == 2:
            u = up_sampler(latent)
            return u.view(latent.size(0), *out_shape) if u.dim() == 2 else u
        elif latent.dim() == 3:
            S, B, D = latent.shape
            u = up_sampler(latent.reshape(-1, D))
            if u.dim() == 2:
                return u.view(S, B, *out_shape)
            return u.view(S, B, *u.shape[1:])
        raise ValueError(f"latent must be 2D or 3D, got shape {latent.shape}")

    def project(self, u: torch.Tensor) -> torch.Tensor:
        """Project image-shaped tensor u onto the perturbation budget."""
        norm = self.norm.lower()
        eps = self.eps
        if norm == "linf":
            return eps * torch.tanh(u)
        elif norm == "l2":
            if u.dim() == 4:
                flat = u.view(u.size(0), -1)
                n = flat.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                return (eps * flat / n).view_as(u)
            elif u.dim() == 5:
                flat = u.view(u.shape[0], u.shape[1], -1)
                n = flat.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                return (eps * flat / n).view_as(u)
        raise ValueError(f"Unsupported norm={norm!r} or u.shape={u.shape}")

    def __call__(self, latent: torch.Tensor, out_shape: tuple,
                 up_sampler=None) -> torch.Tensor:
        """Convenience: decode then project in one call."""
        return self.project(self.decode(latent, out_shape, up_sampler))


# --------------------------------------------------
#               Perturbation Model
# --------------------------------------------------


class PerturbationModel(MixedNoiseDistribution):
    """
    Extends ``MixedNoiseDistribution`` with the perturbation pipeline and
    persistence.  Training and evaluation logic lives in standalone functions
    (``pr_loss``, ``evaluate_pr``, ``compute_pr``) that can also be called
    directly without going through this class.

    Added responsibilities vs ``MixedNoiseDistribution``
    -----------------------------------------------------
    - ``decoder``: ``PerturbationDecoder`` for budget config + decode/project.
    - ``up_sampler``: optional frozen spatial decoder (state-dict registered).
    - ``sample()``: sample perturbations end-to-end.
    - ``save()`` / ``load_from_checkpoint()``: persistence.
    - Thin method wrappers for ``pr_loss``, ``evaluate_pr``, ``compute_pr``
      that delegate to the standalone module-level functions.

    Parameters
    ----------
    K, latent_dim, device, logstd_bounds
        Forwarded to ``MixedNoiseDistribution``.
    """

    def __init__(self, K: int, latent_dim: int, device,
                 logstd_bounds: tuple = (-3.0, 1.0)):
        super().__init__(K, latent_dim, device, logstd_bounds)
        self.decoder = PerturbationDecoder()   # plain object, not nn.Module
        self.up_sampler = None                 # nn.Module when set; kept here
                                               # so state-dict paths are stable

    # ── budget and decoder setup ──────────────────────────────────────────────

    def set_up_sampler(self, up_sampler) -> None:
        """Attach a decoder that maps latent vectors to image tensors."""
        self.up_sampler = up_sampler.to(self.device)

    def set_budget(self, norm: str = "linf", eps: float = 8 / 255) -> None:
        """Configure the perturbation budget (delegates to decoder)."""
        self.decoder.norm = norm
        self.decoder.eps = float(eps)

    @property
    def budget(self) -> dict:
        """Read-only dict view of the current budget (backward compat)."""
        return {"norm": self.decoder.norm, "eps": self.decoder.eps}

    # ── internal helper ───────────────────────────────────────────────────────

    def _to_delta(self, eps: torch.Tensor,
                  out_shape: tuple) -> torch.Tensor:
        """Decode latent eps and project to budget using the model's up_sampler."""
        return self.decoder(eps, out_shape, self.up_sampler)

    # ── perturbation sampling ─────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, x=None, y=None, num_samples: int = 1,
               out_shape=None, chunk_size=None) -> dict:
        """
        Sample perturbations from the learned mixed distribution.

        Parameters
        ----------
        x : Tensor[B, C, H, W] or None
        y : Tensor[B] or None
        num_samples : int
        out_shape : (C, H, W) or None  — required when x is None
        chunk_size : int or None  — samples in chunks; cache is built only once

        Returns
        -------
        dict with keys ``"eps"`` [S,B,D], ``"u"`` [S,B,C,H,W],
        ``"delta"`` [S,B,C,H,W].
        """
        self._require_initialized()
        strategy = self._strategy
        if isinstance(strategy, (XStrategy, XYStrategy)) and x is None:
            raise ValueError(f"{type(strategy).__name__} requires x input.")
        if isinstance(strategy, (YStrategy, XYStrategy)) and y is None:
            raise ValueError(f"{type(strategy).__name__} requires y input.")

        cache = self.forward(x=x, y=y)["cache"]

        if out_shape is None:
            if x is not None:
                out_shape = x.shape[1:]
            else:
                raise ValueError(
                    "out_shape must be provided when x is None. "
                    "Example: out_shape=(3, 32, 32)"
                )

        def _decode_one_chunk(c, n):
            eps_c = self._sample_hard(c, n)
            u_c = self.decoder.decode(eps_c, out_shape, self.up_sampler)
            delta_c = self.decoder.project(u_c)
            return eps_c, u_c, delta_c

        if chunk_size is None or num_samples <= chunk_size:
            eps, u, delta = _decode_one_chunk(cache, num_samples)
            return {"eps": eps, "u": u, "delta": delta}

        eps_list, u_list, delta_list = [], [], []
        for i in range((num_samples + chunk_size - 1) // chunk_size):
            chunk_n = min(chunk_size, num_samples - i * chunk_size)
            ec, uc, dc = _decode_one_chunk(cache, chunk_n)
            eps_list.append(ec); u_list.append(uc); delta_list.append(dc)

        return {
            "eps":   torch.cat(eps_list,   dim=0),
            "u":     torch.cat(u_list,     dim=0),
            "delta": torch.cat(delta_list, dim=0),
        }

    # ── training / evaluation (delegate to standalone functions) ─────────────

    def pr_loss(self, x: torch.Tensor, y: torch.Tensor, classifier,
                **kwargs) -> dict:
        """
        Compute the probabilistic robustness training loss.
        Delegates to the module-level ``pr_loss()`` function.
        See ``pr_loss`` for full parameter documentation.
        """
        return pr_loss(self, x, y, classifier, **kwargs)

    @torch.no_grad()
    def evaluate_pr(self, x: torch.Tensor, y: torch.Tensor, classifier,
                    **kwargs) -> torch.Tensor:
        """
        Evaluate Probabilistic Robustness on a batch of images.
        Delegates to the module-level ``evaluate_pr()`` function.
        See ``evaluate_pr`` for full parameter documentation.
        """
        return evaluate_pr(self, x, y, classifier, **kwargs)

    @staticmethod
    def compute_pr(predictions: torch.Tensor, y: torch.Tensor,
                   reduction: str = 'mean') -> torch.Tensor:
        """Compute PR metric.  Delegates to the module-level ``compute_pr``."""
        return compute_pr(predictions, y, reduction)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str, extra=None) -> None:
        """
        Save model checkpoint.

        Raises
        ------
        ValueError
            If any component has an empty ``registry_name``.

        Note
        ----
        Regularizer functions are **not** saved.  Re-register them after
        loading.
        """
        self._require_initialized()
        for comp in self._components:
            if not comp.registry_name:
                raise ValueError(
                    f"Component {type(comp).__name__} has empty registry_name. "
                    "Set the class attribute before saving."
                )

        cfg = dict(
            K=self.K,
            latent_dim=self.latent_dim,
            logstd_bounds=self.logstd_bounds,
            budget=self.budget,
            components_config=[
                {"type": comp.registry_name, "K": comp.K,
                 "config": comp.get_config()}
                for comp in self._components
            ],
            has_y_emb=(self.y_emb is not None),
            y_emb_dim=(self.y_emb.embedding_dim
                       if self.y_emb is not None else None),
            y_emb_normalize=self.y_emb_normalize,
            cond_mode=self._strategy.mode_name,
            feat_dim=self.feat_dim,
            num_cls=self.num_cls,
            hidden_dim=self.hidden_dim,
        )
        if extra:
            cfg.update(extra)
        torch.save({"state_dict": self.state_dict(), "config": cfg}, path)
        print(f"Model saved to {path}")

    @classmethod
    def load_from_checkpoint(cls, path: str, feat_extractor=None,
                             up_sampler=None, map_location: str = "cpu",
                             strict: bool = True):
        """
        Load a ``PerturbationModel`` / ``MixedNoise4PR`` from a checkpoint.

        For custom component types call ``register_component(name, cls)``
        before this method.  Regularizers are not restored — re-register them
        after loading if needed.

        Returns
        -------
        PerturbationModel (or the calling subclass)
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ckpt["config"]

        model = cls(
            K=cfg["K"],
            latent_dim=cfg["latent_dim"],
            device=map_location,
            logstd_bounds=cfg.get("logstd_bounds", (-3.0, 1.0)),
        )

        if cfg.get("has_y_emb", False):
            model.set_y_embedding(
                num_cls=cfg["num_cls"],
                y_dim=cfg["y_emb_dim"],
                normalize=cfg.get("y_emb_normalize", True),
            )

        components = []
        for comp_cfg in cfg.get("components_config", []):
            comp_type = comp_cfg["type"]
            if comp_type not in _NOISE_REGISTRY:
                raise ValueError(
                    f"Unknown component type '{comp_type}'. "
                    f"Call register_component('{comp_type}', YourClass) "
                    "before loading."
                )
            components.append(_NOISE_REGISTRY[comp_type](
                K=comp_cfg["K"],
                latent_dim=cfg["latent_dim"],
                device=map_location,
                **comp_cfg.get("config", {}),
            ))

        cond_mode = cfg["cond_mode"]
        if cond_mode not in _CONDITIONING_REGISTRY:
            raise ValueError(
                f"Unknown cond_mode '{cond_mode}' in checkpoint. "
                "Register a custom strategy before loading."
            )
        model.set_condition(
            _CONDITIONING_REGISTRY[cond_mode](),
            feat_dim=cfg.get("feat_dim", 0),
            num_cls=cfg.get("num_cls", 0),
            hidden_dim=cfg.get("hidden_dim", 128),
            components=components if components else None,
        )

        if any("feat_extractor" in k for k in ckpt["state_dict"].keys()):
            if feat_extractor is not None:
                model.set_feat_extractor(feat_extractor)

        if up_sampler is not None:
            model.set_up_sampler(up_sampler)

        model.set_budget(**cfg.get("budget", {"norm": "linf", "eps": 8 / 255}))
        model.load_state_dict(ckpt["state_dict"], strict=strict)

        if not any("feat_extractor" in k for k in ckpt["state_dict"].keys()):
            if feat_extractor is not None:
                model.set_feat_extractor(feat_extractor)

        print(f"Model loaded from {path}")
        return model


# Backward-compatible alias
MixedNoise4PR = PerturbationModel

### SECTION 7: Training and Evaluation ### 
# --------------------------------------------------
#          Training and Evaluation Functions 
# --------------------------------------------------

#
# These functions operate on a PerturbationModel but live outside the class so
# they can be used in notebooks, trainer loops, or evaluation scripts without
# importing the model class itself.  PerturbationModel.pr_loss() and
# PerturbationModel.evaluate_pr() delegate here.
#
# Usage::
#
#   from src.mixture4pr import pr_loss, evaluate_pr, compute_pr
#
#   result = pr_loss(model, x, y, classifier, num_samples=8)
#   per_image_pr = evaluate_pr(model, x, y, classifier, num_samples=100)
#   scalar_pr = compute_pr(predictions, y, reduction="mean")


def compute_pr(predictions: torch.Tensor, y: torch.Tensor,
               reduction: str = 'mean') -> torch.Tensor:
    """
    Compute Probabilistic Robustness from classifier predictions.

    Parameters
    ----------
    predictions : Tensor[S, B] or Tensor[S*B]
        Predicted class indices from S perturbation samples.
    y : Tensor[B]
        Ground-truth labels.
    reduction : "mean" | "sum" | "none"

    Returns
    -------
    Scalar or Tensor[B].
    """
    if predictions.dim() == 1:
        S_times_B = predictions.size(0)
        B = y.size(0)
        if S_times_B % B != 0:
            raise ValueError(
                f"predictions size {S_times_B} not divisible by y size {B}"
            )
        predictions = predictions.view(S_times_B // B, B)
    elif predictions.dim() == 2:
        if predictions.shape[1] != y.size(0):
            raise ValueError(
                f"predictions batch size {predictions.shape[1]} != "
                f"y size {y.size(0)}"
            )
    else:
        raise ValueError(
            f"predictions must be 1D or 2D, got shape {predictions.shape}"
        )

    y_exp = y.unsqueeze(0).expand(predictions.shape[0], -1)
    per_image_pr = predictions.eq(y_exp).float().mean(dim=0)

    if reduction == 'mean':
        return per_image_pr.mean()
    elif reduction == 'sum':
        return per_image_pr.sum()
    elif reduction == 'none':
        return per_image_pr
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}")


def _sample_and_classify(model: PerturbationModel,
                         x: torch.Tensor, num_samples: int,
                         classifier, cache: DistCache,
                         gumbel_temperature: float = 1.0) -> torch.Tensor:
    """
    Sample perturbations (differentiable), apply to images, classify.
    Returns logits [S, B, num_classes].
    """
    B = x.size(0)
    eps = model._rsample(cache, num_samples,
                         gumbel_temperature=gumbel_temperature)
    delta = model._to_delta(eps, x.shape[1:])
    x_rep = x.unsqueeze(0).expand_as(delta)
    logits = classifier((x_rep + delta).flatten(0, 1))
    return logits.view(num_samples, B, -1)


def pr_loss(model: PerturbationModel,
            x: torch.Tensor, y: torch.Tensor, classifier,
            num_samples: int = 8, loss_variant: str = "cw",
            kappa: float = 0.0, chunk_size=None,
            return_reg_details: bool = False,
            gumbel_temperature: float = 1.0) -> dict:
    """
    Compute the probabilistic robustness training loss.

    Can be called as a standalone function or via ``model.pr_loss(...)``.

    Parameters
    ----------
    model : PerturbationModel
    x : Tensor[B, C, H, W]   clean images
    y : Tensor[B]             ground-truth labels
    classifier : nn.Module    the classifier being robustified
    num_samples : int         perturbation samples per image
    loss_variant : "cw" | "ce"
    kappa : float             margin constant for CW loss
    chunk_size : int or None  max samples per chunk (None = adaptive)
    return_reg_details : bool include ``"reg_details"`` and ``"pi_probs"``
    gumbel_temperature : float  Gumbel-Softmax temperature

    Returns
    -------
    dict with keys ``"pr"``, ``"loss"``, ``"main"``, ``"reg"``
    and optionally ``"reg_details"``, ``"pi_probs"``.
    """
    out = model.forward(x=x, y=y)
    cache = out["cache"]
    B = x.size(0)

    if chunk_size is None:
        chunk_size = max(1, 32 // B)

    if num_samples <= chunk_size:
        logits = _sample_and_classify(
            model, x, num_samples, classifier, cache, gumbel_temperature)
    else:
        logits_list = []
        for i in range((num_samples + chunk_size - 1) // chunk_size):
            chunk_n = min(chunk_size, num_samples - i * chunk_size)
            logits_list.append(_sample_and_classify(
                model, x, chunk_n, classifier, cache, gumbel_temperature))
        logits = torch.cat(logits_list, dim=0)

    logits = logits - logits.max(dim=-1, keepdim=True).values

    if loss_variant == "cw":
        y_rep = y.unsqueeze(0).expand(num_samples, -1)
        logit_y = logits.gather(-1, y_rep.unsqueeze(-1)).squeeze(-1)
        mask = F.one_hot(y_rep, logits.size(-1)).bool()
        max_others = logits.masked_fill(mask, float("-inf")).max(-1).values
        main_loss = F.softplus(logit_y - max_others + kappa).mean()
    else:
        main_loss = 1 - F.cross_entropy(
            logits.flatten(0, 1),
            y.unsqueeze(0).expand(num_samples, -1).flatten()
        )

    total_reg = model.compute_regularization(cache)
    total_loss = main_loss + total_reg

    predictions = logits.argmax(dim=-1)
    pr_val = compute_pr(predictions, y, reduction='mean').item()

    result = {
        "pr":   pr_val,
        "loss": total_loss,
        "main": main_loss.detach(),
        "reg":  total_reg.detach(),
    }
    if return_reg_details:
        result["reg_details"] = total_reg.detach().item()
        result["pi_probs"] = (
            F.softmax(cache.pi_logits, dim=-1).mean(dim=0).detach()
        )
    return result


@torch.no_grad()
def evaluate_pr(model: PerturbationModel,
                x: torch.Tensor, y: torch.Tensor, classifier,
                num_samples: int = 100, use_soft_sampling: bool = False,
                temperature: float = 1.0, reduction: str = 'none',
                chunk_size=None) -> torch.Tensor:
    """
    Evaluate Probabilistic Robustness on a batch of images.

    Can be called as a standalone function or via ``model.evaluate_pr(...)``.

    Parameters
    ----------
    model : PerturbationModel
    x : Tensor[B, C, H, W]
    y : Tensor[B]
    classifier : nn.Module  (should be in eval mode)
    num_samples : int
    use_soft_sampling : bool
        If True, use Gumbel-Softmax (differentiable).
        If False, use hard categorical sampling (true distribution).
    temperature : float
        Gumbel-Softmax temperature (only when use_soft_sampling=True).
    reduction : "mean" | "sum" | "none"
    chunk_size : int or None

    Returns
    -------
    Scalar or Tensor[B].
    """
    B = x.size(0)
    if chunk_size is None:
        chunk_size = max(1, 32 // B)

    def _eval_chunk(n):
        cache = model.forward(x=x, y=y)["cache"]
        eps = (model._rsample(cache, n, gumbel_temperature=temperature)
               if use_soft_sampling
               else model._sample_hard(cache, n))
        delta = model._to_delta(eps, x.shape[1:])
        x_rep = x.unsqueeze(0).expand_as(delta)
        logits = classifier((x_rep + delta).flatten(0, 1))
        return logits.argmax(dim=-1).view(n, B)

    if num_samples <= chunk_size:
        predictions = _eval_chunk(num_samples)
    else:
        preds_list = []
        for i in range((num_samples + chunk_size - 1) // chunk_size):
            chunk_n = min(chunk_size, num_samples - i * chunk_size)
            preds_list.append(_eval_chunk(chunk_n))
        predictions = torch.cat(preds_list, dim=0)

    return compute_pr(predictions, y, reduction=reduction)

### SECTION 8: Build Decoder ### 
# --------------------------------------------------
#          Build Decoder From Flag 
# --------------------------------------------------

def build_decoder_from_flag(backend: str, latent_dim: int,
                             out_shape: tuple, device):
    """
    Build decoder that maps latent_dim -> out_shape.

    Args:
        backend: Decoder type ('bicubic', 'bicubic_trainable')
        latent_dim: Dimensionality of latent space
        out_shape: Output shape (C, H, W)
        device: Target device

    Returns:
        decoder: nn.Module that maps [B, latent_dim] -> [B, C, H, W]
    """
    C, H, W = out_shape

    def calc_init_size(target_size):
        if target_size <= 32:
            return 4
        elif target_size <= 64:
            return 7
        else:
            return target_size // 32

    if backend == "bicubic":
        class BicubicDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.init_size = calc_init_size(min(H, W))
                self.init_dim = C * self.init_size * self.init_size

            def forward(self, z):
                B = z.size(0)
                assert z.size(1) == self.init_dim, (
                    f"Expected latent_dim={self.init_dim}, got {z.size(1)}"
                )
                z = z.view(B, C, self.init_size, self.init_size)
                return F.interpolate(z, size=(H, W), mode='bicubic',
                                     align_corners=False)

        decoder = BicubicDecoder().to(device)
        print(f"[Decoder 'bicubic'] "
              f"{sum(p.numel() for p in decoder.parameters()):,} params")

    elif backend == "bicubic_trainable":
        class BicubicDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.init_size = calc_init_size(min(H, W))
                init_dim = C * self.init_size * self.init_size
                self.latent_to_spatial = nn.Linear(latent_dim, init_dim)

            def forward(self, z):
                B = z.size(0)
                h = self.latent_to_spatial(z)
                h = h.view(B, C, self.init_size, self.init_size)
                return F.interpolate(h, size=(H, W), mode='bicubic',
                                     align_corners=False)

        decoder = BicubicDecoder().to(device)
        print(f"[Decoder 'bicubic_trainable'] "
              f"{sum(p.numel() for p in decoder.parameters()):,} params")

    else:
        raise ValueError(f"Unknown decoder backend: '{backend}'.")

    return decoder
