"""
GMM4PR: Gaussian Mixture Model for Probabilistic Robustness
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    Categorical, Normal, Independent, MixtureSameFamily,
    LowRankMultivariateNormal, MultivariateNormal
)


class GMM4PR(nn.Module):
    """
    Gaussian Mixture Model for Probabilistic Robustness (GMM4PR).

    Models the distribution of adversarial perturbations as a K-component GMM
    in a (possibly compressed) latent space. Perturbation samples are decoded
    back to image space and projected onto a norm-ball budget (L-inf or L2),
    making the model suitable for probabilistic robustness training and
    evaluation of image classifiers.

    Architecture overview
    ---------------------
    - Optional frozen encoder (``feat_extractor``): maps raw images x to
      feature vectors that condition the GMM parameters.
    - Shared trunk: a single linear→BN→ReLU layer that processes the image
      features before branching into separate π / μ / Σ heads. Absent in
      unconditional and y-only modes.
    - Three output heads (or parameters when unconditional):
        * π head  – mixture-weight logits  [B, K]
        * μ head  – component means        [B, K, latent_dim]
        * Σ head  – covariance parameters  (type-dependent, see below)
    - Optional frozen decoder (``up_sampler``): maps latent samples back to
      image-shaped perturbations.

    Conditioning modes (set via ``set_condition``)
    -----------------------------------------------
    ``None``  Unconditional GMM; π, μ, Σ are free ``nn.Parameter`` tensors.
    ``"x"``   Conditioned on image features; all three heads use the shared
              trunk output.
    ``"y"``   Conditioned on class label only; π is a linear head over the
              label embedding / one-hot vector, while μ and Σ remain free
              parameters (closest to unconditional GMM).
    ``"xy"``  Joint conditioning: π uses the label embedding directly (no
              trunk), while μ and Σ use the shared trunk over image features.

    Covariance types (``cov_type``)
    --------------------------------
    ``"diag"``      Diagonal Gaussian; one log-std per dimension per component.
    ``"lowrank"``   Low-rank + diagonal: Σ = UUᵀ + diag(σ²), where U has rank
                    ``cov_rank``.
    ``"full"``      Full covariance via a learned Cholesky factor L (lower-
                    triangular with softplus diagonal).

    Temperature scaling
    -------------------
    Independent temperatures ``T_pi``, ``T_mu``, ``T_sigma``, and ``T_shared``
    scale the logits / parameters at inference time, allowing sharpening or
    flattening of the distribution without retraining.

    Sampling & perturbation generation
    ------------------------------------
    Differentiable samples are drawn via the Gumbel-Softmax trick for component
    selection combined with the reparameterisation trick for Gaussian sampling
    (``_rsample_from_gmm``). Latent samples are then decoded and projected onto
    the perturbation budget (``_project_to_budget``).

    Parameters
    ----------
    K : int
        Number of mixture components.
    latent_dim : int
        Dimensionality of the latent (perturbation) space.
    device : torch.device or str
        Device on which all sub-modules and parameters are allocated.
    T_pi : float, optional
        Temperature for mixture-weight logits (default 1.0).
    T_mu : float, optional
        Temperature for component means (default 1.0).
    T_sigma : float, optional
        Temperature for covariance parameters (default 1.0).
    T_shared : float, optional
        Temperature applied to the shared trunk output (default 1.0).
    logstd_bounds : tuple of (float, float), optional
        (min, max) clamp range for log-standard-deviation (default (-3.0, 1.0)).

    Key methods
    -----------
    set_condition(cond_mode, cov_type, cov_rank, feat_dim, num_cls, hidden_dim)
        Build all network heads / parameters for the chosen conditioning mode
        and covariance type. Must be called before any forward pass.
    set_feat_extractor(feat_extractor)
        Attach a frozen image encoder used in ``"x"`` and ``"xy"`` modes.
    set_up_sampler(up_sampler)
        Attach a frozen decoder that maps latent vectors to image-shaped tensors.
    set_y_embedding(num_cls, y_dim)
        Install a learnable label-embedding table (alternative to one-hot).
    set_budget(norm, eps)
        Configure the perturbation budget (norm type and radius).
    forward(x, y) -> dict
        Returns ``{"dist": MixtureSameFamily, "cache": dict}`` where ``cache``
        stores the raw π logits, μ, and covariance parameters for downstream
        use (regularisation, reparameterised sampling, etc.).
    """
    def __init__(self, K, latent_dim, device, T_pi=1.0, T_mu=1.0, T_sigma=1.0, 
                 T_shared=1.0, logstd_bounds=(-3.0, 1.0)):
        super().__init__()
        self.K = K
        self.latent_dim = latent_dim
        self.device = device 

        # Temperature parameters
        self.T_pi = T_pi
        self.T_mu = T_mu
        self.T_sigma = T_sigma
        self.T_shared = T_shared
        self.logstd_bounds = logstd_bounds
        self.budget = {"norm": "linf", "eps": 8/255}

        # Regularization coefficients with default values
        self.reg_coeffs = {
            'pi_entropy': 0.01,
            # 'component_usage': 0.001,
            'mean_diversity': 0.001,
            # 'kl_prior': 0.0001,
        }

        # Components
        # for inputs
        self.feat_extractor = None # this can be an external frozen encoder
        self.shared_trunk = None # shared trunk do not take y_embedding

        # for labels
        self.y_emb = None
        self.y_emb_normalize = True
        self.num_cls = None

        # for outputs of GMM
        self.up_sampler = None # this can be an external frozen decoder

    def _make_shared_trunk(self, in_dim, h_dim):
        """Create shared trunk for all heads. This is the alternative for external encoder.
           It takes in the feature from the CLS before making dependent for pi/mu/Sigma.
        """
        return nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
        ).to(self.device)

    def _make_head(self, h_dim, out_dim): # for pi, mu, Sigma heads
        """Create a single output head (last layer only)."""
        return nn.Linear(h_dim, out_dim).to(self.device)

    def set_y_embedding(self, num_cls, y_dim, normalize=True):
        """Install a learnable label embedding table."""
        self.y_emb = nn.Embedding(num_cls, y_dim).to(self.device)
        self.y_emb_normalize = normalize
        self.num_cls = num_cls

    def set_temperatures(self, T_pi=None, T_mu=None, T_sigma=None, T_shared=None):
        """Set temperature parameters."""
        if T_pi is not None:
            self.T_pi = T_pi
        if T_mu is not None:
            self.T_mu = T_mu
        if T_sigma is not None:
            self.T_sigma = T_sigma
        if T_shared is not None:
            self.T_shared = T_shared

    def set_regularization(self, **coeffs):
        """Set regularization coefficients."""
        self.reg_coeffs.update(coeffs)

    def set_condition(self, cond_mode, cov_type, cov_rank, feat_dim, num_cls, hidden_dim):
        self.cond_mode = cond_mode
        self.hidden_dim = hidden_dim # hidden dimension for shared trunk

        # input dimensions
        self.feat_dim = feat_dim
        y_dim = (self.y_emb.embedding_dim if (self.y_emb is not None) else num_cls)
        self.num_cls = num_cls

        # for covariance
        self.cov_type = cov_type
        self.cov_rank = cov_rank

        if cond_mode is None: # unconditional GMM
            self.pi = nn.Parameter(torch.randn(self.K, device=self.device) * 0.01) 
            self.mu = nn.Parameter(torch.zeros(self.K, self.latent_dim, device=self.device))
            self._init_cov_params(cov_type, cov_rank)

        elif cond_mode == "x":
            # Create SHARED TRUNK
            self.shared_trunk = self._make_shared_trunk(feat_dim, hidden_dim)

            # Create SEPARATE OUTPUT HEADS
            self.pi = self._make_head(hidden_dim, self.K)
            self.mu = self._make_head(hidden_dim, self.K * self.latent_dim)
            self._init_cov_heads(cov_type, cov_rank, hidden_dim)

        elif cond_mode == "xy": # for xy conditioning, pi head bypasses trunk and uses y embedding directly
            # Create SHARED TRUNK
            self.shared_trunk = self._make_shared_trunk(feat_dim, hidden_dim)
            
            # pi head bypasses trunk and uses y embedding directly
            self.pi = self._make_head(y_dim, self.K)

            # Create SEPARATE OUTPUT HEADS for mu and Sigma only
            self.mu = self._make_head(hidden_dim, self.K * self.latent_dim)
            self._init_cov_heads(cov_type, cov_rank, hidden_dim)

        elif cond_mode == "y": # for y conditioning only, this is the mode that most similar to unconditional GMM
            # check shared trunk is None in this case 
            assert self.shared_trunk is None, "For cond_mode='y', shared_trunk should be None"
            
            # pi head bypasses trunk and uses y embedding directly
            self.pi = self._make_head(y_dim, self.K)

            # unconditional mu and Sigma
            self.mu = nn.Parameter(torch.zeros(self.K, self.latent_dim, device=self.device))
            self._init_cov_params(cov_type, cov_rank)

        else:
            raise ValueError(f"cond_mode must be x/y/xy/none, got {cond_mode}")           
            
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        if self.cond_mode in ["x", "xy"]:
            trunk_params = sum(p.numel() for p in self.shared_trunk.parameters())
            pi_params = sum(p.numel() for p in self.pi.parameters())
            mu_params = sum(p.numel() for p in self.mu.parameters())

        elif self.cond_mode == "y":
            trunk_params = 0
            pi_params = sum(p.numel() for p in self.pi.parameters())
            mu_params = self.mu.numel()
        
        else: # unconditional
            trunk_params = 0
            pi_params = self.pi.numel()
            mu_params = self.mu.numel()

        cov_params = total_params - pi_params - mu_params
        print(f"[Params] Shared trunk: {trunk_params:,} | pi: {pi_params:,} | mu: {mu_params:,}, | cov: {cov_params:,} | Total: {total_params:,}")

    def _init_cov_params(self, cov_type, cov_rank):
        """Initialize unconditional covariance parameters."""
        if cov_type == "diag": # [K, D]
            self.log_sigma = nn.Parameter(torch.zeros(self.K, self.latent_dim, device=self.device))

        elif cov_type == "lowrank": # [K, D] and [K, D, r]
            self.log_sigma = nn.Parameter(torch.zeros(self.K, self.latent_dim, device=self.device))
            self.U = nn.Parameter(torch.zeros(self.K, self.latent_dim, cov_rank, device=self.device))

        elif cov_type == "full": # [K, D, D]
            self.L_raw = nn.Parameter(torch.zeros(self.K, self.latent_dim, self.latent_dim, device=self.device))
        else:
            raise ValueError(f"cov_type must be diag/full/lowrank, got {cov_type}")

    def _init_cov_heads(self, cov_type, cov_rank, hidden_dim): # the outputs needs reshape to original shape later
        """Initialize conditional covariance heads."""
        if cov_type == "diag":
            self.logsig = self._make_head(hidden_dim, self.K * self.latent_dim)
        elif cov_type == "lowrank":
            self.logsig = self._make_head(hidden_dim, self.K * self.latent_dim)
            self.U = self._make_head(hidden_dim, self.K * self.latent_dim * cov_rank)
        elif cov_type == "full":
            self.L = self._make_head(hidden_dim, self.K * self.latent_dim * self.latent_dim)
        else:
            raise ValueError(f"cov_type must be diag/full/lowrank, got {cov_type}")

    def set_feat_extractor(self, feat_extractor):
        """Set frozen feature extractor."""
        self.feat_extractor = feat_extractor.to(self.device)
        for p in self.feat_extractor.parameters():
            p.requires_grad = False

    def set_up_sampler(self, up_sampler): 
        """Set frozen decoder."""
        self.up_sampler = up_sampler.to(self.device)

    def set_budget(self, norm="linf", eps=8/255):
        """Set perturbation budget."""
        self.budget = {"norm": norm, "eps": float(eps)}

    def _make_condition(self, x=None, y=None): # x, y can be both None
        """Build conditioning vector from x and/or y."""
        mode = self.cond_mode
        part_x = None
        part_y = None

        # setup batch size
        B = x.size(0) if x is not None else (y.size(0) if y is not None else 1) # batch size            

        # similar structure as method of set_condition
        if mode is None: # create dummy condition
            part_x = part_y = torch.ones(B, 1, device=self.device)

        elif mode == "x":
            # check necessary inputs
            if self.feat_extractor is None:
                raise ValueError(f"cond_mode={mode} requires feat_extractor")
            if x is None:
                raise ValueError(f"cond_mode={mode} requires x input")
            with torch.no_grad():
                part_x = self.feat_extractor(x).view(x.size(0), -1)
            part_y = torch.ones(B, 1, device=self.device) # dummy y part

        elif mode == "xy": 
            if self.feat_extractor is None:
                raise ValueError(f"cond_mode={mode} requires feat_extractor")
            if x is None or y is None:
                raise ValueError(f"cond_mode={mode} requires both x and y inputs")
            with torch.no_grad():
                part_x = self.feat_extractor(x).view(x.size(0), -1)

            if self.num_cls is None:
                raise ValueError("self.num_cls must be set for xy-conditioning")

            if self.y_emb is not None: # embedding the y
                yvec = self.y_emb(y)
                if self.y_emb_normalize:
                    yvec = F.normalize(yvec, dim=-1)
            else: # one-hot encoding for the y
                yvec = F.one_hot(y, num_classes=self.num_cls).float().to(self.device)            
            part_y = yvec

        elif mode == "y":
            if y is None:
                raise ValueError(f"cond_mode={mode} requires y input")
            if self.num_cls is None:
                raise ValueError("self.num_cls must be set for y-conditioning")

            if self.y_emb is not None:
                yvec = self.y_emb(y)
                if self.y_emb_normalize:
                    yvec = F.normalize(yvec, dim=-1)
            else:
                yvec = F.one_hot(y, num_classes=self.num_cls).float().to(self.device)

            part_x = torch.ones(B, 1, device=self.device)
            part_y = yvec
        else:
            raise ValueError(f"cond_mode must be x/y/xy/none, got {mode}")

        if (part_x is None) or (part_y is None):
            raise ValueError(f"No enough conditioning created for mode={mode}")

        return part_x, part_y

    def _decode_latent(self, eps, out_shape):
        """Map latent eps to image-shaped u."""
        if self.up_sampler is None:
            assert eps.size(-1) == np.prod(out_shape), \
                f"Latent vector size {eps.size(-1)} does not match output shape {out_shape}"
            return eps.view(eps.shape[:-1] + out_shape)

        if eps.dim() == 2: # for no batch or single sample
            u = self.up_sampler(eps)
            return u.view(eps.size(0), *out_shape) if u.dim() == 2 else u
        elif eps.dim() == 3:
            S, B, D = eps.shape
            u = self.up_sampler(eps.reshape(-1, D)) # (S*B, D) -> (S*B, C, H, W)
            if u.dim() == 2: # MLP decoder needs reshaping
                return u.view(S, B, *out_shape)
            return u.view(S, B, *u.shape[1:])
        else:
            raise ValueError(f"eps must be 2D or 3D, got shape {eps.shape}")

    def _project_to_budget(self, u):
        """Project u to perturbation budget."""
        norm = self.budget["norm"].lower()
        eps = float(self.budget["eps"])

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
        
        raise ValueError(f"Unsupported norm={norm} or shape={u.shape}")

    # build GMM distribution with temperature scaling
    def _build_dist(self, part_x, part_y): 
        """Build GMM with temperature scaling."""
        B = part_x.size(0)
        # part_x and part_y should always have the same batch size
        if part_y is not None and part_y.size(0) != B:
            raise ValueError(f"Batch size mismatch between x_cond ({B}) and y_cond ({part_y.size(0)})")
        K, D = self.K, self.latent_dim

        # shared trunk with temperature
        if self.cond_mode is None:
            assert self.shared_trunk is None, "Unconditional GMM should not have shared trunk"
            h_shared = None

            assert all(isinstance(getattr(self, name, None), nn.Parameter) for name in ("pi","mu")), \
                "Unconditional GMM requires pi and mu to be parameters"
            pi_logits = self.pi.unsqueeze(0).expand(B, -1) * self.T_pi
            mu = self.mu.unsqueeze(0).expand(B, K, D) * self.T_mu

        elif self.cond_mode == "x":
            assert self.shared_trunk is not None, "cond_mode='x' requires shared_trunk"
            h_shared = self.shared_trunk(part_x) * self.T_shared

            assert all(callable(getattr(self, name, None)) for name in ("pi", "mu")), \
                "Conditional GMM of x requires pi and mu to be mappings"            
            pi_logits = self.pi(h_shared) * self.T_pi
            mu_flat = self.mu(h_shared) * self.T_mu
            mu = mu_flat.view(B, K, D)

        elif self.cond_mode == "xy":
            assert self.shared_trunk is not None, "cond_mode='xy' requires shared_trunk"
            h_shared = self.shared_trunk(part_x) * self.T_shared

            assert all(callable(getattr(self, name, None)) for name in ("pi", "mu")), \
                "Conditional GMM of xy requires pi and mu to be mappings"            
            pi_logits = self.pi(part_y) * self.T_pi
            mu_flat = self.mu(h_shared) * self.T_mu
            mu = mu_flat.view(B, K, D)

        elif self.cond_mode == "y":
            assert self.shared_trunk is None, "conditional GMM of y should not have shared trunk"
            h_shared = None

            assert callable(getattr(self, "pi", None)), "Conditional GMM of y requires pi to be a mapping"
            pi_logits = self.pi(part_y) * self.T_pi

            assert isinstance(self.mu, nn.Parameter), "Conditional GMM of y requires mu to be a parameter"
            mu = self.mu.unsqueeze(0).expand(B, K, D) * self.T_mu

        else:
            raise ValueError(f"Unknown cond_mode: {self.cond_mode}")
        
        # setup mixtures
        mix = Categorical(logits=pi_logits)
        cache = {"pi_logits": pi_logits, "mu": mu}

        # Covariance with T_sigma
        ## no need to separate the case of x/xy/y/none, since h_shared is either None or from shared trunk
        ## all the case have been included in _get_log_std/_get_U/_get_cholesky
        if self.cov_type == "diag":
            log_std = self._get_log_std(h_shared, B, K, D)
            log_std = log_std + torch.log(torch.tensor(self.T_sigma, device=part_x.device))
            std = torch.exp(log_std)
            comp = Independent(Normal(mu, std), 1)
            cache["log_std"] = log_std

        elif self.cov_type == "lowrank":
            log_std = self._get_log_std(h_shared, B, K, D)
            log_std = log_std + torch.log(torch.tensor(self.T_sigma, device=part_x.device))
            std2 = torch.exp(2.0 * log_std)
            
            U = self._get_U(h_shared, B, K, D)
            U = U * np.sqrt(self.T_sigma)
            
            comp = LowRankMultivariateNormal(loc=mu, cov_factor=U, cov_diag=std2)
            cache.update({"log_std": log_std, "U": U})

        elif self.cov_type == "full":
            L = self._get_cholesky(h_shared, B, K, D)
            L = L * np.sqrt(self.T_sigma)
            comp = MultivariateNormal(loc=mu, scale_tril=L)
            cache["L"] = L

        else:
            raise ValueError(f"Unknown cov_type: {self.cov_type}")

        dist = MixtureSameFamily(mixture_distribution=mix, component_distribution=comp)
        return dist, cache

    def _get_log_std(self, h_shared, B, K, D):
        """Get clamped log standard deviation."""
        if callable(getattr(self, "logsig", None)):
            log_std = self.logsig(h_shared).view(B, K, D)
        else:
            log_std = self.log_sigma.unsqueeze(0).expand(B, K, D)
        
        lo, hi = self.logstd_bounds
        return torch.clamp(log_std, lo, hi)

    def _get_U(self, h_shared, B, K, D):
        """Get low-rank factor U."""
        if callable(getattr(self, "U", None)):
            U = self.U(h_shared).view(B, K, D, self.cov_rank)
        else:
            U = self.U.unsqueeze(0).expand(B, K, D, self.cov_rank)
        return U

    def _get_cholesky(self, h_shared, B, K, D):
        """Get Cholesky factor for full covariance."""
        if callable(getattr(self, "L", None)):
            L_raw = self.L(h_shared).view(B, K, D, D)
        else:
            L_raw = self.L_raw.unsqueeze(0).expand(B, K, D, D)

        tril_mask = torch.tril(torch.ones(D, D, device=self.device, dtype=torch.bool))
        L = torch.zeros_like(L_raw)
        L[..., tril_mask] = L_raw[..., tril_mask]

        diag_idx = torch.arange(D, device=self.device)
        L[..., diag_idx, diag_idx] = F.softplus(L[..., diag_idx, diag_idx]) + 1e-4
        
        return L

    def compute_regularization(self, cache):
        """Compute regularization terms."""
        reg_terms = {}
        pi_logits = cache['pi_logits']
        mu = cache['mu']
        B, K, D = mu.shape

        # Pi entropy
        pi_probs = F.softmax(pi_logits, dim=-1)
        pi_entropy = Categorical(probs=pi_probs).entropy()
        norm_entropy_loss = (1.0 - pi_entropy / torch.log(torch.tensor(pi_probs.size(-1), dtype=torch.float32)))

        reg_terms['pi_entropy'] = norm_entropy_loss.mean() # minimize it encourages more spread pi

        # Mean diversity
        # mu: [B, K, D]
        K = mu.size(1)

        # 1) Normalization：care about diversity
        mu_hat = F.normalize(mu, p=2, dim=-1, eps=1e-8)  # [B, K, D]

        # 2) Gram matrix + stabilization
        gram = torch.bmm(mu_hat, mu_hat.transpose(1, 2))  # [B, K, K]
        gram = gram + 1e-6 * torch.eye(K, device=mu.device, dtype=gram.dtype).unsqueeze(0)

        # 3) Stable log determinant
        sign, logabsdet = torch.linalg.slogdet(gram)  # [B], [B]

        # 5) Diversity loss（maximal volume = minimal log det）
        target_diversity = torch.log(torch.tensor(float(K), device=mu.device))
        diversity_loss = F.relu(target_diversity - logabsdet).mean() # make it non-negative


        # (Optional) Check during debugging
        if self.training and torch.rand(1).item() < 0.05:  # 5% chance to check
            if (sign < 0).any():
                print(f"[Warning] Negative det: sign={sign.min().item()}")

        reg_terms['mean_diversity'] = diversity_loss

        return reg_terms

    def forward(self, x=None, y=None):
        """Build distribution."""
        part_x, part_y = self._make_condition(x=x, y=y)
        dist, cache = self._build_dist(part_x, part_y)
        return {"dist": dist, "cache": cache}

    def _rsample_from_gmm(self, cache, num_samples, temperature=1.0):
        """
        Reparameterized sampling from GMM using Gumbel-Softmax trick.
        
        Args:
            cache: Distribution cache with pi_logits, mu, and covariance parameters
            num_samples: Number of samples to draw
            temperature: Temperature for Gumbel-Softmax (lower = more discrete)
        
        Returns:
            samples: [num_samples, B, D] with gradients
        """
        pi_logits = cache['pi_logits']  # [B, K]
        mu = cache['mu']  # [B, K, D]
        B, K, D = mu.shape
        
        # ============ STEP 1: Gumbel-Softmax for component selection ============
        # Sample Gumbel noise: G = -log(-log(U)) where U ~ Uniform(0,1)
        gumbel_noise = -torch.log(-torch.log(
            torch.rand(num_samples, B, K, device=pi_logits.device) + 1e-20
        ) + 1e-20)
        
        # Add to logits and apply softmax with temperature
        logits_with_gumbel = (pi_logits.unsqueeze(0) + gumbel_noise) / temperature
        
        soft_component_weights = F.softmax(logits_with_gumbel, dim=-1)  # [S, B, K], doing on the K dimension
        
        # ============ STEP 2: Reparameterized Gaussian sampling ============
        # Sample from standard normal
        eps_std = torch.randn(num_samples, B, K, D, device=mu.device)  # [S, B, K, D]
        
        # Reparameterize based on covariance type
        if self.cov_type == "diag":
            log_std = cache['log_std']  # [B, K, D]
            std = torch.exp(log_std)
            
            # Expand dimensions for broadcasting
            mu_expanded = mu.unsqueeze(0)  # [1, B, K, D]
            std_expanded = std.unsqueeze(0)  # [1, B, K, D]
            
            # Sample: z_k = μ_k + σ_k * ε for each component
            component_samples = mu_expanded + std_expanded * eps_std  # [S, B, K, D]
        
        elif self.cov_type == "lowrank":
            log_std = cache['log_std']  # [B, K, D]
            U = cache['U']  # [B, K, D, r]
            
            std = torch.exp(log_std)
            
            # Sample from low-rank: z = μ + U*η + σ*ε
            # where η ~ N(0, I_r) and ε ~ N(0, I_D)
            eta = torch.randn(num_samples, B, K, self.cov_rank, device=mu.device) # [S, B, K, r]
            
            mu_expanded = mu.unsqueeze(0)  # [1, B, K, D]
            std_expanded = std.unsqueeze(0)  # [1, B, K, D]
            U_expanded = U.unsqueeze(0)  # [1, B, K, D, r]
            
            # Low-rank component: U @ η, [1, B, K, D, r] @ [S, B, K, r, 1] -> [S, B, K, D, 1] -> [S, B, K, D]
            lowrank_term = torch.matmul(U_expanded, eta.unsqueeze(-1)).squeeze(-1)
            
            # Diagonal component: σ * ε_diag
            eps_diag = torch.randn(num_samples, B, K, D, device=mu.device)
            
            # Full sample: μ + U*η + σ*ε
            component_samples = mu_expanded + lowrank_term + std_expanded * eps_diag
        
        elif self.cov_type == "full":
            L = cache['L']  # [B, K, D, D] - Cholesky factor
            
            mu_expanded = mu.unsqueeze(0)  # [1, B, K, D]
            L_expanded = L.unsqueeze(0)  # [1, B, K, D, D]
            
            # Sample: z = μ + L @ ε
            # eps_std: [S, B, K, D] -> [S, B, K, D, 1]
            component_samples = mu_expanded + torch.matmul(
                L_expanded, eps_std.unsqueeze(-1)
            ).squeeze(-1)  # [1, B, K, D, D] @ [S, B, K, D, 1] -> [S, B, K, D, 1] -> [S, B, K, D]
        
        else:
            raise ValueError(f"Unknown cov_type: {self.cov_type}")
        
        # ============ STEP 3: Weighted combination using soft weights ============
        # Expand weights: [S, B, K] -> [S, B, K, 1]
        weights = soft_component_weights.unsqueeze(-1)
        # Weighted sum over components: [S, B, K, D] * [S, B, K, 1] -> [S, B, D]
        samples = (component_samples * weights).sum(dim=2)
        
        return samples

    def _sample_and_classify(self, x, num_samples, classifier, cache, temperature=1.0):
        """
        Sample perturbations with GRADIENTS, apply to images, and classify.
        
        Args:
            x: Clean images [B, C, H, W]
            num_samples: Number of samples to draw
            classifier: Classifier model
            cache: Distribution cache from forward() containing pi_logits, mu, and cov params
            temperature: Temperature for Gumbel-Softmax (default=1.0)
        
        Returns:
            logits: [num_samples, B, num_classes]
        """
        B = x.size(0)
        
        # Reparameterized sampling (WITH GRADIENTS)
        eps = self._rsample_from_gmm(cache, num_samples, temperature=temperature)
        
        # Decode to image space
        u = self._decode_latent(eps, out_shape=x.shape[1:])
        
        # Project to budget
        delta = self._project_to_budget(u)
        # delta.view(delta.shape[0], delta.shape[1], -1).norm(p=2, dim=-1, keepdim=True)
        # Replicate clean images for broadcasting
        x_rep = x.unsqueeze(0).expand_as(delta)
        
        # Classify perturbed images
        # Flatten samples and batch: [S, B, C, H, W] -> [S*B, C, H, W]
        logits = classifier((x_rep + delta).flatten(0, 1))
        
        # Reshape to [num_samples, B, num_classes]
        logits = logits.view(num_samples, B, -1)
        
        return logits


    def pr_loss(self, x, y, classifier, num_samples=8, loss_variant="cw", kappa=0.0, 
                chunk_size=None, return_reg_details=False, gumbel_temperature=1.0):
        """Compute loss with regularization."""
        out = self.forward(x=x, y=y)
        # self.out_test = out  # for debugging

        cache = out["cache"]
        B = x.size(0)

        # Adaptive chunking
        if chunk_size is None:
            max_batch = 32
            chunk_size = max(1, max_batch // B)
        
        # Sample and classify (with optional chunking)
        if num_samples <= chunk_size:
            # No chunking needed
            logits = self._sample_and_classify(x, num_samples, classifier, cache, gumbel_temperature)
        else:
            # Chunking for memory efficiency
            logits_list = []
            num_chunks = (num_samples + chunk_size - 1) // chunk_size
            
            for i in range(num_chunks):
                chunk_samples = min(chunk_size, num_samples - i * chunk_size)
                logits_chunk = self._sample_and_classify(x, chunk_samples, classifier, cache, gumbel_temperature)
                logits_list.append(logits_chunk)
            
            logits = torch.cat(logits_list, dim=0)

        logits = logits - logits.max(dim=-1, keepdim=True).values  # shift-invariant; avoids huge magnitudes
        # Main loss
        if loss_variant == "cw":
            # Carlini-Wagner loss: minimize margin between correct class and best other class
            y_rep = y.unsqueeze(0).expand(num_samples, -1)
            logit_y = logits.gather(-1, y_rep.unsqueeze(-1)).squeeze(-1)
            mask = F.one_hot(y_rep, logits.size(-1)).bool()
            max_others = logits.masked_fill(mask, float("-inf")).max(-1).values

            margin = logit_y - max_others + kappa
            # main_loss = F.softplus(margin).mean()
            main_loss = F.softplus(margin).mean()

        else:
            main_loss = 1 - F.cross_entropy(
                logits.flatten(0, 1),
                y.unsqueeze(0).expand(num_samples, -1).flatten()
            )

        # Regularization
        reg_terms = self.compute_regularization(cache)
        total_reg = sum(self.reg_coeffs.get(k, 0.0) * v for k, v in reg_terms.items())

        total_loss = main_loss + total_reg

        # running PR metrics - using canonical compute_pr() method
        predictions = logits.argmax(dim=-1)  # [num_samples, B]
        pr = self.compute_pr(predictions, y, reduction='mean').item()

        result = {
            "pr": pr,
            "loss": total_loss,
            "main": main_loss.detach(),
            "reg": total_reg.detach(),
        }
        
        if return_reg_details:
            result["reg_details"] = {k: v.detach().item() for k, v in reg_terms.items()}
            result["pi_probs"] = F.softmax(cache['pi_logits'], dim=-1).mean(dim=0).detach()
        
        return result

    @staticmethod
    def compute_pr(predictions, y, reduction='mean'):
        """
        Compute Probabilistic Robustness from predictions.

        This is the canonical PR computation method. All other PR calculations
        should use this method to ensure consistency.

        Args:
            predictions: [S, B] or [S*B] predicted labels
                        S = number of samples per image
                        B = batch size (number of images)
            y: [B] ground truth labels
            reduction: 'mean' | 'sum' | 'none'
                      - 'mean': return scalar average PR across all images
                      - 'sum': return sum of per-image PR
                      - 'none': return per-image PR values

        Returns:
            pr: Scalar (if reduction='mean'/'sum') or [B] tensor (if reduction='none')

        """
        # Ensure consistent shape
        if predictions.dim() == 1:
            # Flat format [S*B] -> reshape to [S, B]
            S_times_B = predictions.size(0)
            B = y.size(0)
            if S_times_B % B != 0:
                raise ValueError(f"predictions size {S_times_B} not divisible by y size {B}")
            S = S_times_B // B
            predictions = predictions.view(S, B)
        elif predictions.dim() == 2:
            S, B = predictions.shape
            if y.size(0) != B:
                raise ValueError(f"predictions batch size {B} != y size {y.size(0)}")
        else:
            raise ValueError(f"predictions must be 1D or 2D, got shape {predictions.shape}")

        # Expand y: [B] -> [S, B]
        y_expanded = y.unsqueeze(0).expand(S, -1)

        # Compute success indicator: [S, B]
        success = predictions.eq(y_expanded).float()

        # Compute per-image success rate: [B]
        per_image_pr = success.mean(dim=0)

        # Handle deprecated per_sample parameter

        # Apply reduction
        if reduction == 'mean':
            return per_image_pr.mean()
        elif reduction == 'sum':
            return per_image_pr.sum()
        elif reduction == 'none':
            return per_image_pr
        else:
            raise ValueError(f"Unknown reduction: {reduction}. Use 'mean', 'sum', or 'none'.")

    @torch.no_grad()
    def evaluate_pr(self, x, y, classifier, num_samples=100,
                    use_soft_sampling=False, temperature=1.0, reduction='none',
                    chunk_size=None):
        """
        Evaluate Probabilistic Robustness on a batch of images.

        This is the unified evaluation method that works consistently for both
        hard (categorical) and soft (Gumbel-Softmax) sampling. Supports chunking
        for memory efficiency with large num_samples.

        Args:
            x: Images [B, C, H, W]
            y: Labels [B]
            classifier: Classifier model (should be in eval mode)
            num_samples: Number of samples per image
            use_soft_sampling: If True, use Gumbel-Softmax (_rsample_from_gmm).
                              If False, use hard categorical sampling (dist.sample).
            temperature: Temperature for Gumbel-Softmax (only used if use_soft_sampling=True)
            reduction: 'mean' | 'sum' | 'none' - how to reduce per-image PR values
            chunk_size: Maximum samples to process at once. If None, uses adaptive chunking.
                       Useful for large num_samples to avoid OOM errors.

        Returns:
            If reduction='none': [B] per-image PR values
            If reduction='mean': scalar mean PR
            If reduction='sum': scalar sum of PR

        """
        B = x.size(0)

        # Adaptive chunking (same logic as pr_loss)
        if chunk_size is None:
            max_batch = 32
            chunk_size = max(1, max_batch // B)

        # Process with chunking if needed
        if num_samples <= chunk_size:
            # No chunking needed - process all samples at once
            predictions = self._evaluate_chunk(
                x, y, num_samples, classifier,
                use_soft_sampling, temperature)
        else:
            # Chunking for memory efficiency
            predictions_list = []
            num_chunks = (num_samples + chunk_size - 1) // chunk_size

            for i in range(num_chunks):
                chunk_samples = min(chunk_size, num_samples - i * chunk_size)
                predictions_chunk = self._evaluate_chunk(
                    x, y, chunk_samples, classifier,
                    use_soft_sampling, temperature)
                predictions_list.append(predictions_chunk)

            # Concatenate all chunks: list of [S_i, B] -> [S_total, B]
            predictions = torch.cat(predictions_list, dim=0)

        # Compute PR using the canonical method
        pr = self.compute_pr(predictions, y, reduction=reduction)

        return pr

    def _evaluate_chunk(self, x, y, num_samples, classifier,
                       use_soft_sampling, temperature):
        """
        Helper method to evaluate a chunk of samples.

        Args:
            x: Images [B, C, H, W]
            forward_out: Output from forward() containing dist and cache
            num_samples: Number of samples for this chunk
            classifier: Classifier model
            use_soft_sampling: Whether to use Gumbel-Softmax
            temperature: Temperature for Gumbel-Softmax
            out_shape: Output shape (C, H, W)

        Returns:
            predictions: [num_samples, B] predicted labels
        """
        B = x.size(0)

        # Build distribution once (reused across chunks)
        forward_out = self.forward(x=x, y=y)

        # Sample perturbations (using the appropriate method)
        if use_soft_sampling:
            # Use Gumbel-Softmax (differentiable, used during training)
            cache = forward_out["cache"]
            eps = self._rsample_from_gmm(cache, num_samples, temperature=temperature)
        else:
            # Use hard categorical sampling (true GMM distribution)
            dist = forward_out["dist"]
            eps = dist.sample((num_samples,))

        # Decode latent to image space
        u = self._decode_latent(eps, out_shape=x.shape[1:])

        # Project to perturbation budget
        delta = self._project_to_budget(u)

        # Apply perturbations and classify
        x_rep = x.unsqueeze(0).expand_as(delta)  # [S, B, C, H, W]

        # Flatten and classify: [S, B, C, H, W] -> [S*B, C, H, W]
        logits = classifier((x_rep + delta).flatten(0, 1))  # [S*B, num_classes]

        # Get predictions: [S*B] -> [S, B]
        predictions = logits.argmax(dim=-1).view(num_samples, B)

        return predictions

    @torch.no_grad()
    def sample(self, x=None, y=None, num_samples=1, out_shape=None, chunk_size=None):
        """
        Sample perturbations from the learned GMM.
        
        Args:
            x: Input images [B, C, H, W] (required for x/xy conditioning)
            y: Labels [B]/[B, Em] (required for y/xy conditioning)  
            num_samples: Number of perturbation samples per image
            out_shape: Output shape (C, H, W). Required if x is None.
            chunk_size: If provided and num_samples > chunk_size, samples in chunks
                        to save memory. Useful for very large num_samples (e.g., 1000+).
        
        Returns:
            dict with keys:
                - 'eps': Latent samples [num_samples, B, D] or [num_samples, D]
                - 'u': Decoded samples [num_samples, B, C, H, W] or [num_samples, C, H, W]
                - 'delta': Projected perturbations (same shape as u)
        
        Example:
            # Conditional sampling (infers shape from x)
            out = gmm.sample(x=images, y=labels, num_samples=10)
            
            # Unconditional sampling (must provide shape)
            out = gmm.sample(x=None, y=None, num_samples=100, out_shape=(3, 32, 32))
            
            # Large-scale sampling with chunking
            out = gmm.sample(x=images, y=labels, num_samples=5000, chunk_size=100)
        """
        # Validate conditioning requirements
        if self.cond_mode in ("x", "xy") and x is None:
            raise ValueError(
                f"GMM trained with cond_mode='{self.cond_mode}' requires x input.\n"
                f"Provide x or train a model with different conditioning."
            )
        if self.cond_mode in ("y", "xy") and y is None:
            raise ValueError(
                f"GMM trained with cond_mode='{self.cond_mode}' requires y input.\n"
                f"Provide y or train a model with different conditioning."
            )
        
        # Build distribution
        out = self.forward(x=x, y=y)
        dist = out["dist"]
        
        # Infer output shape
        if x is not None:
            out_shape = x.shape[1:]  # (C, H, W)
        elif out_shape is None:
            raise ValueError(
                "out_shape must be provided when x is None.\n"
                "Pass out_shape=(C, H, W) explicitly, e.g., out_shape=(3, 32, 32)"
            )
        
        # Decide whether to chunk
        use_chunking = (chunk_size is not None and num_samples > chunk_size)
        
        if not use_chunking:
            # Simple path: sample all at once
            eps = dist.sample((num_samples,))
            u = self._decode_latent(eps, out_shape=out_shape)
            delta = self._project_to_budget(u)
            return {"eps": eps, "u": u, "delta": delta}
        
        # Chunked path: sample in batches
        eps_list, u_list, delta_list = [], [], []
        num_chunks = (num_samples + chunk_size - 1) // chunk_size # standard ceiling division trick
        
        for i in range(num_chunks):
            chunk_samples = min(chunk_size, num_samples - i * chunk_size)
            
            # Sample chunk
            eps_chunk = dist.sample((chunk_samples,))
            eps_list.append(eps_chunk)
            
            # Decode and project chunk
            u_chunk = self._decode_latent(eps_chunk, out_shape=out_shape)
            delta_chunk = self._project_to_budget(u_chunk)
            u_list.append(u_chunk)
            delta_list.append(delta_chunk)
        
        # Concatenate all chunks
        eps = torch.cat(eps_list, dim=0)
        u = torch.cat(u_list, dim=0)
        delta = torch.cat(delta_list, dim=0)
        
        return {"eps": eps, "u": u, "delta": delta}

    def save(self, path, extra=None):
        """Save checkpoint."""
        cfg = dict(
            ## initialization parameters ##
            K=self.K, 
            latent_dim=self.latent_dim,
            # Do not save device info, as it may differ on load

            # temperature parameters
            T_pi=self.T_pi,
            T_mu=self.T_mu, 
            T_sigma=self.T_sigma, 
            T_shared=self.T_shared,
            logstd_bounds=self.logstd_bounds, 
            budget=self.budget,
            
            # regularization coefficients
            reg_coeffs=self.reg_coeffs,
            ## condition parameters (end) ##

            # for set_y_embedding
            has_y_emb=(self.y_emb is not None),
            y_emb_dim=(self.y_emb.embedding_dim if self.y_emb is not None else None),
            y_emb_normalize=self.y_emb_normalize,

            # for set_condition
            cond_mode=self.cond_mode, 
            cov_type=self.cov_type,
            cov_rank=self.cov_rank, 
            feat_dim=self.feat_dim,
            num_cls=self.num_cls, 
            hidden_dim=self.hidden_dim,  
        )
        if extra:
            cfg.update(extra)
        torch.save({"state_dict": self.state_dict(), "config": cfg}, path)
        print(f"Model saved to {path}")

    @classmethod
    def load_from_checkpoint(cls, path, feat_extractor=None, up_sampler=None, 
                            map_location="cpu", strict=True):
        """
        Load model from checkpoint with automatic architecture reconstruction.
        
        Args:
            path: Path to checkpoint
            feat_extractor: Optional frozen feature extractor
            up_sampler: Optional frozen decoder
            map_location: Device to load to
            strict: Whether to strictly enforce state dict loading
        
        Returns:
            Loaded GMM4PR model
        """
        ckpt = torch.load(path, map_location=map_location)
        cfg = ckpt["config"]
        
        # Create model with basic config (initialization parameters)
        model = cls(
            K=cfg["K"],
            latent_dim=cfg["latent_dim"],
            device=map_location,
            T_pi=cfg.get("T_pi", 1.0),
            T_mu=cfg.get("T_mu", 1.0),
            T_sigma=cfg.get("T_sigma", 1.0),
            T_shared=cfg.get("T_shared", 1.0),
            logstd_bounds=cfg.get("logstd_bounds", (-3.0, 1.0))
        )
        
        # Set up label embedding if it was used
        if cfg.get("has_y_emb", False):
            model.set_y_embedding(
                num_cls=cfg["num_cls"],
                y_dim=cfg["y_emb_dim"],
                normalize=cfg.get("y_emb_normalize", True)
            )
        
        # Set up architecture
        model.set_condition(
            cond_mode=cfg["cond_mode"],
            cov_type=cfg["cov_type"],
            cov_rank=cfg.get("cov_rank", 0),
            feat_dim=cfg["feat_dim"],
            num_cls=cfg["num_cls"],
            hidden_dim=cfg["hidden_dim"]
        )

        # this is my fault, since some of the loaded model save the feat_extractor inside
        # but older ones do not have it, shit! especially the unconditional ones
        if any("feat_extractor" in k for k in ckpt["state_dict"].keys()):
            # Set external components if provided
            if feat_extractor is not None:
                model.set_feat_extractor(feat_extractor)
        if up_sampler is not None:
            model.set_up_sampler(up_sampler)
        
        # Set budget and regularization
        model.set_budget(**cfg.get("budget", {"norm": "linf", "eps": 8/255}))
        model.set_regularization(**cfg.get("reg_coeffs", {}))
        
        # Load weights
        model.load_state_dict(ckpt["state_dict"], strict=strict)
        
        if not any("feat_extractor" in k for k in ckpt["state_dict"].keys()):
            # Set external components if provided
            if feat_extractor is not None:
                model.set_feat_extractor(feat_extractor)


        print(f"Model loaded from {path}")
        return model
    
    

def build_decoder_from_flag(backend: str, latent_dim: int, out_shape: tuple, device):
    """
    Build decoder that maps latent_dim -> out_shape.
    
    Args:
        backend: Decoder type ('bicubic', 'wavelet', 'dct', 'nearest_blur', 
                                'conv', 'upsample', 'tiny', 'mlp')
        latent_dim: Dimensionality of latent space
        out_shape: Output shape (C, H, W)
        device: Target device
    
    Returns:
        decoder: nn.Module that maps [B, latent_dim] -> [B, C, H, W]
    """
    C, H, W = out_shape
    
    # Helper function to calculate adaptive sizes
    def calc_init_size(target_size):
        """Calculate initial spatial size for progressive upsampling."""
        # Start from 4x4 or 7x7, whichever is more appropriate
        if target_size <= 32:
            return 4  # For CIFAR-10, MNIST, etc.
        elif target_size <= 64:
            return 7  # For 64x64 images
        else:
            return target_size // 32  # For larger images
    
    
    if backend == "bicubic":
        class BicubicDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.init_size = calc_init_size(min(H, W))
                self.init_dim = C * self.init_size * self.init_size
            
            def forward(self, z):
                B = z.size(0)
                assert z.size(1) == self.init_dim, f"Expected latent_dim={self.init_dim}, got {z.size(1)}"

                z = z.view(B, C, self.init_size, self.init_size)
                return F.interpolate(z, size=(H, W), mode='bicubic', align_corners=False)

        decoder = BicubicDecoder().to(device)
        print(f"[Decoder 'bicubic'] {sum(p.numel() for p in decoder.parameters()):,} params")

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
                return F.interpolate(h, size=(H, W), mode='bicubic', align_corners=False)
        
        decoder = BicubicDecoder().to(device)
        print(f"[Decoder 'bicubic'] {sum(p.numel() for p in decoder.parameters()):,} params")
    
    
    else:
        raise ValueError(
            f"Unknown decoder backend: '{backend}'."
        )
    
    return decoder