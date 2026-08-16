"""
Configuration file for GMM4PR training.
Basic default configurations only.
Use command-line arguments to override these defaults.
"""


import torch
import torch.nn as nn
import numpy as np


def initialize_gmm_parameters(gmm, init_mode='spread'):
    """Initialize GMM parameters with different strategies."""
    with torch.no_grad():
        if hasattr(gmm, 'mu') and isinstance(gmm.mu, nn.Parameter):
            K, D = gmm.mu.shape

            if init_mode == 'spread':
                # Original binary pattern
                gmm.mu.data.normal_(0, 0.5)
                if K <= 8 and D >= 3:
                    for k in range(min(K, 8)):
                        binary = format(k, '03b')
                        for d, bit in enumerate(binary):
                            if d < D:
                                gmm.mu.data[k, d] = 1.0 if bit == '1' else -1.0

            elif init_mode == 'random':
                gmm.mu.data.normal_(0, 1.0)

            elif init_mode == 'grid':
                # Evenly spaced grid
                if D >= 2:
                    side = int(np.ceil(K ** (1/2)))
                    for k in range(K):
                        i, j = k // side, k % side
                        gmm.mu.data[k, 0] = (i / side) * 2 - 1
                        gmm.mu.data[k, 1] = (j / side) * 2 - 1

            elif init_mode == 'uniform':
                # Uniform in [-1, 1]
                gmm.mu.data.uniform_(-1, 1)

    print(f"[init] GMM means initialized with mode='{init_mode}'")


class TemperatureScheduler:
    """Temperature scheduler for GMM distribution parameters."""
    def __init__(self, gmm, initial_T_pi=1.0, final_T_pi=1.0,
                 initial_T_mu=1.0, final_T_mu=1.0,
                 initial_T_sigma=1.0, final_T_sigma=1.0,
                 initial_T_shared=1.0, final_T_shared=1.0,
                 warmup_epochs=50):
        self.gmm = gmm
        self.initial_T_pi = initial_T_pi
        self.final_T_pi = final_T_pi
        self.initial_T_mu = initial_T_mu
        self.final_T_mu = final_T_mu
        self.initial_T_sigma = initial_T_sigma
        self.final_T_sigma = final_T_sigma
        self.initial_T_shared = initial_T_shared
        self.final_T_shared = final_T_shared
        self.warmup_epochs = warmup_epochs

    def step(self, epoch):
        """Update temperatures based on current epoch."""
        if epoch < self.warmup_epochs:
            alpha = epoch / self.warmup_epochs
            T_pi = self.initial_T_pi + alpha * (self.final_T_pi - self.initial_T_pi)
            T_mu = self.initial_T_mu + alpha * (self.final_T_mu - self.initial_T_mu)
            T_sigma = self.initial_T_sigma + alpha * (self.final_T_sigma - self.initial_T_sigma)
            T_shared = self.initial_T_shared + alpha * (self.final_T_shared - self.initial_T_shared)
        else:
            T_pi = self.final_T_pi
            T_mu = self.final_T_mu
            T_sigma = self.final_T_sigma
            T_shared = self.final_T_shared

        self.gmm.set_temperatures(T_pi=T_pi, T_mu=T_mu, T_sigma=T_sigma, T_shared=T_shared)
        return T_pi, T_mu, T_sigma, T_shared


class Config:
    """Basic configuration with default values."""

    # Device
    device: str = "cuda"
    num_workers: int = 2
    seed: int = 42

    # Dataset
    dataset: str = "cifar10"  # cifar10, cifar100, tinyimagenet
    data_root: str = "./dataset"
    resize: bool = False

    # Model Architecture
    arch: str = "resnet18"
    clf_ckpt: str = "./tests/standard_training/resnet18_cifar10_standard.pth"

    # GMM settings
    K: int = 3
    latent_dim: int = 128

    # Condition settings
    cond_mode: str = "xy"  # x, y, xy, None
    cov_type: str = "diag"  # diag, lowrank, full
    cov_rank: int = 0  # For lowrank only
    hidden_dim: int = 512

    # Label Embedding
    use_y_embedding: bool = True
    y_emb_dim: int = 128
    y_emb_normalize: bool = True

    # Decoder
    use_decoder: bool = True
    decoder_backend: str = "bicubic_trainable"

    # Perturbation Budget
    norm: str = "linf"
    epsilon: float = 16/255

    # Training
    epochs: int = 50
    batch_size: int = 256
    batch_index_max: int = float("inf")

    lr: float = 5e-4
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    accumulate_grad: int = 1

    # Learning Rate Scheduler
    use_lr_scheduler: bool = False
    lr_warmup_epochs: int = 5
    lr_min: float = 2e-6

    # Loss
    loss_variant: str = "cw"  # cw, ce
    kappa: float = 1.0

    # Sampling
    num_samples: int = 32
    chunk_size: int = 32

    # Regularization
    reg_pi_entropy: float = 0.0
    reg_mean_div: float = 0.0

    # Temperature Schedule
    T_pi_init: float = 3.0
    T_pi_final: float = 1.0
    T_mu_init: float = 3.0
    T_mu_final: float = 1.0
    T_sigma_init: float = 1.5
    T_sigma_final: float = 1.0
    T_shared_init: float = 1.5
    T_shared_final: float = 1.0
    warmup_epochs: int = 50

    # Gumbel-Softmax Temperature
    use_gumbel_anneal: bool = True
    gumbel_temp_init: float = 1.0
    gumbel_temp_final: float = 0.1

    # Initialization
    init_mode: str = "uniform"  # spread, random, grid, uniform

    # Monitoring & Logging
    check_collapse_every: int = 10
    ckp_dir: str = "./tests/standard_training/by_gmm"
    exp_name: str = "gmm4pr"

    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def __repr__(self):
        """Pretty print configuration."""
        lines = ["Configuration:"]
        lines.append("=" * 60)
        for key, value in self.__dict__.items():
            lines.append(f"  {key:25s}: {value}")
        lines.append("=" * 60)
        return "\n".join(lines)


def get_default_config():
    """Get default configuration."""
    return Config()


# Backward compatibility: old scripts may call get_config(name)
def get_config(name: str = None):
    """
    Get configuration (backward compatible).

    Args:
        name: Config name (ignored, kept for backward compatibility)

    Returns:
        Config object with default values
    """
    if name is not None:
        print(f"[Warning] Named configs are deprecated. Using default config. "
              f"Override values using command-line arguments.")
    return get_default_config()


# Backward compatibility: old scripts may call list_configs()
def list_configs():
    """List available configs (backward compatible)."""
    print("\nConfiguration:")
    print("=" * 60)
    print("Using default config. Override values with command-line arguments.")
    print("Example: --epochs 100 --K 5 --dataset cifar100 --arch resnet50")
    print("=" * 60)
