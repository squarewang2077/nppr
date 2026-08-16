import torch

# --------------------------------------------------
#   Helper function to build sigma_list for GMM prior
# --------------------------------------------------
def build_sigma_list(
    epsilon: float,
    K: int,
    mode_type: str = "linear",
    *,
    min_ratio: float = 0.4,
    rho: float = 0.5,
):
    """
    Build sigma_list for GMM prior (L_inf setting).

    Args:
        epsilon: perturbation budget (L_inf radius)
        K: number of mixture modes
        mode_type:
            - "linear"    : evenly spaced from min_ratio*ε to ε  (recommended)
            - "geometric" : geometric progression ending at ε
            - "full"      : evenly spaced from ε/K to ε
        min_ratio: smallest sigma as fraction of epsilon (for linear)
        rho: geometric ratio (for geometric)

    Returns:
        sigma_list: list of length K
    """

    if K <= 0:
        raise ValueError("K must be positive.")

    mode_type = mode_type.lower()

    # -------------------------
    # 1) Linear (recommended)
    # -------------------------
    if mode_type == "linear":
        sigma_list = torch.linspace(
            min_ratio * epsilon,
            epsilon,
            steps=K
        ).tolist()

    # -------------------------
    # 2) Geometric
    # -------------------------
    elif mode_type == "geometric":
        # ensures largest is epsilon
        sigma_list = [
            epsilon * (rho ** (K - 1 - k))
            for k in range(K)
        ]

    # -------------------------
    # 3) Full coverage
    # -------------------------
    elif mode_type == "full":
        sigma_list = [
            epsilon * (k + 1) / K
            for k in range(K)
        ]

    else:
        raise ValueError(f"Unknown mode_type: {mode_type}")

    return sigma_list