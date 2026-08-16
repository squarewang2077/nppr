"""
utils/corrupter.py — Image corruption functions and batch-level utilities.

All per-image functions operate on HWC uint8 numpy arrays (values 0-255).
Use apply_corruption_batch() to apply a named corruption to a batch of
(B, C, H, W) float32 tensors in [0, 1].

Severity levels 1-5 follow the ImageNet-C convention:
  1 = mildest,  5 = strongest.

Available corruptions (keys of CORRUPTION_FNS):
  "salt_pepper"   — salt-and-pepper impulse noise
  "motion_blur"   — horizontal motion blur (linear kernel)
  "brightness"    — global brightness reduction
  "jpeg"          — JPEG compression artefacts
"""

import cv2
import numpy as np
import torch


# ------------------------------------------------------------------
#                   Per-image corruption functions
# ------------------------------------------------------------------

def salt_pepper_noise(img, amount=0.02, salt_vs_pepper=0.5):
    """Add salt-and-pepper impulse noise to a HWC uint8 image."""
    out = img.copy()
    H, W, _ = out.shape
    num = int(amount * H * W)

    coords = (np.random.randint(0, H, num), np.random.randint(0, W, num))
    out[coords[0], coords[1], :] = 255

    coords = (np.random.randint(0, H, num), np.random.randint(0, W, num))
    out[coords[0], coords[1], :] = 0
    return out


def motion_blur(img, ksize=15):
    """Apply horizontal motion blur to a HWC uint8 image."""
    kernel = np.zeros((ksize, ksize))
    kernel[ksize // 2, :] = np.ones(ksize) / ksize
    return cv2.filter2D(img, -1, kernel)


def brightness_change(img, factor=0.6):
    """Scale image brightness by factor (< 1 darkens, > 1 brightens)."""
    out = img.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def jpeg_compression(img, quality=20):
    """Apply JPEG compression artefacts to a HWC uint8 image."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encimg, 1)


# ------------------------------------------------------------------
#                         Registries
# ------------------------------------------------------------------

CORRUPTION_FNS = {
    "salt_pepper": salt_pepper_noise,
    "motion_blur": motion_blur,
    "brightness":  brightness_change,
    "jpeg":        jpeg_compression,
}

# Severity-indexed parameter dicts (index 0 = severity 1, index 4 = severity 5)
SEVERITY_PARAMS = {
    "salt_pepper": [
        {"amount": 0.01},
        {"amount": 0.02},
        {"amount": 0.05},
        {"amount": 0.10},
        {"amount": 0.20},
    ],
    "motion_blur": [
        {"ksize": 5},
        {"ksize": 9},
        {"ksize": 15},
        {"ksize": 21},
        {"ksize": 31},
    ],
    "brightness": [
        {"factor": 0.90},
        {"factor": 0.75},
        {"factor": 0.60},
        {"factor": 0.40},
        {"factor": 0.20},
    ],
    "jpeg": [
        {"quality": 75},
        {"quality": 50},
        {"quality": 30},
        {"quality": 15},
        {"quality":  5},
    ],
}


# ------------------------------------------------------------------
#                     Batch-level wrapper
# ------------------------------------------------------------------

def apply_corruption_batch(x, corruption_name, severity=1):
    """
    Apply a named corruption to a batch of float32 tensors.

    Args:
        x              : (B, C, H, W) float32 tensor in [0, 1], any device.
        corruption_name: key in CORRUPTION_FNS.
        severity       : int in [1, 5] — perturbation strength.

    Returns:
        x_corrupt : (B, C, H, W) float32 tensor in [0, 1], same device as x.
    """
    if corruption_name not in CORRUPTION_FNS:
        raise ValueError(
            f"Unknown corruption '{corruption_name}'. "
            f"Available: {list(CORRUPTION_FNS)}"
        )
    if not (1 <= severity <= 5):
        raise ValueError(f"severity must be 1–5, got {severity}")

    fn     = CORRUPTION_FNS[corruption_name]
    params = SEVERITY_PARAMS[corruption_name][severity - 1]
    device = x.device

    # (B, C, H, W) float32 [0,1] → (B, H, W, C) uint8
    imgs_np = (x.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    out = []
    for img in imgs_np:
        corrupted = fn(img, **params)
        out.append(corrupted.astype(np.float32) / 255.0)

    # (B, H, W, C) float32 → (B, C, H, W), back on original device
    return (
        torch.from_numpy(np.stack(out))
        .permute(0, 3, 1, 2)
        .to(device)
        .clamp(0.0, 1.0)
    )
