# model_zoo - Self-contained model definitions for training from scratch
#
# Requirements:
#   torch >= 2.0
#   torchvision >= 0.15

import torch
import torch.nn as nn

from .resnets import resnet18, resnet50, wide_resnet50_2
from .densenet import densenet121
from .vgg import vgg16
from .mobilenet import mobilenet_v3_large
from .efficientnet import efficientnet_b0
from .vit import vit_b_16

MODEL_REGISTRY = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "wide_resnet50_2": wide_resnet50_2,
    "densenet121": densenet121,
    "vgg16": vgg16,
    "mobilenet_v3_large": mobilenet_v3_large,
    "efficientnet_b0": efficientnet_b0,
    "vit_b_16": vit_b_16,
}


class NormalizedModel(nn.Module):
    """Wraps a backbone with a dataset-specific normalization layer.

    Inputs are expected to be raw images in [0, 1].  The normalization
    is registered as a buffer so it moves with the model on .to(device)
    and is included in state_dict() checkpoints.
    """
    def __init__(self, backbone: nn.Module, mean, std):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(std,  dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone((x - self.mean) / self.std)


def build_model(arch: str, num_classes: int, dataset: str) -> nn.Module:
    """Build a normalized model by architecture name and dataset.

    The returned model expects raw [0, 1] inputs and applies the
    dataset-specific normalization internally.
    """
    # Deferred import to break the circular dependency:
    # model_zoo -> utils.__init__ -> utils.utils -> fit_classifiers -> model_zoo
    from utils.data_preprocessing import get_norm_stats

    arch = arch.lower()
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {arch}. Available: {list(MODEL_REGISTRY.keys())}")
    backbone = MODEL_REGISTRY[arch](num_classes=num_classes)
    mean, std = get_norm_stats(dataset)
    return NormalizedModel(backbone, mean, std)
