# model_zoo - Self-contained model definitions for training from scratch
#
# Requirements:
#   torch >= 2.0
#   torchvision >= 0.15

import inspect

import torch
import torch.nn as nn

from .resnets import resnet18, resnet34, resnet50, wide_resnet50_2
from .densenet import densenet121
from .vgg import vgg16
from .mobilenet import mobilenet_v3_large
from .efficientnet import efficientnet_b0
from .vit import vit_tiny, vit_small, convit_tiny, convit_small

MODEL_REGISTRY = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "wide_resnet50_2": wide_resnet50_2,
    "densenet121": densenet121,
    "vgg16": vgg16,
    "mobilenet_v3_large": mobilenet_v3_large,
    "efficientnet_b0": efficientnet_b0,
    # Transformers: from-scratch, sized for 32/64px inputs.
    "vit_tiny": vit_tiny,
    "vit_small": vit_small,
    "convit_tiny": convit_tiny,
    "convit_small": convit_small,
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
        C = len(mean)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, C, 1, 1))
        self.register_buffer("std",  torch.tensor(std,  dtype=torch.float32).view(1, C, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone((x - self.mean) / self.std)


def build_model(arch: str, num_classes: int, dataset: str, pretrained: bool = False) -> nn.Module:
    """Build a normalized model by architecture name and dataset.

    The returned model expects raw [0, 1] inputs and applies the
    dataset-specific normalization internally.
    """
    # Deferred import to break the circular dependency:
    # model_zoo -> utils.__init__ -> utils.utils -> fit_classifiers -> model_zoo
    from utils.preprocess_data import get_norm_stats, get_img_size

    arch = arch.lower()
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {arch}. Available: {list(MODEL_REGISTRY.keys())}")
    builder = MODEL_REGISTRY[arch]
    kwargs = {}
    # Small-input ViTs size their patch grid from the resolution they will see;
    # CNNs are fully convolutional and ignore it.
    if "img_size" in inspect.signature(builder).parameters:
        kwargs["img_size"] = get_img_size(dataset)
    backbone = builder(num_classes=num_classes, pretrained=pretrained, **kwargs)
    mean, std = get_norm_stats(dataset)
    return NormalizedModel(backbone, mean, std)


def build_feat_extractor(arch: str, num_classes: int, dataset: str,
                         backbone=None) -> nn.Module:
    """Build a feature extractor (backbone minus classification head).

    The feature extractor outputs a flat feature vector for each input image.

    Args:
        arch:        Architecture name (must be a key in MODEL_REGISTRY).
        num_classes: Number of output classes for the backbone.
        dataset:     Dataset name used to look up normalization statistics.
        backbone:    Optional pre-built backbone to use directly.  When
                     provided (e.g. ``model.backbone`` after loading weights),
                     the returned extractor shares the same parameter objects
                     as that backbone — no separate weight loading needed.
                     When None, a fresh backbone is built from scratch.

    Returns:
        feat_extractor: headless backbone as an nn.Module.
    """
    arch = arch.lower()

    if backbone is None:
        backbone = build_model(arch=arch, num_classes=num_classes, dataset=dataset).backbone

    if arch.startswith("resnet") or arch.startswith("wide_resnet"):
        # Module order is conv1, bn1, relu, maxpool, layer1..4, avgpool, fc —
        # dropping the trailing fc leaves exactly the feature trunk.
        return nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())

    elif arch == "vgg16":
        return nn.Sequential(backbone.features, backbone.avgpool, nn.Flatten())

    elif arch == "densenet121":
        return nn.Sequential(
            backbone.features,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    elif arch == "mobilenet_v3_large":
        return nn.Sequential(
            backbone.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    elif arch == "efficientnet_b0":
        return nn.Sequential(
            backbone.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    elif hasattr(backbone, "forward_features"):
        # ViT / DeiT / ConViT all expose forward_features -> (B, embed_dim).
        class TransformerFeat(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                return self.model.forward_features(x)

        return TransformerFeat(backbone)

    else:
        raise ValueError(f"Unsupported arch: {arch}")
