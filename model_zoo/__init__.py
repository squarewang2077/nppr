# model_zoo - Self-contained model definitions for training from scratch
#
# Requirements:
#   torch >= 2.0
#   torchvision >= 0.15

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

def build_model(arch: str, num_classes: int):
    """Build a model by architecture name."""
    arch = arch.lower()
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {arch}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[arch](num_classes=num_classes)
