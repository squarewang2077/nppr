# VGG models
#
# Requirements:
#   torch >= 2.0
#   torchvision >= 0.15

import torch.nn as nn
from torchvision.models import vgg16 as _vgg16, VGG16_Weights


def vgg16(num_classes: int, pretrained: bool = False) -> nn.Module:
    weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = _vgg16(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
