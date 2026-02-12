# VGG models
#
# Requirements:
#   torch >= 2.0
#   torchvision >= 0.15

import torch.nn as nn
from torchvision.models import vgg16 as _vgg16


def vgg16(num_classes: int) -> nn.Module:
    """VGG-16 trained from scratch."""
    model = _vgg16(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
