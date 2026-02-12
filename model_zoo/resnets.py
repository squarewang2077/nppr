# ResNet family models
#
# Requirements:
#   torch >= 2.0
#   torchvision >= 0.15

import torch.nn as nn
from torchvision.models import resnet18 as _resnet18
from torchvision.models import resnet50 as _resnet50
from torchvision.models import wide_resnet50_2 as _wide_resnet50_2


def resnet18(num_classes: int) -> nn.Module:
    """ResNet-18 trained from scratch."""
    model = _resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def resnet50(num_classes: int) -> nn.Module:
    """ResNet-50 trained from scratch."""
    model = _resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def wide_resnet50_2(num_classes: int) -> nn.Module:
    """Wide ResNet-50-2 trained from scratch."""
    model = _wide_resnet50_2(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
