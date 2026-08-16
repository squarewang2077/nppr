# ResNet family models
#
# Requirements:
#   torch >= 2.0
#   torchvision >= 0.15

import torch.nn as nn
from torchvision.models import resnet18 as _resnet18, ResNet18_Weights
from torchvision.models import resnet50 as _resnet50, ResNet50_Weights
from torchvision.models import wide_resnet50_2 as _wide_resnet50_2, Wide_ResNet50_2_Weights


def resnet18(num_classes: int, pretrained: bool = False) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = _resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def resnet50(num_classes: int, pretrained: bool = False) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = _resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def wide_resnet50_2(num_classes: int, pretrained: bool = False) -> nn.Module:
    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2 if pretrained else None
    model = _wide_resnet50_2(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
