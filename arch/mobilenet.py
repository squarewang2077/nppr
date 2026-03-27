# MobileNet models
#
# Requirements:
#   torch >= 2.0
#   torchvision >= 0.15

import torch.nn as nn
from torchvision.models import mobilenet_v3_large as _mobilenet_v3_large, MobileNet_V3_Large_Weights


def mobilenet_v3_large(num_classes: int, pretrained: bool = False) -> nn.Module:
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = _mobilenet_v3_large(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
