# EfficientNet models
#
# Requirements:
#   torch >= 2.0
#   torchvision >= 0.15

import torch.nn as nn
from torchvision.models import efficientnet_b0 as _efficientnet_b0


def efficientnet_b0(num_classes: int) -> nn.Module:
    """EfficientNet-B0 trained from scratch."""
    model = _efficientnet_b0(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
