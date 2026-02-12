# MobileNet models
#
# Requirements:
#   torch >= 2.0
#   torchvision >= 0.15

import torch.nn as nn
from torchvision.models import mobilenet_v3_large as _mobilenet_v3_large


def mobilenet_v3_large(num_classes: int) -> nn.Module:
    """MobileNetV3-Large trained from scratch."""
    model = _mobilenet_v3_large(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
