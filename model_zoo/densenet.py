# DenseNet models
#
# Requirements:
#   torch >= 2.0
#   torchvision >= 0.15

import torch.nn as nn
from torchvision.models import densenet121 as _densenet121


def densenet121(num_classes: int) -> nn.Module:
    """DenseNet-121 trained from scratch."""
    model = _densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model
