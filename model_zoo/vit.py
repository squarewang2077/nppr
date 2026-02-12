# Vision Transformer models
#
# Requirements:
#   torch >= 2.0
#   torchvision >= 0.15

import torch.nn as nn
from torchvision.models import vit_b_16 as _vit_b_16


def vit_b_16(num_classes: int) -> nn.Module:
    """ViT-B/16 trained from scratch."""
    model = _vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model
