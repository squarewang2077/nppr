# Vision Transformer models
#
# Requirements:
#   torch >= 2.0
#   torchvision >= 0.15

import torch.nn as nn
from torchvision.models import vit_b_16 as _vit_b_16, ViT_B_16_Weights


def vit_b_16(num_classes: int, pretrained: bool = False) -> nn.Module:
    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = _vit_b_16(weights=weights)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model
