# `arch/` — Model Zoo

All models are built through one entry point and share the same interface.
Models take **raw `[0, 1]` images**; normalization happens inside the model, so
attacks can clamp in pixel space without a normalization mismatch.

## Quick start

```python
import torch
from arch import build_model

model = build_model(arch="resnet18", num_classes=10, dataset="cifar10")

x = torch.rand(8, 3, 32, 32)      # raw [0, 1], NOT normalized
logits = model(x)                 # (8, 10)
```

That is the whole API for training. `dataset` picks the normalization
statistics and the input resolution, so you never pass `img_size` yourself.

## Available architectures

| Family | Names | Params (CIFAR-10) |
|---|---|---|
| ResNet | `resnet18`, `resnet34`, `resnet50`, `wide_resnet50_2` | 11.2M / 21.3M / 23.5M / 66.9M |
| Other CNNs | `densenet121`, `vgg16`, `mobilenet_v3_large`, `efficientnet_b0` | 7.0M / 134.3M / 4.2M / 4.0M |
| Transformers | `vit_tiny`, `vit_small`, `convit_tiny`, `convit_small` | 5.4M / 21.3M / 5.4M / 27.0M |

Supported datasets: `cifar10`, `cifar100`, `svhn` (32px), `tinyimagenet` (64px).

The CNNs wrap torchvision. The transformers are written from scratch in
[vit.py](vit.py) because torchvision's ViT has no stochastic depth and cannot
express ConViT's gated positional attention — see the header of that file.

**Picking a transformer:** `convit_tiny` is the safer default for training from
scratch. It starts with a convolutional prior (each attention head initialised
to a kernel offset) and can learn its way out, which is what makes ViTs
converge on CIFAR-sized data. A plain `vit_tiny` is less stable, more so under
adversarial training.

## Feature extraction

`build_feat_extractor` returns the backbone without its classification head,
producing a flat feature vector per image:

```python
from arch import build_model, build_feat_extractor

model = build_model("resnet18", 10, "cifar10")
# pass model.backbone to share weights with an already-loaded model
feat = build_feat_extractor("resnet18", 10, "cifar10", backbone=model.backbone)

f = feat(x)                       # (8, 512) — note: expects NORMALIZED input
```

Feature dimensions: ResNet-18/34 512, ResNet-50/WRN 2048, DenseNet 1024,
EfficientNet 1280, MobileNet 960, ViT/ConViT-tiny 192, `vit_small` 384,
`convit_small` 432.

> The extractor takes already-normalized input (`(x - model.mean) / model.std`),
> unlike `build_model`, which normalizes internally.

## Checkpoints

`build_model` returns a `NormalizedModel` whose `state_dict` includes the `mean`
and `std` buffers, so a checkpoint is self-contained:

```python
torch.save({"arch": "resnet18", "dataset": "cifar10",
            "model_state": model.state_dict()}, "ckpt.pth")

ckpt = torch.load("ckpt.pth", map_location="cpu")
model = build_model(ckpt["arch"], 10, ckpt["dataset"])
model.load_state_dict(ckpt["model_state"])
```

## Notes

- `pretrained=True` loads ImageNet weights for the CNNs. The transformers are
  small-input models with no pretrained weights — they warn and initialise from
  scratch.
- Transformers are built for one resolution and will raise a clear error if
  given another; rebuild with the right `dataset`.

## Adding a model

Add a builder with the signature `(num_classes, pretrained=False)` — plus
`img_size` if it is resolution-dependent, which `build_model` then passes
automatically — and register the name in `MODEL_REGISTRY` in
[\_\_init\_\_.py](__init__.py). If the model is not a ResNet/VGG/DenseNet-style
CNN, give it a `forward_features(x) -> (B, dim)` method and
`build_feat_extractor` will pick it up with no further changes.
