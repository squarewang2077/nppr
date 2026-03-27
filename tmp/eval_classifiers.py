import os
import re
import argparse
from typing import Tuple, Optional

import torch
import torchvision.transforms as T

# Reuse your helper functions from training
from scripts.train_classifiers import (
    build_model,
    get_dataset,          # (name, root, train, img_size, use_imnet_stats) -> (dataset, num_classes)
    get_norm_stats,       # (dataset, use_imnet_stats) -> (mean, std)
    evaluate,             # (model, loader, device) -> acc
)


IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _parse_from_filename(fname: str) -> Optional[Tuple[str, str]]:
    """
    Fallback when ckpt metadata is missing: parse "<arch>_<dataset>.pth".
    """
    m = re.match(r"(.+)_([a-z0-9]+)\.pth$", fname)
    if not m:
        return None
    return m.group(1), m.group(2)


def _build_eval_transform(dataset: str, img_size: int, use_imnet_stats: bool) -> T.Compose:
    """
    Exactly mirror your eval-time preprocessing in training:
      Resize -> CenterCrop -> ToTensor -> Normalize
    """
    mean, std = (IMNET_MEAN, IMNET_STD) if use_imnet_stats else get_norm_stats(dataset, False)
    return T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def load_and_eval_ckpt(ckpt_path: str, data_root: str, device: torch.device, batch_size: int, num_workers: int):
    """
    Load one checkpoint, rebuild dataset & model with the SAME preprocessing as training,
    evaluate top-1 accuracy on the test/val split, and return (filename, acc).
    """
    state = torch.load(ckpt_path, map_location="cpu")

    # Prefer metadata saved by the trainer
    arch = state.get("arch")
    dataset = state.get("dataset")
    img_size = state.get("img_size")
    was_pretrained = bool(state.get("pretrained", False))

    # Fallback: parse from filename and infer img_size
    if arch is None or dataset is None:
        parsed = _parse_from_filename(os.path.basename(ckpt_path))
        if parsed is None:
            print(f"[skip] cannot parse arch/dataset from: {ckpt_path}")
            return None
        arch, dataset = parsed
    if img_size is None:
        # If trained with ImageNet weights, they were resized to 224; else native sizes.
        img_size = 224 if was_pretrained else (64 if dataset == "tinyimagenet" else 32)

    print(f"\n[eval] {os.path.basename(ckpt_path)} "
          f"(arch={arch}, dataset={dataset}, img={img_size}, pretrained={was_pretrained})")

    # -------- dataset (test/val) with the SAME transforms as training eval --------
    test_set, num_classes = get_dataset(
        dataset, data_root, train=False,
        img_size=img_size, use_imnet_stats=was_pretrained
    )
    # Force the exact pipeline (safeguard against internal defaults)
    test_set.transform = _build_eval_transform(dataset, img_size, was_pretrained)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # -------- model skeleton & load weights --------
    # IMPORTANT: we DO NOT load ImageNet weights here; we load your checkpoint weights instead.
    model = build_model(arch, num_classes, device, pretrained=False)
    # Handle DataParallel-trained checkpoints
    state_dict = state.get("model_state", state)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"  [warn] load_state: missing={len(missing)} unexpected={len(unexpected)}")

    # -------- evaluate --------
    acc = evaluate(model, test_loader, device)
    print(f"  Acc = {acc * 100:.2f}%")
    return os.path.basename(ckpt_path), acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="./model_zoo/trained_model",
                        help="Folder containing <arch>_<dataset>.pth checkpoints")
    parser.add_argument("--data_root", type=str, default="./dataset",
                        help="Root folder for datasets (same as training)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    all_results = []
    for fn in sorted(os.listdir(args.ckpt_dir)):
        if not fn.endswith(".pth"):
            continue
        path = os.path.join(args.ckpt_dir, fn)
        out = load_and_eval_ckpt(
            ckpt_path=path,
            data_root=args.data_root,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        if out is not None:
            all_results.append(out)

    if not all_results:
        print("\n[warn] no checkpoints evaluated.")
        return

    # Sort by accuracy desc for a neat summary
    all_results.sort(key=lambda x: x[1], reverse=True)

    print("\n================ Summary (Top-1) ================")
    width = max(len(n) for n, _ in all_results) + 2
    for name, acc in all_results:
        print(f"{name:<{width}} {acc*100:6.2f}%")
    print("=================================================")


if __name__ == "__main__":
    main()
