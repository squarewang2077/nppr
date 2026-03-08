# data_preprocessing.py - Dataset loading and preprocessing utilities
#
# Requirements:
#   torch >= 2.0
#   torchvision >= 0.15

import os
import torchvision.transforms as T
import torchvision.datasets as dsets

# ------------------------------ Normalization Stats ------------------------------

CIFAR10_MEAN, CIFAR10_STD   = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN, CIFAR100_STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
TINY_MEAN, TINY_STD         = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)

# ------------------------------ Utility Functions ------------------------------

def get_norm_stats(dataset: str):
    """Return (mean, std) for normalization."""
    dataset = dataset.lower()
    if dataset == "cifar10":
        return CIFAR10_MEAN, CIFAR10_STD
    if dataset == "cifar100":
        return CIFAR100_MEAN, CIFAR100_STD
    if dataset == "tinyimagenet":
        return TINY_MEAN, TINY_STD
    raise ValueError(f"Unknown dataset {dataset}")


def get_img_size(dataset: str, manual_override: int = None) -> int:
    """Return native image size for the dataset."""
    if manual_override is not None:
        return manual_override
    if dataset.lower() == "tinyimagenet":
        return 64
    return 32  # CIFAR10/100


def get_dataset(name: str, root: str, train: bool, img_size: int, augment: bool = True):
    """Build dataset + num_classes.

    Training transforms (when train=True and augment=True):
        RandomCrop (with padding) → RandomHorizontalFlip → RandAugment → ToTensor → RandomErasing
    Test transforms:
        Resize → ToTensor
    """
    name = name.lower()

    if train and augment:
        padding = max(4, img_size // 8)   # 4 for CIFAR-32, 8 for TinyImageNet-64
        tf = T.Compose([
            T.Resize(img_size),
            T.RandomCrop(img_size, padding=padding, padding_mode="reflect"),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=9),  # AutoAugment-style policy
            T.ToTensor(),           # outputs [0, 1]; normalization is handled inside the model
            T.RandomErasing(p=0.25, scale=(0.02, 0.2)),  # occlusion robustness
        ])
    else:
        tf = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),           # outputs [0, 1]; normalization is handled inside the model
        ])

    if name == "cifar10":
        ds = dsets.CIFAR10(root=root, train=train, download=True, transform=tf)
        num_classes = 10
    elif name == "cifar100":
        ds = dsets.CIFAR100(root=root, train=train, download=True, transform=tf)
        num_classes = 100
    elif name == "tinyimagenet":
        split = "train" if train else "val"
        base = os.path.join(root, "tiny-imagenet-200", split)
        if not os.path.isdir(base):
            raise FileNotFoundError(
                f"TinyImageNet folder not found at {base}.\n"
                "Download from http://cs231n.stanford.edu/tiny-imagenet-200.zip "
                "and extract to <root>/tiny-imagenet-200/"
            )
        ds = dsets.ImageFolder(base, transform=tf)
        num_classes = 200
    else:
        raise ValueError(f"Unknown dataset {name}")

    return ds, num_classes
