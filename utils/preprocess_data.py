# data_preprocessing.py - Dataset loading and preprocessing utilities
#
# Requirements:
#   torch >= 2.0
#   torchvision >= 0.15

import os
import torch
import torchvision
import torchvision.transforms as T
import torchvision.datasets as dsets

# ------------------------------------------------------------------
#                       Normalization Stats
# ------------------------------------------------------------------


CIFAR10_MEAN, CIFAR10_STD   = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN, CIFAR100_STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
TINY_MEAN, TINY_STD         = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
MNIST_MEAN, MNIST_STD       = (0.1307,),                 (0.3081,)

# ------------------------------------------------------------------
#                       Data Utility Functions
# ------------------------------------------------------------------


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
    dataset = dataset.lower()
    if dataset == "tinyimagenet":
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


# ------------------------------------------------------------------
#                       Indexed Dataset Utilities
# ------------------------------------------------------------------


def _build_transform(mean, std, resize: bool, channels: int = 3):
    """Build a normalizing transform, optionally resizing to 224."""
    if resize:
        return T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([T.ToTensor(), T.Normalize(mean, std)])


class WithIndex(torch.utils.data.Dataset):
    """
    Wrap an existing dataset so __getitem__ returns (..., idx).
    Works whether the base dataset returns (img, label) or a dict.
    """
    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx: int):
        item = self.base_ds[idx]
        if isinstance(item, dict):
            item = dict(item)
            item["idx"] = idx
            return item
        elif isinstance(item, (list, tuple)):
            return (*item, idx)
        else:
            raise TypeError(
                f"WithIndex: unexpected item type {type(item)} from base dataset. "
                "Expected a tuple/list (img, label) or a dict."
            )


def get_dataset_with_index(name, root="./dataset", train=False, resize=False):
    """
    Get a dataset by name.
    Returns (dataset, num_classes, input_shape)
    """
    name = name.lower()

    if name == "cifar10":
        tf = _build_transform(CIFAR10_MEAN, CIFAR10_STD, resize)
        input_shape = (3, 224, 224) if resize else (3, 32, 32)
        num_classes = 10
        ds = WithIndex(torchvision.datasets.CIFAR10(root=root, train=train, download=True, transform=tf))

    elif name == "cifar100":
        tf = _build_transform(CIFAR100_MEAN, CIFAR100_STD, resize)
        input_shape = (3, 224, 224) if resize else (3, 32, 32)
        num_classes = 100
        ds = WithIndex(torchvision.datasets.CIFAR100(root=root, train=train, download=True, transform=tf))

    elif name == "mnist":
        tf = _build_transform(MNIST_MEAN, MNIST_STD, resize)
        input_shape = (1, 224, 224) if resize else (1, 28, 28)
        num_classes = 10
        ds = WithIndex(torchvision.datasets.MNIST(root=root, train=train, download=True, transform=tf))

    elif name == "tinyimagenet":
        tf = _build_transform(TINY_MEAN, TINY_STD, resize)
        input_shape = (3, 224, 224) if resize else (3, 64, 64)
        num_classes = 200
        split = "train" if train else "val"
        ds = WithIndex(torchvision.datasets.ImageFolder(
            os.path.join(root, "tiny-imagenet-200", split), transform=tf
        ))

    else:
        raise ValueError(f"Unknown dataset {name}")

    return ds, num_classes, input_shape
