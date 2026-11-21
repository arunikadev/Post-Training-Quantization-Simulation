"""Optional torchvision dataset adapters (CIFAR-10, MNIST, ImageFolder)."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader, Subset

from ptq_sim.data.registry import register_dataset


def _get_torchvision():
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise ImportError("torchvision is required for vision datasets.") from exc
    return datasets, transforms


@register_dataset("cifar10")
def build_cifar10_loaders(
    calib_batch_size: int = 32,
    eval_batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
    root: str = "data",
    num_calib: int = 512,
    download: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Build CIFAR-10 calibration (train subset) and validation DataLoaders."""
    datasets, transforms = _get_torchvision()
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_ds = datasets.CIFAR10(root=root, train=True, download=download, transform=tf)
    val_ds = datasets.CIFAR10(root=root, train=False, download=download, transform=tf)
    calib_subset = Subset(train_ds, list(range(min(num_calib, len(train_ds)))))
    return (
        DataLoader(calib_subset, batch_size=calib_batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers),
    )


@register_dataset("mnist")
def build_mnist_loaders(
    calib_batch_size: int = 32,
    eval_batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
    root: str = "data",
    num_calib: int = 512,
    download: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Build MNIST calibration (train subset) and validation DataLoaders."""
    datasets, transforms = _get_torchvision()
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_ds = datasets.MNIST(root=root, train=True, download=download, transform=tf)
    val_ds = datasets.MNIST(root=root, train=False, download=download, transform=tf)
    calib_subset = Subset(train_ds, list(range(min(num_calib, len(train_ds)))))
    return (
        DataLoader(calib_subset, batch_size=calib_batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers),
    )


@register_dataset("image_folder")
def build_image_folder_loaders(
    calib_batch_size: int = 32,
    eval_batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
    calib_dir: str = "data/calib",
    eval_dir: str = "data/val",
    image_size: int = 224,
) -> tuple[DataLoader, DataLoader]:
    """Build DataLoaders from standard ImageFolder directory structures."""
    datasets, transforms = _get_torchvision()
    tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    if not Path(calib_dir).is_dir() or not Path(eval_dir).is_dir():
        raise FileNotFoundError(
            f"ImageFolder paths must exist: calib_dir='{calib_dir}', eval_dir='{eval_dir}'."
        )
    calib_ds = datasets.ImageFolder(calib_dir, transform=tf)
    eval_ds = datasets.ImageFolder(eval_dir, transform=tf)
    return (
        DataLoader(calib_ds, batch_size=calib_batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(eval_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers),
    )
