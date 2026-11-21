"""Deterministic learnable synthetic datasets for image, tabular, and sequence PTQ experiments."""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from ptq_sim.data.registry import register_dataset


def _make_synthetic_image_tensors(
    num_samples: int,
    image_size: int = 32,
    in_channels: int = 3,
    num_classes: int = 10,
    noise_std: float = 0.4,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    labels = torch.arange(num_samples, dtype=torch.long) % num_classes
    perm = torch.randperm(num_samples, generator=gen)
    labels = labels[perm]

    # Build deterministic class prototypes with distinct spatial frequency & channel signatures.
    yy = torch.linspace(-math.pi, math.pi, steps=image_size).unsqueeze(1).expand(image_size, image_size)
    xx = torch.linspace(-math.pi, math.pi, steps=image_size).unsqueeze(0).expand(image_size, image_size)
    prototypes = torch.zeros(num_classes, in_channels, image_size, image_size, dtype=torch.float32)

    for c in range(num_classes):
        freq_x = 1.0 + 0.6 * (c % 4)
        freq_y = 1.0 + 0.6 * ((c // 2) % 4)
        phase = 0.7 * c
        for ch in range(in_channels):
            wave = torch.sin(freq_x * xx + phase + 0.8 * ch) + torch.cos(freq_y * yy - 0.5 * c)
            prototypes[c, ch] = 0.8 * wave
        # Add localized class indicator patch.
        r0 = (c * 3) % max(1, image_size - 6)
        c0 = ((c * 5) + 4) % max(1, image_size - 6)
        prototypes[c, c % in_channels, r0 : r0 + 5, c0 : c0 + 5] += 1.5

    noise = torch.randn(num_samples, in_channels, image_size, image_size, generator=gen) * noise_std
    images = prototypes[labels] + noise
    return images, labels


def _make_synthetic_tabular_tensors(
    num_samples: int,
    input_dim: int = 64,
    num_classes: int = 4,
    noise_std: float = 0.65,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    proto_gen = torch.Generator().manual_seed(12345)
    prototypes = torch.randn(num_classes, input_dim, generator=proto_gen) * 1.4

    gen = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, num_classes, (num_samples,), generator=gen, dtype=torch.long)
    noise = torch.randn(num_samples, input_dim, generator=gen) * noise_std
    features = prototypes[labels] + noise
    return features, labels


def _make_synthetic_sequence_tensors(
    num_samples: int,
    seq_len: int = 32,
    vocab_size: int = 128,
    num_classes: int = 4,
    signal_ratio: float = 0.65,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, num_classes, (num_samples,), generator=gen, dtype=torch.long)
    tokens = torch.randint(0, vocab_size, (num_samples, seq_len), generator=gen, dtype=torch.long)

    band_size = max(1, vocab_size // num_classes)
    mask = torch.rand(num_samples, seq_len, generator=gen) < signal_ratio
    for c in range(num_classes):
        low = c * band_size
        high = min(vocab_size, (c + 1) * band_size) if c < num_classes - 1 else vocab_size
        cls_tokens = torch.randint(low, max(low + 1, high), (num_samples, seq_len), generator=gen, dtype=torch.long)
        cls_mask = (labels == c).unsqueeze(1) & mask
        tokens = torch.where(cls_mask, cls_tokens, tokens)
    return tokens, labels


def _build_split_loaders(
    calib_ds: TensorDataset,
    eval_ds: TensorDataset,
    calib_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    calib_loader = DataLoader(calib_ds, batch_size=calib_batch_size, shuffle=False, num_workers=num_workers)
    eval_loader = DataLoader(eval_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)
    return calib_loader, eval_loader


@register_dataset("synthetic_image")
def build_synthetic_image_loaders(
    calib_batch_size: int = 32,
    eval_batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
    num_calib: int = 256,
    num_eval: int = 512,
    image_size: int = 32,
    in_channels: int = 3,
    num_classes: int = 10,
    noise_std: float = 0.4,
) -> tuple[DataLoader, DataLoader]:
    """Create deterministic learnable synthetic image (calib, eval) DataLoaders."""
    cx, cy = _make_synthetic_image_tensors(num_calib, image_size, in_channels, num_classes, noise_std, seed=seed + 101)
    ex, ey = _make_synthetic_image_tensors(num_eval, image_size, in_channels, num_classes, noise_std, seed=seed + 202)
    return _build_split_loaders(
        TensorDataset(cx, cy), TensorDataset(ex, ey), calib_batch_size, eval_batch_size, num_workers
    )


@register_dataset("synthetic_tabular")
def build_synthetic_tabular_loaders(
    calib_batch_size: int = 32,
    eval_batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
    num_calib: int = 256,
    num_eval: int = 512,
    input_dim: int = 64,
    num_classes: int = 4,
    noise_std: float = 0.65,
) -> tuple[DataLoader, DataLoader]:
    """Create deterministic learnable synthetic tabular (calib, eval) DataLoaders."""
    cx, cy = _make_synthetic_tabular_tensors(num_calib, input_dim, num_classes, noise_std, seed=seed + 101)
    ex, ey = _make_synthetic_tabular_tensors(num_eval, input_dim, num_classes, noise_std, seed=seed + 202)
    return _build_split_loaders(
        TensorDataset(cx, cy), TensorDataset(ex, ey), calib_batch_size, eval_batch_size, num_workers
    )


@register_dataset("synthetic_sequence")
def build_synthetic_sequence_loaders(
    calib_batch_size: int = 32,
    eval_batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
    num_calib: int = 256,
    num_eval: int = 512,
    seq_len: int = 32,
    vocab_size: int = 128,
    num_classes: int = 4,
    signal_ratio: float = 0.65,
) -> tuple[DataLoader, DataLoader]:
    """Create deterministic learnable synthetic token sequence (calib, eval) DataLoaders."""
    cx, cy = _make_synthetic_sequence_tensors(
        num_calib, seq_len, vocab_size, num_classes, signal_ratio, seed=seed + 101
    )
    ex, ey = _make_synthetic_sequence_tensors(
        num_eval, seq_len, vocab_size, num_classes, signal_ratio, seed=seed + 202
    )
    return _build_split_loaders(
        TensorDataset(cx, cy), TensorDataset(ex, ey), calib_batch_size, eval_batch_size, num_workers
    )


def build_synthetic_train_loader(
    dataset_name: str,
    batch_size: int = 64,
    num_samples: int = 1500,
    seed: int = 42,
    **kwargs,
) -> DataLoader:
    """Create a training DataLoader matching one of the built-in synthetic datasets."""
    key = dataset_name.lower()
    if key == "synthetic_image":
        x, y = _make_synthetic_image_tensors(num_samples=num_samples, seed=seed, **kwargs)
    elif key == "synthetic_tabular":
        x, y = _make_synthetic_tabular_tensors(num_samples=num_samples, seed=seed, **kwargs)
    elif key == "synthetic_sequence":
        x, y = _make_synthetic_sequence_tensors(num_samples=num_samples, seed=seed, **kwargs)
    else:
        raise ValueError(f"Unsupported synthetic training dataset '{dataset_name}'.")
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
