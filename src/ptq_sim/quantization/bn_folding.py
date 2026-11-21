"""BatchNorm folding for Conv2d+BatchNorm2d and Linear+BatchNorm1d pairs."""

from __future__ import annotations

import copy
import logging

import torch
from torch import Tensor, fx, nn

logger = logging.getLogger("ptq_sim")


def fold_conv_bn_weights(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[Tensor, Tensor]:
    """Compute folded (weight, bias) absorbing an eval-mode BatchNorm2d into Conv2d."""
    w = conv.weight.detach()
    std = torch.sqrt(bn.running_var.detach() + bn.eps)
    gamma = bn.weight.detach() if bn.affine and bn.weight is not None else torch.ones_like(std)
    beta = bn.bias.detach() if bn.affine and bn.bias is not None else torch.zeros_like(std)
    mean = bn.running_mean.detach() if bn.running_mean is not None else torch.zeros_like(std)

    factor = gamma / std
    w_fold = w * factor.reshape(-1, 1, 1, 1)
    b_orig = conv.bias.detach() if conv.bias is not None else torch.zeros_like(mean)
    b_fold = (b_orig - mean) * factor + beta
    return w_fold, b_fold


def fold_linear_bn_weights(linear: nn.Linear, bn: nn.BatchNorm1d) -> tuple[Tensor, Tensor]:
    """Compute folded (weight, bias) absorbing an eval-mode BatchNorm1d into Linear."""
    w = linear.weight.detach()
    std = torch.sqrt(bn.running_var.detach() + bn.eps)
    gamma = bn.weight.detach() if bn.affine and bn.weight is not None else torch.ones_like(std)
    beta = bn.bias.detach() if bn.affine and bn.bias is not None else torch.zeros_like(std)
    mean = bn.running_mean.detach() if bn.running_mean is not None else torch.zeros_like(std)

    factor = gamma / std
    w_fold = w * factor.reshape(-1, 1)
    b_orig = linear.bias.detach() if linear.bias is not None else torch.zeros_like(mean)
    b_fold = (b_orig - mean) * factor + beta
    return w_fold, b_fold


def _get_submodule_by_path(model: nn.Module, path: str) -> nn.Module:
    return model.get_submodule(path)


def _replace_submodule_by_path(model: nn.Module, path: str, new_module: nn.Module) -> None:
    parts = path.split(".")
    parent = model if len(parts) == 1 else model.get_submodule(".".join(parts[:-1]))
    attr = parts[-1]
    if isinstance(parent, nn.Sequential) and attr.isdigit():
        parent[int(attr)] = new_module
    else:
        setattr(parent, attr, new_module)


def _apply_fold_pair(model: nn.Module, primary_name: str, bn_name: str) -> bool:
    """Fold primary_mod + bn_mod in-place and replace bn_mod with nn.Identity()."""
    primary_mod = _get_submodule_by_path(model, primary_name)
    bn_mod = _get_submodule_by_path(model, bn_name)

    if isinstance(primary_mod, nn.Conv2d) and isinstance(bn_mod, nn.BatchNorm2d):
        w_fold, b_fold = fold_conv_bn_weights(primary_mod, bn_mod)
    elif isinstance(primary_mod, nn.Linear) and isinstance(bn_mod, nn.BatchNorm1d):
        w_fold, b_fold = fold_linear_bn_weights(primary_mod, bn_mod)
    else:
        return False

    with torch.no_grad():
        primary_mod.weight.copy_(w_fold)
        if primary_mod.bias is None:
            primary_mod.bias = nn.Parameter(b_fold.clone().to(dtype=primary_mod.weight.dtype, device=primary_mod.weight.device))
        else:
            primary_mod.bias.copy_(b_fold)

    _replace_submodule_by_path(model, bn_name, nn.Identity())
    return True


def _discover_pairs_fx(model: nn.Module) -> list[tuple[str, str]]:
    """Trace model via torch.fx to find foldable (Conv2d/Linear, BatchNorm) pairs."""
    traced = fx.symbolic_trace(model)
    modules = dict(traced.named_modules())
    pairs: list[tuple[str, str]] = []

    for node in traced.graph.nodes:
        if node.op != "call_module":
            continue
        bn_mod = modules.get(node.target)
        if not isinstance(bn_mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
            continue
        if not node.args or not isinstance(node.args[0], fx.Node):
            continue
        prev_node = node.args[0]
        if prev_node.op != "call_module" or len(prev_node.users) != 1:
            continue

        prev_mod = modules.get(prev_node.target)
        valid_conv = isinstance(prev_mod, nn.Conv2d) and isinstance(bn_mod, nn.BatchNorm2d)
        valid_lin = isinstance(prev_mod, nn.Linear) and isinstance(bn_mod, nn.BatchNorm1d)
        if valid_conv or valid_lin:
            pairs.append((str(prev_node.target), str(node.target)))

    return pairs


def _discover_pairs_sequential(model: nn.Module, prefix: str = "") -> list[tuple[str, str]]:
    """Fallback pair discovery by scanning adjacent modules in nn.Sequential containers."""
    pairs: list[tuple[str, str]] = []
    if isinstance(model, nn.Sequential):
        items = list(model.named_children())
        for idx in range(len(items) - 1):
            name_a, mod_a = items[idx]
            name_b, mod_b = items[idx + 1]
            full_a = f"{prefix}.{name_a}" if prefix else name_a
            full_b = f"{prefix}.{name_b}" if prefix else name_b
            valid_conv = isinstance(mod_a, nn.Conv2d) and isinstance(mod_b, nn.BatchNorm2d)
            valid_lin = isinstance(mod_a, nn.Linear) and isinstance(mod_b, nn.BatchNorm1d)
            if valid_conv or valid_lin:
                pairs.append((full_a, full_b))

    for child_name, child_mod in model.named_children():
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        pairs.extend(_discover_pairs_sequential(child_mod, child_prefix))
    return pairs


def fold_batch_norms(model: nn.Module) -> tuple[nn.Module, list[tuple[str, str]]]:
    """Deep-copy model in eval mode, fold eligible BatchNorm layers, and return (model, folded_pairs)."""
    folded_model = copy.deepcopy(model)
    folded_model.eval()

    try:
        candidate_pairs = _discover_pairs_fx(folded_model)
    except Exception as exc:
        logger.warning(
            "torch.fx symbolic_trace failed during BN folding (%s); falling back to Sequential traversal.",
            exc,
        )
        candidate_pairs = _discover_pairs_sequential(folded_model)

    folded_pairs: list[tuple[str, str]] = []
    for primary_name, bn_name in candidate_pairs:
        if _apply_fold_pair(folded_model, primary_name, bn_name):
            folded_pairs.append((primary_name, bn_name))

    return folded_model, folded_pairs
