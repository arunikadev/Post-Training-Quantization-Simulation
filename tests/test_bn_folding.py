"""Tests for Conv2d+BatchNorm2d and Linear+BatchNorm1d folding (FX + Sequential fallback)."""

from __future__ import annotations

import unittest

import torch
from torch import Tensor, nn

from ptq_sim.models.examples.resnet_mini import MiniResNet
from ptq_sim.models.examples.simple_cnn import SimpleCNN
from ptq_sim.models.examples.simple_mlp import SimpleMLP
from ptq_sim.quantization.bn_folding import fold_batch_norms


def _populate_non_trivial_bn(model: nn.Module) -> None:
    """Initialize BN running stats and affine params to non-identity values so folding is tested rigorously."""
    torch.manual_seed(99)
    for mod in model.modules():
        if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
            with torch.no_grad():
                mod.running_mean.normal_(mean=0.3, std=0.4)
                mod.running_var.uniform_(0.2, 2.5)
                if mod.weight is not None:
                    mod.weight.uniform_(0.5, 1.8)
                if mod.bias is not None:
                    mod.bias.normal_(mean=-0.1, std=0.3)


class DynamicControlFlowNet(nn.Module):
    """Non-FX-traceable model containing Sequential(Conv2d, BatchNorm2d) to test fallback path."""

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.head = nn.Linear(8, 4)

    def forward(self, x: Tensor) -> Tensor:
        # Data-dependent branch raises fx.TraceError, exercising sequential fallback.
        if x.mean() > 0:
            h = self.block(x)
        else:
            h = self.block(x * 0.5)
        return self.head(h.mean(dim=(2, 3)))


class TestBatchNormFolding(unittest.TestCase):
    def test_simple_cnn_bn_folding_equivalence(self) -> None:
        model = SimpleCNN(in_channels=3, num_classes=10)
        _populate_non_trivial_bn(model)
        model.eval()

        folded, pairs = fold_batch_norms(model)
        self.assertEqual(len(pairs), 3)
        x = torch.randn(4, 3, 32, 32)
        with torch.inference_mode():
            y_orig = model(x)
            y_fold = folded(x)
        self.assertTrue(torch.allclose(y_orig, y_fold, atol=1e-5, rtol=1e-4))

    def test_mini_resnet_fx_bn_folding_equivalence(self) -> None:
        model = MiniResNet(in_channels=3, num_classes=10)
        _populate_non_trivial_bn(model)
        model.eval()

        folded, pairs = fold_batch_norms(model)
        # stem + 3 blocks * 2 + 2 downsample projections = 9 Conv+BN pairs
        self.assertEqual(len(pairs), 9)
        x = torch.randn(4, 3, 32, 32)
        with torch.inference_mode():
            y_orig = model(x)
            y_fold = folded(x)
        self.assertTrue(torch.allclose(y_orig, y_fold, atol=1e-5, rtol=1e-4))

    def test_simple_mlp_linear_bn1d_folding_equivalence(self) -> None:
        model = SimpleMLP(input_dim=64, hidden_dims=(128, 64), num_classes=4, use_bn=True)
        _populate_non_trivial_bn(model)
        model.eval()

        folded, pairs = fold_batch_norms(model)
        self.assertEqual(len(pairs), 2)
        x = torch.randn(8, 64)
        with torch.inference_mode():
            y_orig = model(x)
            y_fold = folded(x)
        self.assertTrue(torch.allclose(y_orig, y_fold, atol=1e-5, rtol=1e-4))

    def test_dynamic_control_flow_sequential_fallback(self) -> None:
        model = DynamicControlFlowNet()
        _populate_non_trivial_bn(model)
        model.eval()

        folded, pairs = fold_batch_norms(model)
        self.assertEqual(pairs, [("block.0", "block.1")])
        for val in (1.0, -1.0):
            x = torch.randn(2, 3, 16, 16) + val * 3.0
            with torch.inference_mode():
                y_orig = model(x)
                y_fold = folded(x)
            self.assertTrue(torch.allclose(y_orig, y_fold, atol=1e-5, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
