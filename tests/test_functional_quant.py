"""Unit tests for core functional uniform quantization and group reshaping."""

from __future__ import annotations

import unittest

import torch

from ptq_sim.quantization.functional import (
    compute_qparams,
    fake_quantize_tensor,
    get_qmin_qmax,
    reshape_for_group_quant,
    restore_from_group_quant,
    uniform_quantize,
)


class TestFunctionalQuantization(unittest.TestCase):
    def test_qmin_qmax_bounds(self) -> None:
        self.assertEqual(get_qmin_qmax(8, signed=True, narrow_range=True), (-127, 127))
        self.assertEqual(get_qmin_qmax(8, signed=True, narrow_range=False), (-128, 127))
        self.assertEqual(get_qmin_qmax(8, signed=False), (0, 255))
        self.assertEqual(get_qmin_qmax(4, signed=True, narrow_range=True), (-7, 7))
        self.assertEqual(get_qmin_qmax(4, signed=False), (0, 15))
        self.assertEqual(get_qmin_qmax(2, signed=True, narrow_range=True), (-1, 1))
        self.assertEqual(get_qmin_qmax(3, signed=False), (0, 7))
        with self.assertRaises(ValueError):
            get_qmin_qmax(1, signed=True)

    def test_symmetric_zero_point_and_asymmetric_zero_preservation(self) -> None:
        min_v = torch.tensor([-1.2])
        max_v = torch.tensor([2.5])

        s_sym, z_sym = compute_qparams(min_v, max_v, bit_width=8, symmetric=True, signed=True, narrow_range=True)
        self.assertEqual(float(z_sym.item()), 0.0)
        self.assertAlmostEqual(float(s_sym.item()), 2.5 / 127.0, places=6)

        s_asym, z_asym = compute_qparams(min_v, max_v, bit_width=8, symmetric=False, signed=False, include_zero=True)
        qmin, qmax = get_qmin_qmax(8, signed=False)
        fq_zero = fake_quantize_tensor(torch.tensor([0.0]), s_asym, z_asym, qmin, qmax)
        self.assertAlmostEqual(float(fq_zero.item()), 0.0, places=6)

    def test_monotonic_mse_across_bitwidths(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(4096)
        mses: list[float] = []
        for bits in (2, 4, 6, 8):
            s, z = compute_qparams(x.min(), x.max(), bit_width=bits, symmetric=True, signed=True, narrow_range=True)
            qmin, qmax = get_qmin_qmax(bits, signed=True, narrow_range=True)
            fq = fake_quantize_tensor(x, s, z, qmin, qmax)
            mses.append(float(torch.mean((x - fq) ** 2).item()))
        self.assertGreater(mses[0], mses[1])
        self.assertGreater(mses[1], mses[2])
        self.assertGreater(mses[2], mses[3])

    def test_per_channel_conv_and_linear(self) -> None:
        torch.manual_seed(1)
        # Conv2d weight: (8, 3, 3, 3) with wildly differing scales per output channel.
        scales = torch.tensor([0.01, 0.1, 1.0, 10.0, 0.05, 0.5, 5.0, 20.0]).reshape(8, 1, 1, 1)
        w = torch.randn(8, 3, 3, 3) * scales
        w_flat = w.reshape(8, -1)
        s, z = compute_qparams(w_flat.min(dim=1).values, w_flat.max(dim=1).values, bit_width=8, symmetric=True)
        self.assertEqual(tuple(s.shape), (8,))
        qmin, qmax = get_qmin_qmax(8, signed=True, narrow_range=True)
        fq = fake_quantize_tensor(w, s.reshape(8, 1, 1, 1), z.reshape(8, 1, 1, 1), qmin, qmax)
        # Small-scale channel 0 should preserve small relative precision.
        ch0_err = float((w[0] - fq[0]).abs().max().item())
        self.assertLess(ch0_err, 5e-4)

    def test_per_group_edge_padding_preserves_strictly_positive_tail(self) -> None:
        # 100 features is not divisible by group_size=64; second group has 36 strictly positive items in [10, 11].
        w = torch.cat(
            [
                torch.linspace(-1.0, 1.0, steps=64),
                torch.linspace(10.0, 11.0, steps=36),
            ]
        ).unsqueeze(0)  # (1, 100)
        grouped, orig_shape, pad_len = reshape_for_group_quant(w, group_size=64)
        self.assertEqual(tuple(grouped.shape), (1, 2, 64))
        self.assertEqual(pad_len, 28)

        g_min = grouped.min(dim=-1).values
        g_max = grouped.max(dim=-1).values
        # Edge-replication padding must NOT pull g_min down to 0.0!
        self.assertGreaterEqual(float(g_min[0, 1].item()), 10.0)

        s, z = compute_qparams(g_min, g_max, bit_width=4, symmetric=False, signed=False, include_zero=False)
        qmin, qmax = get_qmin_qmax(4, signed=False)
        fq_grouped = fake_quantize_tensor(grouped, s.unsqueeze(-1), z.unsqueeze(-1), qmin, qmax)
        restored = restore_from_group_quant(fq_grouped, orig_shape, pad_len)
        self.assertEqual(tuple(restored.shape), (1, 100))
        tail_err = float((w[0, 64:] - restored[0, 64:]).abs().max().item())
        self.assertLess(tail_err, 0.1)

    def test_constant_and_zero_tensors_are_finite(self) -> None:
        for val in (0.0, 3.14, -2.5):
            t = torch.full((16, 16), val)
            s, z = compute_qparams(t.min(), t.max(), bit_width=8, symmetric=False, signed=False)
            qmin, qmax = get_qmin_qmax(8, signed=False)
            q = uniform_quantize(t, s, z, qmin, qmax)
            fq = fake_quantize_tensor(t, s, z, qmin, qmax)
            self.assertTrue(torch.all(torch.isfinite(q)))
            self.assertTrue(torch.all(torch.isfinite(fq)))
            self.assertTrue(torch.allclose(t, fq, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
