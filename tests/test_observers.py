"""Unit tests for PTQ calibration observers (MinMax, EMA, Percentile, MSE, KL)."""

from __future__ import annotations

import unittest

import torch

from ptq_sim.config.schema import QuantizerConfig
from ptq_sim.quantization.functional import fake_quantize_tensor, get_qmin_qmax
from ptq_sim.quantization.observers import build_observer


class TestObservers(unittest.TestCase):
    def test_all_observers_per_tensor_multibatch(self) -> None:
        torch.manual_seed(42)
        batches = [torch.randn(16, 32) for _ in range(4)]
        for obs_name in ("minmax", "ema", "percentile", "mse", "kl"):
            cfg = QuantizerConfig(
                bit_width=8,
                symmetric=True,
                signed=True,
                narrow_range=True,
                granularity="per_tensor",
                observer=obs_name,
            )
            obs = build_observer(cfg)
            for b in batches:
                obs.observe(b)
            scale, zp = obs.calculate_qparams()
            self.assertTrue(torch.isfinite(scale).all(), msg=f"{obs_name} scale not finite")
            self.assertTrue(torch.isfinite(zp).all(), msg=f"{obs_name} zp not finite")
            self.assertGreater(float(scale.item()), 0.0)
            obs.reset()
            self.assertFalse(obs.has_observed)

    def test_outlier_robustness_percentile_kl_mse_beat_minmax(self) -> None:
        torch.manual_seed(7)
        bulk = torch.randn(20000) * 0.5
        # Inject extreme outliers so MinMax range explodes.
        outliers = torch.tensor([800.0, -750.0, 900.0, -850.0])
        x = torch.cat([bulk, outliers])

        def _bulk_mse(tensor_in: torch.Tensor, obs_name: str, bit_width: int = 8, **kwargs) -> float:
            cfg = QuantizerConfig(
                bit_width=bit_width,
                symmetric=True,
                signed=True,
                narrow_range=True,
                granularity="per_tensor",
                observer=obs_name,
                observer_kwargs=kwargs,
            )
            obs = build_observer(cfg)
            obs.observe(tensor_in)
            scale, zp = obs.calculate_qparams()
            qmin, qmax = get_qmin_qmax(bit_width, signed=True, narrow_range=True)
            fq_bulk = fake_quantize_tensor(bulk, scale, zp, qmin, qmax)
            return float(torch.mean((bulk - fq_bulk) ** 2).item())

        # Percentile clips even extreme 1800-sigma outliers cleanly.
        mm_extreme = _bulk_mse(x, "minmax", bit_width=8)
        pct_extreme = _bulk_mse(x, "percentile", bit_width=8, percentile=99.9)
        self.assertLess(pct_extreme, mm_extreme * 0.01)

        # KL entropy calibration beats MinMax decisively on strong activation outliers (~80-sigma).
        x_kl = torch.cat([bulk, torch.tensor([40.0, -38.0, 45.0, -42.0])])
        mm_kl = _bulk_mse(x_kl, "minmax", bit_width=8)
        kl_err = _bulk_mse(x_kl, "kl", bit_width=8)
        self.assertLess(kl_err, mm_kl * 0.02)

        # MSE observer minimizes global L2 error on 4-bit distributions with clipping-worthy tails.
        bulk_unit = torch.randn(20000)
        x_mse4 = torch.cat([bulk_unit, torch.tensor([8.5, -8.5])])
        cfg_mm4 = QuantizerConfig(bit_width=4, symmetric=True, signed=True, granularity="per_tensor", observer="minmax")
        cfg_mse4 = QuantizerConfig(bit_width=4, symmetric=True, signed=True, granularity="per_tensor", observer="mse")
        obs_mm4 = build_observer(cfg_mm4)
        obs_mse4 = build_observer(cfg_mse4)
        obs_mm4.observe(x_mse4)
        obs_mse4.observe(x_mse4)
        s_mm4, z_mm4 = obs_mm4.calculate_qparams()
        s_mse4, z_mse4 = obs_mse4.calculate_qparams()
        qmin4, qmax4 = get_qmin_qmax(4, signed=True, narrow_range=True)
        err_mm4 = float(torch.mean((bulk_unit - fake_quantize_tensor(bulk_unit, s_mm4, z_mm4, qmin4, qmax4)) ** 2).item())
        err_mse4 = float(torch.mean((bulk_unit - fake_quantize_tensor(bulk_unit, s_mse4, z_mse4, qmin4, qmax4)) ** 2).item())
        self.assertLess(err_mse4, err_mm4 * 0.25)

    def test_per_channel_and_per_group_observers(self) -> None:
        torch.manual_seed(11)
        w = torch.randn(16, 128)
        for gran in ("per_channel", "per_group"):
            for obs_name in ("minmax", "ema", "percentile", "mse", "kl"):
                cfg = QuantizerConfig(
                    bit_width=4,
                    symmetric=True,
                    signed=True,
                    granularity=gran,
                    group_size=32,
                    observer=obs_name,
                )
                obs = build_observer(cfg)
                obs.observe(w)
                scale, zp = obs.calculate_qparams()
                expected_shape = (16,) if gran == "per_channel" else (16, 4)
                self.assertEqual(tuple(scale.shape), expected_shape, msg=f"{gran}/{obs_name}")
                self.assertEqual(tuple(zp.shape), expected_shape, msg=f"{gran}/{obs_name}")
                self.assertTrue(torch.isfinite(scale).all())

    def test_kl_observer_does_not_collapse_on_normal_distribution(self) -> None:
        torch.manual_seed(123)
        x = torch.randn(20000)
        cfg = QuantizerConfig(
            bit_width=8,
            symmetric=True,
            signed=True,
            narrow_range=True,
            granularity="per_tensor",
            observer="kl",
        )
        obs = build_observer(cfg)
        obs.observe(x)
        scale, zp = obs.calculate_qparams()
        threshold = float(scale.item()) * 127.0
        self.assertEqual(float(zp.item()), 0.0)
        self.assertGreater(threshold, 2.2)
        self.assertLessEqual(threshold, float(x.abs().max().item()) + 0.1)


if __name__ == "__main__":
    unittest.main()
