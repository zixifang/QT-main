"""量价因子：正常序列、缺失、窗口不足、方差为零。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from factors.price_volume import compute_factor
from preprocess.adjust import adjust_ohlcv


def sample(rows: int = 30, volume=None, close=None) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=rows)
    close_values = np.arange(1, rows + 1, dtype=float) if close is None else np.asarray(close, dtype=float)
    volume_values = np.linspace(100, 100 + rows - 1, rows) if volume is None else np.asarray(volume, dtype=float)
    return pd.DataFrame(
        {
            "open": close_values * 0.99,
            "high": close_values * 1.02,
            "low": close_values * 0.98,
            "close": close_values,
            "adjusted_close": close_values,
            "volume": volume_values,
        },
        index=index,
    )


class PriceVolumeTest(unittest.TestCase):
    def test_roc_matches_shift_and_leaves_short_window_empty(self) -> None:
        frame = sample(8)
        values = compute_factor(frame, "ROC_N", {"N": 3})
        self.assertTrue(values.iloc[:3].isna().all())
        self.assertAlmostEqual(float(values.iloc[3]), 4 / 1 - 1, places=8)
        self.assertEqual(len(values), len(frame))

    def test_nan_in_price_is_not_filled(self) -> None:
        frame = sample(12)
        frame.iloc[5, frame.columns.get_loc("adjusted_close")] = np.nan
        frame.iloc[5, frame.columns.get_loc("close")] = np.nan
        values = compute_factor(frame, "ROC_N", {"N": 2})
        self.assertTrue(pd.isna(values.iloc[5]))
        self.assertTrue(pd.isna(values.iloc[7]))

    def test_zero_variance_correlation_is_nan(self) -> None:
        frame = sample(12, volume=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], close=[5] * 12)
        values = compute_factor(frame, "CCV_N", {"N": 5})
        self.assertTrue(values.dropna().empty)

    def test_cov_uses_previous_volume(self) -> None:
        frame = sample(10)
        frame["volume"] = np.arange(10, dtype=float)
        frame["open"] = frame["adjusted_close"].shift(1).fillna(frame["adjusted_close"]) + frame["volume"].shift(1).fillna(0)
        values = compute_factor(frame, "COV_N", {"N": 4})
        self.assertGreater(len(values.dropna()), 0)
        self.assertTrue(values.iloc[:4].isna().all())

    def test_adjust_factor_scales_ohlc_and_keeps_volume(self) -> None:
        frame = sample(4, close=[10, 10.1, 10.2, 10.3])
        frame["adjusted_close"] = frame["close"] * 2
        frame["volume"] = [10, 20, 30, 40]
        adjusted, jumps = adjust_ohlcv(frame)
        self.assertAlmostEqual(float(adjusted["open"].iloc[0]), float(frame["open"].iloc[0] * 2))
        self.assertEqual(float(adjusted["volume"].iloc[1]), 20)
        self.assertFalse(bool(jumps.any()))

    def test_morphology_does_not_divide_by_zero_range(self) -> None:
        frame = sample(6)
        frame["high"] = frame["low"] = frame["adjusted_close"]
        values = compute_factor(frame, "CLOSE_POS")
        self.assertTrue(values.isna().all())


if __name__ == "__main__":
    unittest.main()
