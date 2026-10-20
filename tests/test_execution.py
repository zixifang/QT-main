"""股数回测：一次买卖、现金不足、停牌、手续费。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from backtest.execution import run_backtest


def prices(rows: list[list[float]]) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=len(rows))
    return pd.DataFrame(rows, index=index, columns=["A", "B"])


class ExecutionTest(unittest.TestCase):
    def test_rebalance_sells_before_buying(self) -> None:
        close = prices([[10, 10], [10, 10], [12, 10]])
        signal = pd.DataFrame({"A": [1.0, 1.0, 0.0], "B": [0.0, 0.0, 1.0]}, index=close.index)
        result = run_backtest(
            close,
            signal,
            {
                "rebalance": "D",
                "top_n": 1,
                "weight": "equal",
                "max_weight": 1,
                "commission": 0,
                "slippage": 0,
                "initial_cash": 10000,
                "fill": "close",
            },
        )
        self.assertEqual(float(result.positions["A"].iloc[0]), 1000)
        self.assertEqual(float(result.positions["B"].iloc[2]), 1200)
        self.assertAlmostEqual(float(result.nav_series.iloc[-1]), 12000, places=6)
        sides = list(result.trades["side"])
        self.assertEqual(sides[:3], ["buy", "sell", "buy"])

    def test_insufficient_cash_buys_a_partial_lot(self) -> None:
        close = prices([[30, 30], [30, 30]])
        signal = pd.DataFrame({"A": [1.0, 1.0], "B": [0.0, 0.0]}, index=close.index)
        result = run_backtest(
            close,
            signal,
            {
                "rebalance": "D",
                "top_n": 1,
                "max_weight": 1,
                "commission": 0,
                "slippage": 0,
                "initial_cash": 100,
                "fill": "close",
            },
        )
        self.assertEqual(float(result.positions["A"].iloc[0]), 3)
        self.assertAlmostEqual(float(result.trades.iloc[0]["cash_after"]), 10, places=6)

    def test_missing_price_keeps_the_position(self) -> None:
        close = prices([[10, 10], [np.nan, 10], [11, 10]])
        signal = pd.DataFrame({"A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 1.0]}, index=close.index)
        result = run_backtest(
            close,
            signal,
            {
                "rebalance": "D",
                "top_n": 1,
                "max_weight": 1,
                "commission": 0,
                "slippage": 0,
                "initial_cash": 1000,
                "fill": "close",
            },
        )
        self.assertEqual(float(result.positions["A"].iloc[1]), 100)
        self.assertEqual(float(result.positions["A"].iloc[2]), 0)
        self.assertGreater(float(result.positions["B"].iloc[2]), 0)

    def test_commission_and_slippage_reduce_cash(self) -> None:
        close = prices([[10, 10], [10, 10]])
        signal = pd.DataFrame({"A": [1.0, 1.0], "B": [0.0, 0.0]}, index=close.index)
        result = run_backtest(
            close,
            signal,
            {
                "rebalance": "D",
                "top_n": 1,
                "max_weight": 1,
                "commission": 0.0003,
                "slippage": 0.0005,
                "initial_cash": 10000,
                "fill": "close",
                "lot": 1,
            },
        )
        trade = result.trades.iloc[0]
        shares = float(trade["shares"])
        self.assertGreater(float(trade["commission"]), 0)
        self.assertAlmostEqual(float(trade["slippage"]), shares * 10 * 0.0005, places=6)
        spent = shares * 10 + float(trade["slippage"]) + float(trade["commission"])
        self.assertAlmostEqual(float(trade["cash_after"]), 10000 - spent, places=4)


if __name__ == "__main__":
    unittest.main()
