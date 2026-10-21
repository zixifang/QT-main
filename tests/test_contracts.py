"""未来函数、幸存者偏差、复权方式和服务契约。"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.portfolio import simulate
from factors.price import residual_momentum
from preprocess.prepare import clean_returns, known_cutoff, neutralize_frame, signal_date
from services.backtest_service import BacktestService
from services.composite_service import CompositeService
from services.config_store import DEFAULT_CONFIG, validate_config
from services.data_service import DataService
from services.errors import AdjustError, ConstituentGapError, DataGapError, LookaheadError
from services.eval_service import EvalService
from services.factor_service import FactorService
import inspect
from services.data_service import DataService as DataSvc
from services.factor_service import FactorService as FactorSvc
from services.eval_service import EvalService as EvalSvc
from services.composite_service import CompositeService as CompSvc
from services.backtest_service import BacktestService as BackSvc
from services.task_service import TaskService


def write_ohlcv(root: Path, symbol: str, n: int = 180, seed: int = 1, close=None) -> None:
    index = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    values = close if close is not None else 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, n))
    frame = pd.DataFrame(
        {
            "Date": index.strftime("%Y-%m-%d"),
            "Open": values,
            "High": values,
            "Low": values,
            "Close": values,
            "Adj Close": np.asarray(values) / 10,
            "Volume": rng.integers(1_000_000, 2_000_000, n),
        }
    )
    folder = root / "data" / "nasdaq100"
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / f"{symbol}.csv", index=False)


def sample_config(**overrides) -> dict:
    config = {
        "universe": ["AAA", "BBB"],
        "benchmark": "NDX",
        "start": "2020-01-01",
        "end": "2020-09-01",
        "adjust": "raw",
        "source_order": ["local_csv"],
        "api_keys": {"nasdaq_data_link": ""},
        "universe_mode": "manual",
        "rebalance": "M",
        "cost_bps": 0,
        "slippage_bps": 0,
        "max_weight": 0.6,
        "diversify_penalty": 0.2,
        "jump_limit": 0.5,
        "horizon": 5,
        "neutralize": ["beta", "liquidity"],
        "weight_scheme": "equal",
        "train_years": 7,
        "valid_years": 3,
        "beta_lookback": 40,
        "liquidity_lookback": 10,
        "factors": {
            "residual_momentum": {"enabled": True, "lookback": 40, "skip": 5},
            "low_volatility": {"enabled": True, "lookback": 40},
            "fwd_eps_growth": {"enabled": False},
        },
    }
    config.update(overrides)
    return validate_config(config)


class ContractTest(unittest.TestCase):
    def test_service_methods_exist(self) -> None:
        for name in ("coverage", "prices", "fundamentals", "constituents", "pull"):
            self.assertTrue(hasattr(DataSvc, name))
        self.assertIn("config", inspect.signature(DataSvc.coverage).parameters)
        self.assertIn("factor_id", inspect.signature(FactorSvc.compute).parameters)
        self.assertIn("factor_id", inspect.signature(EvalSvc.single).parameters)
        self.assertTrue(callable(CompSvc.combine))
        self.assertTrue(callable(BackSvc.run))
        self.assertTrue(callable(TaskService.start) and callable(TaskService.cancel) and callable(TaskService.poll))
        rows = FactorSvc(Path(".")).catalog()
        ids = {row["id"] for row in rows}
        self.assertIn("residual_momentum", ids)
        self.assertIn("fwd_eps_growth", ids)
        for row in rows:
            for key in ("id", "name", "formula", "fields", "direction", "role", "configurable"):
                self.assertIn(key, row)

    def test_reject_forward_adjust(self) -> None:
        with self.assertRaises(AdjustError) as caught:
            DataService(Path(".")).prices(["AAA"], "2020-01-01", "2020-02-01", adjust="qfq")
        self.assertIn("前复权", str(caught.exception))
        with self.assertRaises(DataGapError) as gap:
            DataService(Path(".")).prices(["AAA"], "2020-01-01", "2020-02-01", adjust="hfq")
        self.assertIn("后复权", str(gap.exception))

    def test_prices_use_close_not_adj(self) -> None:
        root = Path(self.id().replace(".", "_"))
        # use a temp dir via unittest
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_ohlcv(folder, "AAA", n=2, close=[10, 11])
            market = DataService(folder).load_market(["AAA"], "AAA", "2020-01-01", "2020-12-31", "raw", 0.5, ["local_csv"], {})
        self.assertEqual(float(market["close"]["AAA"].iloc[-1]), 11)

    def test_signal_waits_one_session(self) -> None:
        calendar = pd.bdate_range("2020-01-01", periods=10)
        self.assertEqual(signal_date(calendar[3], calendar), calendar[4])
        self.assertGreater(signal_date(calendar[3], calendar), pd.Timestamp(calendar[3]))
        with self.assertRaises(LookaheadError):
            signal_date(calendar[-1], calendar)

    def test_known_cutoff_hides_unfinished_ic(self) -> None:
        calendar = pd.bdate_range("2020-01-01", periods=100)
        self.assertEqual(known_cutoff(calendar[80], calendar, 21), calendar[59])

    def test_historical_constituents_are_not_backfilled(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fetch = root / "data_fetch"
            fetch.mkdir()
            (fetch / "nasdaq100_constituents.txt").write_text("# now\nAAPL\nMSFT\n", encoding="utf-8")
            service = DataService(root)
            with self.assertRaises(ConstituentGapError):
                service.constituents("2018-06-01")
            current = service.constituents("2026-10-15")
            self.assertEqual(current["symbols"], ["AAPL", "MSFT"])
            self.assertIn("2026-10", current["warning"])
            snap = root / "data" / "nasdaq100" / "snapshots"
            snap.mkdir(parents=True)
            (snap / "2018-06-01.txt").write_text("CSCO\nINTC\n", encoding="utf-8")
            historical = service.constituents("2018-06-01")
            self.assertEqual(historical["symbols"], ["CSCO", "INTC"])
            self.assertIsNone(historical["warning"])

    def test_residual_matches_definition_and_ignores_future(self) -> None:
        index = pd.bdate_range("2020-01-01", periods=120)
        rng = np.random.default_rng(2)
        market = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 120)), index=index)
        stock = pd.Series(50 * np.cumprod(1 + rng.normal(0, 0.012, 120)), index=index)
        lookback, skip = 30, 5
        factor = residual_momentum(stock, market, lookback, skip, 0.5)
        both = pd.concat({"s": stock, "m": market}, axis=1).dropna()
        rs = clean_returns(both["s"], 0.5)
        rm = clean_returns(both["m"], 0.5)
        i = 80
        y = rs.iloc[i - skip - lookback + 1 : i - skip + 1].to_numpy()
        x = rm.iloc[i - skip - lookback + 1 : i - skip + 1].to_numpy()
        centered = x - x.mean()
        beta = float(np.dot(centered, y - y.mean()) / np.dot(centered, centered))
        expected = float(np.sum(y - beta * x))
        self.assertAlmostEqual(float(factor.iloc[i]), expected, places=6)
        changed = stock.copy()
        changed.iloc[-10:] = changed.iloc[-10:] * 10
        later = residual_momentum(changed, market, lookback, skip, 0.5)
        pd.testing.assert_series_equal(factor.iloc[:-5], later.iloc[:-5])
        self.assertFalse(np.isclose(float(factor.iloc[-1]), float(later.iloc[-1]), equal_nan=True))

    def test_neutralize_needs_enough_names(self) -> None:
        day = pd.Timestamp("2020-01-31")
        columns = list("ABCDEF")
        beta = pd.DataFrame([[1, 2, 3, 4, 5, 6]], index=[day], columns=columns, dtype=float)
        raw = beta * 2 + 0.3
        out, warnings = neutralize_frame(raw, {"beta": beta})
        self.assertFalse(warnings)
        self.assertTrue(np.allclose(out.iloc[0].to_numpy(), 0, atol=1e-8))
        small, notes = neutralize_frame(raw.iloc[:, :2], {"beta": beta.iloc[:, :2]})
        self.assertTrue(any("中性化" in note for note in notes))
        self.assertTrue(np.allclose(small.iloc[0].to_numpy(), raw.iloc[0, :2].to_numpy()))

    def test_backtest_does_not_earn_same_day_return(self) -> None:
        index = pd.bdate_range("2020-01-01", periods=20)
        prices = pd.DataFrame(100.0, index=index, columns=["A", "B"])
        prices.loc[index[5], "A"] = 110
        prices.loc[index[6]:, "A"] = 121
        scores = pd.DataFrame({"A": 1.0, "B": 0.0}, index=index)
        result = simulate(
            prices,
            scores,
            pd.Series(100.0, index=index),
            "M",
            0,
            0,
            1,
            0,
            0.5,
            rebalance_dates=[index[5]],
        )
        self.assertAlmostEqual(float(result["net"].iloc[5]), 0.0, places=8)
        self.assertAlmostEqual(float(result["net"].iloc[6]), 0.10, places=6)

    def test_pipeline_on_synthetic_files(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ohlcv(root, "AAA", seed=1)
            write_ohlcv(root, "BBB", seed=2)
            write_ohlcv(root, "NDX", seed=3)
            config = sample_config()
            degraded = FactorService(root).compute(config, "fwd_eps_growth")
            self.assertTrue(degraded["degraded"])
            self.assertEqual(degraded["latest"], [])
            first = FactorService(root).compute(config, "low_volatility")
            second = FactorService(root).compute(config, "low_volatility")
            self.assertEqual(first["cache"], "miss")
            self.assertEqual(second["cache"], "hit")
            self.assertAlmostEqual(first["latest"][0]["score"], second["latest"][0]["score"], places=6)
            self.assertTrue(any("中性化" in note for note in first["warnings"]))
            combined = CompositeService(root).combine(config)
            self.assertAlmostEqual(sum(combined["last_weights"].values()), 1.0, places=6)
            evaluated = EvalService(root).single(config, "low_volatility")
            self.assertEqual(evaluated["n_buckets"], 2)
            self.assertTrue(any("分层" in note for note in evaluated["warnings"]))
            backtest = BacktestService(root).run(config)
            self.assertIsNotNone(backtest["full"]["cumulative"])
            self.assertTrue(any("幸存者" in note for note in backtest["warnings"]))
            blocked = sample_config(universe_mode="snapshot")
            with self.assertRaises(ConstituentGapError):
                BacktestService(root).run(blocked)


if __name__ == "__main__":
    unittest.main()
