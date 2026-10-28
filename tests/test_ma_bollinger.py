"""双均线 / 布林带两个算法写出的 JSON 是否符合 `docs/界面数据契约.md`。

运行：python -m unittest tests.test_ma_bollinger -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from algorithms import bollinger, contract, ma_cross

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outdata" / "equity-2018"
STAT_KEYS = {
    "cumulative_return",
    "annualized_return",
    "volatility",
    "sharpe",
    "max_drawdown",
    "calmar",
    "trades",
    "win_rate",
}


def synthetic_close(size: int = 400, seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    values = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size)))
    index = pd.bdate_range("2020-01-02", periods=size)
    return pd.Series(values, index=index, name="close")


class ContractTest(unittest.TestCase):
    @unittest.skipUnless((OUT_DIR / "index.json").is_file(), "2018 样本结果已不在仓库里")
    def test_published_json_follows_contract(self) -> None:
        index = json.loads((OUT_DIR / "index.json").read_text(encoding="utf-8"))
        self.assertEqual(index["schema_version"], 1)
        self.assertEqual(index["dataset_id"], "equity-2018")
        self.assertEqual(index["source"], "data/Equity-Portfolio-Management-master")
        ids = [item["id"] for item in index["algorithms"]]
        self.assertIn("rsi", ids, "合并时不能删掉已有算法")
        self.assertIn("ma_cross", ids)
        self.assertIn("bollinger", ids)

        for entry in index["algorithms"]:
            if entry["id"] == "rsi":
                continue
            self.assertIn("2018", entry["years"])
            year = entry["years"]["2018"]
            self.assertEqual(set(year), {"symbols", "strategy", "benchmark"})
            # 年表字段以契约里的样例为准：策略块多一个 beat_benchmark。
            self.assertEqual(
                set(year["strategy"]),
                STAT_KEYS - {"volatility", "calmar"} | {"count", "winners", "losers", "profit_ratio", "median_return", "beat_benchmark"},
            )
            self.assertEqual(
                set(year["benchmark"]),
                STAT_KEYS - {"volatility", "calmar"} | {"count", "winners", "losers", "profit_ratio", "median_return"},
            )
            self.assertIsNone(year["benchmark"]["win_rate"], "买入持有不写胜率")
            self.assertEqual([row["symbol"] for row in year["symbols"]], index["symbols"])
            history = entry["history"]
            self.assertEqual(len(history["dates"]), len(history["strategy"]), "history 三条数组要等长")
            self.assertEqual(len(history["dates"]), len(history["benchmark"]))
            self.assertEqual(history["dates"], sorted(history["dates"]), "history 日期要从早到晚")

        for symbol in index["symbols"]:
            record = json.loads((OUT_DIR / f"{symbol}.json").read_text(encoding="utf-8"))
            self.assertEqual(record["symbol"], symbol)
            self.assertEqual(record["dataset_id"], index["dataset_id"])
            dates = record["dates"]
            self.assertEqual(len(dates), len(record["close"]))
            self.assertEqual(len(dates), len(record["volume"]))
            self.assertIn("rsi", record["algorithms"])
            for algorithm_id in ("ma_cross", "bollinger"):
                algo = record["algorithms"][algorithm_id]
                for key in ("indicator", "signal", "position", "equity", "benchmark_equity"):
                    self.assertEqual(len(algo[key]), len(dates), f"{symbol}.{algorithm_id}.{key} 长度不对")
                self.assertEqual(algo["position"][0], 0.0, "第一天不该有持仓")
                self.assertEqual(algo["position"][1:], algo["signal"][:-1], "position 应为 signal 后移一天")
                self.assertTrue(set(algo["signal"]) <= {0.0, 1.0}, "只做多，权重只能是 0 或 1")
                known = set(dates)
                self.assertTrue(set(algo["buy_dates"]) <= known)
                self.assertTrue(set(algo["sell_dates"]) <= known)
                self.assertEqual(algo["equity"][0], 1.0, "净值起点为 1")
                self.assertEqual(set(algo["stats"]["strategy"]), STAT_KEYS)
                self.assertIsNone(algo["stats"]["benchmark"]["win_rate"])
                self.assertEqual(algo["id"], algorithm_id)
                if algorithm_id == "bollinger":
                    self.assertEqual(algo["params"]["oversold"], -bollinger.K)
                    self.assertEqual(algo["params"]["overbought"], bollinger.K)

    def test_no_lookahead(self) -> None:
        close = synthetic_close()
        full = {
            "ma_cross": ma_cross.ma_position(close).to_numpy(),
            "bollinger": bollinger.bollinger_position(close).to_numpy(),
        }
        for cut in (120, 280):
            piece = close.iloc[:cut]
            self.assertTrue(np.array_equal(ma_cross.ma_position(piece).to_numpy(), full["ma_cross"][:cut]))
            self.assertTrue(np.array_equal(bollinger.bollinger_position(piece).to_numpy(), full["bollinger"][:cut]))

    def test_build_is_deterministic(self) -> None:
        close = synthetic_close(300, seed=5)
        first = ma_cross.build(close)[0]
        second = ma_cross.build(close)[0]
        self.assertEqual(first, second)
        self.assertEqual(len(first["equity"]), len(close))

    def test_merge_keeps_other_algorithms(self) -> None:
        close = synthetic_close(120, seed=7)
        frame = pd.DataFrame(
            {
                "date": close.index,
                "open": close.to_numpy(),
                "high": close.to_numpy(),
                "low": close.to_numpy(),
                "close": close.to_numpy(),
                "volume": 1000,
            }
        )
        payload, _yearly, _equity, _benchmark = bollinger.build(close)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "AAA.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset_id": "demo",
                        "symbol": "AAA",
                        "price_field": "close",
                        "dates": [day.strftime("%Y-%m-%d") for day in close.index],
                        "open": [], "high": [], "low": [], "close": [], "volume": [],
                        "algorithms": {"rsi": {"id": "rsi", "name": "RSI", "note": "旧算法"}},
                    }
                ),
                encoding="utf-8",
            )
            contract.merge_symbol(path, "AAA", "demo", frame, close, payload)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("rsi", saved["algorithms"], "别的算法条目被清掉了")
            self.assertIn("bollinger", saved["algorithms"])

    def test_reject_mismatched_dates(self) -> None:
        close = synthetic_close(120, seed=3)
        frame = pd.DataFrame({"date": close.index, "close": close.to_numpy(), "volume": 1})
        payload, _y, _e, _b = ma_cross.build(close)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "AAA.json"
            path.write_text(json.dumps({"dates": ["2020-01-02"]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                contract.merge_symbol(path, "AAA", "demo", frame, close, payload)

    def test_history_uses_common_dates(self) -> None:
        index = pd.bdate_range("2020-01-01", periods=10)
        short = pd.Series(1.0, index=index[:8])
        long = pd.Series(2.0, index=index)
        history = contract.build_history({"A": short}, {"A": long})
        self.assertEqual(len(history["dates"]), 8, "共同交易日不足时退回并集前应先用交集")

    def test_history_falls_back_to_union(self) -> None:
        index = pd.bdate_range("2020-01-01", periods=10)
        left = pd.Series(1.0, index=index[:4])
        right = pd.Series(2.0, index=index[6:])
        history = contract.build_history({"A": left, "B": right}, {"A": left, "B": right})
        self.assertEqual(len(history["dates"]), 8, "没有共同交易日时应退回并集")
        self.assertEqual(history["dates"][0], index[0].strftime("%Y-%m-%d"))
        self.assertEqual(history["dates"][-1], index[-1].strftime("%Y-%m-%d"))


if __name__ == "__main__":
    unittest.main()
