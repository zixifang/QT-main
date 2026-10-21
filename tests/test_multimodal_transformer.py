"""多模态 Transformer 的截面标准化、公告日对齐、滑窗和 Rank IC 损失。"""

from __future__ import annotations

import math
import unittest

try:
    import torch
except ImportError:  # pragma: no cover - 环境没装 PyTorch 时跳过整组测试
    torch = None

import numpy as np
import pandas as pd

if torch is not None:
    from algorithms.multimodal_transformer import (
        BINARY_COLUMNS,
        CONTINUOUS_COLUMNS,
        OHLCV_FEATURES,
        CrossSectionBatchSampler,
        ModelConfig,
        MultimodalReturnTransformer,
        RankICLoss,
        StockMultimodalDataset,
        _average_rank,
        align_fundamentals_asof,
        config_for_dataset,
        cross_sectional_zscore,
        encode_ordinal,
        fill_missing_with_mask,
        make_loader,
        make_synthetic_market,
        orient_factors,
        run_epoch,
        split_indices,
    )


def _prices(dates: pd.DatetimeIndex, n_stocks: int = 3) -> pd.DataFrame:
    frames = []
    for i in range(n_stocks):
        close = 100.0 + np.arange(len(dates), dtype=float)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": f"S{i}",
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1000 * (i + 1),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _factors(dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    early = [1.0, 2.0, 3.0]
    late = [100.0, 0.0, 0.0]
    for i in range(3):
        base = {name: 1.0 for name in CONTINUOUS_COLUMNS}
        base.update({name: 1 for name in BINARY_COLUMNS})
        base["FScore"] = 6
        first = dict(base)
        first.update(
            {
                "ticker": f"S{i}",
                "filed_date": dates[12],
                "period_end": pd.Timestamp("2019-12-31"),
                "Accruals": early[i],
            }
        )
        second = dict(base)
        second.update(
            {
                "ticker": f"S{i}",
                "filed_date": dates[-10],
                "period_end": pd.Timestamp("2020-03-31"),
                "Accruals": late[i],
            }
        )
        rows.extend([first, second])
    return pd.DataFrame(rows)


@unittest.skipUnless(torch is not None, "需要安装 PyTorch")
class MultimodalTransformerTest(unittest.TestCase):
    def test_cross_sectional_zscore_ignores_missing(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["d", "d", "d", "e", "e"],
                "x": [1.0, np.nan, 3.0, 5.0, 5.0],
            }
        )
        got = cross_sectional_zscore(frame, ["x"], by="date", clip=None)["x"].to_numpy()
        self.assertTrue(np.allclose(got[:3], [-1.0, np.nan, 1.0], equal_nan=True))
        self.assertTrue(np.allclose(got[3:], [0.0, 0.0]))

    def test_three_names_match_population_std(self) -> None:
        frame = pd.DataFrame({"date": ["d"] * 3, "x": [1.0, 2.0, 3.0]})
        got = cross_sectional_zscore(frame, ["x"], clip=None)["x"].to_numpy()
        raw = np.array([1.0, 2.0, 3.0])
        expected = (raw - raw.mean()) / raw.std(ddof=0)
        self.assertTrue(np.allclose(got, expected))

    def test_fill_missing_keeps_zero_distinct_from_nan(self) -> None:
        frame = pd.DataFrame({"a": [1.0, np.nan, np.inf]})
        filled, mask = fill_missing_with_mask(frame, ["a"])
        self.assertTrue(np.allclose(filled[:, 0], [1.0, 0.0, 0.0]))
        self.assertEqual(mask[:, 0].tolist(), [True, False, False])

    def test_encode_ordinal_reserves_zero_for_missing(self) -> None:
        codes = encode_ordinal(pd.Series([np.nan, 0, 1, 9, -3, 12]), 9)
        self.assertEqual(codes.tolist(), [0, 1, 2, 10, 1, 10])

    def test_low_is_better_factors_are_negated(self) -> None:
        frame = pd.DataFrame(
            {
                "Accruals": [1.0, -2.0],
                "AssetGrowth": [0.2, 0.1],
                "NetStockIssues": [0.0, 4.0],
            }
        )
        got = orient_factors(frame)
        self.assertTrue(np.allclose(got["Accruals"], [-1.0, 2.0]))
        self.assertTrue(np.allclose(got["AssetGrowth"], [-0.2, -0.1]))
        self.assertTrue(np.allclose(got["NetStockIssues"], [0.0, -4.0]))

    def test_asof_uses_filing_date_not_period_end(self) -> None:
        calendar = pd.bdate_range("2020-01-02", periods=15)
        keys = pd.DataFrame({"ticker": ["A"] * len(calendar), "date": calendar})
        fund = pd.DataFrame(
            {
                "ticker": ["A", "B"],
                "filed_date": [calendar[4], calendar[4]],
                "period_end": [calendar[4] + pd.Timedelta(days=400), calendar[4]],
                "ROE": [0.33, 9.0],
            }
        )
        aligned = align_fundamentals_asof(keys, fund, calendar, announcement_lag=0, factor_columns=["ROE"])
        by_date = aligned.set_index("date")["ROE"]
        self.assertTrue(math.isnan(by_date.loc[calendar[3]]))
        self.assertAlmostEqual(by_date.loc[calendar[4]], 0.33)
        self.assertTrue(aligned["ROE"].isna().all() or "B" not in set(aligned["ticker"]))

    def test_weekend_filing_and_lag_and_later_period(self) -> None:
        calendar = pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07"])
        keys = pd.DataFrame({"ticker": ["A", "A", "A"], "date": calendar})
        fund = pd.DataFrame(
            {
                "ticker": ["A", "A"],
                "filed_date": [pd.Timestamp("2020-01-04"), pd.Timestamp("2020-01-05")],
                "period_end": [pd.Timestamp("2020-03-31"), pd.Timestamp("2019-12-31")],
                "ROE": [4.0, 1.0],
            }
        )
        aligned = align_fundamentals_asof(keys, fund, pd.DatetimeIndex(calendar), factor_columns=["ROE"])
        roe = aligned.set_index("date")["ROE"]
        self.assertTrue(math.isnan(roe.loc[pd.Timestamp("2020-01-03")]))
        # 周六那条报告期更晚，两条都要到下周一才可见，应留下 ROE 4。
        self.assertAlmostEqual(roe.loc[pd.Timestamp("2020-01-06")], 4.0)

        monday = pd.bdate_range("2020-01-06", periods=3)
        keys = pd.DataFrame({"ticker": ["A"] * 3, "date": monday})
        fund = pd.DataFrame(
            {
                "ticker": ["A"],
                "filed_date": [monday[0]],
                "period_end": [pd.Timestamp("2019-12-31")],
                "ROE": [0.5],
            }
        )
        lagged = align_fundamentals_asof(keys, fund, monday, announcement_lag=1, factor_columns=["ROE"])
        lagged_roe = lagged.set_index("date")["ROE"]
        self.assertTrue(math.isnan(lagged_roe.loc[monday[0]]))
        self.assertAlmostEqual(lagged_roe.loc[monday[1]], 0.5)

    def test_dataset_window_label_and_point_in_time(self) -> None:
        dates = pd.bdate_range("2020-01-02", periods=40)
        dataset = StockMultimodalDataset(
            _prices(dates),
            _factors(dates),
            window=10,
            horizon=3,
            announcement_lag=0,
            z_clip=None,
        )
        self.assertEqual(len(dataset.calendar), 40)
        self.assertFalse(bool(dataset.pit_audit()["leak"].any()))

        def item(ticker: str, day: pd.Timestamp) -> dict:
            day = pd.Timestamp(day)
            for index, (name, _) in enumerate(dataset.samples):
                if name == ticker and dataset.decision_dates[index] == day:
                    return dataset[index]
            self.fail(f"缺少样本 {ticker} {day.date()}")

        hidden = item("S0", dates[9])
        self.assertEqual(int(hidden["fscore"]), 0)
        self.assertEqual(float(hidden["continuous_mask"].sum()), 0.0)
        self.assertEqual(tuple(hidden["ohlcv"].shape), (10, len(OHLCV_FEATURES)))

        visible = item("S0", dates[12])
        self.assertEqual(int(visible["fscore"]), 7)
        self.assertTrue(torch.equal(visible["binary"], torch.full((9,), 2)))
        self.assertGreater(float(visible["continuous_mask"].sum()), 0.0)

        audit = dataset.pit_audit()
        early = audit[(audit["ticker"] == "S0") & (audit["decision_date"] == dates[15])].iloc[0]
        self.assertEqual(pd.Timestamp(early["filed_date"]), dates[12])
        late = audit[(audit["ticker"] == "S0") & (audit["decision_date"] == dates[31])].iloc[0]
        self.assertEqual(pd.Timestamp(late["filed_date"]), dates[30])

        accrual = CONTINUOUS_COLUMNS.index("Accruals")
        early_z = [float(item(f"S{i}", dates[15])["continuous"][accrual]) for i in range(3)]
        self.assertGreater(early_z[0], 0.5)
        self.assertAlmostEqual(early_z[1], 0.0, places=4)
        self.assertLess(early_z[2], -0.5)
        late_z = [float(item(f"S{i}", dates[31])["continuous"][accrual]) for i in range(3)]
        self.assertLess(late_z[0], late_z[1])
        self.assertAlmostEqual(late_z[1], late_z[2], places=4)

        end = 15
        expected = (100.0 + end + 3) / (100.0 + end) - 1.0
        self.assertAlmostEqual(float(item("S0", dates[end])["label"]), expected, places=5)

        index_by_end = {end_index: index for index, (name, end_index) in enumerate(dataset.samples) if name == "S0"}
        left = dataset[index_by_end[17]]["ohlcv"]
        right = dataset[index_by_end[18]]["ohlcv"]
        self.assertTrue(torch.allclose(left[1:], right[:-1]))

        volume_index = OHLCV_FEATURES.index("log_volume")
        raw = np.log1p(np.array([1000.0, 2000.0, 3000.0]))
        expected_z = (raw - raw.mean()) / raw.std(ddof=0)
        got_z = float(item("S1", dates[15])["ohlcv"][-1, volume_index])
        self.assertAlmostEqual(got_z, float(expected_z[1]), places=4)

    def test_announcement_lag_is_applied_by_dataset(self) -> None:
        dates = pd.bdate_range("2020-01-02", periods=30)
        fundamentals = _factors(dates)
        fundamentals = fundamentals[fundamentals["period_end"] == pd.Timestamp("2019-12-31")].copy()
        fundamentals["filed_date"] = dates[10]
        dataset = StockMultimodalDataset(
            _prices(dates),
            fundamentals,
            window=8,
            horizon=2,
            announcement_lag=1,
        )
        audit = dataset.pit_audit()
        self.assertFalse(bool(audit["leak"].any()))
        on_filing = audit[(audit["ticker"] == "S0") & (audit["decision_date"] == dates[10])]
        self.assertTrue(pd.isna(on_filing["filed_date"].iloc[0]))
        next_day = audit[(audit["ticker"] == "S0") & (audit["decision_date"] == dates[11])]
        self.assertEqual(pd.Timestamp(next_day["filed_date"].iloc[0]), dates[10])

    def test_split_leaves_an_embargo(self) -> None:
        dates = pd.bdate_range("2020-01-02", periods=40)
        dataset = StockMultimodalDataset(_prices(dates), _factors(dates), window=10, horizon=3)
        parts = split_indices(dataset, train_end=str(dates[20].date()), val_end=str(dates[35].date()))
        train_ends = {dataset.samples[index][1] for index in parts["train"]}
        val_ends = {dataset.samples[index][1] for index in parts["val"]}
        self.assertIn(17, train_ends)
        self.assertTrue({18, 19, 20}.isdisjoint(train_ends))
        self.assertTrue(val_ends)
        self.assertTrue(all(end > 20 and end + dataset.horizon <= 35 for end in val_ends))
        self.assertTrue(all(end + dataset.horizon <= 20 for end in train_ends))

    def test_average_rank_and_soft_spearman_loss(self) -> None:
        ranks = _average_rank(torch.tensor([10.0, 30.0, 20.0, 20.0]))
        self.assertTrue(torch.allclose(ranks, torch.tensor([0.0, 3.0, 1.5, 1.5])))

        target = torch.linspace(-1.0, 1.0, 12)
        loss_fn = RankICLoss(temperature=0.05)
        aligned = loss_fn(target.clone().requires_grad_(True), target)
        reversed_loss = loss_fn((-target).requires_grad_(True), target)
        self.assertLess(float(aligned.detach()), -0.999)
        self.assertGreater(float(reversed_loss.detach()), 0.999)

        flat = loss_fn(torch.randn(8, requires_grad=True), torch.zeros(8))
        self.assertEqual(float(flat.detach()), 0.0)
        flat.backward()

        pred = (target + torch.randn(12) * 0.05).requires_grad_(True)
        loss = loss_fn(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(float(pred.grad.abs().sum()), 0.0)

    def test_model_shapes_dropout_and_one_epoch(self) -> None:
        torch.manual_seed(0)
        for pool, length in (("cls", 21), ("last", 20)):
            config = ModelConfig(ohlcv_dim=8, window=20, pool=pool, head_dropout=0.3)
            model = MultimodalReturnTransformer(config)
            self.assertEqual(tuple(model.position.shape), (1, length, config.d_model))
            self.assertEqual(model.fusion.in_features, config.d_model * 2)
            self.assertEqual(model.ohlcv_proj.in_features, 16)
            dropouts = [module for module in model.head if isinstance(module, torch.nn.Dropout)]
            linears = [module for module in model.head if isinstance(module, torch.nn.Linear)]
            self.assertEqual(len(linears), 2)
            self.assertEqual(dropouts[0].p, 0.3)
            batch = 6
            pred = model(
                torch.randn(batch, 20, 8),
                torch.ones(batch, 20, dtype=torch.bool),
                torch.ones(batch, 20, 8, dtype=torch.bool),
                torch.randn(batch, 8),
                torch.ones(batch, 8),
                torch.randint(0, 11, (batch,)),
                torch.randint(0, 3, (batch, 9)),
            )
            self.assertEqual(tuple(pred.shape), (batch,))
            self.assertTrue(torch.isfinite(pred).all())

        prices, fundamentals = make_synthetic_market(n_stocks=12, n_days=48, filing_index=6)
        dataset = StockMultimodalDataset(prices, fundamentals, window=20, horizon=5)
        self.assertFalse(bool(dataset.pit_audit()["leak"].any()))
        model = MultimodalReturnTransformer(config_for_dataset(dataset))
        loader = make_loader(dataset, shuffle=False, min_names=8)
        self.assertGreater(len(loader), 0)
        before = {name: param.detach().clone() for name, param in model.named_parameters()}
        metrics = run_epoch(
            model,
            loader,
            RankICLoss(),
            torch.optim.AdamW(model.parameters(), lr=1e-3),
        )
        self.assertGreater(int(metrics["batches"]), 0)
        self.assertTrue(math.isfinite(float(metrics["loss"])))
        changed = any(not torch.allclose(before[name], param) for name, param in model.named_parameters())
        self.assertTrue(changed)
        self.assertGreater(len(CrossSectionBatchSampler(dataset.decision_dates, min_names=8)), 0)


if __name__ == "__main__":
    unittest.main()
