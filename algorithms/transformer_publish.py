"""把各数据集已训练的 Transformer 在测试集上回测，写入界面可读的 JSON。

运行：python -m algorithms.transformer_publish

只用测试集分数。训练和早停用过的决策日不画进右侧锁定区间。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import numpy as np
import pandas as pd

from algorithms.index_enhance import _describe_returns, _num, _overview_side, _series, _share_portfolio, _trade_dates
from algorithms.multimodal_transformer import ModelConfig, MultimodalReturnTransformer, StockMultimodalDataset, make_loader, split_indices
from algorithms.train_multimodal import TRAIN_END, VAL_END, output_dir
from algorithms.transformer_backtest import ALGO_ID, ALGO_NAME, FEE, KEEP, STEP, _align, _sleeve, _targets, _year_return
from algorithms.transformer_splits import DATASETS, empty_factors, listed_symbols, load_price_file, ratio_train_end
from engine import run_backtest

ROOT = Path(__file__).resolve().parents[1]
PUBLISH = (
    "nasdaq100",
    "tech-us-10y",
    "manufacturing-us-10y",
    "healthcare-us-10y",
    "us-agri",
)


def _test_indices(dataset: StockMultimodalDataset, spec: dict) -> list[int]:
    if spec.get("train_ratio"):
        train_end = ratio_train_end(dataset.calendar, dataset.window, dataset.horizon, float(spec["train_ratio"]))
        parts = split_indices(dataset, train_end, train_end)
    else:
        parts = split_indices(dataset, TRAIN_END, VAL_END)
    if not parts["test"]:
        raise RuntimeError(f"{spec['id']} 没有测试集样本")
    return parts["test"]


def _scores(dataset: StockMultimodalDataset, model: MultimodalReturnTransformer, indices: list[int], min_names: int) -> pd.DataFrame:
    loader = make_loader(dataset, indices, shuffle=False, min_names=min_names)
    rows: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pred = model.forward_batch(batch).detach().cpu().numpy()
            for ticker, day, score in zip(batch["ticker"], batch["date"], pred):
                rows.append({"date": pd.Timestamp(day), "ticker": ticker, "score": float(score)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("测试集没有打出分数")
    return frame.pivot(index="date", columns="ticker", values="score").sort_index()


def _benchmark(prices: pd.DataFrame, calendar: pd.DatetimeIndex, asset: pd.DataFrame, dataset_id: str) -> tuple[pd.Series, pd.Series, str]:
    ndx_path = ROOT / "data" / "nasdaq100" / "NDX.csv"
    if dataset_id == "nasdaq100" and ndx_path.exists():
        ndx = pd.read_csv(ndx_path, usecols=["Date", "Close"])
        ndx["Date"] = pd.to_datetime(ndx["Date"])
        benchmark_price = ndx.set_index("Date")["Close"].astype(float).reindex(calendar)
        result = run_backtest(benchmark_price, pd.Series(1.0, index=calendar), fee=0.0)
        return result["return"], result["equity"], "纳斯达克指数"
    listed = prices.pivot(index="date", columns="ticker", values="close").sort_index().reindex(index=calendar).notna()
    equal_weight = listed.div(listed.sum(axis=1).where(lambda count: count > 0), axis=0).fillna(0.0)
    benchmark_return = (equal_weight.shift(1).fillna(0.0) * asset).sum(axis=1)
    benchmark_return.iloc[0] = 0.0
    bench_equity = (1.0 + benchmark_return).cumprod()
    return benchmark_return, bench_equity, "买入持有"


def _note(dataset_id: str, train_end: str, score_start: str, score_end: str, price_end: str) -> str:
    if dataset_id == "us-agri":
        trained = "按决策日先后，前 80% 训练，其中末尾约 20% 只用于早停。"
    else:
        trained = f"实验集训练到 {train_end}，验证集 2023-01-03 至 2024-12-23 只用于早停。"
    return (
        f"{trained}\n"
        f"测试集决策日 {score_start} 至 {score_end}，不参与训练和挑权重。\n"
        f"每 5 个交易日买入分数最高的 40%，次日生效。右侧只画测试集，行情到 {price_end}。"
    )


def publish_one(spec: dict) -> None:
    dataset_id = spec["id"]
    weights = output_dir(dataset_id) / "best.pt"
    out_dir = ROOT / "outdata" / dataset_id
    if not weights.exists():
        raise FileNotFoundError(weights)
    if not (out_dir / "index.json").exists():
        raise FileNotFoundError(out_dir / "index.json")
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    model = MultimodalReturnTransformer(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    prices = load_price_file(spec["path"], spec["skip"], listed_symbols(dataset_id))
    factors = empty_factors()
    dataset = StockMultimodalDataset(prices, factors, window=20, horizon=5, require_fundamental=False)
    min_names = 8 if dataset_id == "healthcare-us-10y" else 20
    scores = _scores(dataset, model, _test_indices(dataset, spec), min_names)
    calendar = pd.DatetimeIndex(pd.to_datetime(prices["date"].unique())).sort_values()
    calendar = calendar[(calendar >= scores.index.min()) & (calendar <= prices["date"].max())]
    closes = prices.pivot(index="date", columns="ticker", values="close").sort_index().reindex(index=calendar)
    asset = closes.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    held = _targets(scores, calendar).reindex(index=calendar, columns=closes.columns).fillna(0.0)
    held = held.where(closes.notna(), 0.0)
    totals = held.sum(axis=1)
    held = held.div(totals.where(totals > 0.0), axis=0).fillna(0.0)
    net, equity = _share_portfolio(held, asset.reindex(columns=held.columns).fillna(0.0), FEE)
    benchmark_return, bench_equity, bench_name = _benchmark(prices, equity.index, asset, dataset_id)
    bench_equity = bench_equity.reindex(equity.index)
    benchmark_return = benchmark_return.reindex(equity.index).fillna(0.0)
    view_start = str(scores.index.min().date())
    view_end = str(calendar.max().date())
    saved = json.loads((weights.parent / "split.json").read_text(encoding="utf-8"))
    train_end = saved.get("train", {}).get("end") or saved.get("experiment", {}).get("train", {}).get("end") or ""
    note = _note(dataset_id, str(train_end), view_start, str(scores.index.max().date()), view_end)

    sleeves = {ticker: _sleeve(held[ticker], asset[ticker]) for ticker in held.columns}
    buyholds = {ticker: (1.0 + asset[ticker].fillna(0.0)).cumprod() for ticker in held.columns}
    catalog = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
    symbols = [item for item in catalog.get("symbols") or [] if (out_dir / f"{item}.json").exists()]
    years: dict[str, dict] = {}
    for year in sorted({int(value) for value in net.index.year}):
        mask = net.index.year == year
        rows = []
        stock_returns = []
        hold_returns = []
        beats = 0
        for ticker in symbols:
            if ticker not in sleeves:
                continue
            own = _year_return(sleeves[ticker], year)
            base = _year_return(buyholds[ticker], year)
            if own is None:
                continue
            rows.append(
                {
                    "symbol": ticker,
                    "cumulative_return": _num(own),
                    "benchmark_cumulative_return": None if base is None else _num(base),
                }
            )
            stock_returns.append(own)
            if base is not None:
                hold_returns.append(base)
                beats += int(own > base)
        if not stock_returns:
            continue
        values = np.asarray(stock_returns, dtype=float)
        strategy_stats = _describe_returns(net.loc[mask].to_numpy(dtype=float), (0, 0))
        benchmark_stats = _describe_returns(benchmark_return.loc[mask].to_numpy(dtype=float), (0, 0), hide_win_rate=True)
        strategy = _overview_side(
            strategy_stats,
            beat=beats,
            count=len(values),
            winners=int((values > 0).sum()),
            losers=int((values <= 0).sum()),
            mean_return=float(values.mean()),
            median_return=float(np.median(values)),
            annualized_return=strategy_stats["annualized_return"],
            sharpe=strategy_stats["sharpe"],
            drawdown=strategy_stats["max_drawdown"],
        )
        strategy["cumulative_return"] = strategy_stats["cumulative_return"]
        years[str(year)] = {
            "symbols": rows,
            "strategy": strategy,
            "benchmark": benchmark_stats,
            "holding": {
                "cumulative_return": _num(float(np.mean(hold_returns))) if hold_returns else None,
                "median_return": _num(float(np.median(hold_returns))) if hold_returns else None,
            },
        }
    history = {
        "dates": [stamp.strftime("%Y-%m-%d") for stamp in equity.index],
        "strategy": _series(equity),
        "benchmark": _series(bench_equity),
        "holding": _series(pd.DataFrame(buyholds).mean(axis=1).reindex(equity.index)),
    }
    entry = {
        "id": ALGO_ID,
        "name": ALGO_NAME,
        "note": note,
        "view_start": view_start,
        "view_end": view_end,
        "years": years,
        "history": history,
    }
    algorithms = [item for item in catalog.get("algorithms") or [] if item.get("id") != ALGO_ID]
    algorithms.append(entry)
    catalog["algorithms"] = algorithms
    if dataset_id == "nasdaq100" and "benchmark" not in catalog:
        catalog["benchmark"] = {"id": "NDX", "name": bench_name}
    (out_dir / "index.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")

    signal = held.shift(-1)
    for index, ticker in enumerate(symbols, start=1):
        path = out_dir / f"{ticker}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        dates = payload.get("dates") or []
        weight = held[ticker] if ticker in held.columns else pd.Series(0.0, index=calendar)
        curve = sleeves.get(ticker, pd.Series(1.0, index=calendar))
        base = buyholds.get(ticker, pd.Series(1.0, index=calendar))
        buys, sells = _trade_dates(weight.index, weight.fillna(0.0))
        body = {
            "id": ALGO_ID,
            "name": ALGO_NAME,
            "note": note,
            "view_start": view_start,
            "view_end": view_end,
            "params": {"keep": KEEP, "step": STEP, "fee": FEE, "trained_on": dataset_id},
            "indicator": _align(scores[ticker] if ticker in scores.columns else pd.Series(dtype=float), dates),
            "signal": _align(signal[ticker] if ticker in getattr(signal, "columns", []) else weight, dates),
            "position": _align(weight, dates),
            "buy_dates": buys,
            "sell_dates": sells,
            "equity": _align(curve, dates),
            "benchmark_equity": _align(base, dates),
            "stats": {
                "strategy": _describe_returns(curve.pct_change(fill_method=None).fillna(0.0).to_numpy(dtype=float), (len(sells), 0)),
                "benchmark": _describe_returns(base.pct_change(fill_method=None).fillna(0.0).to_numpy(dtype=float), (0, 0), hide_win_rate=True),
            },
        }
        algorithms_map = payload.get("algorithms") if isinstance(payload.get("algorithms"), dict) else {}
        algorithms_map[ALGO_ID] = body
        payload["algorithms"] = algorithms_map
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if index % 20 == 0 or index == len(symbols):
            print(f"已写入 {dataset_id} {index}/{len(symbols)}", flush=True)
    print(
        f"{dataset_id} 测试集 {view_start} 至 {view_end}，"
        f"组合收益 {_num(float((1 + net).prod() - 1))}，{bench_name} {_num(float((1 + benchmark_return.fillna(0)).prod() - 1))}",
        flush=True,
    )


def main() -> None:
    specs = {item["id"]: item for item in DATASETS}
    for dataset_id in PUBLISH:
        print(f"发布 {dataset_id}", flush=True)
        publish_one(specs[dataset_id])


if __name__ == "__main__":
    main()
