"""用已训练的多模态 Transformer 在验证集上回测，并写入纳斯达克 XBRL。

运行：python -m algorithms.transformer_backtest

只给验证集决策日打分。每 5 个交易日买入分数最高的 40%，权重次日收盘生效。
右侧图表通过 view_start / view_end 只画这一段。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import numpy as np
import pandas as pd

from algorithms.index_enhance import (
    _describe_returns,
    _num,
    _overview_side,
    _pair_stats,
    _series,
    _share_portfolio,
    _trade_dates,
)
from algorithms.multimodal_transformer import (
    ModelConfig,
    MultimodalReturnTransformer,
    make_loader,
    split_indices,
    StockMultimodalDataset,
)
from algorithms.train_multimodal import TRAIN_END, load_annual_factors, load_prices
from engine import run_backtest

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outdata" / "nasdaq-xbrl"
WEIGHTS = ROOT / "outdata" / "multimodal" / "best.pt"
ALGO_ID = "transformer"
ALGO_NAME = "【多模态Transformer】"
VIEW_START = "2023-01-03"
KEEP = 0.40
FEE = 0.001
STEP = 5


def _note(first: str, last: str, price_end: str) -> str:
    return (
        "第1步 对齐行情和已公告年报：训练和验证都做。\n"
        "第2步 当天截面标准化：训练、验证各自只看当天。\n"
        "第3步 训练模型：训练集，决策日 2015-01-30 至 2022-12-22。\n"
        "第4步 挑选权重：仍用当时验证集，决策日 2023-01-03 至 2024-12-23，没有重训。\n"
        f"第5步 回测：验证集决策日从 2023-01-03 延到 {last}，行情画到 {price_end}。"
        "每5个交易日买入最高的40%，次日生效。右侧只画这段验证集。"
    )


def _scores(dataset: StockMultimodalDataset, model: MultimodalReturnTransformer) -> pd.DataFrame:
    parts = split_indices(dataset, TRAIN_END, None)
    loader = make_loader(dataset, parts["val"], shuffle=False, min_names=20)
    rows: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pred = model.forward_batch(batch).detach().cpu().numpy()
            for ticker, day, score in zip(batch["ticker"], batch["date"], pred):
                rows.append({"date": pd.Timestamp(day), "ticker": ticker, "score": float(score)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("验证集没有打出分数")
    return frame.pivot(index="date", columns="ticker", values="score").sort_index()


def _targets(scores: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    signal = pd.DataFrame(np.nan, index=calendar, columns=scores.columns, dtype="float64")
    days = list(scores.index.intersection(calendar))
    for day in days[::STEP]:
        row = scores.loc[day].replace([np.inf, -np.inf], np.nan).dropna()
        if len(row) < 5:
            continue
        keep = max(1, int(round(len(row) * KEEP)))
        chosen = row.nlargest(keep)
        signal.loc[day, :] = 0.0
        signal.loc[day, chosen.index] = 1.0 / len(chosen)
    held = signal.ffill().fillna(0.0).shift(1).fillna(0.0)
    return held


def _sleeve(weight: pd.Series, asset: pd.Series) -> pd.Series:
    held = weight.reindex(asset.index).fillna(0.0)
    change = asset.where(held > 1e-8, 0.0).fillna(0.0)
    return (1.0 + change).cumprod()


def _align(values: pd.Series, dates: list[str]) -> list:
    lookup = {stamp.strftime("%Y-%m-%d"): _num(value) for stamp, value in values.items() if pd.notna(value)}
    return [lookup.get(day) for day in dates]


def _year_return(curve: pd.Series, year: int) -> float | None:
    part = curve[curve.index.year == year].dropna()
    if len(part) < 2 or float(part.iloc[0]) == 0.0:
        return None
    return float(part.iloc[-1] / part.iloc[0] - 1.0)


def main() -> None:
    checkpoint = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    model = MultimodalReturnTransformer(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    prices = load_prices()
    factors = load_annual_factors()
    dataset = StockMultimodalDataset(prices, factors, window=20, horizon=5, require_fundamental=True)
    scores = _scores(dataset, model)
    calendar = pd.DatetimeIndex(pd.to_datetime(prices["date"].unique())).sort_values()
    calendar = calendar[calendar >= VIEW_START]
    closes = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    closes = closes.reindex(index=calendar, columns=scores.columns)
    asset = closes.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    held = _targets(scores, calendar).reindex(columns=scores.columns).fillna(0.0)
    held = held.where(closes.notna(), 0.0)
    totals = held.sum(axis=1)
    held = held.div(totals.where(totals > 0.0), axis=0).fillna(0.0)
    net, equity = _share_portfolio(held, asset, FEE)
    ndx = pd.read_csv(ROOT / "data" / "nasdaq100" / "NDX.csv", usecols=["Date", "Close"])
    ndx["Date"] = pd.to_datetime(ndx["Date"])
    benchmark_price = ndx.set_index("Date")["Close"].astype(float).reindex(calendar)
    benchmark = run_backtest(benchmark_price, pd.Series(1.0, index=calendar), fee=0.0)
    bench_equity = benchmark["equity"]
    bench_return = benchmark["return"]

    index_path = OUT_DIR / "index.json"
    catalog = json.loads(index_path.read_text(encoding="utf-8"))
    symbols = [item for item in catalog.get("symbols") or [] if (OUT_DIR / f"{item}.json").exists()]
    sleeves: dict[str, pd.Series] = {}
    buyholds: dict[str, pd.Series] = {}
    for ticker in scores.columns:
        sleeves[ticker] = _sleeve(held[ticker], asset[ticker])
        buyholds[ticker] = (1.0 + asset[ticker].fillna(0.0)).cumprod()

    note = _note(
        str(scores.index.min().date()),
        str(scores.index.max().date()),
        str(calendar.max().date()),
    )
    years: dict[str, dict] = {}
    for year in sorted({int(value) for value in net.index.year}):
        mask = net.index.year == year
        if not bool(mask.any()):
            continue
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
        summary = _pair_stats(net.loc[mask], bench_return.loc[mask])
        values = np.asarray(stock_returns, dtype=float)
        strategy = _overview_side(
            summary["strategy"],
            beat=beats,
            count=len(values),
            winners=int((values > 0).sum()) if len(values) else 0,
            losers=int((values <= 0).sum()) if len(values) else 0,
            mean_return=float(values.mean()) if len(values) else None,
            median_return=float(np.median(values)) if len(values) else None,
            annualized_return=summary["strategy"]["annualized_return"],
            sharpe=summary["strategy"]["sharpe"],
            drawdown=summary["strategy"]["max_drawdown"],
        )
        strategy["cumulative_return"] = summary["strategy"]["cumulative_return"]
        benchmark_side = dict(summary["benchmark"])
        benchmark_side["count"] = None
        holding = {
            "cumulative_return": _num(float(np.mean(hold_returns))) if hold_returns else None,
            "median_return": _num(float(np.median(hold_returns))) if hold_returns else None,
        }
        years[str(year)] = {
            "symbols": rows,
            "strategy": strategy,
            "benchmark": benchmark_side,
            "holding": holding,
        }

    entry = {
        "id": ALGO_ID,
        "name": ALGO_NAME,
        "note": note,
        "view_start": VIEW_START,
        "view_end": str(calendar.max().date()),
        "years": years,
        "history": {
            "dates": [stamp.strftime("%Y-%m-%d") for stamp in equity.index],
            "strategy": _series(equity),
            "benchmark": _series(bench_equity.reindex(equity.index)),
            "holding": _series(pd.DataFrame(buyholds).mean(axis=1).reindex(equity.index)),
        },
    }
    algorithms = [item for item in catalog.get("algorithms") or [] if item.get("id") != ALGO_ID]
    algorithms.append(entry)
    catalog["algorithms"] = algorithms
    index_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")

    signal = held.shift(-1)
    for index, ticker in enumerate(symbols, start=1):
        path = OUT_DIR / f"{ticker}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        dates = payload.get("dates") or []
        weight = held[ticker] if ticker in held.columns else pd.Series(0.0, index=calendar)
        curve = sleeves.get(ticker, pd.Series(1.0, index=calendar))
        base = buyholds.get(ticker, pd.Series(1.0, index=calendar))
        buys, sells = _trade_dates(weight.index, weight.fillna(0.0))
        own_return = curve.pct_change().fillna(0.0)
        base_return = base.pct_change().fillna(0.0)
        body = {
            "id": ALGO_ID,
            "name": ALGO_NAME,
            "note": note,
            "view_start": VIEW_START,
            "view_end": str(calendar.max().date()),
            "params": {"keep": KEEP, "step": STEP, "fee": FEE, "horizon": 5},
            "indicator": _align(scores[ticker] if ticker in scores.columns else pd.Series(dtype=float), dates),
            "signal": _align(signal[ticker] if ticker in signal.columns else weight, dates),
            "position": _align(weight, dates),
            "buy_dates": buys,
            "sell_dates": sells,
            "equity": _align(curve, dates),
            "benchmark_equity": _align(base, dates),
            "stats": {
                "strategy": _describe_returns(own_return.to_numpy(dtype=float), (len(sells), 0)),
                "benchmark": _describe_returns(base_return.to_numpy(dtype=float), (0, 0), hide_win_rate=True),
            },
        }
        algorithms_map = payload.get("algorithms")
        if not isinstance(algorithms_map, dict):
            algorithms_map = {}
        algorithms_map[ALGO_ID] = body
        payload["algorithms"] = algorithms_map
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if index % 20 == 0 or index == len(symbols):
            print(f"已写入 {index}/{len(symbols)}", flush=True)

    ndx_path = OUT_DIR / "NDX.json"
    if ndx_path.exists():
        payload = json.loads(ndx_path.read_text(encoding="utf-8"))
        dates = payload.get("dates") or []
        algorithms_map = payload.get("algorithms") if isinstance(payload.get("algorithms"), dict) else {}
        algorithms_map[ALGO_ID] = {
            "id": ALGO_ID,
            "name": ALGO_NAME,
            "note": note,
            "view_start": VIEW_START,
            "view_end": str(calendar.max().date()),
            "equity": _align(equity, dates),
            "benchmark_equity": _align(bench_equity, dates),
            "stats": _pair_stats(net, bench_return),
            "years": {year: {"strategy": block["strategy"], "benchmark": block["benchmark"]} for year, block in years.items()},
        }
        payload["algorithms"] = algorithms_map
        ndx_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"验证集组合收益 {_num(float((1 + net).prod() - 1))}，"
        f"指数收益 {_num(float((1 + bench_return.fillna(0)).prod() - 1))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
