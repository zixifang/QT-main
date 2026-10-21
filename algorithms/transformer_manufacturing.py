"""用纳斯达克上训好的多模态 Transformer，对美国制造业近十年日线打分并回测。

运行：python -m algorithms.transformer_manufacturing

权重不再训练。制造业里能对上纳斯达克 10-K 的公司使用已公告年报，
其余公司只走量价分支，基本面按缺失处理。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import numpy as np
import pandas as pd

from algorithms.index_enhance import _describe_returns, _num, _overview_side, _trade_dates
from algorithms.multimodal_transformer import (
    ModelConfig,
    MultimodalReturnTransformer,
    StockMultimodalDataset,
    make_loader,
)
from algorithms.train_multimodal import JUMP_LIMIT, load_annual_factors
from algorithms.transformer_backtest import ALGO_ID, ALGO_NAME, FEE, _align, _sleeve, _targets, _year_return

ROOT = Path(__file__).resolve().parents[1]
PRICE_PATH = ROOT / "data" / "manufacturing-us-10y" / "manufacturing_us_10y.csv"
OUT_DIR = ROOT / "outdata" / "manufacturing-us-10y"
WEIGHTS = ROOT / "outdata" / "multimodal" / "best.pt"
NOTE = (
    "权重来自纳斯达克100：训练集 2015-01-30 至 2022-12-22，验证集 2023-01-03 至 2024-12-23。\n"
    "这里不再训练，直接对美国制造业近十年日线打分。\n"
    "与纳斯达克因子表重合的公司使用已公告 10-K，其余公司只用量价，基本面记为缺失。\n"
    "每5个交易日买入分数最高的40%，次日收盘换仓。"
)


def load_manufacturing_prices() -> pd.DataFrame:
    frame = pd.read_csv(PRICE_PATH)
    frame = frame.rename(columns={"Ticker": "ticker", "Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    frame["ticker"] = frame["ticker"].astype(str).str.strip()
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    previous = frame.groupby("ticker", sort=False)["close"].shift(1)
    jump = (frame["close"] / previous - 1.0).abs() > JUMP_LIMIT
    frame.loc[jump.fillna(False), ["open", "high", "low", "close", "volume"]] = np.nan
    return frame.reset_index(drop=True)


def _score_all(dataset: StockMultimodalDataset, model: MultimodalReturnTransformer) -> pd.DataFrame:
    loader = make_loader(dataset, shuffle=False, min_names=8)
    rows: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pred = model.forward_batch(batch).detach().cpu().numpy()
            for ticker, day, score in zip(batch["ticker"], batch["date"], pred):
                rows.append({"date": pd.Timestamp(day), "ticker": ticker, "score": float(score)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("没有打出分数")
    return frame.pivot(index="date", columns="ticker", values="score").sort_index()


def main() -> None:
    checkpoint = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    model = MultimodalReturnTransformer(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    prices = load_manufacturing_prices()
    names = set(prices["ticker"].unique())
    factors = load_annual_factors()
    factors = factors[factors["ticker"].isin(names)].copy()
    covered = sorted(factors["ticker"].unique())
    print(f"制造业 {len(names)} 只，其中 {len(covered)} 只能用到已有 10-K：{', '.join(covered) or '无'}", flush=True)
    dataset = StockMultimodalDataset(prices, factors, window=20, horizon=5, require_fundamental=False)
    scores = _score_all(dataset, model)
    calendar = pd.DatetimeIndex(scores.index.union(pd.to_datetime(prices["date"].unique()))).sort_values()
    calendar = calendar[(calendar >= scores.index.min()) & (calendar <= prices["date"].max())]
    closes = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    closes = closes.reindex(index=calendar)
    asset = closes.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    held = _targets(scores, calendar).reindex(index=calendar, columns=closes.columns).fillna(0.0)
    held = held.where(closes.notna(), 0.0)
    totals = held.sum(axis=1)
    held = held.div(totals.where(totals > 0.0), axis=0).fillna(0.0)
    from algorithms.index_enhance import _share_portfolio

    net, equity = _share_portfolio(held, asset, FEE)
    listed = closes.notna()
    equal_weight = listed.div(listed.sum(axis=1).where(lambda count: count > 0), axis=0).fillna(0.0)
    benchmark_return = (equal_weight.shift(1).fillna(0.0) * asset).sum(axis=1)
    benchmark_return.iloc[0] = 0.0
    bench_equity = (1.0 + benchmark_return).cumprod()
    span = f"{calendar.min().date()} 至 {calendar.max().date()}"
    print(
        f"{span} 组合收益 {_num(float((1 + net).prod() - 1))}，"
        f"等权持有 {_num(float((1 + benchmark_return).prod() - 1))}",
        flush=True,
    )

    sleeves = {ticker: _sleeve(held[ticker], asset[ticker]) for ticker in held.columns}
    buyholds = {ticker: (1.0 + asset[ticker].fillna(0.0)).cumprod() for ticker in held.columns}
    catalog = json.loads((OUT_DIR / "index.json").read_text(encoding="utf-8"))
    symbols = [item for item in catalog.get("symbols") or [] if (OUT_DIR / f"{item}.json").exists()]
    years: dict[str, dict] = {}
    for year in sorted(set(net.index.year)):
        mask = net.index.year == int(year)
        rows = []
        stock_returns = []
        hold_returns = []
        beats = 0
        for ticker in symbols:
            if ticker not in sleeves:
                continue
            own = _year_return(sleeves[ticker], int(year))
            base = _year_return(buyholds[ticker], int(year))
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
        years[str(int(year))] = {
            "symbols": rows,
            "strategy": strategy,
            "benchmark": benchmark_stats,
            "holding": {
                "cumulative_return": _num(float(np.mean(hold_returns))) if hold_returns else None,
                "median_return": _num(float(np.median(hold_returns))) if hold_returns else None,
            },
        }
    algorithms = [item for item in catalog.get("algorithms") or [] if item.get("id") != ALGO_ID]
    algorithms.append({"id": ALGO_ID, "name": ALGO_NAME, "note": NOTE, "years": years})
    catalog["algorithms"] = algorithms
    (OUT_DIR / "index.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")

    signal = held.shift(-1)
    for index, ticker in enumerate(symbols, start=1):
        path = OUT_DIR / f"{ticker}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        dates = payload.get("dates") or []
        weight = held[ticker] if ticker in held.columns else pd.Series(0.0, index=calendar)
        curve = sleeves.get(ticker, pd.Series(1.0, index=calendar))
        base = buyholds.get(ticker, pd.Series(1.0, index=calendar))
        buys, sells = _trade_dates(weight.index, weight.fillna(0.0))
        body = {
            "id": ALGO_ID,
            "name": ALGO_NAME,
            "note": NOTE,
            "params": {"keep": 0.4, "step": 5, "fee": FEE, "trained_on": "nasdaq100"},
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
        if index % 15 == 0 or index == len(symbols):
            print(f"已写入 {index}/{len(symbols)}", flush=True)


if __name__ == "__main__":
    main()
