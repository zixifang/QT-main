"""把 XBRL 因子逐个回测，写到数据集「纳斯达克XBRL」。

运行：python -m algorithms.xbrl_factors
界面不调用本模块。本程序只写 outdata/nasdaq-xbrl，不改纳斯达克100那张表。
已有的其他算法会保留，同名因子用这次的结果覆盖。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from algorithms.index_enhance import (
    BENCHMARK_NAME,
    FEE,
    INDEX_STEM,
    OOS_START,
    PRICE_FIELD,
    SCHEMA_VERSION,
    WEIGHT_KEEP,
    _cleaned_panel,
    _describe_returns,
    _equity,
    _month_ends,
    _overview_year,
    _pair_stats,
    _round_trips,
    _series,
    _share_equity,
    _share_portfolio,
    _stock_returns,
    _symbol_year_rows,
    _trade_dates,
    _write_merged_json,
    load_prices,
)
from data_fetch.build_factors import load_wide
from engine import run_backtest

ROOT = Path(__file__).resolve().parents[1]
FACTOR_PATH = ROOT / "data" / "factors" / "factors_nq100.parquet"
LONG_PATH = ROOT / "data" / "xbrl" / "nq100_xbrl_fundamentals.parquet"
PRICE_DIR = ROOT / "data" / "nasdaq100"
OUT_DIR = ROOT / "outdata" / "nasdaq-xbrl"
DATASET_ID = "nasdaq-xbrl"
DATASET_NAME = "纳斯达克XBRL"
MIN_NAMES = 5
UNIVERSE_NOTE = (
    "没有调仓日的历史成分名单，不能称为指数增强。"
    "股票池是 2026 年 10 月有年报 XBRL、并且本地有未复权日线的幸存者，已去掉 GOOG。"
    "因子只用 10-K，并且从 filed_date 当天起才能使用。"
)

# direction 为 1 表示数值越高越好；为 -1 表示排序前取反。
FACTORS = (
    {"id": "xbrl-roe", "name": "【ROE】", "column": "ROE", "direction": 1, "note": "年报 ROE，越高越好。"},
    {"id": "xbrl-roa", "name": "【ROA】", "column": "ROA", "direction": 1, "note": "年报 ROA，越高越好。"},
    {"id": "xbrl-gp", "name": "【毛利润因子】", "column": "GrossProfitability", "direction": 1, "note": "毛利润除以总资产，越高越好。"},
    {"id": "xbrl-bm", "name": "【账面市值比】", "column": "BM", "direction": 1, "note": "已披露股东权益除以当天收盘价乘以已披露股数，越高越便宜。"},
    {"id": "xbrl-accrual", "name": "【应计利润】", "column": "Accruals", "direction": -1, "note": "应计利润越低越好，排序前取反。"},
    {"id": "xbrl-turn", "name": "【资产周转率】", "column": "AssetTurnover", "direction": 1, "note": "营业收入除以平均总资产，越高越好。"},
    {"id": "xbrl-f-roa", "name": "【F-ROA】", "column": "F_ROA", "direction": 1, "note": "Piotroski 分项：ROA 为正记 1。"},
    {"id": "xbrl-f-cfo", "name": "【F-经营现金流】", "column": "F_CFO", "direction": 1, "note": "Piotroski 分项：经营现金流为正记 1。"},
    {"id": "xbrl-f-droa", "name": "【F-ROA改善】", "column": "F_dROA", "direction": 1, "note": "Piotroski 分项：ROA 高于上年记 1。"},
    {"id": "xbrl-f-accrual", "name": "【F-应计】", "column": "F_ACCRUAL", "direction": 1, "note": "Piotroski 分项：经营现金流高于净利润记 1。"},
    {"id": "xbrl-f-lever", "name": "【F-杠杆下降】", "column": "F_dLEVER", "direction": 1, "note": "Piotroski 分项：长期负债率下降记 1。"},
    {"id": "xbrl-f-liquid", "name": "【F-流动比率改善】", "column": "F_dLIQUID", "direction": 1, "note": "Piotroski 分项：流动比率上升记 1。"},
    {"id": "xbrl-f-issue", "name": "【F-未增发】", "column": "F_EQ_ISSUE", "direction": 1, "note": "Piotroski 分项：流通股没有增加记 1。"},
    {"id": "xbrl-f-margin", "name": "【F-毛利率改善】", "column": "F_dGROSS_MARGIN", "direction": 1, "note": "Piotroski 分项：毛利率上升记 1。"},
    {"id": "xbrl-f-turn", "name": "【F-周转率改善】", "column": "F_dASSET_TURN", "direction": 1, "note": "Piotroski 分项：资产周转率上升记 1。"},
    {"id": "xbrl-fscore", "name": "【F-Score】", "column": "FScore", "direction": 1, "note": "九个分项里有数据的那些相加，越高越好。"},
    {"id": "xbrl-growth", "name": "【资产增长】", "column": "AssetGrowth", "direction": -1, "note": "总资产增长越低越好，排序前取反。"},
    {"id": "xbrl-issues", "name": "【净股票发行】", "column": "NetStockIssues", "direction": -1, "note": "流通股增长越低越好，排序前取反。拆股当年会失真。"},
)


def event_panel(events: pd.DataFrame, calendar: pd.DatetimeIndex, column: str) -> pd.DataFrame:
    """每个交易日只看见已经提交的最近一条。同一天多条时留下报告期较晚的那条。"""
    usable = events.dropna(subset=[column, "filed_date", "ticker"]).copy()
    usable["filed_date"] = pd.to_datetime(usable["filed_date"])
    usable["period_end"] = pd.to_datetime(usable["period_end"])
    usable = usable.sort_values(["filed_date", "period_end"]).drop_duplicates(["ticker", "filed_date"], keep="last")
    columns = {}
    for ticker, group in usable.groupby("ticker", sort=False):
        series = pd.Series(group[column].to_numpy(dtype=float), index=pd.DatetimeIndex(group["filed_date"]))
        series = series[~series.index.duplicated(keep="last")].sort_index()
        merged = calendar.union(series.index).sort_values()
        columns[str(ticker)] = series.reindex(merged).ffill().reindex(calendar)
    return pd.DataFrame(columns, index=calendar)


def select_weights(scores: pd.Series) -> pd.Series:
    """大约保留最高的 40%。分数卡在门槛上的股票全部留下，避免 0/1 因子被名单顺序裁掉。"""
    scores = scores.replace([np.inf, -np.inf], np.nan).dropna().astype("float64")
    weights = pd.Series(0.0, index=scores.index, dtype="float64")
    if len(scores) < MIN_NAMES:
        return weights
    keep = max(1, int(round(len(scores) * WEIGHT_KEEP)))
    cutoff = float(scores.nlargest(keep).min())
    selected = scores[scores >= cutoff]
    centered = selected - float(selected.mean())
    scale = float(centered.std(ddof=0))
    tilted = centered / scale if scale > 0.0 else centered * 0.0
    raw = 1.0 + tilted
    lowest = float(raw.min())
    if lowest <= 0.0:
        raw = raw - lowest + 1.0
    total = float(raw.sum())
    if total <= 0.0:
        return weights
    weights.loc[raw.index] = (raw / total).to_numpy()
    return weights


def target_signal(closes: pd.DataFrame, factor: pd.DataFrame) -> pd.DataFrame:
    signal = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns, dtype="float64")
    aligned = factor.reindex(index=closes.index, columns=closes.columns)
    for day in _month_ends(closes.index):
        eligible = closes.loc[day].dropna().index.intersection(aligned.loc[day].dropna().index)
        signal.loc[day] = 0.0
        if len(eligible):
            weights = select_weights(aligned.loc[day, eligible])
            signal.loc[day, weights.index] = weights.to_numpy()
    return signal.ffill().fillna(0.0)


def run_factor_portfolio(closes: pd.DataFrame, factor: pd.DataFrame, ndx: pd.Series, asset: pd.DataFrame) -> dict:
    signal = target_signal(closes, factor)
    held = signal.shift(1).fillna(0.0)
    held = held.where(closes.notna(), 0.0)
    totals = held.sum(axis=1)
    held = held.div(totals.where(totals > 0.0), axis=0).fillna(0.0)
    net, equity = _share_portfolio(held, asset, FEE)
    benchmark = run_backtest(ndx.reindex(closes.index), pd.Series(1.0, index=closes.index), fee=0.0)
    return {
        "signal": signal,
        "position": held,
        "return": net,
        "equity": equity,
        "benchmark_return": benchmark["return"],
        "benchmark_equity": benchmark["equity"],
    }


def _annual_factors() -> pd.DataFrame:
    factors = pd.read_parquet(FACTOR_PATH)
    annual = factors[factors["form"] == "10-K"].copy()
    annual["filed_date"] = pd.to_datetime(annual["filed_date"])
    annual["period_end"] = pd.to_datetime(annual["period_end"])
    return annual


def _book_to_market(closes: pd.DataFrame) -> pd.DataFrame:
    annual, _quarterly = load_wide(LONG_PATH)
    annual = annual[annual["form"] == "10-K"]
    equity = event_panel(annual, closes.index, "StockholdersEquity").reindex(columns=closes.columns)
    shares = event_panel(annual, closes.index, "Shares").reindex(columns=closes.columns)
    market_cap = closes * shares
    return equity.div(market_cap.where(market_cap > 0.0))


def build_factor_panels(closes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    annual = _annual_factors()
    panels = {}
    for spec in FACTORS:
        if spec["column"] == "BM":
            raw = _book_to_market(closes)
        else:
            raw = event_panel(annual, closes.index, spec["column"])
        panels[spec["column"]] = raw.reindex(index=closes.index, columns=closes.columns)
    return panels


def _symbol_payload(symbol: str, frame: pd.DataFrame, panels: dict[str, pd.DataFrame], books: dict[str, dict]) -> tuple[dict, dict, dict]:
    algorithms = {}
    year_maps: dict[str, dict[str, dict]] = {}
    trade_maps: dict[str, dict[str, tuple[int, int]]] = {}
    level = _equity(_stock_returns(frame["close"]))
    for spec in FACTORS:
        book = books[spec["id"]]
        position = book["position"][symbol].reindex(frame.index).fillna(0.0)
        equity = _share_equity(level, position, FEE)
        window_return = equity.pct_change().fillna(0.0)
        window_return.iloc[0] = 0.0
        bench_return = _stock_returns(frame["close"])
        buy_dates, sell_dates = _trade_dates(frame.index, position)
        trades, by_year = _round_trips(frame.index, position, equity)
        year_maps[spec["id"]] = _symbol_year_rows(frame, position, equity)
        trade_maps[spec["id"]] = by_year
        algorithms[spec["id"]] = {
            "id": spec["id"],
            "name": spec["name"],
            "note": _note(spec),
            "params": {"fee": FEE, "weight_keep": WEIGHT_KEEP, "direction": spec["direction"], "form": "10-K"},
            "indicator": _series(panels[spec["column"]][symbol].reindex(frame.index)),
            "signal": _series(book["signal"][symbol].reindex(frame.index).fillna(0.0)),
            "position": _series(position),
            "buy_dates": buy_dates,
            "sell_dates": sell_dates,
            "equity": _series(equity),
            "benchmark_equity": _series(_equity(bench_return)),
            "stats": {
                "strategy": _describe_returns(window_return.to_numpy(dtype=float), trades),
                "benchmark": _describe_returns(bench_return.to_numpy(dtype=float), (0, 0), hide_win_rate=True),
            },
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "symbol": symbol,
        "price_field": PRICE_FIELD,
        "dates": [timestamp.strftime("%Y-%m-%d") for timestamp in frame.index],
        "open": _series(frame.get("open")),
        "high": _series(frame.get("high")),
        "low": _series(frame.get("low")),
        "close": _series(frame["close"]),
        "return_index": _series(level),
        "volume": _series(frame.get("volume")),
        "algorithms": algorithms,
    }
    return payload, year_maps, trade_maps


def _note(spec: dict) -> str:
    return (
        f"{spec['note']}只用 10-K，filed_date 当天及之后才进入横截面。"
        "月末持有得分较高的大约 40%，卡在分数线上的股票全部保留，权重次日按收盘价成交。"
        "股票池是当前幸存者名单，没有历史成分。"
    )


def _benchmark_payload(books: dict[str, dict], frame: pd.DataFrame) -> dict:
    calendar = frame.index
    close = frame["close"].astype("float64")
    level = _equity(_stock_returns(close))
    algorithms = {}
    for spec in FACTORS:
        book = books[spec["id"]]
        strategy = book["return"].reindex(calendar).fillna(0.0)
        benchmark = book["benchmark_return"].reindex(calendar).fillna(0.0)
        years = {}
        for year in sorted(set(strategy.index.year)):
            mask = strategy.index.year == int(year)
            if bool(mask.any()):
                years[str(int(year))] = _pair_stats(strategy.loc[mask], benchmark.loc[mask])
        since = strategy.index >= pd.Timestamp(OOS_START)
        algorithms[spec["id"]] = {
            "id": spec["id"],
            "name": spec["name"],
            "note": _note(spec),
            "params": {"fee": FEE, "weight_keep": WEIGHT_KEEP, "direction": spec["direction"], "form": "10-K"},
            "equity": _series(book["equity"].reindex(calendar)),
            "benchmark_equity": _series(book["benchmark_equity"].reindex(calendar)),
            "stats": _pair_stats(strategy, benchmark),
            "stats_since_2021": _pair_stats(strategy.loc[since], benchmark.loc[since]),
            "years": years,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "symbol": INDEX_STEM,
        "name": BENCHMARK_NAME,
        "kind": "index",
        "price_field": PRICE_FIELD,
        "dates": [timestamp.strftime("%Y-%m-%d") for timestamp in calendar],
        "open": _series(frame.get("open")),
        "high": _series(frame.get("high")),
        "low": _series(frame.get("low")),
        "close": _series(close),
        "return_index": _series(level),
        "volume": _series(frame.get("volume")),
        "universe_note": UNIVERSE_NOTE,
        "algorithms": algorithms,
    }


def export_dataset() -> Path:
    ndx, frames = load_prices(PRICE_DIR)
    annual_names = set(_annual_factors()["ticker"].astype(str))
    symbols = [symbol for symbol in frames if symbol in annual_names]
    if not symbols:
        raise FileNotFoundError("没有同时具备年报因子和日线的股票")
    frames = {symbol: frames[symbol] for symbol in symbols}
    calendar = ndx.index
    closes = pd.DataFrame({symbol: frames[symbol]["close"] for symbol in symbols}).reindex(calendar)
    returns = _cleaned_panel(closes)
    panels = build_factor_panels(closes)
    books = {
        spec["id"]: run_factor_portfolio(closes, panels[spec["column"]] * spec["direction"], ndx, returns)
        for spec in FACTORS
    }
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    overviews = {spec["id"]: {} for spec in FACTORS}
    trade_totals = {spec["id"]: {} for spec in FACTORS}
    print(f"开始写入 {len(symbols)} 只股票、{len(FACTORS)} 个因子")
    for symbol in symbols:
        payload, year_maps, trade_maps = _symbol_payload(symbol, frames[symbol], panels, books)
        _write_merged_json(out_dir / f"{symbol}.json", payload)
        for spec in FACTORS:
            for year, row in year_maps[spec["id"]].items():
                overviews[spec["id"]].setdefault(year, {})[symbol] = row
            for year, pair in trade_maps[spec["id"]].items():
                bucket = trade_totals[spec["id"]].setdefault(year, [0, 0])
                bucket[0] += pair[0]
                bucket[1] += pair[1]
    algorithms = []
    for spec in FACTORS:
        full_return = books[spec["id"]]["return"]
        bench_return = books[spec["id"]]["benchmark_return"]
        full = _describe_returns(full_return.to_numpy(dtype=float), (0, 0))
        bench = _describe_returns(bench_return.to_numpy(dtype=float), (0, 0))
        paired = _pair_stats(full_return, bench_return)
        since = full_return.index >= pd.Timestamp(OOS_START)
        reported = _pair_stats(full_return.loc[since], bench_return.loc[since])
        algorithms.append(
            {
                "id": spec["id"],
                "name": spec["name"],
                "note": _note(spec),
                "years": {
                    year: _overview_year(
                        symbols,
                        rows,
                        tuple(trade_totals[spec["id"]].get(year, (0, 0))),
                        full_return.loc[full_return.index.year == int(year)],
                        bench_return.loc[bench_return.index.year == int(year)],
                    )
                    for year, rows in sorted(overviews[spec["id"]].items())
                },
            }
        )
        print(
            f"{spec['name']} 累计 {full['cumulative_return'] * 100:+.2f}%，"
            f"基准 {bench['cumulative_return'] * 100:+.2f}%，"
            f"可交易区间资金比 {paired['strategy'].get('wealth_ratio')}，"
            f"2021年以来资金比 {reported['strategy'].get('wealth_ratio')}"
        )
    raw_index = pd.read_csv(PRICE_DIR / f"{INDEX_STEM}.csv")
    raw_index = raw_index.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    raw_index["date"] = pd.to_datetime(raw_index["date"])
    index_frame = raw_index.set_index("date").reindex(calendar)
    index_frame["close"] = ndx.reindex(calendar)
    _write_merged_json(out_dir / f"{INDEX_STEM}.json", _benchmark_payload(books, index_frame))
    _write_merged_json(
        out_dir / "index.json",
        {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "name": DATASET_NAME,
            "source": FACTOR_PATH.relative_to(ROOT).as_posix(),
            "price_field": PRICE_FIELD,
            "symbols": symbols,
            "benchmark": {"id": INDEX_STEM, "name": BENCHMARK_NAME},
            "algorithms": algorithms,
            "universe_note": UNIVERSE_NOTE,
        },
        catalog=True,
    )
    print(f"已写入 {out_dir.relative_to(ROOT).as_posix()}")
    return out_dir


def main() -> None:
    export_dataset()


if __name__ == "__main__":
    main()
