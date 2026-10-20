"""用纳斯达克100成分股做指数增强，并把界面所需 JSON 写到 outdata/nasdaq100。

运行：python -m algorithms.index_enhance
界面不调用本模块。本程序会重写该数据集的 JSON。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine import prepare_close, run_backtest, summarize
from engine.core import TRADING_DAYS
from preprocess.prepare import clean_returns

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "nasdaq100"
OUT_DIR = ROOT / "outdata" / "nasdaq100"
CONSTITUENTS_FILE = ROOT / "data_fetch" / "nasdaq100_constituents.txt"

SCHEMA_VERSION = 1
PRICE_FIELD = "close"
FEE = 0.001
WEIGHT_KEEP = 0.40
# 与行情接口相同：未复权序列里单日涨跌超过 50% 视为拆股或坏点，收益记 0。价格本身不改。
JUMP_LIMIT = 0.5
DATASET_ID = "nasdaq100"
DATASET_NAME = "纳斯达克100"
INDEX_STEM = "NDX"
BENCHMARK_NAME = "纳斯达克指数"

MOMENTUM_LONG = 252
MOMENTUM_SKIP = 21
REVERSAL_WINDOW = 21
VOL_WINDOW = 60
LIQUIDITY_WINDOW = 20
CORR_WINDOW = 20
FACTOR_VOL_WINDOW = 20

PORTFOLIOS = (
    {
        "id": "enhance-g1",
        "name": "【指数增强+动量】",
        "note": "月末按12-1动量倾斜配置，只持有得分最高的40%，其余权重为0。权重次日生效。新纳入为买入，调出为卖出。单股净值用该股自己的现金买卖，赚到的差价下次再买入；全局表对比这些股票各自的买入持有。",
        "factor": "momentum",
        "params": {"long": MOMENTUM_LONG, "skip": MOMENTUM_SKIP, "fee": FEE, "weight_keep": WEIGHT_KEEP},
    },
    {
        "id": "enhance-g2",
        "name": "【指数增强+反转】",
        "note": "月末按21日反转倾斜配置，近期跌得多的超配，只持有得分最高的40%。新纳入为买入，调出为卖出。单股净值用该股自己的现金买卖，赚到的差价下次再买入；全局表对比这些股票各自的买入持有。",
        "factor": "reversal",
        "params": {"window": REVERSAL_WINDOW, "fee": FEE, "weight_keep": WEIGHT_KEEP},
    },
    {
        "id": "enhance-g3",
        "name": "【指数增强+低波动】",
        "note": "月末按60日低波动倾斜配置，波动更低的超配，只持有得分最高的40%。新纳入为买入，调出为卖出。单股净值用该股自己的现金买卖，赚到的差价下次再买入；全局表对比这些股票各自的买入持有。",
        "factor": "low_vol",
        "params": {"window": VOL_WINDOW, "fee": FEE, "weight_keep": WEIGHT_KEEP},
    },
    {
        "id": "enhance-g4",
        "name": "【指数增强+流动性】",
        "note": "月末按20日平均成交额倾斜配置，成交更活跃的超配，只持有得分最高的40%。新纳入为买入，调出为卖出。单股净值用该股自己的现金买卖，赚到的差价下次再买入；全局表对比这些股票各自的买入持有。",
        "factor": "liquidity",
        "params": {"window": LIQUIDITY_WINDOW, "fee": FEE, "weight_keep": WEIGHT_KEEP},
    },
    {
        "id": "enhance-g5",
        "name": "【指数增强+动量低波】",
        "note": "月末把12-1动量和60日低波动的横截面排名等权合成后再倾斜，只持有得分最高的40%。新纳入为买入，调出为卖出。单股净值用该股自己的现金买卖，赚到的差价下次再买入；全局表对比这些股票各自的买入持有。",
        "factor": "momentum_low_vol",
        "params": {"long": MOMENTUM_LONG, "skip": MOMENTUM_SKIP, "window": VOL_WINDOW, "fee": FEE, "weight_keep": WEIGHT_KEEP},
    },
    {
        "id": "enhance-g6",
        "name": "【指数增强+等权组合】",
        "note": "月末把四项横截面排名等权合成，各占25%：21日反转、60日低波动、20日平均成交额取反、20日收盘价与成交量相关系数。只持有得分最高的40%。新纳入为买入，调出为卖出。单股净值用该股自己的现金买卖，赚到的差价下次再买入；全局表对比这些股票各自的买入持有。",
        "factor": "equal_blend",
        "params": {
            "reversal": REVERSAL_WINDOW,
            "low_vol": VOL_WINDOW,
            "liquidity": LIQUIDITY_WINDOW,
            "price_volume_corr": CORR_WINDOW,
            "weights": [0.25, 0.25, 0.25, 0.25],
            "fee": FEE,
            "weight_keep": WEIGHT_KEEP,
        },
    },
    {
        "id": "enhance-g7",
        "name": "【指数增强+下影线】",
        "note": "月末按下影线倾斜配置。下影线是（开盘价与收盘价的较小值减最低价）除以开盘价，下影线更长的超配，只持有得分最高的40%。新纳入为买入，调出为卖出。单股净值用该股自己的现金买卖，赚到的差价下次再买入；全局表对比这些股票各自的买入持有。",
        "factor": "lower_shadow",
        "params": {"fee": FEE, "weight_keep": WEIGHT_KEEP},
    },
    {
        "id": "enhance-g8",
        "name": "【指数增强+20日波动】",
        "note": "月末按20日收益标准差倾斜配置，波动更高的超配，只持有得分最高的40%。这一项不把单日涨跌超过50%的日子记成0。新纳入为买入，调出为卖出。单股净值用该股自己的现金买卖，赚到的差价下次再买入；全局表对比这些股票各自的买入持有。",
        "factor": "vol_20",
        "params": {"window": FACTOR_VOL_WINDOW, "fee": FEE, "weight_keep": WEIGHT_KEEP},
    },
    {
        "id": "enhance-g9",
        "name": "【指数增强+全日振幅】",
        "note": "月末按全日振幅倾斜配置。振幅是（最高价减最低价）除以开盘价，振幅更大的超配，只持有得分最高的40%。新纳入为买入，调出为卖出。单股净值用该股自己的现金买卖，赚到的差价下次再买入；全局表对比这些股票各自的买入持有。",
        "factor": "klen",
        "params": {"fee": FEE, "weight_keep": WEIGHT_KEEP},
    },
)


def read_constituents(path: Path = CONSTITUENTS_FILE) -> list[str]:
    """读成分股名单，忽略空行和注释。"""
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip().upper()
        if text:
            names.append(text)
    if not names:
        raise ValueError(f"成分股名单是空的：{path}")
    return names


def load_prices(data_dir: Path = DATA_DIR) -> tuple[pd.Series, dict[str, pd.DataFrame]]:
    """读 NDX 和名单里的成分股。日历用指数交易日，个股只保留两边都有的日期。"""
    index_path = data_dir / f"{INDEX_STEM}.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"没有指数行情：{index_path}")
    ndx = prepare_close(_read_csv(index_path).loc[:, ["date", "close"]])
    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for symbol in read_constituents():
        path = data_dir / f"{symbol}.csv"
        if not path.exists():
            missing.append(symbol)
            continue
        frame = _read_csv(path)
        close = prepare_close(frame.loc[:, ["date", "close"]])
        shared = close.index.intersection(ndx.index)
        if shared.empty:
            missing.append(symbol)
            continue
        aligned = frame.set_index("date").reindex(shared)
        aligned["close"] = close.reindex(shared)
        frames[symbol] = aligned
    if missing:
        print("缺少或无法对齐的成分股：" + ", ".join(missing))
    if not frames:
        raise FileNotFoundError(f"没有可用的成分股 CSV：{data_dir}")
    return ndx, frames


def build_factor_panels(
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    returns: pd.DataFrame,
    opens: pd.DataFrame,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """因子都只用当天及以前的数据。收益已去掉超过 50% 的跳变，20日波动除外。"""
    momentum = _window_return(returns, MOMENTUM_LONG - MOMENTUM_SKIP).shift(MOMENTUM_SKIP)
    reversal = -_window_return(returns, REVERSAL_WINDOW)
    low_vol = -returns.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std()
    liquidity = (closes * volumes).rolling(LIQUIDITY_WINDOW, min_periods=LIQUIDITY_WINDOW).mean()
    price_volume = closes.rolling(CORR_WINDOW, min_periods=CORR_WINDOW).corr(volumes)
    momentum_rank = momentum.rank(axis=1, pct=True)
    low_vol_rank = low_vol.rank(axis=1, pct=True)
    combined = ((momentum_rank + low_vol_rank) / 2.0).where(momentum_rank.notna() & low_vol_rank.notna())
    equal_blend = _equal_ranks(reversal, low_vol, -liquidity, price_volume)
    lower_shadow = _safe_div(np.minimum(opens, closes) - lows, opens)
    amplitude = _safe_div(highs - lows, opens)
    vol_20 = (closes / closes.shift(1) - 1.0).rolling(FACTOR_VOL_WINDOW, min_periods=FACTOR_VOL_WINDOW).std()
    return {
        "momentum": momentum,
        "reversal": reversal,
        "low_vol": low_vol,
        "liquidity": liquidity,
        "momentum_low_vol": combined,
        "equal_blend": equal_blend,
        "lower_shadow": lower_shadow,
        "vol_20": vol_20,
        "klen": amplitude,
    }


def _panel_field(frames: dict[str, pd.DataFrame], field: str, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    columns = {}
    for symbol, frame in frames.items():
        if field in frame.columns:
            columns[symbol] = pd.to_numeric(frame[field], errors="coerce")
        else:
            columns[symbol] = pd.Series(np.nan, index=frame.index)
    return pd.DataFrame(columns).reindex(calendar)


def _safe_div(num: pd.DataFrame, den: pd.DataFrame) -> pd.DataFrame:
    return num / den.where(den != 0)


def _equal_ranks(*panels: pd.DataFrame) -> pd.DataFrame:
    """每项先做横截面百分比排名，再等权。任一项缺失则该股票当天不参与。"""
    ranks = [panel.rank(axis=1, pct=True) for panel in panels]
    ready = ranks[0].notna()
    total = ranks[0]
    for rank in ranks[1:]:
        ready = ready & rank.notna()
        total = total.add(rank)
    return (total / len(ranks)).where(ready)


def run_portfolio(
    closes: pd.DataFrame,
    factor: pd.DataFrame,
    ndx: pd.Series,
    asset: pd.DataFrame,
) -> dict[str, pd.DataFrame | pd.Series]:
    """月末打分，权重次日生效。两次调仓之间股份数量不变，差价留在股票里。"""
    signal = _target_signal(closes, factor)
    held = signal.shift(1).fillna(0.0)
    held = held.where(closes.notna(), 0.0)
    totals = held.sum(axis=1)
    held = held.div(totals.where(totals > 0.0), axis=0).fillna(0.0)
    net, equity = _share_portfolio(held, asset)
    benchmark = run_backtest(ndx.reindex(closes.index), pd.Series(1.0, index=closes.index), fee=0.0)
    return {
        "signal": signal,
        "position": held,
        "return": net,
        "equity": equity,
        "benchmark_return": benchmark["return"],
        "benchmark_equity": benchmark["equity"],
    }


def _share_portfolio(target: pd.DataFrame, asset: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """目标权重变了才按前收盘价换股。没变的日子股份不动，涨出来的钱留在原来的股票里。"""
    weights = target.to_numpy(dtype=float)
    returns = asset.reindex(index=target.index, columns=target.columns).fillna(0.0).to_numpy(dtype=float)
    days, names = weights.shape
    nav = np.empty(days, dtype=float)
    out = np.empty(days, dtype=float)
    cash = 1.0
    shares = np.zeros(names, dtype=float)
    price = np.ones(names, dtype=float)
    previous = np.zeros(names, dtype=float)
    nav[0] = 1.0
    out[0] = 0.0
    for day in range(1, days):
        goal = weights[day]
        if float(np.max(np.abs(goal - previous))) > 1e-12:
            wealth = cash + float(shares @ price)
            held_now = shares * price / wealth if wealth > 0.0 else np.zeros(names, dtype=float)
            wealth *= max(0.0, 1.0 - float(np.abs(goal - held_now).sum()) * FEE)
            shares = np.divide(goal * wealth, price, out=np.zeros(names, dtype=float), where=price > 0.0)
            cash = max(0.0, wealth - float(shares @ price))
            previous = goal.copy()
        price = price * (1.0 + returns[day])
        wealth = cash + float(shares @ price)
        nav[day] = wealth
        out[day] = wealth / nav[day - 1] - 1.0 if nav[day - 1] > 0.0 else 0.0
    return (
        pd.Series(out, index=target.index, name="return"),
        pd.Series(nav, index=target.index, name="equity"),
    )


def export_dataset(data_dir: Path = DATA_DIR, out_dir: Path = OUT_DIR) -> Path:
    ndx, frames = load_prices(data_dir)
    symbols = list(frames)
    calendar = ndx.index
    closes = pd.DataFrame({symbol: frames[symbol]["close"] for symbol in symbols}).reindex(calendar)
    volumes = _panel_field(frames, "volume", calendar)
    opens = _panel_field(frames, "open", calendar)
    highs = _panel_field(frames, "high", calendar)
    lows = _panel_field(frames, "low", calendar)
    returns = _cleaned_panel(closes)
    factors = build_factor_panels(closes, volumes, returns, opens, highs, lows)
    books = {
        spec["id"]: run_portfolio(closes, factors[spec["factor"]], ndx, returns)
        for spec in PORTFOLIOS
    }
    _check_weights(books)

    out_dir.mkdir(parents=True, exist_ok=True)
    overviews = {spec["id"]: {} for spec in PORTFOLIOS}
    trade_totals = {spec["id"]: {} for spec in PORTFOLIOS}
    print(f"开始写入 {len(symbols)} 只成分股")
    for symbol in symbols:
        payload, year_maps, trade_maps = _symbol_payload(symbol, frames[symbol], factors, books)
        _write_json(out_dir / f"{symbol}.json", payload)
        for spec in PORTFOLIOS:
            for year, row in year_maps[spec["id"]].items():
                overviews[spec["id"]].setdefault(year, {})[symbol] = row
            for year, pair in trade_maps[spec["id"]].items():
                bucket = trade_totals[spec["id"]].setdefault(year, [0, 0])
                bucket[0] += pair[0]
                bucket[1] += pair[1]

    algorithms = []
    for spec in PORTFOLIOS:
        entry = {"id": spec["id"], "name": spec["name"], "note": spec["note"]}
        entry["years"] = {
            year: _overview_year(
                symbols,
                rows,
                tuple(trade_totals[spec["id"]].get(year, (0, 0))),
                books[spec["id"]]["return"].loc[books[spec["id"]]["return"].index.year == int(year)],
                books[spec["id"]]["benchmark_return"].loc[books[spec["id"]]["benchmark_return"].index.year == int(year)],
            )
            for year, rows in sorted(overviews[spec["id"]].items())
        }
        algorithms.append(entry)
        full = _describe_returns(books[spec["id"]]["return"].to_numpy(dtype=float), (0, 0))
        bench = _describe_returns(books[spec["id"]]["benchmark_return"].to_numpy(dtype=float), (0, 0))
        print(
            f"{spec['name']} 组合累计 {full['cumulative_return'] * 100:+.2f}%，"
            f"基准 {bench['cumulative_return'] * 100:+.2f}%"
        )

    _write_json(out_dir / f"{INDEX_STEM}.json", _benchmark_payload(data_dir, books))
    index = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "name": DATASET_NAME,
        "source": data_dir.relative_to(ROOT).as_posix(),
        "price_field": PRICE_FIELD,
        "symbols": symbols,
        "benchmark": {"id": INDEX_STEM, "name": BENCHMARK_NAME},
        "algorithms": algorithms,
    }
    _write_json(out_dir / "index.json", index)
    print(f"已写入 {out_dir.relative_to(ROOT).as_posix()}，{len(symbols)} 只，{len(PORTFOLIOS)} 个组合")
    return out_dir


def _benchmark_payload(data_dir: Path, books: dict[str, dict]) -> dict:
    """指数行情，以及每个组合相对纳斯达克指数的净值。界面只读这份 JSON。"""
    calendar = books[PORTFOLIOS[0]["id"]]["return"].index
    raw = _read_csv(data_dir / f"{INDEX_STEM}.csv").set_index("date").sort_index()
    frame = raw.reindex(calendar)
    close = frame["close"].astype("float64")
    level = _equity(_stock_returns(close))
    dates = [timestamp.strftime("%Y-%m-%d") for timestamp in calendar]
    algorithms = {}
    for spec in PORTFOLIOS:
        book = books[spec["id"]]
        strategy = book["return"].reindex(calendar).fillna(0.0)
        benchmark = book["benchmark_return"].reindex(calendar).fillna(0.0)
        years = {}
        for year in sorted(set(strategy.index.year)):
            mask = strategy.index.year == int(year)
            if bool(mask.any()):
                years[str(int(year))] = _pair_stats(strategy.loc[mask], benchmark.loc[mask])
        algorithms[spec["id"]] = {
            "id": spec["id"],
            "name": spec["name"],
            "note": spec["note"],
            "params": {**spec["params"], "keep": WEIGHT_KEEP},
            "equity": _series(book["equity"].reindex(calendar)),
            "benchmark_equity": _series(book["benchmark_equity"].reindex(calendar)),
            "stats": _pair_stats(strategy, benchmark),
            "stats_10y": _pair_stats(*_trailing(strategy, benchmark, 10)),
            "years": years,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "symbol": INDEX_STEM,
        "name": BENCHMARK_NAME,
        "kind": "index",
        "price_field": PRICE_FIELD,
        "dates": dates,
        "open": _series(frame.get("open")),
        "high": _series(frame.get("high")),
        "low": _series(frame.get("low")),
        "close": _series(close),
        "return_index": _series(level),
        "volume": _series(frame.get("volume")),
        "algorithms": algorithms,
    }


def _trailing(strategy: pd.Series, benchmark: pd.Series, years: int) -> tuple[pd.Series, pd.Series]:
    end = strategy.index.max()
    start = end - pd.DateOffset(years=int(years))
    mask = strategy.index >= start
    return strategy.loc[mask], benchmark.loc[mask]


def _pair_stats(strategy: pd.Series, benchmark: pd.Series) -> dict:
    """组合和指数各自的区间指标。超额收益只写在组合一侧。"""
    strategy_values = strategy.to_numpy(dtype=float)
    benchmark_values = benchmark.to_numpy(dtype=float)
    strategy_stats = _describe_returns(strategy_values, (0, 0))
    benchmark_stats = _describe_returns(benchmark_values, (0, 0), hide_win_rate=True)
    strategy_stats["excess_return"] = _num(_compound(strategy_values) - _compound(benchmark_values))
    benchmark_stats["excess_return"] = None
    return {"strategy": strategy_stats, "benchmark": benchmark_stats}


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ("open", "high", "low", "close", "adj_close", "volume"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date")
    frame = frame.drop_duplicates(subset=["date"], keep="last")
    return frame.reset_index(drop=True)


def _month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    stamps = pd.Series(index, index=index)
    return pd.DatetimeIndex(stamps.groupby([index.year, index.month]).tail(1).to_numpy())


def _target_weights(scores: pd.Series) -> pd.Series:
    """只保留得分最高的 40%，组内按 z 分倾斜。落选的权重为 0。"""
    scores = scores.dropna().astype("float64")
    weights = pd.Series(0.0, index=scores.index, dtype="float64")
    if scores.empty:
        return weights
    keep = max(1, int(round(len(scores) * WEIGHT_KEEP)))
    selected = scores.nlargest(keep)
    centered = selected - float(selected.mean())
    scale = float(centered.std(ddof=0))
    zscore = centered / scale if scale > 0.0 else centered * 0.0
    raw = 1.0 + zscore
    lowest = float(raw.min())
    if lowest <= 0.0:
        raw = raw - lowest + 1.0
    total = float(raw.sum())
    if total <= 0.0:
        return weights
    weights.loc[raw.index] = (raw / total).to_numpy()
    return weights


def _target_signal(closes: pd.DataFrame, factor: pd.DataFrame) -> pd.DataFrame:
    signal = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns, dtype="float64")
    for day in _month_ends(closes.index):
        eligible = closes.loc[day].dropna().index.intersection(factor.loc[day].dropna().index)
        signal.loc[day] = 0.0
        if len(eligible):
            signal.loc[day, eligible] = _target_weights(factor.loc[day, eligible]).to_numpy()
    return signal.ffill().fillna(0.0)


def _check_weights(books: dict[str, dict]) -> None:
    for name, book in books.items():
        totals = book["position"].sum(axis=1)
        invested = totals[totals > 0.0]
        if invested.empty:
            raise ValueError(f"{name} 没有产生任何持仓")
        drift = (invested - 1.0).abs().max()
        if float(drift) > 1e-6:
            raise ValueError(f"{name} 持仓权重没有归一化，最大偏差 {drift}")


def _symbol_payload(
    symbol: str,
    frame: pd.DataFrame,
    factors: dict[str, pd.DataFrame],
    books: dict[str, dict],
) -> tuple[dict, dict[str, dict[str, dict]], dict[str, dict[str, tuple[int, int]]]]:
    dates = [timestamp.strftime("%Y-%m-%d") for timestamp in frame.index]
    algorithms = {}
    year_maps: dict[str, dict[str, dict]] = {}
    trade_maps: dict[str, dict[str, tuple[int, int]]] = {}
    level = _equity(_stock_returns(frame["close"]))
    for spec in PORTFOLIOS:
        book = books[spec["id"]]
        position = book["position"][symbol].reindex(frame.index).fillna(0.0)
        equity = _share_equity(level, position)
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
            "note": spec["note"],
            "params": {**spec["params"], "keep": WEIGHT_KEEP},
            "indicator": _series(factors[spec["factor"]][symbol].reindex(frame.index)),
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
        "dates": dates,
        "open": _series(frame.get("open")),
        "high": _series(frame.get("high")),
        "low": _series(frame.get("low")),
        "close": _series(frame["close"]),
        "return_index": _series(level),
        "volume": _series(frame.get("volume")),
        "algorithms": algorithms,
    }
    return payload, year_maps, trade_maps


def _symbol_year_rows(frame: pd.DataFrame, position: pd.Series, equity: pd.Series) -> dict[str, dict]:
    """当年收益直接取单股净值的涨跌，和净值图、综合盈利率用同一条曲线。"""
    sleeve = equity.pct_change().fillna(0.0)
    sleeve.iloc[0] = 0.0
    stock_return = _stock_returns(frame["close"])
    held = position.reindex(frame.index).fillna(0.0) > 1e-8
    rows: dict[str, dict] = {}
    years = sorted({int(value) for value in frame.index.year})
    for year in years:
        mask = frame.index.year == year
        if not bool(mask.any()):
            continue
        days = int(mask.sum())
        strategy = _path_stats(sleeve.loc[mask], days)
        benchmark = _path_stats(stock_return.loc[mask], days)
        rows[str(year)] = {
            "cumulative_return": strategy[0],
            "benchmark_cumulative_return": benchmark[0],
            "annualized_return": strategy[1],
            "sharpe": strategy[2],
            "max_drawdown": strategy[3],
            "benchmark_annualized_return": benchmark[1],
            "benchmark_sharpe": benchmark[2],
            "benchmark_max_drawdown": benchmark[3],
            "held": bool(held.loc[mask].any()),
        }
    return rows


def _overview_year(
    symbols: list[str],
    rows: dict[str, dict],
    trades: tuple[int, int],
    strategy_daily: pd.Series,
    benchmark_daily: pd.Series,
) -> dict:
    """股票数、中位数和柱状图来自各股。策略与指数的盈利率是同一条净值。持有行是各股买入持有。"""
    ordered = [symbol for symbol in symbols if symbol in rows]
    ordered.extend(symbol for symbol in rows if symbol not in ordered)
    trade_count, wins = trades
    strategy_path = _describe_returns(strategy_daily.to_numpy(dtype=float), (int(trade_count), int(wins)))
    benchmark_path = _describe_returns(benchmark_daily.to_numpy(dtype=float), (0, 0), hide_win_rate=True)
    strategy = {
        "trades": int(trade_count),
        "win_rate": _num((wins / trade_count) if trade_count else 0.0),
    }
    bench = {"trades": 0, "win_rate": None}
    counts = _held_counts(rows)
    return {
        "symbols": [
            {
                "symbol": symbol,
                "cumulative_return": _num(rows[symbol]["cumulative_return"]),
                "benchmark_cumulative_return": _num(rows[symbol]["benchmark_cumulative_return"]),
            }
            for symbol in ordered
        ],
        "strategy": _overview_side(
            strategy,
            beat=counts["beat"],
            count=counts["universe"],
            winners=counts["winners"],
            losers=counts["losers"],
            mean_return=strategy_path["cumulative_return"],
            median_return=counts["strategy_median"],
            annualized_return=strategy_path["annualized_return"],
            sharpe=strategy_path["sharpe"],
            drawdown=strategy_path["max_drawdown"],
        ),
        "benchmark": _overview_side(
            bench,
            beat=None,
            count=None,
            mean_return=benchmark_path["cumulative_return"],
            median_return=None,
            annualized_return=benchmark_path["annualized_return"],
            sharpe=benchmark_path["sharpe"],
            drawdown=benchmark_path["max_drawdown"],
        ),
        "holding": _overview_side(
            {"trades": 0, "win_rate": None},
            beat=None,
            count=counts["universe"],
            winners=counts["bench_winners"],
            losers=counts["bench_losers"],
            mean_return=counts["bench_mean"],
            median_return=counts["bench_median"],
            annualized_return=_annualize(counts["bench_mean"] or 0.0, max(int(strategy_daily.size), 1)),
            sharpe=counts["bench_sharpe"],
            drawdown=counts["bench_drawdown"],
        ),
    }


def _held_counts(rows: dict[str, dict]) -> dict:
    """汇总各股当年净值。没持有过的股票收益为 0，也不算跑赢。"""
    empty = _empty_counts()
    if not rows:
        return empty
    strategy_cums: list[float] = []
    strategy_annuals: list[float] = []
    strategy_sharpes: list[float] = []
    strategy_drawdowns: list[float] = []
    bench_cums: list[float] = []
    bench_annuals: list[float] = []
    bench_sharpes: list[float] = []
    bench_drawdowns: list[float] = []
    winners = losers = bench_winners = bench_losers = beat = 0
    for row in rows.values():
        strategy_cum = float(row["cumulative_return"])
        bench_cum = float(row["benchmark_cumulative_return"])
        strategy_cums.append(strategy_cum)
        strategy_annuals.append(float(row["annualized_return"]))
        strategy_sharpes.append(float(row["sharpe"]))
        strategy_drawdowns.append(float(row["max_drawdown"]))
        bench_cums.append(bench_cum)
        bench_annuals.append(float(row["benchmark_annualized_return"]))
        bench_sharpes.append(float(row["benchmark_sharpe"]))
        bench_drawdowns.append(float(row["benchmark_max_drawdown"]))
        winners += int(strategy_cum > 0.0)
        losers += int(strategy_cum < 0.0)
        bench_winners += int(bench_cum > 0.0)
        bench_losers += int(bench_cum < 0.0)
        beat += int(bool(row["held"]) and strategy_cum > bench_cum)
    strategy_mean, strategy_median = _center(strategy_cums)
    bench_mean, bench_median = _center(bench_cums)
    return {
        "beat": beat,
        "winners": winners,
        "losers": losers,
        "bench_winners": bench_winners,
        "bench_losers": bench_losers,
        "strategy_mean": strategy_mean,
        "strategy_median": strategy_median,
        "strategy_annual": _mean(strategy_annuals),
        "strategy_sharpe": _mean(strategy_sharpes),
        "strategy_drawdown": _mean(strategy_drawdowns),
        "bench_mean": bench_mean,
        "bench_median": bench_median,
        "bench_annual": _mean(bench_annuals),
        "bench_sharpe": _mean(bench_sharpes),
        "bench_drawdown": _mean(bench_drawdowns),
        "universe": len(rows),
    }


def _empty_counts() -> dict:
    return {
        "beat": 0,
        "winners": 0,
        "losers": 0,
        "bench_winners": 0,
        "bench_losers": 0,
        "strategy_mean": None,
        "strategy_median": None,
        "strategy_annual": None,
        "strategy_sharpe": None,
        "strategy_drawdown": None,
        "bench_mean": None,
        "bench_median": None,
        "bench_annual": None,
        "bench_sharpe": None,
        "bench_drawdown": None,
        "universe": 0,
    }


def _path_stats(daily: pd.Series, year_days: int) -> tuple[float, float, float, float]:
    """累计、按当年交易日年化、夏普、最大回撤。整年没有波动时夏普和回撤为 0。"""
    values = daily.to_numpy(dtype=float)
    cumulative = _compound(values)
    annualized = _annualize(cumulative, year_days)
    if values.size > 1:
        std = float(np.std(values, ddof=1))
        sharpe = 0.0 if std == 0.0 else float(np.mean(values) / std * np.sqrt(TRADING_DAYS))
    else:
        sharpe = 0.0
    equity = np.cumprod(1.0 + values) if values.size else np.array([1.0])
    drawdown = float(np.min(equity / np.maximum.accumulate(equity) - 1.0))
    return cumulative, annualized, sharpe, drawdown


def _annualize(cumulative: float, periods: int) -> float:
    """按当年交易日数折成年化。样本正好是 252 天时，年化等于这段累计收益。"""
    growth = 1.0 + cumulative
    if growth <= 0.0:
        return -1.0
    return float(growth ** (TRADING_DAYS / max(int(periods), 1)) - 1.0)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _center(values: list[float]) -> tuple[float | None, float | None]:
    """等权平均和中位数。只有一只股票时两者相同，多只股票分开算。"""
    if not values:
        return None, None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[middle]
    else:
        median = (ordered[middle - 1] + ordered[middle]) / 2.0
    return sum(values) / len(values), median


def _cleaned_panel(closes: pd.DataFrame) -> pd.DataFrame:
    columns = {name: clean_returns(closes[name], JUMP_LIMIT) for name in closes.columns}
    returns = pd.DataFrame(columns, index=closes.index)
    return returns.where(closes.notna() & closes.shift(1).notna(), 0.0).fillna(0.0)


def _window_return(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """用清洗后的日收益合成一段区间收益。窗口里有缺失则留空。"""
    valid = returns.notna()
    grown = np.log1p(returns.fillna(0.0)).rolling(window, min_periods=window).sum()
    enough = valid.rolling(window, min_periods=window).sum() >= window
    return np.expm1(grown).where(enough)


def _stock_returns(close: pd.Series) -> pd.Series:
    """买入持有的日收益。首日为 0，使净值从 1 起算。拆股日收益为 0。"""
    returns = clean_returns(close, JUMP_LIMIT).reindex(close.index).fillna(0.0)
    returns.iloc[0] = 0.0
    return returns


def _share_equity(price: pd.Series, position: pd.Series) -> pd.Series:
    """按前一日收盘价买卖。卖出后现金留在这只股票，下次用全部现金再买入。"""
    held_flags = (position.reindex(price.index).fillna(0.0) > 1e-8).to_numpy()
    prices = price.to_numpy(dtype=float)
    values = np.empty(len(prices), dtype=float)
    cash = 1.0
    shares = 0.0
    prev = False
    for i, (level, held) in enumerate(zip(prices, held_flags)):
        if i > 0 and prices[i - 1] > 0.0:
            if held and not prev:
                shares = cash * (1.0 - FEE) / prices[i - 1]
                cash = 0.0
            elif prev and not held:
                cash = shares * prices[i - 1] * (1.0 - FEE)
                shares = 0.0
        values[i] = shares * level if held and shares else cash
        prev = bool(held)
    return pd.Series(values, index=price.index, name="equity")


def _equity(returns: pd.Series) -> pd.Series:
    return pd.Series(np.cumprod(1.0 + returns.to_numpy(dtype=float)), index=returns.index, name="equity")


def _compound(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    return float(np.prod(1.0 + returns) - 1.0)


def _describe_returns(returns: np.ndarray, trades: tuple[int, int], hide_win_rate: bool = False) -> dict:
    values = np.asarray(returns, dtype=float)
    if values.size == 0:
        values = np.zeros(1, dtype=float)
    equity = np.cumprod(1.0 + values)
    frame = pd.DataFrame({"equity": equity, "return": values})
    base = summarize(frame)
    periods = max(int(values.size), 1)
    cumulative = _compound(values)
    annualized = float((1.0 + cumulative) ** (TRADING_DAYS / periods) - 1.0)
    volatility = float(np.std(values, ddof=1) * np.sqrt(TRADING_DAYS)) if values.size > 1 else 0.0
    drawdown = base["max_drawdown"]
    calmar = float(annualized / abs(drawdown)) if drawdown < 0 else 0.0
    trade_count, wins = trades
    rate = None if hide_win_rate else ((wins / trade_count) if trade_count else 0.0)
    return {
        "cumulative_return": _num(cumulative),
        "annualized_return": _num(annualized),
        "volatility": _num(volatility),
        "sharpe": _num(base["sharpe"]),
        "max_drawdown": _num(drawdown),
        "calmar": _num(calmar),
        "trades": int(trade_count),
        "win_rate": None if rate is None else _num(rate),
    }


def _overview_side(
    stats: dict,
    beat: int | None,
    count: int | None = 1,
    winners: int = 0,
    losers: int = 0,
    mean_return: float | None = None,
    median_return: float | None = None,
    annualized_return: float | None = None,
    sharpe: float | None = None,
    drawdown: float | None = None,
) -> dict:
    payload = {
        "count": None if count is None else int(count),
        "winners": int(winners),
        "losers": int(losers),
        "profit_ratio": None if not count else _num(winners / count),
        "win_rate": None if beat is None or not stats["trades"] else stats["win_rate"],
        "cumulative_return": _num(mean_return),
        "median_return": _num(median_return),
        "annualized_return": _num(annualized_return),
        "sharpe": None if sharpe is None else _num(sharpe),
        "max_drawdown": None if drawdown is None else _num(drawdown),
        "trades": stats["trades"],
    }
    if beat is not None:
        payload["beat_benchmark"] = beat
    return payload


def _trade_dates(index: pd.DatetimeIndex, held: pd.Series) -> tuple[list[str], list[str]]:
    previous = held.shift(1).fillna(0.0)
    buys = index[(previous == 0.0) & (held > 0.0)]
    sells = index[(previous > 0.0) & (held == 0.0)]
    return (
        [timestamp.strftime("%Y-%m-%d") for timestamp in buys],
        [timestamp.strftime("%Y-%m-%d") for timestamp in sells],
    )


def _round_trips(
    index: pd.DatetimeIndex,
    held: pd.Series,
    equity: pd.Series,
) -> tuple[tuple[int, int], dict[str, tuple[int, int]]]:
    """卖出后现金多于买入前才算赢，手续费算在里面。记在卖出的那年，年末未卖出的不结案。"""
    full_trades = 0
    full_wins = 0
    by_year: dict[str, list[int]] = {}
    entry_cash = None
    previous = 0.0
    values = equity.reindex(index).to_numpy(dtype=float)
    years = index.year.to_numpy()
    weights = held.reindex(index).fillna(0.0).to_numpy(dtype=float)

    def add(year_key: str, won: bool) -> None:
        nonlocal full_trades, full_wins
        bucket = by_year.setdefault(year_key, [0, 0])
        bucket[0] += 1
        bucket[1] += int(won)
        full_trades += 1
        full_wins += int(won)

    for i, current in enumerate(weights):
        if previous == 0.0 and current > 0.0:
            entry_cash = float(values[0] if i == 0 else values[i - 1])
        elif previous > 0.0 and current == 0.0 and entry_cash is not None:
            add(str(int(years[i])), float(values[i]) > entry_cash)
            entry_cash = None
        previous = float(current)
    return (full_trades, full_wins), {year: (pair[0], pair[1]) for year, pair in by_year.items()}


def _series(values) -> list:
    if values is None:
        return []
    return [_num(value) for value in pd.Series(values).to_numpy()]


def _num(value):
    if value is None or pd.isna(value):
        return None
    return round(float(value), 6)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    export_dataset()


if __name__ == "__main__":
    main()
