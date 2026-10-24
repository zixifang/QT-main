"""技术类算法的共享导出逻辑。

三个算法（MACD 趋势、12-1 动量、唐奇安突破）只负责给出 signal 和 indicator，
本模块负责读取行情、调用 engine 回测、合并 outdata JSON，并按自然年汇总。

共用约定：
- 信号在当日收盘确定，engine 把仓位后移一天，避免未来函数；
- 只向已有 JSON 追加自己的 algorithm id，不覆盖别人的结果；
- index.json 的 algorithms 数组按 id 追加或替换，不整文件覆盖。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from engine import prepare_close, run_backtest, summarize
from engine.core import TRADING_DAYS

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
OUT_ROOT = ROOT / "outdata"
REPORT_ROOT = ROOT / "reports"

SCHEMA_VERSION = 1
PRICE_FIELD = "close"
DEFAULT_FEE = 0.001

# 数据目录 -> 界面数据集信息。其他目录用目录名作为 dataset_id。
DATASETS = {
    "Equity-Portfolio-Management-master": {
        "dataset_id": "equity-2018",
        "name": "2018 股票组合样本",
    },
    "tech-us-10y": {
        "dataset_id": "tech-us-10y",
        "name": "科技美股近十年日线",
    },
    "manufacturing-us-10y": {
        "dataset_id": "manufacturing-us-10y",
        "name": "美国制造业近十年日线",
    },
    "recent-us-2024-2026": {
        "dataset_id": "recent-us-2024-2026",
        "name": "美股大盘 2024–2026",
    },
}

# 合并长表不是单只股票；纳斯达克100 数据集不在本工作区处理。
SKIP_STEMS = {"tech_us_10y", "manufacturing_us_10y", "recent_us_2024_2026", "nasdaq100"}
SKIP_FOLDERS = {"nasdaq100"}

Compute = Callable[[pd.DataFrame], tuple[pd.Series, pd.Series]]


def ema(values: pd.Series, span: int) -> pd.Series:
    """指数移动平均。前 span-1 个值没有定义，返回 NaN。"""
    return values.ewm(span=span, adjust=False, min_periods=span).mean()


def wilder_atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    """Wilder ATR：真实波幅的指数平滑，alpha = 1/window。"""
    high = pd.to_numeric(frame["high"], errors="coerce").astype("float64")
    low = pd.to_numeric(frame["low"], errors="coerce").astype("float64")
    close = pd.to_numeric(frame["close"], errors="coerce").astype("float64")
    previous = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous).abs(), (low - previous).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def load_frames(data_dir: Path) -> dict[str, pd.DataFrame]:
    """读取目录里的单只股票 CSV，返回按代码索引的行情表。

    合并长表（含 Ticker 列）和没有 Close 列的文件会被跳过。
    """
    found: dict[str, Path] = {}
    for path in sorted(data_dir.glob("*.csv")):
        if path.stem.lower() in SKIP_STEMS:
            continue
        try:
            header = pd.read_csv(path, nrows=0)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            continue
        columns = {str(name).strip().lower() for name in header.columns}
        if "ticker" in columns or "close" not in columns:
            continue
        found[path.stem.upper()] = path

    frames: dict[str, pd.DataFrame] = {}
    for symbol, path in found.items():
        try:
            raw = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            continue
        renamed = {}
        for column in raw.columns:
            key = str(column).strip().lower()
            if key == "adj close":
                key = "adj_close"
            renamed[column] = key
        frame = raw.rename(columns=renamed)
        if "date" not in frame.columns or "close" not in frame.columns:
            continue
        frame = frame.loc[:, [column for column in ("date", "open", "high", "low", "close", "volume") if column in frame.columns]].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for column in ("open", "high", "low", "close", "volume"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["date", "close"]).sort_values("date")
        frame = frame.drop_duplicates(subset=["date"], keep="last")
        if frame.empty:
            continue
        # ATR 和通道需要最高/最低价；缺失时退化为收盘价，保证数组长度一致。
        for column in ("open", "high", "low"):
            if column not in frame.columns:
                frame[column] = frame["close"]
            frame[column] = frame[column].fillna(frame["close"])
        if "volume" not in frame.columns:
            frame["volume"] = 0.0
        frame["volume"] = frame["volume"].fillna(0.0)
        frames[symbol] = frame.reset_index(drop=True)
    return frames


def return_index(close: pd.Series, jump_limit: float = 0.5) -> pd.Series:
    """供界面左侧整体涨跌使用的净值。

    单日涨跌超过 jump_limit 视为拆股或坏点，该日收益记 0；原始 close 不改。
    """
    daily = close.pct_change().fillna(0.0)
    daily = daily.mask(daily.abs() > jump_limit, 0.0)
    return (1.0 + daily).cumprod().rename("return_index")


def export_algorithm(
    algorithm: dict,
    params: dict,
    compute: Compute,
    *,
    fee: float = DEFAULT_FEE,
    data_root: Path = DATA_ROOT,
    out_root: Path = OUT_ROOT,
    report_root: Path = REPORT_ROOT,
    dataset_folders: list[str] | None = None,
) -> dict:
    """把算法结果写入 outdata，并返回数据集级汇总。

    compute(frame) 返回 (signal, indicator)，两者都以 frame 的行顺序对齐。
    """
    if fee < 0:
        raise ValueError("fee 不能为负")
    if not data_root.is_dir():
        raise FileNotFoundError(f"没有数据目录：{data_root}")

    folders = _select_folders(data_root, dataset_folders)
    rows: list[dict] = []
    dataset_summaries: dict[str, dict] = {}

    for folder in folders:
        data_dir = data_root / folder
        frames = load_frames(data_dir)
        if not frames:
            continue
        meta = DATASETS.get(folder, {"dataset_id": folder, "name": folder})
        dataset_id = meta["dataset_id"]
        out_dir = out_root / dataset_id
        out_dir.mkdir(parents=True, exist_ok=True)

        existing_index = _read_json(out_dir / "index.json") or {}
        dataset_id = str(existing_index.get("dataset_id") or dataset_id)
        dataset_name = str(existing_index.get("name") or meta["name"])
        symbol_order = _symbol_order(existing_index, frames)

        yearly: dict[str, dict] = {}
        dataset_rows: list[dict] = []
        written = 0
        for symbol in symbol_order:
            frame = frames.get(symbol)
            if frame is None:
                continue
            payload, symbol_yearly = _symbol_payload(
                symbol, frame, dataset_id, algorithm, params, compute, fee
            )
            if payload is None:
                continue
            path = out_dir / f"{symbol}.json"
            if not _merge_symbol_json(path, payload):
                print(f"跳过 {dataset_id}/{symbol}：已有 JSON 的 dates 与行情不一致")
                continue
            written += 1
            for year, stats in symbol_yearly.items():
                yearly.setdefault(year, {})
                yearly[year][symbol] = stats
            row = _summary_row(dataset_id, symbol, payload, fee)
            dataset_rows.append(row)
            rows.append(row)

        if written == 0:
            print(f"跳过 {dataset_id}：没有写入任何股票")
            continue

        _upsert_index(
            out_dir / "index.json",
            existing_index,
            dataset_id,
            dataset_name,
            data_dir,
            list(frames),
            algorithm,
            yearly,
        )
        dataset_summaries[dataset_id] = {
            "name": dataset_name,
            "symbols": written,
            "strategy": _aggregate(dataset_rows, "strategy", with_beat=True),
            "benchmark": _aggregate(dataset_rows, "benchmark", with_beat=False),
        }
        print(
            f"[{algorithm['id']}] {dataset_id}: {written} 只股票，"
            f"策略平均累计 {_pct(dataset_summaries[dataset_id]['strategy']['cumulative_return'])}，"
            f"平均夏普 {_fmt(dataset_summaries[dataset_id]['strategy']['sharpe'])}"
        )

    if not rows:
        raise FileNotFoundError(f"没有可用的 CSV：{data_root}")

    report_root.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(report_root / f"{algorithm['id']}_summary.csv", rows)
    summary = {
        "algorithm": algorithm,
        "params": params,
        "fee": fee,
        "datasets": dataset_summaries,
    }
    _write_json(report_root / f"{algorithm['id']}_summary.json", summary)
    return summary


def _select_folders(data_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        folders = []
        for name in requested:
            path = data_root / name
            if not path.is_dir():
                raise FileNotFoundError(f"没有数据目录：{path}")
            folders.append(name)
        return folders
    return [
        path.name
        for path in sorted(data_root.iterdir())
        if path.is_dir() and path.name not in SKIP_FOLDERS
    ]


def _symbol_order(existing_index: dict, frames: dict[str, pd.DataFrame]) -> list[str]:
    saved = [str(symbol) for symbol in existing_index.get("symbols") or []]
    order = list(saved)
    order.extend(symbol for symbol in sorted(frames) if symbol not in order)
    return order


def _symbol_payload(
    symbol: str,
    frame: pd.DataFrame,
    dataset_id: str,
    algorithm: dict,
    params: dict,
    compute: Compute,
    fee: float,
) -> tuple[dict | None, dict[str, dict]]:
    close = prepare_close(frame.loc[:, ["date", "close"]])
    raw_signal, raw_indicator = compute(frame)
    dates_index = pd.to_datetime(frame["date"])
    if len(raw_signal) != len(frame) or len(raw_indicator) != len(frame):
        raise ValueError("compute() 返回的 signal 和 indicator 必须与行情行数一致")
    signal = pd.Series(np.asarray(raw_signal, dtype="float64"), index=dates_index)
    indicator = pd.Series(np.asarray(raw_indicator, dtype="float64"), index=dates_index)
    signal = signal.reindex(close.index).fillna(0.0)

    aligned = frame.set_index("date").reindex(close.index)
    strategy = _performance(close, signal, fee)
    benchmark = _performance(close, pd.Series(1.0, index=close.index), 0.0)
    yearly = _yearly_stats(strategy["frame"], benchmark["frame"])

    dates = [timestamp.strftime("%Y-%m-%d") for timestamp in close.index]
    buy_dates, sell_dates = _trade_dates(close.index, strategy["position"])

    algorithm_payload = {
        "id": algorithm["id"],
        "name": algorithm["name"],
        "note": algorithm["note"],
        "params": params,
        "indicator": _series(indicator, close.index),
        "signal": _series(signal, close.index),
        "position": _series(strategy["position"], close.index),
        "buy_dates": buy_dates,
        "sell_dates": sell_dates,
        "equity": _series(strategy["equity"], close.index),
        "benchmark_equity": _series(benchmark["equity"], close.index),
        "stats": {
            "strategy": _stats(strategy),
            "benchmark": _stats(benchmark, hide_win_rate=True),
        },
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "symbol": symbol,
        "price_field": PRICE_FIELD,
        "dates": dates,
        "open": _series(aligned.get("open", close), close.index),
        "high": _series(aligned.get("high", close), close.index),
        "low": _series(aligned.get("low", close), close.index),
        "close": _series(close, close.index),
        "volume": _series(aligned.get("volume"), close.index),
        "return_index": _series(return_index(close), close.index),
        "algorithms": {algorithm["id"]: algorithm_payload},
    }
    return payload, yearly


def _merge_symbol_json(path: Path, payload: dict) -> bool:
    """把 payload 的算法合并进已有 JSON；已有 dates 不一致时不写。"""
    if not path.exists():
        _write_json(path, payload)
        return True
    existing = _read_json(path)
    if not isinstance(existing, dict):
        _write_json(path, payload)
        return True
    if existing.get("dates") and existing["dates"] != payload["dates"]:
        return False
    algorithms = existing.get("algorithms")
    if not isinstance(algorithms, dict):
        algorithms = {}
    algorithms.update(payload["algorithms"])
    existing.update(
        {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": payload["dataset_id"],
            "symbol": payload["symbol"],
            "price_field": PRICE_FIELD,
            "dates": payload["dates"],
            "open": payload["open"],
            "high": payload["high"],
            "low": payload["low"],
            "close": payload["close"],
            "volume": payload["volume"],
            "return_index": payload["return_index"],
            "algorithms": algorithms,
        }
    )
    _write_json(path, existing)
    return True


def _upsert_index(
    path: Path,
    existing: dict,
    dataset_id: str,
    name: str,
    data_dir: Path,
    symbols: list[str],
    algorithm: dict,
    yearly: dict[str, dict],
) -> None:
    index = dict(existing) if isinstance(existing, dict) else {}
    index["schema_version"] = SCHEMA_VERSION
    index["dataset_id"] = dataset_id
    index["name"] = name
    index.setdefault("source", _relative_source(data_dir))
    index["price_field"] = PRICE_FIELD
    saved = [str(symbol) for symbol in index.get("symbols") or []]
    order = list(saved)
    order.extend(symbol for symbol in symbols if symbol not in order)
    index["symbols"] = order
    algorithms = [item for item in index.get("algorithms") or [] if isinstance(item, dict)]
    entry = dict(algorithm)
    entry["years"] = {
        year: _overview_year(order, year_rows)
        for year, year_rows in sorted(yearly.items())
    }
    replaced = False
    for position, item in enumerate(algorithms):
        if item.get("id") == algorithm["id"]:
            algorithms[position] = entry
            replaced = True
            break
    if not replaced:
        algorithms.append(entry)
    index["algorithms"] = algorithms
    _write_json(path, index)


def _relative_source(data_dir: Path) -> str:
    try:
        return data_dir.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return data_dir.as_posix()


def _performance(close: pd.Series, signal: pd.Series, fee: float) -> dict:
    result = run_backtest(close, signal, fee=fee)
    described = _describe(result)
    described["frame"] = result
    return described


def _describe(result: pd.DataFrame, trades: tuple[int, int] | None = None) -> dict:
    base = summarize(result)
    returns = result["return"].to_numpy(dtype="float64")
    periods = max(int(returns.size), 1)
    cumulative = base["cumulative_return"]
    annualized = float((1.0 + cumulative) ** (TRADING_DAYS / periods) - 1.0)
    volatility = float(np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS)) if returns.size > 1 else 0.0
    drawdown = base["max_drawdown"]
    calmar = float(annualized / abs(drawdown)) if drawdown < 0 else 0.0
    if trades is None:
        trades, wins = _round_trips(result)
    else:
        trades, wins = trades
    return {
        "cumulative_return": cumulative,
        "annualized_return": annualized,
        "volatility": volatility,
        "sharpe": base["sharpe"],
        "max_drawdown": drawdown,
        "calmar": calmar,
        "trades": trades,
        "win_rate": (wins / trades) if trades else 0.0,
        "equity": result["equity"],
        "position": result["position"],
    }


def _round_trips(result: pd.DataFrame) -> tuple[int, int]:
    held = result["position"].to_numpy(dtype="float64")
    equity = result["equity"].to_numpy(dtype="float64")
    trades = 0
    wins = 0
    entry: float | None = None
    previous = 0.0
    for index, current in enumerate(held):
        if previous == 0.0 and current != 0.0:
            trades += 1
            entry = float(equity[index - 1]) if index else 1.0
        elif previous != 0.0 and current == 0.0 and entry is not None:
            if equity[index] > entry:
                wins += 1
            entry = None
        previous = current
    if entry is not None and equity[-1] > entry:
        wins += 1
    return trades, wins


def _round_trips_for_year(result: pd.DataFrame, year: int) -> tuple[int, int]:
    held = result["position"].to_numpy(dtype="float64")
    equity = result["equity"].to_numpy(dtype="float64")
    years = result.index.year.to_numpy()
    trades = 0
    wins = 0
    entry: float | None = None
    entry_year: int | None = None
    previous = 0.0
    for index, current in enumerate(held):
        this_year = int(years[index])
        if previous == 0.0 and current != 0.0:
            entry = float(equity[index - 1]) if index else 1.0
            entry_year = this_year
        elif previous != 0.0 and current == 0.0 and entry is not None:
            if entry_year == year:
                trades += 1
                if float(equity[index]) > entry:
                    wins += 1
            entry = None
            entry_year = None
        elif entry_year == year and this_year != year:
            trades += 1
            if float(equity[index - 1]) > entry:
                wins += 1
            entry = None
            entry_year = None
        previous = current
    if entry_year == year and entry is not None:
        trades += 1
        if float(equity[-1]) > entry:
            wins += 1
    return trades, wins


def _yearly_stats(strategy_frame: pd.DataFrame, benchmark_frame: pd.DataFrame) -> dict[str, dict]:
    stored: dict[str, dict] = {}
    years = sorted({int(value) for value in strategy_frame.index.year})
    for year in years:
        strategy_returns = _year_returns(strategy_frame, year)
        benchmark_returns = _year_returns(benchmark_frame, year)
        if strategy_returns is None or benchmark_returns is None:
            continue
        stored[str(year)] = {
            "strategy": _stats_from_returns(
                strategy_returns, _round_trips_for_year(strategy_frame, year)
            ),
            "benchmark": _stats_from_returns(benchmark_returns, (0, 0), hide_win_rate=True),
        }
    return stored


def _year_returns(frame: pd.DataFrame, year: int) -> np.ndarray | None:
    piece = frame.loc[frame.index.year == year, "return"]
    if piece.empty:
        return None
    return piece.to_numpy(dtype="float64")


def _stats_from_returns(
    returns: np.ndarray, trades: tuple[int, int], hide_win_rate: bool = False
) -> dict:
    periods = max(int(returns.size), 1)
    cumulative = float(np.prod(1.0 + returns) - 1.0) if returns.size else 0.0
    annualized = float((1.0 + cumulative) ** (TRADING_DAYS / periods) - 1.0)
    volatility = float(np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS)) if returns.size > 1 else 0.0
    equity = np.cumprod(1.0 + returns) if returns.size else np.array([1.0])
    drawdown = float(np.min(equity / np.maximum.accumulate(equity) - 1.0))
    if returns.size < 2:
        sharpe = 0.0
    else:
        std = float(np.std(returns, ddof=1))
        sharpe = 0.0 if std == 0.0 else float(np.mean(returns) / std * np.sqrt(TRADING_DAYS))
    calmar = float(annualized / abs(drawdown)) if drawdown < 0 else 0.0
    trade_count, wins = trades
    return _stats(
        {
            "cumulative_return": cumulative,
            "annualized_return": annualized,
            "volatility": volatility,
            "sharpe": sharpe,
            "max_drawdown": drawdown,
            "calmar": calmar,
            "trades": trade_count,
            "win_rate": (wins / trade_count) if trade_count else 0.0,
        },
        hide_win_rate=hide_win_rate,
    )


def _overview_year(symbol_order: list[str], year_rows: dict[str, dict]) -> dict:
    order = [symbol for symbol in symbol_order if symbol in year_rows]
    order.extend(symbol for symbol in year_rows if symbol not in order)
    rows = [{"symbol": symbol, **year_rows[symbol]} for symbol in order]
    return {
        "symbols": [
            {
                "symbol": row["symbol"],
                "cumulative_return": row["strategy"]["cumulative_return"],
                "benchmark_cumulative_return": row["benchmark"]["cumulative_return"],
            }
            for row in rows
        ],
        "strategy": _aggregate(rows, "strategy", with_beat=True),
        "benchmark": _aggregate(rows, "benchmark", with_beat=False),
    }


def _aggregate(rows: list[dict], side: str, with_beat: bool) -> dict:
    def numbers(key: str) -> list[float]:
        return [float(row[side][key]) for row in rows if row[side].get(key) is not None]

    returns = numbers("cumulative_return")
    count = len(rows)
    winners = sum(1 for value in returns if value > 0)
    weight = 0.0
    won = 0.0
    for row in rows:
        trades = row[side].get("trades")
        rate = row[side].get("win_rate")
        if trades is None or rate is None or float(trades) <= 0:
            continue
        weight += float(trades)
        won += float(rate) * float(trades)
    payload = {
        "count": count,
        "winners": winners,
        "losers": sum(1 for value in returns if value < 0),
        "profit_ratio": None if not returns else _num(winners / len(returns)),
        "win_rate": None if weight <= 0 else _num(won / weight),
        "cumulative_return": _mean_num(returns),
        "median_return": _median_num(returns),
        "annualized_return": _mean_num(numbers("annualized_return")),
        "sharpe": _mean_num(numbers("sharpe")),
        "max_drawdown": _mean_num(numbers("max_drawdown")),
        "trades": int(sum(numbers("trades"))) if numbers("trades") else 0,
    }
    if with_beat:
        beat = 0
        for row in rows:
            strategy = row["strategy"].get("cumulative_return")
            benchmark = row["benchmark"].get("cumulative_return")
            if strategy is not None and benchmark is not None and float(strategy) > float(benchmark):
                beat += 1
        payload["beat_benchmark"] = beat
    return payload


def _summary_row(dataset_id: str, symbol: str, payload: dict, fee: float) -> dict:
    algo = payload["algorithms"][next(iter(payload["algorithms"]))]
    strategy = algo["stats"]["strategy"]
    benchmark = algo["stats"]["benchmark"]
    position = pd.to_numeric(pd.Series(algo["position"]), errors="coerce")
    return {
        "dataset": dataset_id,
        "symbol": symbol,
        "start": payload["dates"][0] if payload["dates"] else "",
        "end": payload["dates"][-1] if payload["dates"] else "",
        "bars": len(payload["dates"]),
        "fee": fee,
        "cumulative_return": strategy["cumulative_return"],
        "benchmark_return": benchmark["cumulative_return"],
        "annualized_return": strategy["annualized_return"],
        "volatility": strategy["volatility"],
        "sharpe": strategy["sharpe"],
        "max_drawdown": strategy["max_drawdown"],
        "calmar": strategy["calmar"],
        "trades": strategy["trades"],
        "win_rate": strategy["win_rate"],
        "exposure": _num(position.mean()) if not position.empty else 0.0,
        "strategy": strategy,
        "benchmark": benchmark,
    }


def _trade_dates(index: pd.DatetimeIndex, held: pd.Series) -> tuple[list[str], list[str]]:
    previous = held.shift(1).fillna(0.0)
    buys = index[(previous == 0.0) & (held > 0.0)]
    sells = index[(previous > 0.0) & (held == 0.0)]
    return (
        [timestamp.strftime("%Y-%m-%d") for timestamp in buys],
        [timestamp.strftime("%Y-%m-%d") for timestamp in sells],
    )


def _stats(result: dict, hide_win_rate: bool = False) -> dict:
    rate = None if hide_win_rate else result["win_rate"]
    return {
        "cumulative_return": _num(result["cumulative_return"]),
        "annualized_return": _num(result["annualized_return"]),
        "volatility": _num(result["volatility"]),
        "sharpe": _num(result["sharpe"]),
        "max_drawdown": _num(result["max_drawdown"]),
        "calmar": _num(result["calmar"]),
        "trades": int(result["trades"]),
        "win_rate": None if rate is None else _num(rate),
    }


def _series(values, index: pd.DatetimeIndex | None = None) -> list:
    if values is None:
        return [] if index is None else [None] * len(index)
    if isinstance(values, pd.Series):
        series = values.astype("float64")
        if index is not None:
            if series.index.equals(index):
                pass
            elif len(series) == len(index):
                series = pd.Series(series.to_numpy(dtype="float64"), index=index)
            else:
                series = series.reindex(index)
    else:
        array = np.asarray(values, dtype="float64")
        series = pd.Series(array, index=index if index is not None and len(array) == len(index) else None)
    return [_num(value) for value in series.to_numpy()]


def _num(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, 6)


def _mean_num(values: list[float]):
    if not values:
        return None
    return _num(sum(values) / len(values))


def _median_num(values: list[float]):
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return _num(ordered[middle])
    return _num((ordered[middle - 1] + ordered[middle]) / 2)


def _write_summary_csv(path: Path, rows: list[dict]) -> None:
    columns = [
        "dataset", "symbol", "start", "end", "bars", "fee",
        "cumulative_return", "benchmark_return", "annualized_return",
        "volatility", "sharpe", "max_drawdown", "calmar",
        "trades", "win_rate", "exposure",
    ]
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row.get(column, "")) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pct(value) -> str:
    return "—" if value is None else f"{float(value) * 100:.1f}%"


def _fmt(value) -> str:
    return "—" if value is None else f"{float(value):.2f}"


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def single_dataset_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="只处理指定数据目录名，可重复；默认扫描 data 下全部单股票目录",
    )
    parser.add_argument("--fee", type=float, default=DEFAULT_FEE, help="单边手续费率")
    return parser
