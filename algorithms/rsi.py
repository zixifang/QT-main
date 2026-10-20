"""用 data 中每个行情目录计算 RSI，并把界面所需 JSON 写到 outdata。

运行：python -m algorithms.rsi
界面不调用本模块。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine import prepare_close, run_backtest, summarize
from engine.core import TRADING_DAYS

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
OUT_ROOT = ROOT / "outdata"

SCHEMA_VERSION = 1
PRICE_FIELD = "close"
FEE = 0.001
RSI_PERIOD = 14
RSI_OVERSOLD = 30.0
RSI_OVERBOUGHT = 70.0
EQUITY_ORDER = ["AAPL", "AMZN", "FB", "GOOG", "IBM", "MSFT", "NFLX", "ORCL", "SAP", "TSLA", "USD_JPY"]
TECH_ORDER = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "AMD", "INTC", "AVGO", "QCOM", "MU", "TSM", "ASML",
    "CRM", "ADBE", "ORCL", "NOW", "SNOW", "PLTR",
    "NFLX", "UBER", "SHOP", "XYZ", "PYPL",
    "CSCO", "IBM", "TXN", "AMAT", "LRCX", "KLAC", "ARM", "SMCI",
    "PANW", "CRWD", "SNPS", "CDNS", "ADI", "MRVL", "ANET",
]
# 合并文件不是单只股票，不能按一只标的做 RSI。
SKIP_STEMS = {"tech_us_10y", "manufacturing_us_10y", "nasdaq100"}
SKIP_FOLDERS = {"nasdaq100"}
DATASETS = {
    "Equity-Portfolio-Management-master": {
        "dataset_id": "equity-2018",
        "name": "2018 股票组合样本",
        "symbol_order": EQUITY_ORDER,
    },
    "tech-us-10y": {
        "dataset_id": "tech-us-10y",
        "name": "科技美股近十年日线",
        "symbol_order": TECH_ORDER,
    },
    "manufacturing-us-10y": {
        "dataset_id": "manufacturing-us-10y",
        "name": "美国制造业近十年日线",
        "symbol_order": [
            "CAT", "DE", "PCAR", "URI", "TEX",
            "ETN", "GEV", "VRT", "PWR", "HUBB", "NVT", "GNRC",
            "ROK", "EMR", "ITW", "PH", "AME", "IR", "FAST", "DOV",
            "GE", "RTX", "LMT", "NOC", "GD", "BA", "TDG", "HWM",
            "GM", "F",
            "TT", "CARR", "JCI", "LII",
            "NUE", "STLD",
            "APH", "TEL",
            "MMM", "HON", "CMI", "WM", "CSX", "UNP", "FDX", "UPS",
        ],
    },
}

ALGORITHM = {
    "id": "rsi",
    "name": "RSI 均值回归",
    "note": "RSI(14) 低于 30 开多，高于 70 平仓",
}


def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder RSI。前 period 根 K 线没有值。"""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return pd.Series(rsi.to_numpy(dtype=float), index=close.index, name="rsi")


def rsi_position(close: pd.Series) -> pd.Series:
    """RSI 跌破超卖线开多，升破超买线平仓。信号在当日收盘确定。"""
    values = compute_rsi(close)
    holding = 0
    position: list[int] = []
    for value in values:
        if np.isnan(value):
            position.append(0)
            continue
        if holding == 0 and value < RSI_OVERSOLD:
            holding = 1
        elif holding == 1 and value > RSI_OVERBOUGHT:
            holding = 0
        position.append(holding)
    return pd.Series(position, index=close.index, dtype=float, name="signal")


def load_frames(data_dir: Path, symbol_order: list[str] | None = None) -> dict[str, pd.DataFrame]:
    found: dict[str, Path] = {}
    for path in data_dir.glob("*.csv"):
        if path.stem.lower() in SKIP_STEMS:
            continue
        columns = {str(name).strip().lower() for name in pd.read_csv(path, nrows=0).columns}
        if "ticker" in columns or "close" not in columns:
            continue
        found[path.stem.upper()] = path
    order = symbol_order or []
    ordered = [name for name in order if name in found]
    ordered.extend(sorted(name for name in found if name not in ordered))
    frames: dict[str, pd.DataFrame] = {}
    for symbol in ordered:
        frame = pd.read_csv(found[symbol])
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
        if frame.empty:
            continue
        frames[symbol] = frame.reset_index(drop=True)
    return frames


def export_dataset(data_dir: Path, dataset_id: str, dataset_name: str, symbol_order: list[str] | None = None) -> Path | None:
    frames = load_frames(data_dir, symbol_order)
    if not frames:
        return None
    out_dir = OUT_ROOT / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)
    by_symbol: dict[str, dict[str, dict]] = {}
    for symbol, frame in frames.items():
        payload, yearly = _symbol_payload(symbol, frame, dataset_id)
        _write_json(out_dir / f"{symbol}.json", payload)
        for year, stats in yearly.items():
            by_symbol.setdefault(year, {})
            by_symbol[year][symbol] = stats
    algorithm = dict(ALGORITHM)
    algorithm["years"] = {
        year: _overview_year(frames, year_rows)
        for year, year_rows in sorted(by_symbol.items())
    }
    index = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "name": dataset_name,
        "source": data_dir.relative_to(ROOT).as_posix(),
        "price_field": PRICE_FIELD,
        "symbols": list(frames),
        "algorithms": [algorithm],
    }
    _write_json(out_dir / "index.json", index)
    return out_dir


def export_all(data_root: Path = DATA_ROOT) -> list[Path]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"没有数据目录：{data_root}")
    written: list[Path] = []
    for folder in sorted(path for path in data_root.iterdir() if path.is_dir()):
        if folder.name in SKIP_FOLDERS:
            print(f"跳过 {folder.relative_to(ROOT).as_posix()}：由 python -m algorithms.index_enhance 计算")
            continue
        meta = DATASETS.get(folder.name, {})
        out_dir = export_dataset(
            folder,
            meta.get("dataset_id", folder.name),
            meta.get("name", folder.name),
            meta.get("symbol_order"),
        )
        if out_dir is None:
            print(f"跳过 {folder.relative_to(ROOT).as_posix()}：没有单只股票的 CSV")
            continue
        print(f"已写入 {out_dir.relative_to(ROOT).as_posix()}，{len(list(out_dir.glob('*.json'))) - 1} 只")
        written.append(out_dir)
    if not written:
        raise FileNotFoundError(f"没有可读的 CSV：{data_root}")
    return written


def _symbol_payload(symbol: str, frame: pd.DataFrame, dataset_id: str) -> tuple[dict, dict[str, dict]]:
    close = prepare_close(frame.loc[:, ["date", PRICE_FIELD]].rename(columns={PRICE_FIELD: "close"}))
    aligned = frame.set_index("date").reindex(close.index)
    signal = rsi_position(close)
    indicator = compute_rsi(close)
    strategy = _performance(close, signal, FEE)
    benchmark = _performance(close, pd.Series(1.0, index=close.index), 0.0)
    yearly = _yearly_stats(strategy["frame"], benchmark["frame"])
    dates = [timestamp.strftime("%Y-%m-%d") for timestamp in close.index]
    buy_dates, sell_dates = _trade_dates(close.index, strategy["position"])
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "symbol": symbol,
        "price_field": PRICE_FIELD,
        "dates": dates,
        "open": _series(aligned.get("open")),
        "high": _series(aligned.get("high")),
        "low": _series(aligned.get("low")),
        "close": _series(close),
        "volume": _series(aligned.get("volume")),
        "algorithms": {
            "rsi": {
                "id": "rsi",
                "name": ALGORITHM["name"],
                "note": ALGORITHM["note"],
                "params": {
                    "period": RSI_PERIOD,
                    "oversold": RSI_OVERSOLD,
                    "overbought": RSI_OVERBOUGHT,
                    "fee": FEE,
                },
                "indicator": _series(indicator),
                "signal": _series(signal),
                "position": _series(strategy["position"]),
                "buy_dates": buy_dates,
                "sell_dates": sell_dates,
                "equity": _series(strategy["equity"]),
                "benchmark_equity": _series(benchmark["equity"]),
                "stats": {
                    "strategy": _stats(strategy),
                    "benchmark": _stats(benchmark, hide_win_rate=True),
                },
            }
        },
    }, yearly


def _performance(close: pd.Series, signal: pd.Series, fee: float) -> dict:
    result = run_backtest(close, signal, fee=fee)
    described = _describe(result)
    described["frame"] = result
    return described


def _describe(result: pd.DataFrame, trades: tuple[int, int] | None = None) -> dict:
    base = summarize(result)
    returns = result["return"].to_numpy(dtype=float)
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
    held = result["position"].to_numpy(dtype=float)
    equity = result["equity"].to_numpy(dtype=float)
    trades = 0
    wins = 0
    entry = None
    previous = 0.0
    for index, current in enumerate(held):
        if previous == 0.0 and current != 0.0:
            trades += 1
            entry = equity[index - 1] if index else 1.0
        elif previous != 0.0 and current == 0.0 and entry is not None:
            if equity[index] > entry:
                wins += 1
            entry = None
        previous = current
    if entry is not None and equity[-1] > entry:
        wins += 1
    return trades, wins


def _round_trips_for_year(result: pd.DataFrame, year: int) -> tuple[int, int]:
    """只统计这一年开仓的交易。跨年仍持有的，用该年最后一根 K 线盯市。"""
    held = result["position"].to_numpy(dtype=float)
    equity = result["equity"].to_numpy(dtype=float)
    years = result.index.year.to_numpy()
    trades = 0
    wins = 0
    entry = None
    entry_year = None
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
    """按自然年切分已经算好的日收益，写入各年的 stats。"""
    stored: dict[str, dict] = {}
    years = sorted({int(value) for value in strategy_frame.index.year})
    for year in years:
        strategy_returns = _year_returns(strategy_frame, year)
        benchmark_returns = _year_returns(benchmark_frame, year)
        if strategy_returns is None or benchmark_returns is None:
            continue
        stored[str(year)] = {
            "strategy": _stats_from_returns(strategy_returns, _round_trips_for_year(strategy_frame, year)),
            "benchmark": _stats_from_returns(benchmark_returns, (0, 0), hide_win_rate=True),
        }
    return stored


def _year_returns(frame: pd.DataFrame, year: int) -> np.ndarray | None:
    piece = frame.loc[frame.index.year == year, "return"]
    if piece.empty:
        return None
    return piece.to_numpy(dtype=float)


def _stats_from_returns(returns: np.ndarray, trades: tuple[int, int], hide_win_rate: bool = False) -> dict:
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
    described = {
        "cumulative_return": cumulative,
        "annualized_return": annualized,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "calmar": calmar,
        "trades": trade_count,
        "win_rate": (wins / trade_count) if trade_count else 0.0,
    }
    return _stats(described, hide_win_rate=hide_win_rate)


def _overview_year(symbol_order, year_rows: dict[str, dict]) -> dict:
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
        "strategy": _aggregate_side(rows, "strategy", with_beat=True),
        "benchmark": _aggregate_side(rows, "benchmark", with_beat=False),
    }


def _aggregate_side(rows: list[dict], side: str, with_beat: bool) -> dict:
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
    export_all()


if __name__ == "__main__":
    main()
