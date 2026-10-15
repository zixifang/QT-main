"""用 data 中的行情计算 RSI，并把界面所需 JSON 写到 outdata。

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
DATA_DIR = ROOT / "data" / "Equity-Portfolio-Management-master"
OUT_DIR = ROOT / "outdata" / "equity-2018"

DATASET_ID = "equity-2018"
DATASET_NAME = "2018 股票组合样本"
SCHEMA_VERSION = 1
PRICE_FIELD = "close"
FEE = 0.001
RSI_PERIOD = 14
RSI_OVERSOLD = 30.0
RSI_OVERBOUGHT = 70.0
SYMBOL_ORDER = ["AAPL", "AMZN", "FB", "GOOG", "IBM", "MSFT", "NFLX", "ORCL", "SAP", "TSLA", "USD_JPY"]

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


def load_frames(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    found = {path.stem.upper(): path for path in data_dir.glob("*.csv")}
    ordered = [name for name in SYMBOL_ORDER if name in found]
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
        frames[symbol] = frame.reset_index(drop=True)
    if not frames:
        raise FileNotFoundError(f"没有可读的 CSV：{data_dir}")
    return frames


def export(data_dir: Path = DATA_DIR, out_dir: Path = OUT_DIR) -> Path:
    frames = load_frames(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "name": DATASET_NAME,
        "source": "data/Equity-Portfolio-Management-master",
        "price_field": PRICE_FIELD,
        "symbols": list(frames),
        "algorithms": [ALGORITHM],
    }
    _write_json(out_dir / "index.json", index)
    for symbol, frame in frames.items():
        _write_json(out_dir / f"{symbol}.json", _symbol_payload(symbol, frame))
    return out_dir


def _symbol_payload(symbol: str, frame: pd.DataFrame) -> dict:
    close = prepare_close(frame.loc[:, ["date", PRICE_FIELD]].rename(columns={PRICE_FIELD: "close"}))
    aligned = frame.set_index("date").reindex(close.index)
    signal = rsi_position(close)
    indicator = compute_rsi(close)
    strategy = _performance(close, signal, FEE)
    benchmark = _performance(close, pd.Series(1.0, index=close.index), 0.0)
    dates = [timestamp.strftime("%Y-%m-%d") for timestamp in close.index]
    buy_dates, sell_dates = _trade_dates(close.index, strategy["position"])
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
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
    }


def _performance(close: pd.Series, signal: pd.Series, fee: float) -> dict:
    result = run_backtest(close, signal, fee=fee)
    base = summarize(result)
    returns = result["return"].to_numpy(dtype=float)
    periods = max(int(returns.size), 1)
    cumulative = base["cumulative_return"]
    annualized = float((1.0 + cumulative) ** (TRADING_DAYS / periods) - 1.0)
    volatility = float(np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS)) if returns.size > 1 else 0.0
    drawdown = base["max_drawdown"]
    calmar = float(annualized / abs(drawdown)) if drawdown < 0 else 0.0
    trades, wins = _round_trips(result)
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
    out_dir = export()
    print(f"已写入 {out_dir}")


if __name__ == "__main__":
    main()
