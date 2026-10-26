"""随机森林涨跌预测：读 data 写 outdata，合并进已有股票 JSON，不覆盖其他算法。

运行：python -m algorithms.random_forest
依赖：scikit-learn。
每只股票单独训练，9 个因子预测次日涨跌，时间顺序 70/15/15 切分，样本外才交易；equity-2018 样本太少，跳过。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from engine import prepare_close, run_backtest, summarize
from engine.core import TRADING_DAYS

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
OUT_ROOT = ROOT / "outdata"

SCHEMA_VERSION = 1
PRICE_FIELD = "close"
FEE = 0.001
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
THRESHOLDS = (0.50, 0.55, 0.60)
N_ESTIMATORS = 300
MAX_DEPTH = 6
MIN_SAMPLES_LEAF = 20
RANDOM_STATE = 42

SKIP_STEMS = {"tech_us_10y", "manufacturing_us_10y", "healthcare_us_10y", "nasdaq100"}
SKIP_FOLDERS = {"nasdaq100", "Equity-Portfolio-Management-master"}
TECH_ORDER = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "AMD", "INTC", "AVGO", "QCOM", "MU", "TSM", "ASML",
    "CRM", "ADBE", "ORCL", "NOW", "SNOW", "PLTR",
    "NFLX", "UBER", "SHOP", "XYZ", "PYPL",
    "CSCO", "IBM", "TXN", "AMAT", "LRCX", "KLAC", "ARM", "SMCI",
    "PANW", "CRWD", "SNPS", "CDNS", "ADI", "MRVL", "ANET",
]
DATASETS = {
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
    "healthcare-us-10y": {
        "dataset_id": "healthcare-us-10y",
        "name": "美国医疗健康近十年日线",
        "symbol_order": [
            "LLY", "JNJ", "ABBV", "MRK", "AZN", "NVO",
            "AMGN", "GILD", "VRTX", "REGN",
            "ABT", "MDT", "SYK", "BSX", "ISRG", "EW",
            "UNH", "ELV", "CI", "HUM", "CVS",
            "TMO", "DHR", "A", "IDXX", "ZTS",
        ],
    },
}

ALGORITHM = {
    "id": "random_forest",
    "name": "随机森林预测",
    "note": "用价量因子预测次日涨跌，时间顺序70/15/15划分，样本外才交易",
}

FEATURE_NAMES = [
    "ret_1", "ret_5", "ret_10", "ret_20",
    "rsi_14", "ma20_gap", "ma60_gap", "vol_20", "volume_ratio",
]


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return pd.Series(rsi.to_numpy(dtype=float), index=close.index, name="rsi")


def build_features(close: pd.Series, volume: pd.Series | None) -> pd.DataFrame:
    daily = close.pct_change()
    feats = pd.DataFrame(index=close.index)
    feats["ret_1"] = daily
    feats["ret_5"] = close.pct_change(5)
    feats["ret_10"] = close.pct_change(10)
    feats["ret_20"] = close.pct_change(20)
    feats["rsi_14"] = compute_rsi(close)
    feats["ma20_gap"] = close / close.rolling(20).mean() - 1.0
    feats["ma60_gap"] = close / close.rolling(60).mean() - 1.0
    feats["vol_20"] = daily.rolling(20).std()
    if volume is not None and float(volume.fillna(0).sum()) > 0:
        feats["volume_ratio"] = volume / volume.rolling(20).mean()
    else:
        feats["volume_ratio"] = 1.0
    feats["label"] = (close.shift(-1) > close).astype(float)
    return feats


def fit_predict(close: pd.Series, volume: pd.Series | None) -> dict:
    """训练一只股票的模型，返回全长 signal、indicator(上涨概率) 和切分信息。"""
    feats = build_features(close, volume).dropna()
    n = len(feats)
    if n < 300:
        raise ValueError(f"可用样本只有 {n} 行，太少，不训练")
    i_train = int(n * TRAIN_RATIO)
    i_valid = int(n * (TRAIN_RATIO + VALID_RATIO))
    x = feats[FEATURE_NAMES].to_numpy(dtype=float)
    y = feats["label"].to_numpy(dtype=int)
    # 训练段最后一行、验证段最后一行的标签都用到了下一段的价格，剔除。
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x[: i_train - 1], y[: i_train - 1])
    # indicator 只写样本外区间的预测概率；训练区间留空，避免把拟合值当成预测。
    proba = pd.Series(np.nan, index=close.index, dtype=float)
    proba.loc[feats.index[i_train:]] = model.predict_proba(x[i_train:])[:, 1]
    oos_proba = proba.loc[feats.index[i_train:]]

    valid_x = feats.index[i_train:i_valid]
    valid_rows = [i for i in range(i_train, i_valid - 1)]
    best_threshold = THRESHOLDS[0]
    best_accuracy = -1.0
    valid_pred_proba = model.predict_proba(x[valid_rows])[:, 1]
    valid_true = y[valid_rows]
    for threshold in THRESHOLDS:
        accuracy = float(np.mean((valid_pred_proba >= threshold).astype(int) == valid_true))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    test_rows = [i for i in range(i_valid, n - 1)]
    test_pred = (model.predict_proba(x[test_rows])[:, 1] >= best_threshold).astype(int)
    test_accuracy = float(np.mean(test_pred == y[test_rows])) if test_rows else 0.0

    signal = pd.Series(0.0, index=close.index, name="signal")
    signal.loc[oos_proba.index] = (oos_proba >= best_threshold).astype(float)
    importance = {
        name: round(float(value), 4)
        for name, value in sorted(
            zip(FEATURE_NAMES, model.feature_importances_),
            key=lambda pair: pair[1],
            reverse=True,
        )
    }
    return {
        "signal": signal,
        "indicator": proba,
        "oos_start": feats.index[i_train],
        "split": {
            "train_end": feats.index[i_train - 1].strftime("%Y-%m-%d"),
            "valid_end": feats.index[i_valid - 1].strftime("%Y-%m-%d"),
            "train_ratio": TRAIN_RATIO,
            "valid_ratio": VALID_RATIO,
            "test_ratio": round(1.0 - TRAIN_RATIO - VALID_RATIO, 2),
            "threshold": best_threshold,
            "valid_accuracy": round(best_accuracy, 4),
            "test_accuracy": round(test_accuracy, 4),
        },
        "feature_importance": importance,
    }


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
    done = 0
    for symbol, frame in frames.items():
        try:
            entry, yearly = _symbol_entry(symbol, frame)
        except ValueError as error:
            print(f"跳过 {dataset_id}/{symbol}：{error}")
            continue
        _merge_symbol_json(out_dir / f"{symbol}.json", symbol, frame, dataset_id, entry)
        for year, stats in yearly.items():
            by_symbol.setdefault(year, {})
            by_symbol[year][symbol] = stats
        done += 1
    if not done:
        return None
    algorithm = dict(ALGORITHM)
    algorithm["years"] = {
        year: _overview_year(list(frames), year_rows)
        for year, year_rows in sorted(by_symbol.items())
    }
    _merge_index_json(out_dir / "index.json", data_dir, dataset_id, dataset_name, list(frames), algorithm)
    return out_dir


def export_all(data_root: Path = DATA_ROOT) -> list[Path]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"没有数据目录：{data_root}")
    written: list[Path] = []
    for folder in sorted(path for path in data_root.iterdir() if path.is_dir()):
        if folder.name == "nasdaq100":
            print("跳过 data/nasdaq100：由 python -m algorithms.index_enhance 计算")
            continue
        if folder.name == "Equity-Portfolio-Management-master":
            print("跳过 data/Equity-Portfolio-Management-master：只有约一年样本，不够学习类算法按时间切分训练/验证/测试")
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
        print(f"已写入 {out_dir.relative_to(ROOT).as_posix()}")
        written.append(out_dir)
    if not written:
        raise FileNotFoundError(f"没有可读的 CSV：{data_root}")
    return written


def _symbol_entry(symbol: str, frame: pd.DataFrame) -> tuple[dict, dict[str, dict]]:
    close = prepare_close(frame.loc[:, ["date", PRICE_FIELD]].rename(columns={PRICE_FIELD: "close"}))
    aligned = frame.set_index("date").reindex(close.index)
    volume = aligned.get("volume")
    fitted = fit_predict(close, volume)
    signal = fitted["signal"]
    strategy_full = _performance(close, signal, FEE)
    benchmark_full = _performance(close, pd.Series(1.0, index=close.index), 0.0)
    start = fitted["oos_start"]
    strategy_stats = _window_stats(strategy_full["frame"], start)
    benchmark_stats = _window_stats(benchmark_full["frame"], start, hide_win_rate=True)
    yearly = _yearly_stats(strategy_full["frame"], benchmark_full["frame"])
    buy_dates, sell_dates = _trade_dates(close.index, strategy_full["position"])
    entry = {
        "id": ALGORITHM["id"],
        "name": ALGORITHM["name"],
        "note": ALGORITHM["note"],
        "params": {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "fee": FEE,
        },
        "split": fitted["split"],
        "feature_importance": fitted["feature_importance"],
        "indicator": _series(fitted["indicator"]),
        "signal": _series(signal),
        "position": _series(strategy_full["position"]),
        "buy_dates": buy_dates,
        "sell_dates": sell_dates,
        "equity": _series(strategy_full["equity"]),
        "benchmark_equity": _series(benchmark_full["equity"]),
        "stats": {
            "strategy": strategy_stats,
            "benchmark": benchmark_stats,
        },
    }
    return entry, yearly


def _window_stats(result: pd.DataFrame, start: pd.Timestamp, hide_win_rate: bool = False) -> dict:
    piece = result.loc[result.index >= start]
    returns = piece["return"].to_numpy(dtype=float)
    equity = np.cumprod(1.0 + returns) if returns.size else np.array([1.0])
    sub = pd.DataFrame(
        {"position": piece["position"].to_numpy(dtype=float), "equity": equity},
        index=piece.index,
    )
    trades, wins = _round_trips(sub)
    described = _describe_from_returns(returns, trades, wins)
    return _stats(described, hide_win_rate=hide_win_rate)


def _describe_from_returns(returns: np.ndarray, trades: int, wins: int) -> dict:
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
    return {
        "cumulative_return": cumulative,
        "annualized_return": annualized,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "calmar": calmar,
        "trades": trades,
        "win_rate": (wins / trades) if trades else 0.0,
    }


def _base_payload(symbol: str, frame: pd.DataFrame, dataset_id: str) -> dict:
    close = prepare_close(frame.loc[:, ["date", PRICE_FIELD]].rename(columns={PRICE_FIELD: "close"}))
    aligned = frame.set_index("date").reindex(close.index)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "symbol": symbol,
        "price_field": PRICE_FIELD,
        "dates": [t.strftime("%Y-%m-%d") for t in close.index],
        "open": _series(aligned.get("open")),
        "high": _series(aligned.get("high")),
        "low": _series(aligned.get("low")),
        "close": _series(close),
        "volume": _series(aligned.get("volume")),
        "algorithms": {},
    }


def _merge_symbol_json(path: Path, symbol: str, frame: pd.DataFrame, dataset_id: str, entry: dict) -> None:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = _base_payload(symbol, frame, dataset_id)
    payload.setdefault("algorithms", {})[ALGORITHM["id"]] = entry
    _write_json(path, payload)


def _merge_index_json(path: Path, data_dir: Path, dataset_id: str, dataset_name: str, symbols: list[str], algorithm: dict) -> None:
    if path.exists():
        index = json.loads(path.read_text(encoding="utf-8"))
    else:
        index = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "name": dataset_name,
            "source": data_dir.relative_to(ROOT).as_posix(),
            "price_field": PRICE_FIELD,
            "symbols": symbols,
            "algorithms": [],
        }
    entries = [a for a in index.get("algorithms", []) if a.get("id") != ALGORITHM["id"]]
    entries.append(algorithm)
    index["algorithms"] = entries
    _write_json(path, index)


def _performance(close: pd.Series, signal: pd.Series, fee: float) -> dict:
    result = run_backtest(close, signal, fee=fee)
    described = _describe(result)
    described["frame"] = result
    return described


def _describe(result: pd.DataFrame) -> dict:
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


def _round_trips_for_year(result: pd.DataFrame, year: int) -> tuple[int, int]:
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
    trade_count, wins = trades
    described = _describe_from_returns(returns, trade_count, wins)
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
