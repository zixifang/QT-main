"""横截面动量轮动：每月末按 6 个月收益排名，取前两名等权持有。

运行：python -m algorithms.momentum_rotation

数据：data/lean-sample/{SPY,IWM,EEM,IBM}.csv（QuantConnect/Lean 官方样例行情，
2003-04 至 2021-03，已按拆股和分红复权）。

输出：outdata/lean-sample/index.json 与四只标的的 JSON，字段遵循 docs/界面数据契约.md。

与 RSI 那份实现的差别只在信号：RSI 是单标的时间序列择时，这里做的是横截面比较，
所以每只股票的 `position` 是它在组合里的权重（0 或 0.5），组合净值写进
index.json 的 `history`，供第五页「历史走势」使用。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine import prepare_close, run_backtest, summarize
from engine.core import TRADING_DAYS

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "lean-sample"
OUT_DIR = ROOT / "outdata" / "lean-sample"

SCHEMA_VERSION = 1
PRICE_FIELD = "close"
FEE = 0.001

DATASET_ID = "lean-sample"
DATASET_NAME = "LEAN 官方样例四资产"

LOOKBACK = 126          # 约 6 个月
TOP_N = 2

ALGORITHM = {
    "id": "mom_rotation",
    "name": "【动量轮动】",
    "note": "每月末按 126 日收益排名取前 2 名等权，收益转负则空仓",
}

COLUMNS = {
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


# ---------------------------------------------------------------- 读数据

def load_frames(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """读目录里所有「一个代码一个文件」的 CSV。

    带 `ticker` 列的合并表不是单只标的（仓库里 data/<数据集>/ 可能同时放一份长表），
    必须跳过——否则它会被当成一只叫「US_AGRI_STOCKS_10Y」的标的，把整个价格序列搅乱。
    与 algorithms/rsi.py 的判断保持一致。
    """
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(data_dir.glob("*.csv")):
        header = {str(name).strip().lower() for name in pd.read_csv(path, nrows=0).columns}
        if "ticker" in header:
            print(f"跳过合并表 {path.name}（含 ticker 列，不是单只标的）")
            continue
        frame = pd.read_csv(path).rename(columns=COLUMNS)
        if "close" not in frame.columns or "date" not in frame.columns:
            continue
        frame["date"] = pd.to_datetime(frame["date"])
        for column in ("open", "high", "low", "close", "adj_close", "volume"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["date", "close"]).sort_values("date")
        frame = frame.drop_duplicates(subset=["date"], keep="last")
        if not frame.empty:
            frames[path.stem.upper()] = frame.set_index("date")
    if not frames:
        raise FileNotFoundError(f"没有可读的 CSV：{data_dir}")
    return frames


def align(frames: dict[str, pd.DataFrame]) -> tuple[pd.DatetimeIndex, dict[str, pd.DataFrame]]:
    """取所有标的共同的交易日，保证组合净值不出现缺口。"""
    common = frames[sorted(frames)[0]].index
    for frame in frames.values():
        common = common.intersection(frame.index)
    common = common.sort_values()
    return common, {symbol: frame.reindex(common) for symbol, frame in frames.items()}


# ---------------------------------------------------------------- 信号

def momentum_weights(close: pd.DataFrame, lookback: int = LOOKBACK, top_n: int = TOP_N) -> pd.DataFrame:
    """返回每个交易日每只标的的目标权重（0 或 1/top_n），在每月的最后一个交易日更新。"""
    months = close.index.to_period("M")
    last_of_month = ~months.duplicated(keep="last")
    momentum = close / close.shift(lookback) - 1.0

    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    target = pd.Series(0.0, index=close.columns)
    for position, day in enumerate(close.index):
        if last_of_month[position]:
            scores = momentum.loc[day].dropna()
            picked = [s for s in scores.sort_values(ascending=False).index if scores[s] > 0][:top_n]
            target = pd.Series(0.0, index=close.columns)
            if picked:
                target[picked] = 1.0 / len(picked)
        weights.loc[day] = target
    return weights


def portfolio_curve(weights: pd.DataFrame, close: pd.DataFrame, fee: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    """组合净值：仓位同样滞后一期，换手率按各标的权重变化绝对值之和计费。"""
    held = weights.shift(1).fillna(0.0)
    asset_return = close.pct_change().fillna(0.0)
    turnover = held.diff().abs().sum(axis=1)
    turnover.iloc[0] = held.iloc[0].abs().sum()
    strategy_return = (held * asset_return).sum(axis=1) - turnover * fee
    equity = (1.0 + strategy_return).cumprod()
    benchmark_return = asset_return.mean(axis=1)
    benchmark_equity = (1.0 + benchmark_return).cumprod()
    position = held.sum(axis=1)
    return equity, benchmark_equity, position


# ---------------------------------------------------------------- 统计

def _num(value):
    if value is None or pd.isna(value):
        return None
    return round(float(value), 6)


def _series(values) -> list:
    if values is None:
        return []
    return [_num(value) for value in pd.Series(values).to_numpy()]


def _stats_block(result: dict, hide_win_rate: bool = False) -> dict:
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


def _round_trips(frame: pd.DataFrame) -> tuple[int, int]:
    held = frame["position"].to_numpy(dtype=float)
    equity = frame["equity"].to_numpy(dtype=float)
    trades = wins = 0
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


def _round_trips_for_year(frame: pd.DataFrame, year: int) -> tuple[int, int]:
    held = frame["position"].to_numpy(dtype=float)
    equity = frame["equity"].to_numpy(dtype=float)
    years = frame.index.year.to_numpy()
    trades = wins = 0
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


def _describe(frame: pd.DataFrame) -> dict:
    base = summarize(frame)
    returns = frame["return"].to_numpy(dtype=float)
    periods = max(int(returns.size), 1)
    cumulative = base["cumulative_return"]
    annualized = float((1.0 + cumulative) ** (TRADING_DAYS / periods) - 1.0)
    volatility = float(np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS)) if returns.size > 1 else 0.0
    drawdown = base["max_drawdown"]
    calmar = float(annualized / abs(drawdown)) if drawdown < 0 else 0.0
    trades, wins = _round_trips(frame)
    return {
        "cumulative_return": cumulative,
        "annualized_return": annualized,
        "volatility": volatility,
        "sharpe": base["sharpe"],
        "max_drawdown": drawdown,
        "calmar": calmar,
        "trades": trades,
        "win_rate": (wins / trades) if trades else 0.0,
    }


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
    return _stats_block({
        "cumulative_return": cumulative,
        "annualized_return": annualized,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "calmar": calmar,
        "trades": trade_count,
        "win_rate": (wins / trade_count) if trade_count else 0.0,
    }, hide_win_rate=hide_win_rate)


def _yearly_stats(strategy: pd.DataFrame, benchmark: pd.DataFrame) -> dict[str, dict]:
    stored: dict[str, dict] = {}
    for year in sorted({int(value) for value in strategy.index.year}):
        piece = strategy.loc[strategy.index.year == year, "return"]
        bench = benchmark.loc[benchmark.index.year == year, "return"]
        if piece.empty or bench.empty:
            continue
        stored[str(year)] = {
            "strategy": _stats_from_returns(piece.to_numpy(dtype=float), _round_trips_for_year(strategy, year)),
            "benchmark": _stats_from_returns(bench.to_numpy(dtype=float), (0, 0), hide_win_rate=True),
        }
    return stored


def _trade_dates(index: pd.DatetimeIndex, held: pd.Series) -> tuple[list[str], list[str]]:
    previous = held.shift(1).fillna(0.0)
    buys = index[(previous == 0.0) & (held > 0.0)]
    sells = index[(previous > 0.0) & (held == 0.0)]
    return (
        [stamp.strftime("%Y-%m-%d") for stamp in buys],
        [stamp.strftime("%Y-%m-%d") for stamp in sells],
    )


def _aggregate(rows: list[dict], side: str, with_beat: bool) -> dict:
    def numbers(key: str) -> list[float]:
        return [float(row[side][key]) for row in rows if row[side].get(key) is not None]

    returns = numbers("cumulative_return")
    winners = sum(1 for value in returns if value > 0)
    weight = won = 0.0
    for row in rows:
        trades = row[side].get("trades")
        rate = row[side].get("win_rate")
        if trades is None or rate is None or float(trades) <= 0:
            continue
        weight += float(trades)
        won += float(rate) * float(trades)
    payload = {
        "count": len(rows),
        "winners": winners,
        "losers": sum(1 for value in returns if value < 0),
        "profit_ratio": None if not returns else _num(winners / len(returns)),
        "win_rate": None if weight <= 0 else _num(won / weight),
        "cumulative_return": _mean(returns),
        "median_return": _median(returns),
        "annualized_return": _mean(numbers("annualized_return")),
        "sharpe": _mean(numbers("sharpe")),
        "max_drawdown": _mean(numbers("max_drawdown")),
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


def _mean(values: list[float]):
    return None if not values else _num(sum(values) / len(values))


def _median(values: list[float]):
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return _num(ordered[middle])
    return _num((ordered[middle - 1] + ordered[middle]) / 2)


# ---------------------------------------------------------------- 写文件

def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def export(
    data_dir: Path | None = None,
    dataset_id: str | None = None,
    dataset_name: str | None = None,
    lookback: int | None = None,
    top_n: int | None = None,
    fee: float | None = None,
    skip: set[str] | None = None,
    universe_note: str | None = None,
) -> Path:
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    dataset_id = dataset_id or DATASET_ID
    dataset_name = dataset_name or DATASET_NAME
    lookback = LOOKBACK if lookback is None else lookback
    top_n = TOP_N if top_n is None else top_n
    fee = FEE if fee is None else fee
    out_dir = ROOT / "outdata" / dataset_id
    algorithm = {
        "id": ALGORITHM["id"],
        "name": ALGORITHM["name"],
        "note": f"每月末按 {lookback} 日收益排名取前 {top_n} 名等权，收益转负则空仓",
    }

    raw = load_frames(data_dir)
    if skip:
        raw = {symbol: frame for symbol, frame in raw.items() if symbol not in skip}
    dates, frames = align(raw)
    close = pd.DataFrame({symbol: frame[PRICE_FIELD] for symbol, frame in frames.items()}, index=dates).dropna()
    dates = close.index
    frames = {symbol: frame.reindex(dates) for symbol, frame in frames.items()}

    weights = momentum_weights(close, lookback, top_n)
    equity, benchmark_equity, _ = portfolio_curve(weights, close, fee)

    out_dir.mkdir(parents=True, exist_ok=True)
    by_year: dict[str, dict[str, dict]] = {}
    for symbol in close.columns:
        symbol_close = prepare_close(frames[symbol].reset_index().loc[:, ["date", PRICE_FIELD]])
        strategy = run_backtest(symbol_close, weights[symbol].reindex(symbol_close.index).fillna(0.0), fee=fee)
        benchmark = run_backtest(symbol_close, pd.Series(1.0, index=symbol_close.index), fee=0.0)
        described = _describe(strategy)
        yearly = _yearly_stats(strategy, benchmark)
        for year, stats in yearly.items():
            by_year.setdefault(year, {})[symbol] = stats
        buy_dates, sell_dates = _trade_dates(symbol_close.index, strategy["position"])
        aligned = frames[symbol].reindex(symbol_close.index)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "symbol": symbol,
            "price_field": PRICE_FIELD,
            "dates": [stamp.strftime("%Y-%m-%d") for stamp in symbol_close.index],
            "open": _series(aligned.get("open")),
            "high": _series(aligned.get("high")),
            "low": _series(aligned.get("low")),
            "close": _series(symbol_close),
            "volume": _series(aligned.get("volume")),
            "return_index": _series(symbol_close / symbol_close.iloc[0]),
            "algorithms": {
                algorithm["id"]: {
                    "id": algorithm["id"],
                    "name": algorithm["name"],
                    "note": algorithm["note"],
                    "params": {
                        "lookback": lookback,
                        "top_n": top_n,
                        "fee": fee,
                        "rebalance": "month_end",
                    },
                    "indicator": _series(symbol_close / symbol_close.shift(lookback) - 1.0),
                    "signal": _series(weights[symbol].reindex(symbol_close.index).fillna(0.0)),
                    "position": _series(strategy["position"]),
                    "buy_dates": buy_dates,
                    "sell_dates": sell_dates,
                    "equity": _series(strategy["equity"]),
                    "benchmark_equity": _series(benchmark["equity"]),
                    "stats": {
                        "strategy": _stats_block(described),
                        "benchmark": _stats_block(_describe(benchmark), hide_win_rate=True),
                    },
                }
            },
        }
        _write(out_dir / f"{symbol}.json", payload)
        print(f"已写入 {symbol}.json：{len(payload['dates'])} 根日线，"
              f"策略累计 {described['cumulative_return']:.2%}，{described['trades']} 笔")

    years = {}
    portfolio_return = equity.pct_change().fillna(0.0)
    benchmark_return = benchmark_equity.pct_change().fillna(0.0)
    for year in sorted(by_year):
        rows = [{"symbol": symbol, **by_year[year][symbol]} for symbol in close.columns if symbol in by_year[year]]
        per_symbol = _aggregate(rows, "strategy", with_beat=True)
        trades = int(per_symbol["trades"])
        wins = int(round((per_symbol["win_rate"] or 0.0) * trades))
        slice_return = portfolio_return[portfolio_return.index.year == int(year)]
        block = _stats_from_returns(slice_return.to_numpy(dtype=float), (trades, wins))
        # 三条路径指标要用组合净值算，才和第五页那条曲线对得上；其余来自各股
        block.update({
            "count": per_symbol["count"],
            "winners": per_symbol["winners"],
            "losers": per_symbol["losers"],
            "profit_ratio": per_symbol["profit_ratio"],
            "median_return": per_symbol["median_return"],
            "beat_benchmark": per_symbol["beat_benchmark"],
        })
        benchmark_slice = benchmark_return[benchmark_return.index.year == int(year)]
        benchmark_block = _stats_from_returns(benchmark_slice.to_numpy(dtype=float), (0, 0), hide_win_rate=True)
        benchmark_block.update({
            "count": None,
            "profit_ratio": None,
            "median_return": None,
            "win_rate": None,
            "trades": 0,
        })
        years[year] = {
            "symbols": [
                {
                    "symbol": row["symbol"],
                    "cumulative_return": row["strategy"]["cumulative_return"],
                    "benchmark_cumulative_return": row["benchmark"]["cumulative_return"],
                }
                for row in rows
            ],
            "strategy": block,
            "benchmark": benchmark_block,
        }

    index = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "name": dataset_name,
        "source": data_dir.relative_to(ROOT).as_posix(),
        "price_field": PRICE_FIELD,
        "symbols": list(close.columns),
        "universe_note": universe_note or "四只标的均为美股 ETF/个股，2003-04 起同日历，已复权",
        "algorithms": [
            {
                **algorithm,
                "history": {
                    "dates": [stamp.strftime("%Y-%m-%d") for stamp in equity.index],
                    "strategy": _series(equity),
                    "benchmark": _series(benchmark_equity),
                },
                "years": years,
            }
        ],
    }
    _write(out_dir / "index.json", index)
    print(f"已写入 index.json：{len(years)} 个年份，组合累计 {equity.iloc[-1] - 1:.2%}")
    return out_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="横截面动量轮动：读 data/<数据集>/ 的 CSV，写 outdata/<数据集>/")
    parser.add_argument("--data", default=None, help="行情目录，默认 data/lean-sample")
    parser.add_argument("--dataset", default=None, help="数据集 id，也是 outdata 下的目录名")
    parser.add_argument("--name", default=None, help="数据集显示名")
    parser.add_argument("--lookback", type=int, default=None, help="动量回看天数，默认 126")
    parser.add_argument("--top-n", type=int, default=None, help="持仓数量，默认 2")
    parser.add_argument("--fee", type=float, default=None, help="单边手续费率，默认 0.001")
    parser.add_argument("--skip", default="", help="不参与选股的文件名，逗号分隔，例如指数 NDX")
    parser.add_argument("--note", default=None, help="universe_note，写在 index.json 里")
    args = parser.parse_args()

    skip = {token.strip().upper() for token in args.skip.split(",") if token.strip()}
    export(
        data_dir=(ROOT / args.data) if args.data else None,
        dataset_id=args.dataset,
        dataset_name=args.name,
        lookback=args.lookback,
        top_n=args.top_n,
        fee=args.fee,
        skip=skip or None,
        universe_note=args.note,
    )


if __name__ == "__main__":
    main()
