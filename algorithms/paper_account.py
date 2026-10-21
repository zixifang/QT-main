"""把已有算法的持仓换成一个共用的 10 万美元整股账户。

运行：python -m algorithms.paper_account
读取 outdata 里已经写好的收盘价和 position，不重新训练。
结果写到 outdata/<dataset_id>/paper/<algorithm_id>.json，不覆盖研究净值。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outdata"
CAPITAL = 100_000.0
FEE_PER_SHARE = 0.005
TRADING_DAYS = 252


def simulate(
    dates: list[str],
    closes: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
    benchmark_close: np.ndarray | None = None,
) -> dict:
    """按前一日收盘价先卖后买。权重加总超过 1 时缩放到 1。"""
    symbols = [name for name in closes if name in weights]
    count = len(dates)
    if count == 0 or not symbols:
        raise ValueError("没有可用的日期或股票")
    price = np.column_stack([_finite(closes[name], count) for name in symbols])
    held_weight = np.column_stack([_finite(weights[name], count) for name in symbols])
    held_weight[held_weight < 0] = 0.0
    shares = np.zeros(len(symbols), dtype=np.int64)
    cost = np.zeros(len(symbols), dtype=float)
    cash = CAPITAL
    nav = np.zeros(count, dtype=float)
    cash_line = np.zeros(count, dtype=float)
    share_book = np.zeros((count, len(symbols)), dtype=np.int64)
    value_book = np.zeros((count, len(symbols)), dtype=float)
    buy_at = {name: [] for name in symbols}
    sell_at = {name: [] for name in symbols}
    year_trades: dict[str, list[int]] = {}
    last_price = np.full(len(symbols), np.nan)

    nav[0] = CAPITAL
    cash_line[0] = CAPITAL
    last_price = np.where(np.isfinite(price[0]), price[0], last_price)

    for index in range(1, count):
        fill = last_price.copy()
        wealth = cash + _marked(shares, fill)
        target = _target_shares(shares, held_weight[index], fill, wealth)
        sold, bought, wins = _rebalance(shares, cost, fill, target)
        cash += sold
        spend, bought_qty = _scale_buys(bought, fill, cash)
        cash -= spend
        _apply_buys(shares, cost, fill, bought_qty)
        mark = np.where(np.isfinite(price[index]), price[index], fill)
        values = _marked_each(shares, mark)
        nav[index] = cash + float(values.sum())
        cash_line[index] = cash
        share_book[index] = shares.copy()
        value_book[index] = values
        _note_trades(dates[index], symbols, share_book[index - 1], shares, bought_qty, wins, buy_at, sell_at, year_trades)
        last_price = np.where(np.isfinite(price[index]), price[index], last_price)

    bench = _benchmark_nav(price, benchmark_close)
    return {
        "dates": dates,
        "nav": nav,
        "cash": cash_line,
        "benchmark_nav": bench,
        "symbols": symbols,
        "shares": share_book,
        "values": value_book,
        "buy_dates": buy_at,
        "sell_dates": sell_at,
        "year_trades": {year: (pair[0], pair[1]) for year, pair in year_trades.items()},
    }


def build_report(result: dict, benchmark_name: str) -> dict:
    nav = result["nav"]
    bench = result["benchmark_nav"]
    returns = _returns(nav)
    bench_returns = _returns(bench)
    trades = _sum_trades(result["year_trades"])
    years = _years(result, returns, bench_returns)
    return {
        "capital": CAPITAL,
        "fee_per_share": FEE_PER_SHARE,
        "currency": "USD",
        "benchmark_name": benchmark_name,
        "dates": result["dates"],
        "nav": [_money(value) for value in nav],
        "cash": [_money(value) for value in result["cash"]],
        "benchmark_nav": [_money(value) for value in bench],
        "symbols": {
            name: {
                "shares": [int(value) for value in result["shares"][:, index]],
                "value": [_money(value) for value in result["values"][:, index]],
                "buy_dates": result["buy_dates"][name],
                "sell_dates": result["sell_dates"][name],
            }
            for index, name in enumerate(result["symbols"])
        },
        "stats": {
            "strategy": _stats(returns, trades),
            "benchmark": _stats(bench_returns, (0, 0), hide_win_rate=True),
        },
        "years": years,
    }


def _finite(values: np.ndarray, count: int) -> np.ndarray:
    out = np.full(count, np.nan, dtype=float)
    size = min(count, len(values))
    out[:size] = values[:size]
    return out


def _marked(shares: np.ndarray, price: np.ndarray) -> float:
    good = np.isfinite(price) & (shares > 0)
    if not np.any(good):
        return 0.0
    return float(np.sum(shares[good] * price[good]))


def _marked_each(shares: np.ndarray, price: np.ndarray) -> np.ndarray:
    out = np.zeros(len(shares), dtype=float)
    good = np.isfinite(price) & (shares > 0)
    out[good] = shares[good] * price[good]
    return out


def _target_shares(current: np.ndarray, weights: np.ndarray, fill: np.ndarray, wealth: float) -> np.ndarray:
    usable = np.nan_to_num(weights, nan=0.0)
    usable[usable < 0] = 0.0
    total = float(usable.sum())
    if total > 1.0:
        usable = usable / total
    target = current.copy()
    tradable = np.isfinite(fill) & (fill > 0)
    dollars = wealth * usable
    target[tradable] = np.floor(dollars[tradable] / fill[tradable]).astype(np.int64)
    target[target < 0] = 0
    return target


def _rebalance(shares: np.ndarray, cost: np.ndarray, fill: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, int]:
    """先卖。返回卖出所得、想买的股数、卖出里赚钱的笔数。"""
    cash = 0.0
    wins = 0
    buy = np.zeros(len(shares), dtype=np.int64)
    for index in range(len(shares)):
        price = fill[index]
        if not np.isfinite(price) or price <= 0:
            continue
        delta = int(target[index]) - int(shares[index])
        if delta < 0:
            qty = -delta
            if price <= FEE_PER_SHARE or qty <= 0:
                continue
            if cost[index] > 0 and price - FEE_PER_SHARE > cost[index]:
                wins += 1
            cash += qty * price - qty * FEE_PER_SHARE
            shares[index] -= qty
            if shares[index] <= 0:
                shares[index] = 0
                cost[index] = 0.0
        elif delta > 0:
            buy[index] = delta
    return cash, buy, wins


def _scale_buys(wanted: np.ndarray, fill: np.ndarray, cash: float) -> tuple[float, np.ndarray]:
    tradable = np.isfinite(fill) & (fill > 0) & (wanted > 0)
    if not np.any(tradable) or cash <= 0:
        return 0.0, np.zeros(len(wanted), dtype=np.int64)
    full_cost = float(np.sum(wanted[tradable] * (fill[tradable] + FEE_PER_SHARE)))
    if full_cost <= cash + 1e-8:
        qty = wanted.copy()
        return full_cost, qty
    low = 0.0
    high = 1.0
    best = np.zeros(len(wanted), dtype=np.int64)
    for _ in range(48):
        mid = (low + high) / 2.0
        qty = np.zeros(len(wanted), dtype=np.int64)
        qty[tradable] = np.floor(wanted[tradable] * mid).astype(np.int64)
        spend = float(np.sum(qty[tradable] * (fill[tradable] + FEE_PER_SHARE)))
        if spend <= cash + 1e-8:
            best = qty
            low = mid
        else:
            high = mid
    spend = float(np.sum(best[tradable] * (fill[tradable] + FEE_PER_SHARE))) if np.any(tradable) else 0.0
    return spend, best


def _apply_buys(shares: np.ndarray, cost: np.ndarray, fill: np.ndarray, qty: np.ndarray) -> None:
    for index, bought in enumerate(qty):
        bought = int(bought)
        if bought <= 0:
            continue
        price = fill[index]
        previous = int(shares[index])
        spent = bought * (price + FEE_PER_SHARE)
        total_shares = previous + bought
        if total_shares <= 0:
            continue
        cost[index] = (cost[index] * previous + spent) / total_shares
        shares[index] = total_shares


def _note_trades(day, symbols, previous, current, bought, wins, buy_at, sell_at, year_trades) -> None:
    year = day[:4]
    bucket = year_trades.setdefault(year, [0, 0])
    sold = False
    for index, name in enumerate(symbols):
        if int(bought[index]) > 0:
            buy_at[name].append(day)
            bucket[0] += 1
        if int(current[index]) < int(previous[index]):
            sell_at[name].append(day)
            bucket[0] += 1
            sold = True
    if sold:
        bucket[1] += int(wins)


def _benchmark_nav(price: np.ndarray, benchmark_close: np.ndarray | None) -> np.ndarray:
    count = price.shape[0]
    nav = np.full(count, CAPITAL, dtype=float)
    if benchmark_close is not None and len(benchmark_close) == count and np.isfinite(benchmark_close[0]) and benchmark_close[0] > 0:
        return _buy_and_hold(benchmark_close, np.array([CAPITAL]))
    first = price[0]
    tradable = np.isfinite(first) & (first > 0)
    if not np.any(tradable):
        return nav
    budgets = np.zeros(price.shape[1], dtype=float)
    budgets[tradable] = CAPITAL / int(np.sum(tradable))
    return _buy_and_hold_many(price, budgets)


def _buy_and_hold(close: np.ndarray, budgets: np.ndarray) -> np.ndarray:
    """单条价格。budgets 只有一个数。"""
    price = close
    nav = np.full(len(price), CAPITAL, dtype=float)
    first = price[0]
    shares = int(np.floor(float(budgets[0]) / (first + FEE_PER_SHARE))) if np.isfinite(first) and first > 0 else 0
    cash = CAPITAL - shares * (first + FEE_PER_SHARE) if shares else CAPITAL
    last = first
    for index in range(len(price)):
        if np.isfinite(price[index]) and price[index] > 0:
            last = price[index]
        nav[index] = cash + shares * last
    return nav


def _buy_and_hold_many(price: np.ndarray, budgets: np.ndarray) -> np.ndarray:
    count, width = price.shape
    shares = np.zeros(width, dtype=np.int64)
    first = price[0]
    for index in range(width):
        if not np.isfinite(first[index]) or first[index] <= 0 or budgets[index] <= 0:
            continue
        shares[index] = int(np.floor(budgets[index] / (first[index] + FEE_PER_SHARE)))
    cash = CAPITAL - float(np.sum(shares * (np.nan_to_num(first, nan=0.0) + FEE_PER_SHARE)))
    if cash < 0:
        cash = 0.0
    nav = np.zeros(count, dtype=float)
    last = first.copy()
    for index in range(count):
        today = price[index]
        good = np.isfinite(today) & (today > 0)
        last[good] = today[good]
        nav[index] = cash + _marked(shares, last)
    return nav


def _returns(nav: np.ndarray) -> np.ndarray:
    out = np.zeros(len(nav), dtype=float)
    for index in range(1, len(nav)):
        previous = nav[index - 1]
        if previous > 0 and np.isfinite(nav[index]):
            out[index] = nav[index] / previous - 1.0
    return out


def _stats(returns: np.ndarray, trades: tuple[int, int], hide_win_rate: bool = False) -> dict:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        values = np.zeros(1, dtype=float)
    cumulative = float(np.prod(1.0 + values) - 1.0)
    periods = max(int(values.size), 1)
    growth = 1.0 + cumulative
    annualized = -1.0 if growth <= 0.0 else float(growth ** (TRADING_DAYS / periods) - 1.0)
    volatility = float(np.std(values, ddof=1) * np.sqrt(TRADING_DAYS)) if values.size > 1 else 0.0
    curve = np.cumprod(1.0 + values)
    drawdown = float(np.min(curve / np.maximum.accumulate(curve) - 1.0))
    if values.size < 2:
        sharpe = 0.0
    else:
        std = float(np.std(values, ddof=1))
        sharpe = 0.0 if std == 0.0 else float(np.mean(values) / std * np.sqrt(TRADING_DAYS))
    calmar = float(annualized / abs(drawdown)) if drawdown < 0 else 0.0
    trade_count, win_count = trades
    rate = None if hide_win_rate else ((win_count / trade_count) if trade_count else 0.0)
    return {
        "cumulative_return": _round(cumulative),
        "annualized_return": _round(annualized),
        "volatility": _round(volatility),
        "sharpe": _round(sharpe),
        "max_drawdown": _round(drawdown),
        "calmar": _round(calmar),
        "trades": int(trade_count),
        "win_rate": None if rate is None else _round(rate),
    }


def _years(result: dict, returns: np.ndarray, bench_returns: np.ndarray) -> dict:
    dates = result["dates"]
    stamps = pd.to_datetime(dates)
    stored = {}
    for year in sorted(set(int(value) for value in stamps.year)):
        mask = np.asarray(stamps.year == year)
        if not bool(mask.any()):
            continue
        key = str(year)
        trade_pair = result["year_trades"].get(key, (0, 0))
        strategy = _stats(returns[mask], trade_pair)
        benchmark = _stats(bench_returns[mask], (0, 0), hide_win_rate=True)
        rows = _symbol_year_returns(result, mask)
        returns_only = [row["cumulative_return"] for row in rows if row["cumulative_return"] is not None]
        winners = sum(1 for value in returns_only if value > 0)
        count = len(returns_only)
        beat = sum(1 for value in returns_only if benchmark["cumulative_return"] is not None and value > benchmark["cumulative_return"])
        strategy.update(
            {
                "count": count,
                "winners": winners,
                "losers": count - winners,
                "profit_ratio": None if not count else _round(winners / count),
                "median_return": _median(returns_only),
                "beat_benchmark": beat,
            }
        )
        benchmark.update({"count": None, "winners": 0, "losers": 0, "profit_ratio": None, "median_return": None, "trades": 0, "win_rate": None})
        stored[key] = {"symbols": rows, "strategy": strategy, "benchmark": benchmark}
    return stored


def _symbol_year_returns(result: dict, mask: np.ndarray) -> list[dict]:
    rows = []
    values = result["values"]
    chosen = np.flatnonzero(mask)
    if len(chosen) == 0:
        return rows
    for index, name in enumerate(result["symbols"]):
        series = values[:, index]
        previous = 0.0
        compounded = 1.0
        held = False
        start = chosen[0]
        if start > 0:
            previous = float(series[start - 1])
        for cursor in chosen:
            current = float(series[cursor])
            if previous > 1.0 and current > 1.0:
                compounded *= current / previous
                held = True
            elif previous > 1.0 and current <= 1.0:
                held = True
            previous = current if current > 1.0 else 0.0
        if not held:
            continue
        rows.append({"symbol": name, "cumulative_return": _round(compounded - 1.0)})
    return rows


def _sum_trades(year_trades: dict[str, tuple[int, int]]) -> tuple[int, int]:
    trades = 0
    wins = 0
    for count, win_count in year_trades.values():
        trades += int(count)
        wins += int(win_count)
    return trades, wins


def _median(values: list[float]):
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return _round(ordered[middle])
    return _round((ordered[middle - 1] + ordered[middle]) / 2.0)


def _money(value: float) -> float:
    return round(float(value), 2)


def _round(value: float) -> float:
    return round(float(value), 6)


def _align(dates: list[str], values: list, calendar: list[str]) -> np.ndarray:
    out = np.full(len(calendar), np.nan, dtype=float)
    index = {day: cursor for cursor, day in enumerate(calendar)}
    for day, value in zip(dates, values):
        cursor = index.get(day)
        if cursor is None or value is None:
            continue
        out[cursor] = float(value)
    return out


def _calendar(frames: dict[str, tuple[list[str], list]]) -> list[str]:
    found = set()
    for dates, _values in frames.values():
        found.update(dates)
    return sorted(found)


def export_dataset(folder: Path) -> list[Path]:
    index_path = folder / "index.json"
    if not index_path.exists():
        return []
    catalog = json.loads(index_path.read_text(encoding="utf-8"))
    algorithms = [item.get("id") for item in catalog.get("algorithms") or [] if item.get("id")]
    symbols = [str(item) for item in catalog.get("symbols") or []]
    if not algorithms or not symbols:
        return []
    benchmark = catalog.get("benchmark") if isinstance(catalog.get("benchmark"), dict) else {}
    benchmark_id = str(benchmark.get("id") or "")
    benchmark_name = str(benchmark.get("name") or "买入持有")
    closes: dict[str, tuple[list[str], list]] = {}
    books: dict[str, dict[str, tuple[list[str], list]]] = {algo: {} for algo in algorithms}
    for symbol in symbols:
        path = folder / f"{symbol}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        dates = payload.get("dates") or []
        close = payload.get("close") or []
        if len(close) != len(dates) or not dates:
            continue
        closes[symbol] = (dates, close)
        for algo in algorithms:
            block = (payload.get("algorithms") or {}).get(algo) or {}
            position = block.get("position")
            if isinstance(position, list) and len(position) == len(dates):
                books[algo][symbol] = (dates, position)
        del payload
    benchmark_frame = None
    if benchmark_id:
        bench_path = folder / f"{benchmark_id}.json"
        if bench_path.exists():
            payload = json.loads(bench_path.read_text(encoding="utf-8"))
            dates = payload.get("dates") or []
            close = payload.get("close") or []
            if len(close) == len(dates) and dates:
                benchmark_frame = (dates, close)
    written = []
    out_dir = folder / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = [symbol for symbol in symbols if symbol in closes]
    for algo in algorithms:
        frames = books[algo]
        names = [symbol for symbol in ordered if symbol in frames]
        if len(names) < 1:
            print(f"跳过 {folder.name}/{algo}：没有持仓", flush=True)
            continue
        calendar = _calendar({symbol: frames[symbol] for symbol in names})
        close_map = {symbol: _align(*closes[symbol], calendar) for symbol in names}
        weight_map = {symbol: _align(*frames[symbol], calendar) for symbol in names}
        bench = _align(*benchmark_frame, calendar) if benchmark_frame else None
        result = simulate(calendar, close_map, weight_map, bench)
        name = benchmark_name if benchmark_frame is not None else "买入持有"
        report = build_report(result, name)
        path = out_dir / f"{algo}.json"
        path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        written.append(path)
        print(f"已写入 {folder.name}/paper/{algo}.json，期末净值 {report['nav'][-1]}", flush=True)
    return written


def export_all(root: Path = OUT_ROOT) -> list[Path]:
    written = []
    if not root.is_dir():
        raise FileNotFoundError(root)
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        if not (folder / "index.json").exists():
            continue
        print(f"处理 {folder.name}", flush=True)
        written.extend(export_dataset(folder))
    return written


def main() -> None:
    export_all()


if __name__ == "__main__":
    main()
