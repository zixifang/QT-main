"""按股数成交的回测。和 backtest.portfolio.simulate 分开，不走优化器。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from services.errors import DataGapError


@dataclass
class BacktestResult:
    nav_series: pd.Series
    trades: pd.DataFrame
    positions: pd.DataFrame
    metrics: dict


def run_backtest(prices: pd.DataFrame, signals: pd.DataFrame, config: dict | None = None) -> BacktestResult:
    """调仓日先卖后买。未成交的卖出留到下一有价格的交易日。现金不计利息。"""
    cfg = _config(config or {})
    close = prices.sort_index().astype(float)
    if close.empty:
        raise DataGapError("没有可用于回测的行情")
    signal = signals.reindex(index=close.index, columns=close.columns).astype(float)
    opened = cfg["open"]
    if opened is None:
        opened = close
    else:
        opened = opened.reindex(index=close.index, columns=close.columns).astype(float)
    names = list(close.columns)
    trades_on = _trade_days(close.index, cfg["rebalance"], cfg["fill"])
    cash = float(cfg["initial_cash"])
    shares = {name: 0.0 for name in names}
    last_price = {name: np.nan for name in names}
    target = {name: 0.0 for name in names}
    warnings: list[str] = list(cfg["warnings"])
    scheme = cfg["weight"]
    if scheme == "cap":
        warnings.append("没有市值数据，市值加权已改为等权")
        scheme = "equal"
    direction = int(cfg["direction"])
    if direction not in (1, -1):
        raise DataGapError("方向只能是 1 或 -1")
    trades: list[dict] = []
    nav_rows = []
    position_rows = []
    pending = False
    for stamp in close.index:
        check = cfg["cancel"]
        if check is not None and check():
            from services.errors import Cancelled

            raise Cancelled("已取消")
        row_close = close.loc[stamp]
        trade_at_open = stamp in trades_on and cfg["fill"] == "next_open"
        if not trade_at_open:
            _remember(last_price, names, row_close)
        trade_today = stamp in trades_on or pending
        if trade_today:
            if stamp in trades_on:
                signal_day = trades_on[stamp]
                quote_for_size = opened.loc[stamp] if cfg["fill"] == "next_open" else close.loc[stamp]
                target = _target_shares(
                    signal.loc[signal_day],
                    _nav(cash, shares, last_price),
                    direction,
                    int(cfg["top_n"]),
                    scheme,
                    float(cfg["max_weight"]),
                    bool(cfg["short"]),
                    quote_for_size,
                    float(cfg["commission"]),
                    float(cfg["slippage"]),
                    int(cfg["lot"]),
                )
            quote = opened.loc[stamp] if stamp in trades_on and cfg["fill"] == "next_open" else close.loc[stamp]
            cash, pending = _trade_toward(
                stamp,
                names,
                shares,
                target,
                quote,
                cash,
                trades,
                float(cfg["commission"]),
                float(cfg["slippage"]),
                float(cfg["sell_tax"]),
                int(cfg["lot"]),
            )
        if trade_at_open:
            _remember(last_price, names, row_close)
        nav_rows.append(_nav(cash, shares, last_price))
        position_rows.append({name: shares[name] for name in names})
    nav = pd.Series(nav_rows, index=close.index, name="nav")
    positions = pd.DataFrame(position_rows, index=close.index)
    trade_frame = pd.DataFrame(
        trades,
        columns=["date", "ticker", "side", "shares", "price", "commission", "slippage", "cash_after"],
    )
    metrics = _metrics(nav, trade_frame, positions, close, cfg["benchmark"])
    metrics["warnings"] = warnings
    metrics["weight"] = scheme
    return BacktestResult(nav, trade_frame, positions, metrics)


def _config(config: dict) -> dict:
    out = {
        "direction": 1,
        "rebalance": "M",
        "top_n": 20,
        "weight": "equal",
        "max_weight": 0.10,
        "commission": 0.0003,
        "slippage": 0.0005,
        "sell_tax": 0.0,
        "lot": 1,
        "fill": "close",
        "initial_cash": 1_000_000.0,
        "short": False,
        "open": None,
        "benchmark": None,
        "warnings": [],
        "cancel": None,
    }
    out.update(config)
    if out["rebalance"] not in {"D", "W", "M"}:
        raise DataGapError("调仓频率只支持 D、W、M")
    if out["fill"] not in {"close", "next_open"}:
        raise DataGapError("成交价只支持 close 或 next_open")
    if int(out["lot"]) < 1:
        raise DataGapError("最小交易单位至少 1 股")
    return out


def _signal_days(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(index).sort_values()
    if freq == "D":
        return dates
    stamps = pd.Series(dates, index=dates)
    if freq == "W":
        groups = dates.isocalendar()
        return pd.DatetimeIndex(stamps.groupby([groups.year, groups.week]).tail(1).to_numpy())
    return pd.DatetimeIndex(stamps.groupby([dates.year, dates.month]).tail(1).to_numpy())


def _trade_days(index: pd.DatetimeIndex, freq: str, fill: str) -> dict[pd.Timestamp, pd.Timestamp]:
    """交易日映射到发信号的那天。次日开盘成交时，下单延后一个交易日。"""
    dates = pd.DatetimeIndex(index).sort_values()
    locator = {pd.Timestamp(day): pos for pos, day in enumerate(dates)}
    mapped: dict[pd.Timestamp, pd.Timestamp] = {}
    for day in _signal_days(dates, freq):
        signal_day = pd.Timestamp(day)
        if fill == "next_open":
            pos = locator[signal_day] + 1
            if pos >= len(dates):
                continue
            mapped[pd.Timestamp(dates[pos])] = signal_day
        else:
            mapped[signal_day] = signal_day
    return mapped


def _target_shares(
    scores: pd.Series,
    nav: float,
    direction: int,
    top_n: int,
    scheme: str,
    cap: float,
    short: bool,
    quote: pd.Series,
    commission: float,
    slippage: float,
    lot: int,
) -> dict[str, float]:
    valid = scores.dropna()
    valid = valid[np.isfinite(quote.reindex(valid.index).to_numpy(dtype=float))]
    target = {name: 0.0 for name in scores.index}
    if valid.empty or nav <= 0:
        return target
    ordered = valid.sort_values(ascending=(direction < 0))
    count = max(int(top_n), 1)
    picked = ordered.head(min(count, len(ordered)))
    long_budget = 0.5 if short else 1.0
    weights = _weights(picked, scheme, cap, long_budget)
    for name, weight in weights.items():
        price = float(quote[name])
        unit = price * (1.0 + slippage) * (1.0 + commission)
        if unit <= 0:
            continue
        raw = nav * float(weight) / unit
        target[name] = float(int(raw // lot) * lot)
    if short:
        opposite = valid.sort_values(ascending=(direction > 0)).head(min(count, len(valid)))
        short_weights = _weights(opposite, scheme, cap, 0.5)
        for name, weight in short_weights.items():
            if name in picked.index:
                continue
            price = float(quote[name])
            unit = price * (1.0 - slippage)
            if unit <= 0:
                continue
            raw = nav * float(weight) / unit
            target[name] = -float(int(raw // lot) * lot)
    return target


def _weights(picked: pd.Series, scheme: str, cap: float, budget: float) -> pd.Series:
    if picked.empty:
        return picked
    if scheme == "factor":
        shifted = picked - float(picked.min())
        total = float(shifted.sum())
        raw = shifted / total if total > 0 else pd.Series(1.0 / len(picked), index=picked.index)
    else:
        raw = pd.Series(1.0 / len(picked), index=picked.index)
    raw = raw * budget
    return _cap(raw, cap)


def _cap(weights: pd.Series, cap: float) -> pd.Series:
    current = weights.astype(float).copy()
    if cap <= 0:
        return current * 0.0
    for _ in range(len(current) + 1):
        over = current > cap + 1e-12
        if not bool(over.any()):
            break
        excess = float((current[over] - cap).sum())
        current.loc[over] = cap
        room = ~over & (current > 0)
        base = float(current[room].sum()) if bool(room.any()) else 0.0
        if base <= 0:
            break
        current.loc[room] = current.loc[room] + excess * current.loc[room] / base
    return current


def _trade_toward(
    stamp,
    names: list[str],
    shares: dict[str, float],
    target: dict[str, float],
    quote: pd.Series,
    cash: float,
    trades: list[dict],
    commission: float,
    slippage: float,
    sell_tax: float,
    lot: int,
) -> tuple[float, bool]:
    pending = False
    for name in names:
        held = shares[name]
        goal = float(target.get(name, 0.0))
        if held <= goal:
            continue
        price = quote.get(name, np.nan)
        if not np.isfinite(price) or float(price) <= 0:
            pending = True
            continue
        cash = _fill(stamp, name, "sell", held - goal, float(price), cash, shares, trades, commission, slippage, sell_tax)
    for name in names:
        held = shares[name]
        goal = float(target.get(name, 0.0))
        if held >= goal:
            continue
        price = quote.get(name, np.nan)
        if not np.isfinite(price) or float(price) <= 0:
            pending = True
            continue
        buy = goal - held
        unit = float(price) * (1.0 + slippage) * (1.0 + commission)
        if unit <= 0:
            continue
        affordable = int(cash // unit // lot) * lot
        if affordable <= 0:
            if buy > 0:
                pending = True
            continue
        if affordable < buy:
            buy = float(affordable)
            pending = True
        cash = _fill(stamp, name, "buy", buy, float(price), cash, shares, trades, commission, slippage, sell_tax)
    return cash, pending


def _fill(
    stamp,
    name: str,
    side: str,
    quantity: float,
    price: float,
    cash: float,
    shares: dict[str, float],
    trades: list[dict],
    commission: float,
    slippage: float,
    sell_tax: float,
) -> float:
    if quantity <= 0 or price <= 0:
        return cash
    slip_cash = quantity * price * slippage
    if side == "buy":
        fee = quantity * price * (1.0 + slippage) * commission
        cash -= quantity * price + slip_cash + fee
        shares[name] = shares.get(name, 0.0) + quantity
    else:
        fee = quantity * price * (1.0 - slippage) * (commission + sell_tax)
        cash += quantity * price - slip_cash - fee
        shares[name] = shares.get(name, 0.0) - quantity
    trades.append(
        {
            "date": pd.Timestamp(stamp).strftime("%Y-%m-%d"),
            "ticker": name,
            "side": side,
            "shares": float(quantity),
            "price": float(price),
            "commission": float(fee),
            "slippage": float(slip_cash),
            "cash_after": float(cash),
        }
    )
    return cash


def _remember(last_price: dict[str, float], names: list[str], row: pd.Series) -> None:
    for name in names:
        price = row.get(name, np.nan)
        if np.isfinite(price):
            last_price[name] = float(price)


def _nav(cash: float, shares: dict[str, float], last_price: dict[str, float]) -> float:
    total = cash
    for name, held in shares.items():
        price = last_price.get(name, np.nan)
        if held and np.isfinite(price):
            total += held * float(price)
    return float(total)


def _metrics(
    nav: pd.Series,
    trades: pd.DataFrame,
    positions: pd.DataFrame,
    close: pd.DataFrame,
    benchmark,
) -> dict:
    returns = nav.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    total = float(nav.iloc[-1] / nav.iloc[0] - 1.0) if len(nav) and nav.iloc[0] else 0.0
    periods = max(len(returns), 1)
    annualized = float((1.0 + total) ** (252.0 / periods) - 1.0) if total > -1 else -1.0
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    volatility = std * np.sqrt(252.0)
    sharpe = 0.0 if std == 0 else float(returns.mean() / std * np.sqrt(252.0))
    peak = nav.cummax()
    drawdown = float((nav / peak - 1.0).min()) if len(nav) else 0.0
    calmar = float(annualized / abs(drawdown)) if drawdown < 0 else 0.0
    turnover = _turnover(trades, nav)
    win_rate, payoff = _round_trips(trades)
    out = {
        "cumulative_return": total,
        "annualized_return": annualized,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "calmar": calmar,
        "turnover": turnover,
        "win_rate": win_rate,
        "payoff": payoff,
        "excess_return": None,
        "excess_annualized": None,
    }
    if benchmark is not None and len(benchmark):
        bench = pd.Series(benchmark).astype(float).reindex(nav.index).dropna()
        if len(bench) > 1:
            bench_nav = bench / float(bench.iloc[0])
            common = nav.reindex(bench_nav.index).dropna()
            bench_nav = bench_nav.reindex(common.index)
            strategy = common / float(common.iloc[0])
            excess = float(strategy.iloc[-1] - bench_nav.iloc[-1])
            bench_total = float(bench_nav.iloc[-1] - 1.0)
            bench_ann = float((1.0 + bench_total) ** (252.0 / max(len(bench_nav) - 1, 1)) - 1.0)
            strat_total = float(strategy.iloc[-1] - 1.0)
            strat_ann = float((1.0 + strat_total) ** (252.0 / max(len(strategy) - 1, 1)) - 1.0)
            out["excess_return"] = excess
            out["excess_annualized"] = float(strat_ann - bench_ann)
    del positions, close
    return out


def _turnover(trades: pd.DataFrame, nav: pd.Series) -> float:
    if trades.empty or nav.empty:
        return 0.0
    traded = 0.0
    for row in trades.itertuples(index=False):
        traded += float(row.shares) * float(row.price)
    years = max(len(nav) / 252.0, 1.0 / 252.0)
    base = float(nav.mean()) if float(nav.mean()) else 1.0
    return float(traded / 2.0 / base / years)


def _round_trips(trades: pd.DataFrame) -> tuple[float | None, float | None]:
    if trades.empty:
        return None, None
    book: dict[str, list[float]] = {}
    profits: list[float] = []
    for row in trades.itertuples(index=False):
        ticker = str(row.ticker)
        qty = float(row.shares)
        cash_flow = qty * float(row.price)
        fee = float(row.commission) + float(row.slippage)
        lots = book.setdefault(ticker, [])
        if row.side == "buy":
            lots.append(cash_flow + fee)
        else:
            cost = lots.pop(0) if lots else 0.0
            profits.append(cash_flow - fee - cost)
    if not profits:
        return None, None
    wins = [item for item in profits if item > 0]
    losses = [item for item in profits if item < 0]
    win_rate = len(wins) / len(profits)
    payoff = None
    if wins and losses:
        payoff = float(np.mean(wins) / abs(np.mean(losses)))
    return win_rate, payoff
