"""月度再平衡。权重在收盘决定，下一日才开始赚收益。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from preprocess.prepare import clean_returns, period_ends
from services.errors import DataGapError
from services.runtime import check_cancel, num


def optimize_weights(scores: np.ndarray, cap: float, penalty: float) -> np.ndarray:
    count = len(scores)
    equal = np.full(count, 1.0 / count)
    if count == 1:
        return equal

    def objective(weights: np.ndarray) -> float:
        gap = weights - equal
        return -float(weights @ scores) + penalty * float(gap @ gap)

    result = minimize(
        objective,
        equal,
        method="SLSQP",
        bounds=[(0.0, cap)] * count,
        constraints={"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},
        options={"maxiter": 200, "ftol": 1e-12},
    )
    weights = np.asarray(result.x if result.success else equal, dtype=float)
    weights = np.clip(weights, 0.0, cap)
    total = float(weights.sum())
    if total <= 0 or float(weights.max()) > cap + 1e-6:
        return equal
    return weights / total


def target_weights(row: pd.Series, previous: pd.Series, cap: float, penalty: float) -> tuple[pd.Series, str | None]:
    valid = row.dropna()
    if valid.empty:
        return previous.reindex(row.index).fillna(0.0), "得分为空，沿用上一期权重"
    if cap * len(valid) < 1 - 1e-9:
        return previous.reindex(row.index).fillna(0.0), "有效股票不足以满足单名上限，沿用上一期权重"
    chosen = optimize_weights(valid.to_numpy(dtype=float), cap, penalty)
    out = pd.Series(0.0, index=row.index, dtype=float)
    out.loc[valid.index] = chosen
    return out, None


def simulate(
    prices: pd.DataFrame,
    scores: pd.DataFrame,
    benchmark: pd.Series,
    rebalance: str,
    cost_bps: float,
    slippage_bps: float,
    max_weight: float,
    diversify_penalty: float,
    jump_limit: float,
    rebalance_dates=None,
    eligible: dict[str, list[str]] | None = None,
    cancel=None,
    progress=None,
) -> dict:
    del progress
    if prices.empty:
        raise DataGapError("没有可用于回测的行情")
    columns = list(prices.columns)
    ends = list(rebalance_dates) if rebalance_dates is not None else list(period_ends(prices.index, rebalance))
    applied: dict[pd.Timestamp, pd.Series] = {}
    held = pd.Series(0.0, index=columns, dtype=float)
    notes: list[str] = []
    for day in ends:
        check_cancel(cancel)
        stamp = pd.Timestamp(day).normalize()
        if stamp not in prices.index:
            continue
        names = columns if eligible is None else [item for item in eligible.get(stamp.strftime("%Y-%m-%d"), []) if item in columns]
        if not names:
            continue
        row = scores.reindex(index=prices.index, columns=columns).loc[stamp].reindex(names)
        chosen, note = target_weights(row, held.reindex(names).fillna(0.0), max_weight, diversify_penalty)
        if note and note not in notes:
            notes.append(note)
        held = pd.Series(0.0, index=columns, dtype=float)
        held.loc[chosen.index] = chosen.to_numpy()
        applied[stamp] = held.copy()
    marked = pd.DataFrame(np.nan, index=prices.index, columns=columns)
    for day, weights in applied.items():
        marked.loc[day] = weights
    daily = marked.ffill().fillna(0.0)
    returns = pd.DataFrame({name: clean_returns(prices[name], jump_limit) for name in columns}).fillna(0.0)
    lagged = daily.shift(1).fillna(0.0)
    gross = (lagged * returns).sum(axis=1)
    change = daily.diff()
    change.iloc[0] = daily.iloc[0]
    traded = change.abs().sum(axis=1)
    cost = (traded * (float(cost_bps) + float(slippage_bps)) / 10000.0).shift(1).fillna(0.0)
    bench = clean_returns(benchmark.reindex(prices.index).astype(float), jump_limit).fillna(0.0)
    return {
        "net": gross - cost,
        "benchmark": bench,
        "one_way": 0.5 * traded,
        "cost": cost,
        "active": lagged.sum(axis=1) > 1e-8,
        "warnings": notes,
    }


def performance(net: pd.Series, bench: pd.Series, one_way: pd.Series, cost: pd.Series) -> dict:
    bench = bench.reindex(net.index).fillna(0.0)
    one_way = one_way.reindex(net.index).fillna(0.0)
    cost = cost.reindex(net.index).fillna(0.0)
    if len(net) < 2:
        return {
            "cumulative": None,
            "annualized": None,
            "volatility": None,
            "sharpe": None,
            "max_drawdown": None,
            "excess_cumulative": None,
            "tracking_error": None,
            "avg_turnover": None,
            "total_cost": num(cost.sum()) if len(cost) else None,
            "days": int(len(net)),
        }
    wealth = float((1.0 + net).prod())
    bench_wealth = float((1.0 + bench).prod())
    std = float(np.std(net.to_numpy(dtype=float), ddof=1))
    excess = net - bench
    excess_std = float(np.std(excess.to_numpy(dtype=float), ddof=1))
    nav = (1.0 + net).cumprod()
    drawdown = float((nav / nav.cummax() - 1.0).min())
    turns = one_way[one_way > 1e-8]
    return {
        "cumulative": num(wealth - 1.0),
        "annualized": num(wealth ** (252 / len(net)) - 1.0) if wealth > 0 else None,
        "volatility": num(std * np.sqrt(252)),
        "sharpe": num(0.0 if std == 0 else float(np.mean(net.to_numpy(dtype=float)) / std * np.sqrt(252))),
        "max_drawdown": num(drawdown),
        "excess_cumulative": num(wealth / bench_wealth - 1.0) if bench_wealth else None,
        "tracking_error": num(excess_std * np.sqrt(252)),
        "avg_turnover": num(turns.mean()) if len(turns) else 0.0,
        "total_cost": num(cost.sum()),
        "days": int(len(net)),
    }


def yearly_table(net: pd.Series, bench: pd.Series) -> list[dict]:
    rows = []
    aligned = bench.reindex(net.index).fillna(0.0)
    for year, part in net.groupby(net.index.year):
        other = aligned.reindex(part.index).fillna(0.0)
        strategy = float((1.0 + part).prod() - 1.0)
        base = float((1.0 + other).prod() - 1.0)
        rows.append(
            {
                "year": int(year),
                "strategy": num(strategy),
                "benchmark": num(base),
                "excess": num((1.0 + strategy) / (1.0 + base) - 1.0) if (1.0 + base) else None,
            }
        )
    return rows
