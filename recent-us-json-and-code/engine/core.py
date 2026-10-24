"""用 pandas 整理行情，用 numpy 计算收益、回撤和夏普。"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def prepare_close(frame: pd.DataFrame) -> pd.Series:
    """把行情表收成按日期升序的收盘价。

    必须有 date、close 两列。date 解析为时间，close 解析为浮点数。
    返回的 Series 以 date 为索引。
    """
    missing = [name for name in ("date", "close") if name not in frame.columns]
    if missing:
        raise ValueError(f"行情表缺少列: {missing}")
    table = frame.loc[:, ["date", "close"]].copy()
    table["date"] = pd.to_datetime(table["date"])
    table["close"] = pd.to_numeric(table["close"], errors="coerce")
    table = table.dropna(subset=["date", "close"]).sort_values("date")
    if table.empty:
        raise ValueError("行情表没有可用的 date 和 close")
    close = table.set_index("date")["close"].astype("float64")
    close.index.name = "date"
    return close


def run_backtest(
    close: pd.Series,
    position: pd.Series,
    fee: float = 0.0,
) -> pd.DataFrame:
    """按收盘价和目标仓位做日频回测。

    position 在当日收盘后确定，实际持仓使用上一期仓位，避免用到未来价格。
    仓位可以是 1、0、-1，也可以是其他权重。换仓成本按仓位变化的绝对值乘 fee。
    """
    if fee < 0:
        raise ValueError("fee 不能为负")
    close = close.astype("float64").sort_index()
    held = position.reindex(close.index).fillna(0.0).astype("float64").shift(1).fillna(0.0)
    asset_return = close.pct_change().fillna(0.0)
    turnover = held.diff().abs()
    turnover.iloc[0] = abs(float(held.iloc[0]))
    strategy_return = held.to_numpy(dtype=float) * asset_return.to_numpy(dtype=float) - turnover.to_numpy(dtype=float) * fee
    strategy_return = pd.Series(strategy_return, index=close.index, name="return")
    equity = pd.Series(np.cumprod(1.0 + strategy_return.to_numpy(dtype=float)), index=close.index, name="equity")
    return pd.DataFrame(
        {
            "close": close,
            "position": held.rename("position"),
            "return": strategy_return,
            "equity": equity,
        }
    )


def summarize(result: pd.DataFrame) -> dict[str, float]:
    """用 numpy 计算累计收益、最大回撤和年化夏普。夏普按日收益、无风险利率为 0。"""
    equity = result["equity"].to_numpy(dtype=float)
    returns = result["return"].to_numpy(dtype=float)
    if equity.size == 0:
        return {"cumulative_return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}
    cumulative = float(equity[-1] / equity[0] - 1.0)
    drawdown = float(np.min(equity / np.maximum.accumulate(equity) - 1.0))
    if returns.size < 2:
        sharpe = 0.0
    else:
        std = float(np.std(returns, ddof=1))
        sharpe = 0.0 if std == 0.0 else float(np.mean(returns) / std * np.sqrt(TRADING_DAYS))
    return {
        "cumulative_return": cumulative,
        "max_drawdown": drawdown,
        "sharpe": sharpe,
    }
