"""用未复权收盘价计算的因子。这里不读文件。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from preprocess.prepare import clean_returns
from services.errors import DataGapError


def _pair(stock: pd.Series, market: pd.Series) -> tuple[pd.Series, pd.Series, pd.DatetimeIndex]:
    both = pd.concat({"s": stock.astype(float), "m": market.astype(float)}, axis=1).dropna()
    return both["s"], both["m"], both.index


def residual_momentum(
    stock: pd.Series,
    market: pd.Series,
    lookback: int,
    skip: int,
    jump_limit: float,
) -> pd.Series:
    if lookback < 10 or skip < 0:
        raise DataGapError("动量窗口至少 10 日，跳过天数不能为负")
    stock_px, market_px, index = _pair(stock, market)
    rs = clean_returns(stock_px, jump_limit)
    rm = clean_returns(market_px, jump_limit)
    lagged_s = rs.shift(skip)
    lagged_m = rm.shift(skip)
    mean_s = lagged_s.rolling(lookback, min_periods=lookback).mean()
    mean_m = lagged_m.rolling(lookback, min_periods=lookback).mean()
    cov = lagged_s.rolling(lookback, min_periods=lookback).cov(lagged_m)
    var = lagged_m.rolling(lookback, min_periods=lookback).var()
    beta = cov / var.where(var > 0)
    factor = lookback * (mean_s - beta * mean_m)
    factor.index = index
    return factor.reindex(stock.index)


def annualized_vol(close: pd.Series, lookback: int, jump_limit: float) -> pd.Series:
    if lookback < 10:
        raise DataGapError("波动窗口至少 10 日")
    returns = clean_returns(close.astype(float), jump_limit)
    vol = returns.rolling(lookback, min_periods=lookback).std(ddof=1) * np.sqrt(252)
    return vol.reindex(close.index)


def rolling_beta(stock: pd.Series, market: pd.Series, lookback: int, jump_limit: float) -> pd.Series:
    if lookback < 10:
        raise DataGapError("贝塔窗口至少 10 日")
    stock_px, market_px, index = _pair(stock, market)
    rs = clean_returns(stock_px, jump_limit)
    rm = clean_returns(market_px, jump_limit)
    cov = rs.rolling(lookback, min_periods=lookback).cov(rm)
    var = rm.rolling(lookback, min_periods=lookback).var()
    beta = cov / var.where(var > 0)
    beta.index = index
    return beta.reindex(stock.index)


def liquidity_proxy(volume: pd.Series, lookback: int) -> pd.Series:
    if lookback < 5:
        raise DataGapError("流动性窗口至少 5 日")
    avg = volume.astype(float).rolling(lookback, min_periods=lookback).mean()
    return np.log(avg.where(avg > 0))
