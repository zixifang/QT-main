"""Piotroski F-Score 的季度多头回测。

调仓日是 1、4、7、10 月的第 10 个交易日。信号只用 filing_date 不晚于调仓日的最新财报。
只买入得分最高的约 10%，等权持有；离开这个区间就卖出。权重不为负。
下一交易日按开盘价或 VWAP 成交。

`python -m algorithms.fscore_long_short` 会把这条策略写到 outdata/nasdaq-xbrl，数据集名称是「纳斯达克XBRL」。

基本面九列默认是原始指标，不是已经打好的 0/1：

- roa、ocf、delta_roa、ocf_gt_ni、delta_current_ratio、delta_gross_margin、delta_asset_turnover：大于 0 记 1
- delta_leverage：杠杆变化小于 0 记 1
- no_new_shares：大于 0 记 1（1 表示没有新发股票）

九项已经是「满足记 1」时，调用 run_fscore_backtest(..., already_flags=True)。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252
QUARTER_MONTHS = (1, 4, 7, 10)
REBALANCE_NTH = 10
ADV_WINDOW = 20
ADV_MIN = 1_000_000.0
BORROW_ANNUAL = 0.15
SIDE_FRACTION = 0.10
MAX_MISSING = 2
JUMP_LIMIT = 0.5
DATASET_ID = "nasdaq-xbrl"
DATASET_NAME = "纳斯达克XBRL"
ALGO_ID = "fscore"
ALGO_NAME = "【F-Score】"
RETIRED_IDS = ("fscore-ls",)
F_COLUMNS = (
    "roa",
    "ocf",
    "delta_roa",
    "ocf_gt_ni",
    "delta_leverage",
    "delta_current_ratio",
    "no_new_shares",
    "delta_gross_margin",
    "delta_asset_turnover",
)
# 1 表示原始值越高越好；-1 表示越低越好。
F_SIGN = {
    "roa": 1,
    "ocf": 1,
    "delta_roa": 1,
    "ocf_gt_ni": 1,
    "delta_leverage": -1,
    "delta_current_ratio": 1,
    "no_new_shares": 1,
    "delta_gross_margin": 1,
    "delta_asset_turnover": 1,
}
OHLCV_COLUMNS = ("date", "ticker", "open", "high", "low", "close", "volume")
FUNDAMENTAL_KEYS = ("ticker", "filing_date", "report_date")
BENCHMARK_TICKERS = ("SPX", "^GSPC", "GSPC", "SPY")


def quarter_rebalance_dates(trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """每季度首月的第 10 个交易日。交易日不够 10 天的月份跳过。"""
    days = pd.DatetimeIndex(pd.to_datetime(trading_days)).unique().sort_values()
    if days.tz is not None:
        days = days.tz_convert("UTC").tz_localize(None)
    chosen = []
    grouped = pd.Series(days, index=days).groupby([days.year, days.month])
    for (year, month), block in grouped:
        if int(month) not in QUARTER_MONTHS or len(block) < REBALANCE_NTH:
            continue
        chosen.append(pd.Timestamp(block.iloc[REBALANCE_NTH - 1]))
    return pd.DatetimeIndex(chosen)


def align_fundamentals(df_fundamentals: pd.DataFrame, asof_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """在每个调仓日只保留 filing_date <= 调仓日的最新一条。同一天多条时留下报告期较晚的那条。"""
    right = _prepare_fundamentals(df_fundamentals)
    dates = pd.DatetimeIndex(pd.to_datetime(asof_dates)).unique().sort_values()
    if dates.tz is not None:
        dates = dates.tz_convert("UTC").tz_localize(None)
    names = pd.Index(right["ticker"].unique()).sort_values()
    left = pd.MultiIndex.from_product([dates, names], names=["rebalance_date", "ticker"]).to_frame(index=False)
    # by 分组时，左右两侧的对齐日期本身也必须整体按时间升序。
    left = left.sort_values(["rebalance_date", "ticker"], kind="mergesort")
    right = right.sort_values(["filing_date", "ticker"], kind="mergesort")
    merged = pd.merge_asof(
        left,
        right,
        left_on="rebalance_date",
        right_on="filing_date",
        by="ticker",
        direction="backward",
        allow_exact_matches=True,
    )
    leaked = merged["filing_date"].notna() & (merged["filing_date"] > merged["rebalance_date"])
    if bool(leaked.any()):
        raise AssertionError("财报对齐用到了调仓日之后的公告")
    return merged.reset_index(drop=True)


def compute_fscore(df_fundamentals: pd.DataFrame, already_flags: bool = False) -> pd.Series:
    """有效项少于 7 项时得分记为空。否则把有效项的通过比例放大到 9 分。"""
    missing = [name for name in F_COLUMNS if name not in df_fundamentals.columns]
    if missing:
        raise ValueError(f"基本面表缺少列: {missing}")
    flags = []
    for column, sign in F_SIGN.items():
        raw = pd.to_numeric(df_fundamentals[column], errors="coerce")
        directed = raw if already_flags or sign > 0 else -raw
        flag = pd.Series(np.where(raw.isna(), np.nan, (directed > 0).astype(float)), index=df_fundamentals.index)
        flags.append(flag)
    table = pd.concat(flags, axis=1)
    available = table.notna().sum(axis=1)
    passed = table.sum(axis=1, min_count=1)
    scaled = passed / available * len(F_COLUMNS)
    return scaled.where(available >= len(F_COLUMNS) - MAX_MISSING)


def average_dollar_volume(df_ohlcv: pd.DataFrame, window: int = ADV_WINDOW) -> pd.DataFrame:
    """每个交易日及之前 window 天的平均成交额，成交额 = close * volume。不足 window 天为空。"""
    prices = _prepare_ohlcv(df_ohlcv)
    prices = prices.copy()
    prices["dollar"] = prices["close"] * prices["volume"]
    wide = prices.pivot(index="date", columns="ticker", values="dollar")
    return wide.rolling(window, min_periods=window).mean()


def target_weights(scores: pd.Series, long_fraction: float = SIDE_FRACTION) -> pd.Series:
    """只买入得分最高的大约 long_fraction。入选股票等权，权重和为 1。分数卡在分位上的股票全部留下。"""
    values = scores.replace([np.inf, -np.inf], np.nan).dropna().astype("float64")
    weights = pd.Series(0.0, index=values.index, dtype="float64")
    count = int(len(values))
    if count < 1:
        return weights
    long_count = max(1, int(np.ceil(count * long_fraction - 1e-12)))
    high_cut = float(values.nlargest(long_count).min())
    long_names = values[values >= high_cut].index
    if len(long_names) == 0:
        return weights
    weights.loc[long_names] = 1.0 / len(long_names)
    return weights


def simulate_open_execution(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    execution: str = "open",
    borrow_annual: float = BORROW_ANNUAL,
    trade_cost: float = 0.0,
) -> pd.DataFrame:
    """信号在调仓日收盘后才知道，下一交易日用开盘价或 VWAP 换成新仓。

    调仓日当天，旧仓只拿到前收盘到成交价的一段，新仓只拿到成交价到收盘的一段。
    权重低于 0 的部分当作空仓。单日价格跳空或日内涨跌超过 50% 时，该段收益记 0。
    """
    if execution not in {"open", "vwap"}:
        raise ValueError("execution 只能是 open 或 vwap")
    if trade_cost < 0 or borrow_annual < 0:
        raise ValueError("费率和借券费不能为负")
    if execution == "vwap" and "vwap" not in prices.columns:
        raise ValueError("按 VWAP 成交时行情表需要 vwap 列")
    close = _wide_price(prices, "close")
    trade = _wide_price(prices, execution).reindex(index=close.index, columns=close.columns)
    calendar = close.index
    # 非调仓日必须是空值。若先填 0，下一交易日的目标仓位无法向后延续。
    held = signal.reindex(index=calendar, columns=close.columns).astype("float64")
    held = held.shift(1).ffill().fillna(0.0).clip(lower=0.0)
    overnight = held.shift(1).fillna(0.0).clip(lower=0.0)
    previous_close = close.shift(1)
    hold_return = _zero_jumps(close / previous_close - 1.0)
    exit_return = _zero_jumps(trade / previous_close - 1.0)
    entry_return = _zero_jumps(close / trade - 1.0)
    execution_rows = held.ne(overnight).any(axis=1).to_numpy()

    long_hold, _ = _book_return(held.clip(lower=0.0), hold_return)
    short_hold, short_gross = _book_return((-held).clip(lower=0.0), hold_return)
    long_exit, _ = _book_return(overnight.clip(lower=0.0), exit_return)
    short_exit, _ = _book_return((-overnight).clip(lower=0.0), exit_return)
    long_entry, _ = _book_return(held.clip(lower=0.0), entry_return)
    short_entry, short_gross_entry = _book_return((-held).clip(lower=0.0), entry_return)

    overnight_pnl = long_exit - short_exit
    intraday_pnl = long_entry - short_entry
    trade_gross = (1.0 + overnight_pnl) * (1.0 + intraday_pnl) - 1.0
    hold_gross = long_hold - short_hold
    gross = np.where(execution_rows, trade_gross, hold_gross)
    long_return = np.where(execution_rows, (1.0 + long_exit) * (1.0 + long_entry) - 1.0, long_hold)
    short_return = np.where(execution_rows, (1.0 + short_exit) * (1.0 + short_entry) - 1.0, short_hold)
    borrow_base = np.where(execution_rows, short_gross_entry, short_gross)
    borrow = borrow_annual / TRADING_DAYS * borrow_base
    turnover = _execution_turnover(held, overnight)
    cost = np.where(execution_rows, turnover * trade_cost, 0.0)
    net = gross - borrow - cost
    daily = pd.DataFrame(
        {
            "long_return": long_return,
            "short_return": short_return,
            "gross_return": gross,
            "borrow_cost": borrow,
            "trade_cost": cost,
            "net_return": net,
            "short_gross": borrow_base,
            "turnover": np.where(execution_rows, turnover, 0.0),
        },
        index=calendar,
    )
    daily.index.name = "date"
    return daily.fillna(0.0)


def performance(returns: pd.Series, benchmark_returns: pd.Series | None = None) -> pd.DataFrame:
    """年化收益按复利和 252 个交易日折算。夏普无风险利率为 0。"""
    rows = {"策略": _path_stats(returns)}
    if benchmark_returns is not None:
        bench = benchmark_returns.reindex(returns.index).fillna(0.0)
        rows["标普500"] = _path_stats(bench)
        rows["日超额"] = _path_stats(returns - bench)
    return pd.DataFrame(rows).T


def run_fscore_backtest(
    df_ohlcv: pd.DataFrame,
    df_fundamentals: pd.DataFrame,
    benchmark_close: pd.Series | None = None,
    *,
    adv_min: float = ADV_MIN,
    adv_window: int = ADV_WINDOW,
    execution: str = "open",
    already_flags: bool = False,
    long_fraction: float = SIDE_FRACTION,
    trade_cost: float = 0.0,
) -> dict[str, pd.DataFrame]:
    """跑完整条季度多头。返回调仓表、得分表、日收益和绩效。"""
    prices = _prepare_ohlcv(df_ohlcv)
    benchmark_close, prices = _split_benchmark(prices, benchmark_close)
    calendar = pd.DatetimeIndex(prices["date"].unique()).sort_values()
    rebalance_dates = _executable_rebalance_dates(calendar)
    aligned = align_fundamentals(df_fundamentals, rebalance_dates)
    aligned["fscore"] = compute_fscore(aligned, already_flags=already_flags)
    adv = average_dollar_volume(prices, adv_window)
    signal, scores = _signals(aligned, adv, calendar, prices["ticker"].unique(), adv_min, long_fraction)
    daily = simulate_open_execution(signal, prices, execution, 0.0, trade_cost)
    evaluated = _from_first_execution(daily)
    bench_returns = _benchmark_returns(benchmark_close, evaluated.index)
    stats = performance(evaluated["net_return"], bench_returns)
    if bench_returns is not None:
        evaluated = evaluated.copy()
        evaluated["benchmark_return"] = bench_returns
        evaluated["excess_return"] = evaluated["net_return"] - bench_returns
    evaluated["equity"] = (1.0 + evaluated["net_return"]).cumprod()
    rebalance = _rebalance_table(scores, daily, rebalance_dates, calendar)
    return {
        "rebalance": rebalance,
        "scores": scores,
        "signal": signal,
        "daily": evaluated,
        "daily_all": daily,
        "stats": stats,
    }


def _executable_rebalance_dates(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    dates = quarter_rebalance_dates(calendar)
    if len(dates) == 0 or len(calendar) < 2:
        return pd.DatetimeIndex([])
    next_day = pd.Series(calendar, index=calendar).shift(-1)
    executable = next_day.reindex(dates).notna()
    return pd.DatetimeIndex(dates[executable.to_numpy()])


def _signals(
    aligned: pd.DataFrame,
    adv: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    tickers: pd.Index,
    adv_min: float,
    long_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal = pd.DataFrame(np.nan, index=calendar, columns=pd.Index(tickers).sort_values(), dtype="float64")
    rows = []
    for day, group in aligned.groupby("rebalance_date", sort=True):
        stamp = pd.Timestamp(day)
        if stamp not in signal.index:
            continue
        liquid = _liquid_names(adv, stamp, adv_min)
        eligible = group[group["ticker"].isin(liquid) & group["fscore"].notna()].set_index("ticker")["fscore"]
        weights = target_weights(eligible, long_fraction)
        row = pd.Series(0.0, index=signal.columns, dtype="float64")
        if len(weights):
            row.loc[weights.index] = weights.to_numpy()
        signal.loc[stamp] = row
        adv_row = adv.loc[stamp] if stamp in adv.index else pd.Series(dtype="float64")
        recorded = group.copy()
        recorded["adv"] = recorded["ticker"].map(adv_row)
        recorded["weight"] = recorded["ticker"].map(weights).fillna(0.0)
        recorded["liquid"] = recorded["ticker"].isin(liquid)
        rows.append(recorded)
    scores = pd.concat(rows, ignore_index=True) if rows else aligned.iloc[0:0].copy()
    return signal, scores


def _liquid_names(adv: pd.DataFrame, day: pd.Timestamp, adv_min: float) -> pd.Index:
    if day not in adv.index:
        return pd.Index([])
    row = adv.loc[day].dropna()
    return row[row > adv_min].index


def _rebalance_table(
    scores: pd.DataFrame,
    daily: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    columns = [
        "rebalance_date",
        "execution_date",
        "n_long",
        "n_short",
        "turnover",
        "period_return",
    ]
    if scores.empty or len(rebalance_dates) == 0:
        return pd.DataFrame(columns=columns)
    next_day = pd.Series(calendar, index=calendar).shift(-1)
    rows = []
    executions = []
    for stamp in rebalance_dates:
        execution = next_day.get(stamp)
        if pd.isna(execution):
            continue
        execution = pd.Timestamp(execution)
        executions.append(execution)
        block = scores[scores["rebalance_date"] == stamp]
        weights = block.set_index("ticker")["weight"]
        rows.append(
            {
                "rebalance_date": pd.Timestamp(stamp),
                "execution_date": execution,
                "n_long": int((weights > 0).sum()),
                "n_short": int((weights < 0).sum()),
                "turnover": np.nan,
                "period_return": np.nan,
            }
        )
    table = pd.DataFrame(rows, columns=columns)
    if table.empty:
        return table
    table["turnover"] = daily["turnover"].reindex(pd.DatetimeIndex(executions)).to_numpy()
    end_points = list(executions[1:]) + [None]
    period_returns = []
    for start, end in zip(executions, end_points):
        if end is None:
            window = daily.loc[daily.index >= start, "net_return"]
        else:
            window = daily.loc[(daily.index >= start) & (daily.index < end), "net_return"]
        period_returns.append(float(np.prod(1.0 + window.to_numpy(dtype=float)) - 1.0) if len(window) else np.nan)
    table["period_return"] = period_returns
    return table


def _from_first_execution(daily: pd.DataFrame) -> pd.DataFrame:
    active = daily.index[daily["short_gross"].gt(0) | daily["turnover"].gt(0)]
    if len(active) == 0:
        return daily.iloc[0:0].copy()
    return daily.loc[active[0] :].copy()


def _benchmark_returns(benchmark_close: pd.Series | None, index: pd.DatetimeIndex) -> pd.Series | None:
    if benchmark_close is None or len(index) == 0:
        return None
    close = pd.to_numeric(benchmark_close, errors="coerce").dropna().sort_index().astype("float64")
    close.index = _as_naive(pd.Series(close.index, index=close.index)).to_numpy()
    returns = close.pct_change().reindex(index)
    return returns.fillna(0.0).astype("float64")


def _split_benchmark(
    prices: pd.DataFrame,
    benchmark_close: pd.Series | None,
) -> tuple[pd.Series | None, pd.DataFrame]:
    if benchmark_close is not None:
        close = benchmark_close.copy()
        close.index = _as_naive(pd.Series(close.index, index=close.index))
        return close.astype("float64"), prices
    for ticker in BENCHMARK_TICKERS:
        block = prices[prices["ticker"] == ticker]
        if block.empty:
            continue
        close = block.drop_duplicates("date").set_index("date")["close"].astype("float64")
        return close, prices[prices["ticker"] != ticker].copy()
    return None, prices


def _path_stats(returns: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    periods = int(values.size)
    if periods == 0:
        return {"cumulative_return": 0.0, "annualized_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "days": 0}
    wealth = np.cumprod(1.0 + values)
    curve = np.concatenate([[1.0], wealth])
    cumulative = float(wealth[-1] - 1.0)
    annualized = float(wealth[-1] ** (TRADING_DAYS / periods) - 1.0)
    deviation = float(np.std(values, ddof=1)) if periods > 1 else 0.0
    if deviation == 0.0:
        sharpe = 0.0 if float(np.mean(values)) == 0.0 else float(np.sign(np.mean(values)) * np.inf)
    else:
        sharpe = float(np.mean(values) / deviation * np.sqrt(TRADING_DAYS))
    drawdown = float(np.min(curve / np.maximum.accumulate(curve) - 1.0))
    return {
        "cumulative_return": cumulative,
        "annualized_return": annualized,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "days": periods,
    }


def _execution_turnover(held: pd.DataFrame, overnight: pd.DataFrame) -> np.ndarray:
    change = (held - overnight).abs().sum(axis=1).to_numpy(dtype=float)
    return 0.5 * change


def _book_return(weight: pd.DataFrame, asset_return: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """把正权重在当天有成交价的股票里重新归一，再对资产收益求加权和。"""
    aligned = asset_return.reindex(index=weight.index, columns=weight.columns)
    raw_weight = weight.to_numpy(dtype=float)
    raw_return = aligned.to_numpy(dtype=float)
    usable = np.isfinite(raw_weight) & np.isfinite(raw_return) & (raw_weight > 0.0)
    kept = np.where(usable, raw_weight, 0.0)
    total = kept.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = np.divide(kept, total, out=np.zeros_like(kept), where=total > 0.0)
    clean_return = np.where(np.isfinite(raw_return), raw_return, 0.0)
    book = (scaled * clean_return).sum(axis=1)
    gross = (total.ravel() > 0.0).astype(float)
    return book, gross


def _wide_price(prices: pd.DataFrame, column: str) -> pd.DataFrame:
    frame = prices.loc[:, ["date", "ticker", column]].copy()
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    wide = frame.pivot(index="date", columns="ticker", values=column).sort_index().astype("float64")
    wide.index.name = "date"
    return wide


def _prepare_ohlcv(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    missing = [name for name in OHLCV_COLUMNS if name not in df_ohlcv.columns]
    if missing:
        raise ValueError(f"行情表缺少列: {missing}")
    columns = list(OHLCV_COLUMNS)
    if "vwap" in df_ohlcv.columns:
        columns.append("vwap")
    frame = df_ohlcv.loc[:, columns].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = _as_naive(frame["date"])
    for column in columns:
        if column in {"date", "ticker"}:
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "ticker", "close"])
    frame = frame.sort_values(["date", "ticker"]).drop_duplicates(["date", "ticker"], keep="last")
    return frame.reset_index(drop=True)


def _prepare_fundamentals(df_fundamentals: pd.DataFrame) -> pd.DataFrame:
    missing = [name for name in FUNDAMENTAL_KEYS + F_COLUMNS if name not in df_fundamentals.columns]
    if missing:
        raise ValueError(f"基本面表缺少列: {missing}")
    frame = df_fundamentals.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["filing_date"] = _as_naive(frame["filing_date"])
    frame["report_date"] = _as_naive(frame["report_date"])
    for column in F_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["ticker", "filing_date"])
    frame = frame.sort_values(["ticker", "filing_date", "report_date"]).drop_duplicates(
        ["ticker", "filing_date"],
        keep="last",
    )
    return frame.reset_index(drop=True)


def _as_naive(values: pd.Series) -> pd.Series:
    stamps = pd.to_datetime(values)
    if getattr(stamps.dtype, "tz", None) is not None:
        stamps = stamps.dt.tz_convert("UTC").dt.tz_localize(None)
    return stamps


def _zero_jumps(returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    values = returns.replace([np.inf, -np.inf], np.nan)
    return values.mask(values.abs() > JUMP_LIMIT, 0.0)


FLAG_SOURCE = {
    "F_ROA": "roa",
    "F_CFO": "ocf",
    "F_dROA": "delta_roa",
    "F_ACCRUAL": "ocf_gt_ni",
    "F_dLEVER": "delta_leverage",
    "F_dLIQUID": "delta_current_ratio",
    "F_EQ_ISSUE": "no_new_shares",
    "F_dGROSS_MARGIN": "delta_gross_margin",
    "F_dASSET_TURN": "delta_asset_turnover",
}
ROOT = Path(__file__).resolve().parents[1]
FACTOR_PATH = ROOT / "data" / "factors" / "factors_nq100.parquet"
PRICE_DIR = ROOT / "data" / "nasdaq100"
OUT_DIR = ROOT / "outdata" / DATASET_ID
UNIVERSE_NOTE = (
    "没有调仓日的历史成分名单。"
    "股票池是 2026 年 10 月有 10-K 或 10-Q、并且本地有未复权日线的幸存者，已去掉 GOOG。"
    "F-Score 从公告日当天起才能使用。只买入，不卖空。"
)
ALGO_NOTE = (
    "每季首月的第 10 个交易日，用已经公告的最新 10-K 或 10-Q 计算 F-Score。"
    "九项里缺失超过两项就剔除，其余按有效项比例放大到 9 分。"
    "过去 20 日平均成交额不超过 100 万美元的股票不参与。"
    "只买入得分最高的约 10%，等权持有，离开这个区间就卖出。"
    "下一交易日按开盘价成交。单日涨跌超过 50% 的收益记 0。"
)


def piotroski_frame(factors: pd.DataFrame) -> pd.DataFrame:
    """把因子表里的九个 0/1 分项改成回测使用的列名。10-K 和 10-Q 都保留。"""
    missing = [name for name in ("ticker", "filed_date", "period_end", *FLAG_SOURCE) if name not in factors.columns]
    if missing:
        raise ValueError(f"因子表缺少列: {missing}")
    frame = factors.loc[factors["form"].isin(["10-K", "10-Q"])].copy() if "form" in factors.columns else factors.copy()
    frame = frame.rename(columns={**FLAG_SOURCE, "filed_date": "filing_date", "period_end": "report_date"})
    keep = ["ticker", "filing_date", "report_date", *F_COLUMNS]
    return frame.loc[:, keep]


def sleeve_returns(close: pd.Series, opened: pd.Series, position: pd.Series) -> pd.Series:
    """单股现金账户。权重大于 0 时按开盘价买入并持有，回到 0 时卖出。超过 50% 的跳变记 0。"""
    close = close.astype("float64")
    opened = opened.reindex(close.index).astype("float64")
    held = position.reindex(close.index).fillna(0.0).astype("float64")
    sign = pd.Series(np.where(held > 1e-8, 1.0, 0.0), index=close.index)
    previous = close.shift(1)
    hold = _zero_jumps(close / previous - 1.0).fillna(0.0)
    gap = _zero_jumps(opened / previous - 1.0).fillna(0.0)
    day = _zero_jumps(close / opened - 1.0).fillna(0.0)
    previous_sign = sign.shift(1).fillna(0.0)
    changed = sign.ne(previous_sign)
    trade_gross = (1.0 + previous_sign * gap) * (1.0 + sign * day) - 1.0
    gross = trade_gross.where(changed, sign * hold).fillna(0.0)
    net = gross.to_numpy(dtype=float)
    if len(net):
        net[0] = 0.0
    return pd.Series(net, index=close.index, name="return")


def export_dataset(
    factor_path: Path = FACTOR_PATH,
    price_dir: Path = PRICE_DIR,
    out_dir: Path = OUT_DIR,
) -> Path:
    """读取纳斯达克行情和 XBRL 因子，写出「纳斯达克XBRL」数据集。"""
    from algorithms.index_enhance import (
        BENCHMARK_NAME,
        INDEX_STEM,
        OOS_START,
        PRICE_FIELD,
        SCHEMA_VERSION,
        _describe_returns,
        _equity,
        _overview_year,
        _pair_stats,
        _read_csv,
        _series,
        _stock_returns,
        _symbol_year_rows,
        _trailing,
        _write_json,
        _write_merged_json,
        load_prices,
    )

    if not factor_path.exists():
        raise FileNotFoundError(f"没有因子表：{factor_path}")
    ndx, frames = load_prices(price_dir)
    factors = piotroski_frame(pd.read_parquet(factor_path))
    names = [symbol for symbol in frames if symbol in set(factors["ticker"].astype(str).str.upper())]
    if len(names) < 2:
        raise FileNotFoundError("没有同时具备 XBRL 因子和日线的股票")
    frames = {symbol: frames[symbol] for symbol in names}
    fundamentals = factors[factors["ticker"].astype(str).str.upper().isin(names)]
    ohlcv = _long_prices(frames)
    result = run_fscore_backtest(
        ohlcv,
        fundamentals,
        benchmark_close=ndx,
        already_flags=True,
        trade_cost=0.0,
    )
    calendar = ndx.index
    signal = result["signal"].reindex(calendar).ffill().fillna(0.0).clip(lower=0.0)
    position = signal.shift(1).fillna(0.0).clip(lower=0.0)
    portfolio = result["daily_all"].reindex(calendar).fillna(0.0)
    strategy_return = portfolio["net_return"].astype("float64")
    benchmark_return = _stock_returns(ndx.reindex(calendar))
    strategy_equity = _equity(strategy_return)
    benchmark_equity = _equity(benchmark_return)
    pit = align_fundamentals(fundamentals, calendar)
    pit["fscore"] = compute_fscore(pit, already_flags=True)
    indicator = pit.pivot(index="rebalance_date", columns="ticker", values="fscore").reindex(calendar)

    out_dir.mkdir(parents=True, exist_ok=True)
    overviews: dict[str, dict] = {}
    trade_totals: dict[str, list[int]] = {}
    print(f"开始写入 {len(names)} 只股票")
    for symbol in names:
        frame = frames[symbol]
        held = position[symbol].reindex(frame.index).fillna(0.0) if symbol in position.columns else pd.Series(0.0, index=frame.index)
        target = signal[symbol].reindex(frame.index).fillna(0.0) if symbol in signal.columns else pd.Series(0.0, index=frame.index)
        sleeve = sleeve_returns(frame["close"], frame.get("open", frame["close"]), held)
        equity = _equity(sleeve)
        buy_dates, sell_dates = _side_dates(frame.index, held)
        trades, by_year = _signed_round_trips(frame.index, held, equity)
        year_rows = _symbol_year_rows(frame, held.abs(), equity)
        for year, row in year_rows.items():
            overviews.setdefault(year, {})[symbol] = row
        for year, pair in by_year.items():
            bucket = trade_totals.setdefault(year, [0, 0])
            bucket[0] += pair[0]
            bucket[1] += pair[1]
        level = _equity(_stock_returns(frame["close"]))
        payload = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "symbol": symbol,
            "price_field": PRICE_FIELD,
            "dates": [timestamp.strftime("%Y-%m-%d") for timestamp in frame.index],
            "open": _series(frame.get("open")),
            "high": _series(frame.get("high")),
            "low": _series(frame.get("low")),
            "close": _series(frame["close"]),
            "return_index": _series(level),
            "volume": _series(frame.get("volume")),
            "algorithms": {
                ALGO_ID: _algorithm_body(
                    target,
                    held,
                    indicator[symbol].reindex(frame.index) if symbol in indicator.columns else pd.Series(np.nan, index=frame.index),
                    equity,
                    level,
                    sleeve,
                    buy_dates,
                    sell_dates,
                    trades,
                )
            },
        }
        _store_json(out_dir / f"{symbol}.json", payload, _write_json, _write_merged_json)
    years = {
        year: _overview_year(
            names,
            rows,
            tuple(trade_totals.get(year, (0, 0))),
            strategy_return.loc[strategy_return.index.year == int(year)],
            benchmark_return.loc[benchmark_return.index.year == int(year)],
        )
        for year, rows in sorted(overviews.items())
    }
    _store_json(
        out_dir / "index.json",
        {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "name": DATASET_NAME,
            "source": price_dir.relative_to(ROOT).as_posix(),
            "price_field": PRICE_FIELD,
            "symbols": names,
            "benchmark": {"id": INDEX_STEM, "name": BENCHMARK_NAME},
            "algorithms": [{"id": ALGO_ID, "name": ALGO_NAME, "note": ALGO_NOTE, "years": years}],
            "universe_note": UNIVERSE_NOTE,
        },
        _write_json,
        _write_merged_json,
        catalog=True,
    )
    _store_json(out_dir / f"{INDEX_STEM}.json", _index_payload(
        price_dir,
        calendar,
        strategy_return,
        benchmark_return,
        strategy_equity,
        benchmark_equity,
        SCHEMA_VERSION,
        PRICE_FIELD,
        INDEX_STEM,
        BENCHMARK_NAME,
        OOS_START,
        _series,
        _pair_stats,
        _trailing,
        _read_csv,
        _stock_returns,
        _equity,
    ), _write_json, _write_merged_json)
    full = _describe_returns(strategy_return.to_numpy(dtype=float), (0, 0))
    bench = _describe_returns(benchmark_return.to_numpy(dtype=float), (0, 0))
    print(
        f"{ALGO_NAME} 累计 {full['cumulative_return'] * 100:+.2f}%，"
        f"纳斯达克指数 {bench['cumulative_return'] * 100:+.2f}%"
    )
    print(f"已写入 {out_dir.relative_to(ROOT).as_posix()}")
    return out_dir


def _store_json(path: Path, payload: dict, write_json, write_merged, catalog: bool = False) -> None:
    """写入结果，并去掉已经停用的做空版本。"""
    import json

    write_merged(path, payload, catalog=catalog)
    saved = json.loads(path.read_text(encoding="utf-8"))
    if catalog:
        saved["algorithms"] = [item for item in saved.get("algorithms") or [] if item.get("id") not in RETIRED_IDS]
    else:
        algorithms = dict(saved.get("algorithms") or {})
        for key in RETIRED_IDS:
            algorithms.pop(key, None)
        saved["algorithms"] = algorithms
    write_json(path, saved)


def _algorithm_body(signal, position, indicator, equity, benchmark_equity, sleeve, buy_dates, sell_dates, trades) -> dict:
    from algorithms.index_enhance import _describe_returns, _series

    bench_return = benchmark_equity.pct_change().fillna(0.0)
    if len(bench_return):
        bench_return.iloc[0] = 0.0
    return {
        "id": ALGO_ID,
        "name": ALGO_NAME,
        "note": ALGO_NOTE,
        "params": {
            "side": SIDE_FRACTION,
            "adv_min": ADV_MIN,
            "adv_window": ADV_WINDOW,
            "execution": "open",
            "fee": 0.0,
            "forms": "10-K,10-Q",
        },
        "indicator": _series(indicator),
        "signal": _series(signal),
        "position": _series(position),
        "buy_dates": buy_dates,
        "sell_dates": sell_dates,
        "equity": _series(equity),
        "benchmark_equity": _series(benchmark_equity),
        "stats": {
            "strategy": _describe_returns(sleeve.to_numpy(dtype=float), trades),
            "benchmark": _describe_returns(bench_return.to_numpy(dtype=float), (0, 0), hide_win_rate=True),
        },
    }


def _index_payload(
    price_dir,
    calendar,
    strategy_return,
    benchmark_return,
    strategy_equity,
    benchmark_equity,
    schema_version,
    price_field,
    index_stem,
    benchmark_name,
    oos_start,
    series,
    pair_stats,
    trailing,
    read_csv,
    stock_returns,
    equity_curve,
) -> dict:
    raw = read_csv(price_dir / f"{index_stem}.csv").set_index("date").sort_index()
    frame = raw.reindex(calendar)
    close = frame["close"].astype("float64")
    years = {}
    for year in sorted(set(strategy_return.index.year)):
        mask = strategy_return.index.year == int(year)
        if bool(mask.any()):
            years[str(int(year))] = pair_stats(strategy_return.loc[mask], benchmark_return.loc[mask])
    since = strategy_return.index >= pd.Timestamp(oos_start)
    return {
        "schema_version": schema_version,
        "dataset_id": DATASET_ID,
        "symbol": index_stem,
        "name": benchmark_name,
        "kind": "index",
        "price_field": price_field,
        "dates": [timestamp.strftime("%Y-%m-%d") for timestamp in calendar],
        "open": series(frame.get("open")),
        "high": series(frame.get("high")),
        "low": series(frame.get("low")),
        "close": series(close),
        "return_index": series(equity_curve(stock_returns(close))),
        "volume": series(frame.get("volume")),
        "universe_note": UNIVERSE_NOTE,
        "algorithms": {
            ALGO_ID: {
                "id": ALGO_ID,
                "name": ALGO_NAME,
                "note": ALGO_NOTE,
                "params": {
                    "side": SIDE_FRACTION,
                    "adv_min": ADV_MIN,
                    "execution": "open",
                    "fee": 0.0,
                },
                "equity": series(strategy_equity.reindex(calendar)),
                "benchmark_equity": series(benchmark_equity.reindex(calendar)),
                "stats": pair_stats(strategy_return, benchmark_return),
                "stats_10y": pair_stats(*trailing(strategy_return, benchmark_return, 10)),
                "stats_since_2021": pair_stats(strategy_return.loc[since], benchmark_return.loc[since]),
                "years": years,
            }
        },
    }


def _long_prices(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for symbol, frame in frames.items():
        block = frame.reset_index()
        if "date" not in block.columns:
            block = block.rename(columns={block.columns[0]: "date"})
        block["ticker"] = symbol
        rows.append(block.loc[:, ["date", "ticker", "open", "high", "low", "close", "volume"]])
    return pd.concat(rows, ignore_index=True)


def _side_dates(index: pd.DatetimeIndex, held: pd.Series) -> tuple[list[str], list[str]]:
    previous = held.shift(1).fillna(0.0)
    current = held.fillna(0.0)
    buys = index[((previous <= 1e-8) & (current > 1e-8)) | ((previous < -1e-8) & (current >= -1e-8))]
    sells = index[((previous >= -1e-8) & (current < -1e-8)) | ((previous > 1e-8) & (current <= 1e-8))]
    return (
        [timestamp.strftime("%Y-%m-%d") for timestamp in buys],
        [timestamp.strftime("%Y-%m-%d") for timestamp in sells],
    )


def _signed_round_trips(index: pd.DatetimeIndex, held: pd.Series, equity: pd.Series) -> tuple[tuple[int, int], dict[str, tuple[int, int]]]:
    """平仓或从多头直接翻成空头时结案。现金增加记为赢，借券费已经在净值里。"""
    full_trades = 0
    full_wins = 0
    by_year: dict[str, list[int]] = {}
    entry_cash = None
    previous = 0.0
    values = equity.reindex(index).to_numpy(dtype=float)
    years = index.year.to_numpy()
    weights = held.reindex(index).fillna(0.0).to_numpy(dtype=float)

    def add(year_key: str, won: bool) -> None:
        nonlocal full_trades, full_wins
        bucket = by_year.setdefault(year_key, [0, 0])
        bucket[0] += 1
        bucket[1] += int(won)
        full_trades += 1
        full_wins += int(won)

    for i, current in enumerate(weights):
        current_sign = 1.0 if current > 1e-8 else (-1.0 if current < -1e-8 else 0.0)
        if previous == 0.0 and current_sign != 0.0:
            entry_cash = float(values[0] if i == 0 else values[i - 1])
        elif previous != 0.0 and current_sign != previous and entry_cash is not None:
            add(str(int(years[i])), float(values[i]) > entry_cash)
            entry_cash = None if current_sign == 0.0 else float(values[i])
        previous = current_sign
    return (full_trades, full_wins), {year: (pair[0], pair[1]) for year, pair in by_year.items()}


def main() -> None:
    export_dataset()


if __name__ == "__main__":
    main()
