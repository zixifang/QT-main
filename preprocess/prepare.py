"""截面处理和时点对齐。

信号日严格晚于公告日。已知信息的截止日用来挡住 IC 加权里的未来收益。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from services.errors import DataGapError, LookaheadError

MAD_SCALE = 1.4826


def signal_date(announce, calendar: pd.DatetimeIndex) -> pd.Timestamp:
    """公告日之后的第一个交易日。公告当天不能用。"""
    day = pd.Timestamp(announce)
    if day.tzinfo is not None:
        day = day.tz_localize(None)
    day = day.normalize()
    dates = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values().normalize()
    later = dates[dates > day]
    if len(later) == 0:
        raise LookaheadError("公告日之后没有交易日，不能使用这条财务数据")
    return pd.Timestamp(later[0])


def known_cutoff(as_of, calendar, horizon: int) -> pd.Timestamp:
    """截止这一天，向前 horizon 个交易日的收益才刚刚走完，更晚的 IC 还不能用。"""
    dates = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values().normalize()
    loc = int(dates.searchsorted(pd.Timestamp(as_of).normalize(), side="right") - 1)
    if loc < 0:
        raise DataGapError("区间里没有交易日")
    known = loc - int(horizon)
    if known < 0:
        return pd.Timestamp(dates[0]) - pd.Timedelta(days=1)
    return pd.Timestamp(dates[known])


def known_history(frame: pd.DataFrame, as_of, calendar, horizon: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    cutoff = known_cutoff(as_of, calendar, horizon)
    return frame.loc[pd.DatetimeIndex(frame.index).normalize() <= cutoff]


def clean_returns(close: pd.Series, jump_limit: float) -> pd.Series:
    """日收益。绝对值超过 jump_limit 的日子记为 0，原始收盘价不改。"""
    returns = close.astype(float).pct_change(fill_method=None)
    return returns.mask(returns.abs() > jump_limit, 0.0)


def jump_dates(close: pd.Series, jump_limit: float) -> list[str]:
    returns = close.astype(float).pct_change(fill_method=None)
    hit = returns.abs() > jump_limit
    return [ts.strftime("%Y-%m-%d") for ts in returns.index[hit.fillna(False)]]


def period_ends(index, freq: str) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(index)).sort_values().normalize()
    if len(dates) == 0:
        return dates
    last = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).tail(1)
    ends = pd.DatetimeIndex(last.to_numpy())
    if freq == "Q":
        ends = ends[ends.month.isin([3, 6, 9, 12])]
    elif freq != "M":
        raise DataGapError("再平衡只支持每月或每季")
    return pd.DatetimeIndex(ends)


def winsorize_frame(panel: pd.DataFrame, width: float = 3.0) -> pd.DataFrame:
    """截面中位数 ± width * 1.4826 * MAD。"""
    out = panel.astype(float).copy()
    for day in out.index:
        values = out.loc[day].to_numpy(dtype=float)
        if values.size == 0 or np.isnan(values).all():
            continue
        med = np.nanmedian(values)
        mad = np.nanmedian(np.abs(values - med))
        if not np.isfinite(mad) or mad == 0 or not np.isfinite(med):
            continue
        bound = width * MAD_SCALE * mad
        out.loc[day] = np.clip(values, med - bound, med + bound)
    return out


def zscore_frame(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.astype(float).copy()
    for day in out.index:
        values = out.loc[day].to_numpy(dtype=float)
        if values.size == 0 or np.isnan(values).all():
            continue
        sd = np.nanstd(values, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            out.loc[day] = np.where(np.isfinite(values), 0.0, np.nan)
            continue
        mu = np.nanmean(values)
        out.loc[day] = (values - mu) / sd
    return out


def neutralize_frame(panel: pd.DataFrame, risks: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    """截面回归取残差。样本不够时保持标准化结果，并留下告警。"""
    warnings: list[str] = []
    out = panel.astype(float).copy()
    usable: dict[str, pd.DataFrame] = {}
    for name, frame in risks.items():
        if frame is None or frame.empty or frame.isna().all().all():
            warnings.append(f"{name} 没有数据，中性化没有使用它")
            continue
        usable[name] = frame.astype(float)
    if not usable:
        warnings.append("没有可用的风险因子，中性化已跳过")
        return out, warnings
    if out.shape[1] <= len(usable) + 1:
        warnings.append(
            f"股票池只有 {out.shape[1]} 只，做不了 {len(usable)} 个风险因子的中性化，本次跳过"
        )
        return out, warnings
    names = list(usable)
    skipped = 0
    for day in out.index:
        y = out.loc[day]
        columns = []
        ready = True
        for name in names:
            risk = usable[name]
            if day not in risk.index:
                ready = False
                break
            columns.append(risk.loc[day].reindex(y.index))
        if not ready:
            skipped += 1
            continue
        design = pd.concat(columns, axis=1)
        mask = y.notna() & design.notna().all(axis=1)
        if int(mask.sum()) <= len(names) + 1:
            skipped += 1
            continue
        yv = y.loc[mask].to_numpy(dtype=float)
        xv = np.column_stack([np.ones(int(mask.sum())), design.loc[mask].to_numpy(dtype=float)])
        coef, *_ = np.linalg.lstsq(xv, yv, rcond=None)
        resid = pd.Series(yv - xv @ coef, index=y.loc[mask].index)
        out.loc[day, resid.index] = resid.to_numpy()
    if skipped:
        warnings.append(f"{skipped} 个交易日风险因子不齐，这些日子保留标准化后的值")
    return out, warnings
