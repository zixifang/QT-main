"""把 OHLC 按 adjusted_close / close 缩放到复权价。成交量保持原始股数。"""

from __future__ import annotations

import pandas as pd

JUMP_LIMIT = 0.5


def adjust_ohlcv(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """返回复权后的行情，以及原始收盘价上疑似拆股的标记。

    因子为 1 时价格不变。当前纳斯达克文件的 Adj Close 与 Close 相同，结果仍是未复权价。
    跳变只做标记，不把收益或因子改成 0。
    """
    close = pd.to_numeric(frame["close"], errors="coerce").astype(float)
    if "adjusted_close" in frame.columns:
        adjusted = pd.to_numeric(frame["adjusted_close"], errors="coerce").astype(float)
    else:
        adjusted = close.copy()
    factor = (adjusted / close).where(close != 0)
    factor = factor.where(adjusted.notna(), 1.0).fillna(1.0)
    out = frame.copy()
    for column in ("open", "high", "low", "close"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype(float) * factor
    out["adjusted_close"] = out["close"]
    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce").astype(float)
    raw_return = close.pct_change(fill_method=None)
    jumps = raw_return.abs() > JUMP_LIMIT
    return out, jumps.fillna(False)
