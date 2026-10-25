"""MACD 趋势跟踪。

规则：
- 空仓时，MACD 柱状图由负转正，且 MACD 线在零轴上方，次日收盘开多；
- 持多时，柱状图转负，或收盘价跌破 ATR 跟踪止损，次日收盘平仓。

信号只使用当日及以前的收盘价、最高价、最低价，engine 再把仓位后移一天。

运行：python -m algorithms.macd_trend
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from algorithms.technical_common import (
    DEFAULT_FEE,
    ema,
    export_algorithm,
    single_dataset_parser,
    wilder_atr,
)

ALGORITHM = {
    "id": "macd_trend",
    "name": "MACD 趋势跟踪",
    "note": "MACD 柱转正且在零轴上方开多；柱转负或跌破 ATR 跟踪止损平仓",
}

PARAMS = {
    "fast": 12,
    "slow": 26,
    "signal": 9,
    "atr_window": 14,
    "atr_mult": 3.0,
    "fee": DEFAULT_FEE,
}


def macd_lines(
    frame: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """返回 MACD 线、信号线、柱状图。"""
    close = pd.to_numeric(frame["close"], errors="coerce").astype("float64")
    macd = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd, signal)
    return macd, signal_line, macd - signal_line


def compute(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """返回 (目标仓位, 指标)。指标用 MACD 柱状图，方便看动能反转。"""
    close = pd.to_numeric(frame["close"], errors="coerce").astype("float64")
    macd, signal_line, hist = macd_lines(frame)
    atr = wilder_atr(frame, window=PARAMS["atr_window"])

    close_values = close.to_numpy(dtype="float64")
    macd_values = macd.to_numpy(dtype="float64")
    hist_values = hist.to_numpy(dtype="float64")
    atr_values = atr.to_numpy(dtype="float64")

    position = np.zeros(len(frame), dtype="float64")
    holding = 0
    peak = np.nan
    for index in range(len(frame)):
        if not np.isfinite(hist_values[index]) or not np.isfinite(atr_values[index]):
            holding = 0
            peak = np.nan
            position[index] = 0.0
            continue
        if holding == 0:
            if hist_values[index] > 0.0 and macd_values[index] > 0.0:
                holding = 1
                peak = close_values[index]
        else:
            peak = max(peak, close_values[index])
            stop = peak - PARAMS["atr_mult"] * atr_values[index]
            if hist_values[index] < 0.0 or close_values[index] < stop:
                holding = 0
                peak = np.nan
        position[index] = float(holding)

    return (
        pd.Series(position, index=frame.index, name="signal"),
        pd.Series(hist_values, index=frame.index, name="indicator"),
    )


def main() -> None:
    parser = single_dataset_parser("MACD 趋势跟踪：写 outdata JSON")
    args = parser.parse_args()
    params = {**PARAMS, "fee": args.fee}
    export_algorithm(
        ALGORITHM,
        params,
        compute,
        fee=args.fee,
        dataset_folders=args.datasets,
    )


if __name__ == "__main__":
    main()
