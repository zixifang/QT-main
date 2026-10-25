"""唐奇安通道突破（简化海龟规则）。

规则：
- 空仓时，收盘价突破前 20 个交易日的最高价，次日收盘开多；
- 持多时，收盘价跌破前 10 个交易日的最低价，或跌破 ATR 跟踪止损，次日收盘平仓。

指标是收盘价在 20 日唐奇安通道中的位置：0 是下轨，100 是上轨，
大于 100 表示当天收盘创出 20 日新高。

运行：python -m algorithms.donchian_breakout
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from algorithms.technical_common import (
    DEFAULT_FEE,
    export_algorithm,
    single_dataset_parser,
    wilder_atr,
)

ALGORITHM = {
    "id": "donchian_breakout",
    "name": "唐奇安通道突破",
    "note": "收盘突破20日最高价开多；跌破10日最低价或 ATR 跟踪止损平仓",
}

PARAMS = {
    "entry_window": 20,
    "exit_window": 10,
    "atr_window": 14,
    "atr_mult": 2.5,
    "fee": DEFAULT_FEE,
}


def donchian_channels(
    frame: pd.DataFrame,
    entry_window: int = 20,
    exit_window: int = 10,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """返回 (入场通道上轨, 出场通道下轨, 20日下轨, 通道位置)。"""
    high = pd.to_numeric(frame["high"], errors="coerce").astype("float64")
    low = pd.to_numeric(frame["low"], errors="coerce").astype("float64")
    close = pd.to_numeric(frame["close"], errors="coerce").astype("float64")

    upper = high.rolling(window=entry_window, min_periods=entry_window).max().shift(1)
    lower_exit = low.rolling(window=exit_window, min_periods=exit_window).min().shift(1)
    lower_entry = low.rolling(window=entry_window, min_periods=entry_window).min().shift(1)
    span = (upper - lower_entry).replace(0.0, np.nan)
    channel_position = 100.0 * (close - lower_entry) / span
    return upper, lower_exit, lower_entry, channel_position


def compute(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    close = pd.to_numeric(frame["close"], errors="coerce").astype("float64")
    upper, lower_exit, _, channel_position = donchian_channels(
        frame,
        entry_window=PARAMS["entry_window"],
        exit_window=PARAMS["exit_window"],
    )
    atr = wilder_atr(frame, window=PARAMS["atr_window"])

    close_values = close.to_numpy(dtype="float64")
    upper_values = upper.to_numpy(dtype="float64")
    lower_values = lower_exit.to_numpy(dtype="float64")
    atr_values = atr.to_numpy(dtype="float64")

    position = np.zeros(len(frame), dtype="float64")
    holding = 0
    peak = np.nan
    for index in range(len(frame)):
        if not np.isfinite(upper_values[index]) or not np.isfinite(atr_values[index]):
            holding = 0
            peak = np.nan
            position[index] = 0.0
            continue
        if holding == 0:
            if close_values[index] > upper_values[index]:
                holding = 1
                peak = close_values[index]
        else:
            peak = max(peak, close_values[index])
            stop = peak - PARAMS["atr_mult"] * atr_values[index]
            if (
                np.isfinite(lower_values[index])
                and close_values[index] < lower_values[index]
            ) or close_values[index] < stop:
                holding = 0
                peak = np.nan
        position[index] = float(holding)

    return (
        pd.Series(position, index=frame.index, name="signal"),
        pd.Series(channel_position.to_numpy(dtype="float64"), index=frame.index, name="indicator"),
    )


def main() -> None:
    parser = single_dataset_parser("唐奇安通道突破：写 outdata JSON")
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
