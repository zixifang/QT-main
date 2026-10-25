"""12-1 时间序列动量。

规则：
- 动量 = 12 个月前的收盘价到 1 个月前的收盘价的涨幅，跳过最近 21 个交易日；
- 空仓时，动量大于 0 且收盘价在 200 日均线上方，次日收盘开多；
- 持多时，动量转负或收盘价跌破 200 日均线，次日收盘平仓。

这是 Moskowitz、Ooi、Pedersen 时间序列动量的一种常见实现。信号只用到
当日及以前的收盘价，engine 再把仓位后移一天。

运行：python -m algorithms.momentum_12_1
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from algorithms.technical_common import (
    DEFAULT_FEE,
    export_algorithm,
    single_dataset_parser,
)

ALGORITHM = {
    "id": "momentum_12_1",
    "name": "12-1 时间序列动量",
    "note": "12-1 动量大于 0 且收盘在 200 日均线上方开多；动量转负或跌破均线平仓",
}

PARAMS = {
    "lookback": 252,
    "skip": 21,
    "trend_window": 200,
    "fee": DEFAULT_FEE,
}


def momentum(
    frame: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
) -> pd.Series:
    """12-1 动量。前 lookback 根 K 线没有值。"""
    close = pd.to_numeric(frame["close"], errors="coerce").astype("float64")
    return close.shift(skip) / close.shift(lookback) - 1.0


def compute(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    close = pd.to_numeric(frame["close"], errors="coerce").astype("float64")
    mom = momentum(frame, lookback=PARAMS["lookback"], skip=PARAMS["skip"])
    trend = close.rolling(
        window=PARAMS["trend_window"], min_periods=PARAMS["trend_window"]
    ).mean()

    close_values = close.to_numpy(dtype="float64")
    mom_values = mom.to_numpy(dtype="float64")
    trend_values = trend.to_numpy(dtype="float64")

    position = np.zeros(len(frame), dtype="float64")
    holding = 0
    for index in range(len(frame)):
        if not np.isfinite(mom_values[index]) or not np.isfinite(trend_values[index]):
            holding = 0
            position[index] = 0.0
            continue
        if holding == 0:
            if mom_values[index] > 0.0 and close_values[index] > trend_values[index]:
                holding = 1
        elif mom_values[index] < 0.0 or close_values[index] < trend_values[index]:
            holding = 0
        position[index] = float(holding)

    return (
        pd.Series(position, index=frame.index, name="signal"),
        pd.Series(mom_values * 100.0, index=frame.index, name="indicator"),
    )


def main() -> None:
    parser = single_dataset_parser("12-1 时间序列动量：写 outdata JSON")
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
