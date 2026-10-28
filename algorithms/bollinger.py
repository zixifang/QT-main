"""布林带均值回归：收盘价跌破下轨（z<-2）开多，回到中轨（z>=0）平仓。

`python -m algorithms.bollinger` 把结果合并进 `outdata/equity-2018/`，
每只股票只替换 `bollinger` 这一条，别的算法原样保留。

界面第二页画的是 z 分数 `(close - MID) / SD`，上下轨正好是 ±k，
所以 `params.oversold` / `params.overbought` 是真实存在的两条轨道，不是编出来的线。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from algorithms import contract, rsi
from engine import prepare_close

DATA_DIR = contract.DATA_ROOT / "Equity-Portfolio-Management-master"
DATASET_ID = "equity-2018"
DATASET_NAME = "2018 股票组合样本"

WINDOW = 20
K = 2.0
FEE = 0.001

ALGORITHM = {
    "id": "bollinger",
    "name": "布林带均值回归",
    "note": "收盘价跌破下轨（z<-2）开多，回到中轨（z>=0）平仓",
}


def compute_bands(
    close: pd.Series,
    window: int = WINDOW,
    k: float = K,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """中轨与上下轨。标准差用总体标准差，与经典布林带一致。"""
    if window < 2:
        raise ValueError("window 至少为 2")
    if k <= 0:
        raise ValueError("k 必须为正")
    values = close.astype("float64")
    middle = values.rolling(window=window, min_periods=window).mean()
    deviation = values.rolling(window=window, min_periods=window).std(ddof=0)
    return (
        middle.rename("middle"),
        (middle + k * deviation).rename("upper"),
        (middle - k * deviation).rename("lower"),
    )


def compute_zscore(close: pd.Series, window: int = WINDOW, k: float = K) -> pd.Series:
    """价格相对中轨的标准化偏离。等于 %B 的线性变换，横轴同一套刻度。"""
    values = close.astype("float64")
    middle, _upper, _lower = compute_bands(values, window, k)
    deviation = values.rolling(window=window, min_periods=window).std(ddof=0)
    zscore = (values - middle) / deviation.replace(0.0, np.nan)
    return pd.Series(zscore.to_numpy(dtype=float), index=close.index, name="zscore")


def bollinger_position(close: pd.Series, window: int = WINDOW, k: float = K) -> pd.Series:
    """跌破下轨开多，回到中轨平仓。指标缺失时维持当前仓位。

    任意一天的仓位只取决于最近一次开仓/平仓事件，所以把事件日标成 1 / 0、
    其余标成缺失再向前填充，与逐行状态机等价，但不必写 Python 循环。
    """
    zscore = compute_zscore(close, window, k).to_numpy(dtype=float)
    state = np.full(zscore.size, np.nan, dtype=float)
    state[zscore < -k] = 1.0
    state[zscore >= 0.0] = 0.0
    holding = pd.Series(state).ffill().fillna(0.0).to_numpy(dtype=float)
    return pd.Series(holding, index=close.index, dtype=float, name="signal")


def build(close: pd.Series) -> tuple[dict, dict, pd.Series, pd.Series]:
    signal = bollinger_position(close)
    return contract.build_payload(
        ALGORITHM,
        close=close,
        signal=signal,
        indicator=compute_zscore(close),
        params={"window": WINDOW, "k": K, "oversold": -K, "overbought": K, "fee": FEE},
        fee=FEE,
    )


def export(data_dir: Path = DATA_DIR, dataset_id: str = DATASET_ID, dataset_name: str = DATASET_NAME) -> Path | None:
    return contract.export_dataset(
        ALGORITHM,
        build,
        data_dir,
        dataset_id,
        dataset_name,
        symbol_order=rsi.DATASETS.get(data_dir.name, {}).get("symbol_order"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="布林带均值回归，把结果合并进 outdata")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--dataset-name", default=None)
    args = parser.parse_args()
    meta = rsi.DATASETS.get(args.data_dir.name, {})
    dataset_id = args.dataset_id or meta.get("dataset_id", args.data_dir.name)
    dataset_name = args.dataset_name or meta.get("name", dataset_id)
    out_dir = export(args.data_dir, dataset_id, dataset_name)
    if out_dir is None:
        raise SystemExit(f"{args.data_dir} 里没有可读的单只股票 CSV")
    print(f"已把 {ALGORITHM['id']} 合并进 {out_dir.relative_to(contract.ROOT).as_posix()}")


if __name__ == "__main__":
    main()
