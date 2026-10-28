"""双均线趋势：SMA(20) 上穿 SMA(60) 开多，下穿平仓。

`python -m algorithms.ma_cross` 把结果合并进 `outdata/equity-2018/`，
每只股票只替换 `ma_cross` 这一条，RSI 等已有算法原样保留。

信号在当日收盘确定，`engine` 用上一期仓位计算收益，不使用未来价格。
界面第二页只画一条指标线，这里画快线；慢线由 `params.slow` 与 `note` 说明。
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

FAST_WINDOW = 20
SLOW_WINDOW = 60
FEE = 0.001

ALGORITHM = {
    "id": "ma_cross",
    "name": "双均线趋势",
    "note": "SMA(20) 上穿 SMA(60) 开多，下穿平仓",
}


def compute_mas(
    close: pd.Series,
    fast: int = FAST_WINDOW,
    slow: int = SLOW_WINDOW,
) -> tuple[pd.Series, pd.Series]:
    """快线与慢线的简单移动平均。窗口不足处为 NaN。"""
    if fast <= 0 or slow <= 0 or fast >= slow:
        raise ValueError("需要 0 < fast < slow")
    values = close.astype("float64")
    return (
        values.rolling(window=fast, min_periods=fast).mean().rename("fast"),
        values.rolling(window=slow, min_periods=slow).mean().rename("slow"),
    )


def ma_position(close: pd.Series, fast: int = FAST_WINDOW, slow: int = SLOW_WINDOW) -> pd.Series:
    """快线在慢线上方持多，否则空仓。两条线都没算出来以前保持空仓。"""
    fast_ma, slow_ma = compute_mas(close, fast, slow)
    above = (fast_ma > slow_ma).to_numpy(dtype="bool")
    ready = (~fast_ma.isna() & ~slow_ma.isna()).to_numpy(dtype="bool")
    holding = (ready & above).astype("float64")
    return pd.Series(holding, index=close.index, dtype=float, name="signal")


def build(close: pd.Series) -> tuple[dict, dict, pd.Series, pd.Series]:
    fast_ma, _slow_ma = compute_mas(close)
    signal = ma_position(close)
    return contract.build_payload(
        ALGORITHM,
        close=close,
        signal=signal,
        indicator=fast_ma,
        params={"fast": FAST_WINDOW, "slow": SLOW_WINDOW, "fee": FEE},
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
    parser = argparse.ArgumentParser(description="双均线趋势，把结果合并进 outdata")
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
