"""按同一套日期，给每个数据集写出 Transformer 的实验集和测试集。

运行：python -m algorithms.transformer_splits

实验集是训练集加验证集，早停只用验证集。测试集不参与训练，也不参与早停。
不重新训练模型。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import numpy as np
import pandas as pd

from algorithms.multimodal_transformer import ALL_FACTOR_COLUMNS, StockMultimodalDataset
from algorithms.train_multimodal import (
    JUMP_LIMIT,
    TRAIN_END,
    VAL_END,
    WINDOW,
    HORIZON,
    build_split,
    load_annual_factors,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "config" / "transformer_splits.json"

DATASETS = (
    {
        "id": "nasdaq-xbrl",
        "name": "纳斯达克XBRL",
        "path": ROOT / "data" / "nasdaq100" / "nasdaq100.csv",
        "fundamentals": True,
        "skip": frozenset({"NDX", "GOOG"}),
    },
    {
        "id": "nasdaq100",
        "name": "纳斯达克100",
        "path": ROOT / "data" / "nasdaq100" / "nasdaq100.csv",
        "fundamentals": False,
        "skip": frozenset({"NDX", "GOOG"}),
    },
    {
        "id": "tech-us-10y",
        "name": "科技美股近十年日线",
        "path": ROOT / "data" / "tech-us-10y" / "tech_us_10y.csv",
        "fundamentals": False,
        "skip": frozenset(),
    },
    {
        "id": "manufacturing-us-10y",
        "name": "美国制造业近十年日线",
        "path": ROOT / "data" / "manufacturing-us-10y" / "manufacturing_us_10y.csv",
        "fundamentals": False,
        "skip": frozenset(),
    },
    {
        "id": "healthcare-us-10y",
        "name": "美国医疗健康近十年日线",
        "path": ROOT / "data" / "healthcare-us-10y" / "healthcare_us_10y.csv",
        "fundamentals": False,
        "skip": frozenset(),
    },
    {
        "id": "us-agri",
        "name": "美国农业股",
        "path": ROOT / "data" / "us-agri" / "us_agri_stocks_10y.csv",
        "fundamentals": False,
        "skip": frozenset(),
        "train_ratio": 0.8,
    },
)


def listed_symbols(dataset_id: str) -> set[str] | None:
    path = ROOT / "outdata" / dataset_id / "index.json"
    if not path.exists():
        return None
    catalog = json.loads(path.read_text(encoding="utf-8"))
    symbols = {str(item) for item in catalog.get("symbols") or []}
    return symbols or None


def load_price_file(path: Path, skip: frozenset[str], keep: set[str] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.rename(columns={column: str(column).strip() for column in frame.columns})
    rename = {"Ticker": "ticker", "Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    frame = frame.rename(columns=rename)
    frame["ticker"] = frame["ticker"].astype(str).str.strip()
    frame = frame[~frame["ticker"].isin(skip)].copy()
    if keep is not None:
        frame = frame[frame["ticker"].isin(keep)].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ("open", "high", "low", "close", "volume"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    previous = frame.groupby("ticker", sort=False)["close"].shift(1)
    jump = (frame["close"] / previous - 1.0).abs() > JUMP_LIMIT
    frame.loc[jump.fillna(False), ["open", "high", "low", "close", "volume"]] = np.nan
    return frame.reset_index(drop=True)


def empty_factors() -> pd.DataFrame:
    return pd.DataFrame(columns=["ticker", "filed_date", "period_end", *ALL_FACTOR_COLUMNS])


def ratio_train_end(calendar: pd.DatetimeIndex, window: int, horizon: int, train_ratio: float = 0.8) -> str:
    """按决策日先后切。后 (1-train_ratio) 是测试，标签跨过分界的几天两边都不用。"""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio 要落在 (0, 1) 内")
    count_days = len(calendar)
    first = window - 1
    last = count_days - horizon - 1
    count = last - first + 1
    if count < 2:
        raise ValueError("决策日太少，无法按比例切开")
    test_count = int(round(count * (1.0 - train_ratio)))
    test_count = min(max(test_count, 1), count - 1)
    test_start = last - test_count + 1
    train_pos = test_start - 1
    return str(pd.Timestamp(calendar[train_pos]).date())


def describe_dataset(spec: dict) -> dict:
    prices = load_price_file(spec["path"], spec["skip"], listed_symbols(spec["id"]))
    factors = load_annual_factors() if spec["fundamentals"] else empty_factors()
    dataset = StockMultimodalDataset(
        prices,
        factors,
        window=WINDOW,
        horizon=HORIZON,
        announcement_lag=0,
        require_fundamental=bool(spec["fundamentals"]),
    )
    ratio = spec.get("train_ratio")
    if ratio:
        train_end = ratio_train_end(dataset.calendar, dataset.window, dataset.horizon, float(ratio))
        summary = build_split(dataset, train_end, train_end)
    else:
        train_end = TRAIN_END
        summary = build_split(dataset, TRAIN_END, VAL_END)
    price_start = str(prices["date"].min().date())
    price_end = str(prices["date"].max().date())
    record = {
        "id": spec["id"],
        "name": spec["name"],
        "price_start": price_start,
        "price_end": price_end,
        "symbols": int(prices["ticker"].nunique()),
        "fundamentals": "10-K" if spec["fundamentals"] else "缺失",
        "experiment": summary["experiment"],
        "test": summary["test"],
        "embargo": summary["embargo"],
    }
    if ratio:
        record["split"] = "ratio"
        record["train_ratio"] = float(ratio)
        record["test_ratio"] = round(1.0 - float(ratio), 2)
        record["train_end"] = train_end
        record["note"] = "按决策日先后切，后20%为测试。标签跨过分界的交易日两边都不用，因此训练决策日略少于80%。没有单独的验证集。"
    return record


def write_splits(path: Path = OUT_PATH) -> dict:
    payload = {
        "window": WINDOW,
        "horizon": HORIZON,
        "train_end": TRAIN_END,
        "val_end": VAL_END,
        "experiment": "训练集加验证集。早停只用验证集。",
        "test_rule": "决策日晚于验证截止日，且标签已在行情里。不参与训练，也不参与早停。",
        "datasets": [describe_dataset(spec) for spec in DATASETS],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    payload = write_splits()
    for item in payload["datasets"]:
        train = item["experiment"]["train"]
        val = item["experiment"]["val"]
        test = item["test"]
        print(
            f"{item['id']} 实验训练 {train['start']} 至 {train['end']}，"
            f"验证 {val['start']} 至 {val['end']}，"
            f"测试 {test['start']} 至 {test['end']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
