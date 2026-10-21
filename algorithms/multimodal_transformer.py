"""多模态 Transformer：用过去 T 日量价和公告日已可见的基本面预测股票收益率。

信息集（决策日为窗口最后一天 t）：

- 量价只使用 t-T+1 到 t，标签使用 t 到 t+horizon 的收盘价。
- 财报按可用日对齐。可用日是 filed_date 当天或之后的第一个交易日，
  再向后移动 announcement_lag 个交易日。period_end 只在同一天有多份财报时
  用来决定留下哪一份，不能代替公告日。
- 应计利润、资产增长、净股票发行在截面标准化之前取负，之后数值越高越好。

依赖：pandas、numpy、PyTorch 2。

    python -m algorithms.multimodal_transformer
"""

from __future__ import annotations

import inspect
import math
import sys
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

# Windows 上必须先加载 PyTorch。若 NumPy 的 OpenMP 运行库先被加载，
# c10.dll 会在初始化时失败。调用方若已经 import 了 numpy，请先 import torch。
if sys.platform == "win32" and "numpy" in sys.modules and "torch" not in sys.modules:
    raise ImportError("在 Windows 上请先 import torch，再 import numpy 或 pandas，然后再导入本模块。")

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

import numpy as np
import pandas as pd

# 连续基本面。前五个越高越好；后三个在进入网络前取负，原始定义是越低越好。
CONTINUOUS_COLUMNS: tuple[str, ...] = (
    "ROE",
    "ROA",
    "GrossProfitability",
    "BM",
    "AssetTurnover",
    "Accruals",
    "AssetGrowth",
    "NetStockIssues",
)
LOW_IS_BETTER: tuple[str, ...] = ("Accruals", "AssetGrowth", "NetStockIssues")

# Piotroski 九个 0/1 分项，顺序与 algorithms.xbrl_factors 一致。
BINARY_COLUMNS: tuple[str, ...] = (
    "F_ROA",
    "F_CFO",
    "F_dROA",
    "F_ACCRUAL",
    "F_dLEVER",
    "F_dLIQUID",
    "F_EQ_ISSUE",
    "F_dGROSS_MARGIN",
    "F_dASSET_TURN",
)
FSCORE_COLUMN = "FScore"
FSCORE_MAX = 9
ALL_FACTOR_COLUMNS: tuple[str, ...] = CONTINUOUS_COLUMNS + BINARY_COLUMNS + (FSCORE_COLUMN,)

# 每个交易日一条。价格水平已消掉，方便不同股价的股票放进同一个截面。
OHLCV_FEATURES: tuple[str, ...] = (
    "ret_1",  # 相对前一交易日收盘的收益率
    "ret_oc",  # 当日开盘到收盘
    "hl_range",  # (高-低) / 收盘
    "close_position",  # 收盘在当日振幅中的位置
    "upper_shadow",
    "lower_shadow",
    "log_volume",
    "volume_chg",
)

_PRICE_ALIASES = {
    "date": "date",
    "trade_date": "date",
    "datetime": "date",
    "ticker": "ticker",
    "symbol": "ticker",
    "code": "ticker",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "vol": "volume",
}
_FUND_KEY_ALIASES = {
    "ticker": "ticker",
    "symbol": "ticker",
    "code": "ticker",
    "filed_date": "filed_date",
    "filing_date": "filed_date",
    "announcement_date": "filed_date",
    "period_end": "period_end",
    "report_period": "period_end",
}


def _factor_aliases() -> dict[str, str]:
    mapping = {name.lower(): name for name in ALL_FACTOR_COLUMNS}
    mapping["f_score"] = FSCORE_COLUMN
    return mapping


def _standardize_columns(
    frame: pd.DataFrame,
    aliases: dict[str, str],
    required: Sequence[str],
) -> pd.DataFrame:
    rename: dict[object, str] = {}
    for column in frame.columns:
        key = str(column).strip().lower().replace(" ", "_")
        if key in aliases:
            rename[column] = aliases[key]
    out = frame.rename(columns=rename).copy()
    duplicated = [str(name) for name in out.columns[out.columns.duplicated()].unique()]
    if duplicated:
        raise ValueError("列名冲突: " + ", ".join(duplicated))
    missing = [name for name in required if name not in out.columns]
    if missing:
        raise KeyError("缺少列: " + ", ".join(missing))
    return out


def _as_naive_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_localize(None)
    return parsed.astype("datetime64[ns]")


def cross_sectional_zscore(
    frame: pd.DataFrame,
    columns: Sequence[str],
    by: str = "date",
    clip: float | None = 5.0,
) -> pd.DataFrame:
    """按日期对连续变量做截面 Z-score。缺失值不参与均值和标准差。

    标准差使用总体口径（除以当日有效股票数 N）。当日有效样本少于 2，
    或标准差接近 0 时，有观测的位置记 0，缺失位置保持为空。
    这样“刚好等于截面均值”和“当天没有这条数据”不会混成同一个数；
    后者留给 fill_missing_with_mask 去补 0 并单独记掩码。
    """
    out = frame.copy()
    if out.empty or not columns:
        return out
    missing = [column for column in columns if column not in out.columns]
    if missing:
        raise KeyError("缺少列: " + ", ".join(missing))
    if by not in out.columns:
        raise KeyError(f"缺少分组列: {by}")

    group_key = out[by]
    for column in columns:
        values = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        grouped = values.groupby(group_key, sort=False)
        mean = grouped.transform("mean")
        count = grouped.transform("count")
        second = values.pow(2).groupby(group_key, sort=False).transform("mean")
        variance = (second - mean.pow(2)).clip(lower=0.0)
        std = variance.pow(0.5)
        present = values.notna()
        usable = present & (count >= 2) & (std > 1e-12)
        z = (values - mean) / std.where(std > 1e-12)
        z = z.where(usable, 0.0)
        z = z.where(present, np.nan)
        if clip is not None:
            z = z.clip(lower=-float(clip), upper=float(clip))
        out[column] = z
    return out


def fill_missing_with_mask(
    frame: pd.DataFrame,
    columns: Sequence[str],
    fill_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """把非有限值视为缺失：数值填 fill_value，并返回同形状的观测掩码。

    截面 Z-score 之后，0 表示“等于当日均值”。缺失如果也填 0，模型会把
    没公告的股票看成平均公司，所以掩码必须和数值一起送进线性层。
    返回 (filled[float32], observed[bool])，行顺序与 frame 一致。
    """
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError("缺少列: " + ", ".join(missing))
    values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    array = values.to_numpy(dtype=np.float64)
    observed = np.isfinite(array)
    filled = np.where(observed, array, fill_value).astype(np.float32)
    return filled, observed


def encode_ordinal(series: pd.Series, upper: int) -> np.ndarray:
    """把 0 到 upper 的评分编成 Embedding 下标。0 号下标固定留给缺失。

    F-Score 的 upper 为 9，因此 0 分存成 1，9 分存成 10，Embedding 大小为 11。
    二元分项的 upper 为 1：缺失=0，得分 0=1，得分 1=2。
    项目里的 FScore 是“有数据的分项之和”，这里保留原值，不把缺失分项补成 0 再重算。
    """
    if upper < 0:
        raise ValueError("评分上界不能为负")
    if pd.api.types.is_bool_dtype(series):
        values = series.astype("float64").to_numpy()
    else:
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    codes = np.zeros(values.shape[0], dtype=np.int64)
    valid = np.isfinite(values)
    rounded = np.floor(values[valid] + 0.5)
    codes[valid] = np.clip(rounded, 0, upper).astype(np.int64) + 1
    return codes


def orient_factors(frame: pd.DataFrame) -> pd.DataFrame:
    """低优因子取相反数。这一步在截面标准化之前完成。

    取负后再标准化，截面上更优的公司仍然对应更大的输入。
    请传入原始因子；调用方不要提前对这三列取负。
    """
    out = frame.copy()
    missing = [column for column in LOW_IS_BETTER if column not in out.columns]
    if missing:
        raise KeyError("缺少列: " + ", ".join(missing))
    for column in LOW_IS_BETTER:
        out[column] = -pd.to_numeric(out[column], errors="coerce")
    return out


def build_ohlcv_features(prices: pd.DataFrame) -> pd.DataFrame:
    """由 OHLCV 生成每个交易日一行的衍生特征。

    shift 只在同一只股票已经观测到的行情上向过去看一天。
    停牌日留到日历对齐时再补空，不把未来价格引进来。
    """
    df = prices.sort_values(["ticker", "date"]).copy()
    grouped = df.groupby("ticker", sort=False)
    prev_close = grouped["close"].shift(1)
    prev_volume = grouped["volume"].shift(1)
    close = df["close"].where(df["close"] > 0)
    opened = df["open"].where(df["open"] > 0)
    high = df["high"]
    low = df["low"]
    span = (high - low).where((high - low) > 0)
    upper = df[["open", "close"]].max(axis=1, skipna=False)
    lower = df[["open", "close"]].min(axis=1, skipna=False)
    volume = df["volume"].where(df["volume"] >= 0)

    df["ret_1"] = close / prev_close.where(prev_close > 0) - 1.0
    df["ret_oc"] = close / opened - 1.0
    df["hl_range"] = (high - low) / close
    df["close_position"] = (close - low) / span
    df["upper_shadow"] = (high - upper) / close
    df["lower_shadow"] = (lower - low) / close
    df["log_volume"] = np.log1p(volume)
    df["volume_chg"] = volume / prev_volume.where(prev_volume > 0) - 1.0
    df.loc[:, list(OHLCV_FEATURES)] = df.loc[:, list(OHLCV_FEATURES)].replace([np.inf, -np.inf], np.nan)
    return df


def _prepare_prices(
    frame: pd.DataFrame,
    feature_columns: Sequence[str] | None,
) -> tuple[pd.DataFrame, list[str]]:
    required = ["date", "ticker", "close"]
    if feature_columns is None:
        required.extend(["open", "high", "low", "volume"])
    out = _standardize_columns(frame, _PRICE_ALIASES, required)
    out["date"] = _as_naive_datetime(out["date"])
    out["ticker"] = out["ticker"].astype("string").fillna("").str.strip()
    out = out.dropna(subset=["date"])
    out = out[out["ticker"] != ""]
    for column in ("open", "high", "low", "close", "volume"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ("open", "high", "low", "close"):
        if column in out.columns:
            out.loc[out[column] <= 0, column] = np.nan
    if "volume" in out.columns:
        out.loc[out["volume"] < 0, "volume"] = np.nan
    if {"high", "low"}.issubset(out.columns):
        invalid_bar = out["high"].notna() & out["low"].notna() & (out["high"] < out["low"])
        price_columns = [column for column in ("open", "high", "low", "close", "volume") if column in out.columns]
        out.loc[invalid_bar, price_columns] = np.nan
    out = out.dropna(subset=["close"])
    out = out.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    out["raw_close"] = out["close"].to_numpy(dtype=np.float64)
    if feature_columns is None:
        out = build_ohlcv_features(out)
        columns = list(OHLCV_FEATURES)
    else:
        columns = list(feature_columns)
        missing = [column for column in columns if column not in out.columns]
        if missing:
            raise KeyError("行情缺少特征列: " + ", ".join(missing))
        for column in columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if out.empty:
        raise ValueError("清洗后没有可用行情")
    return out.reset_index(drop=True), columns


def _prepare_fundamentals(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {**_FUND_KEY_ALIASES, **_factor_aliases()}
    required = ["ticker", "filed_date", *ALL_FACTOR_COLUMNS]
    out = _standardize_columns(frame, aliases, required)
    out["ticker"] = out["ticker"].astype("string").fillna("").str.strip()
    out["filed_date"] = _as_naive_datetime(out["filed_date"])
    if "period_end" not in out.columns:
        out["period_end"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    else:
        out["period_end"] = _as_naive_datetime(out["period_end"])
    out = out.dropna(subset=["filed_date"])
    out = out[out["ticker"] != ""]
    return out.reset_index(drop=True)


def _trading_calendar(dates: pd.Series) -> pd.DatetimeIndex:
    calendar = pd.DatetimeIndex(pd.unique(_as_naive_datetime(dates).dropna())).sort_values()
    if calendar.empty:
        raise ValueError("行情没有有效日期")
    return pd.DatetimeIndex(calendar)


def _on_calendar(frame: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """每只股票都铺到同一交易日历上。没有成交的日期整行是空，后面用掩码丢掉。"""
    pieces: list[pd.DataFrame] = []
    for ticker, group in frame.groupby("ticker", sort=False):
        group = group.drop_duplicates("date", keep="last").set_index("date").sort_index()
        group = group.reindex(calendar)
        group["ticker"] = ticker
        group.index.name = "date"
        pieces.append(group.reset_index())
    if not pieces:
        raise ValueError("没有可用的行情")
    return pd.concat(pieces, ignore_index=True)


def align_fundamentals_asof(
    keys: pd.DataFrame,
    fundamentals: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    announcement_lag: int = 0,
    factor_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """把每个 (股票, 交易日) 对齐到最近一条已经可用的财报。

    可用日 = 公告日当天或之后的第一个交易日，再加 announcement_lag 个交易日。
    lag 为 0 时，交易日当天公告的财报可以在当天使用；公告落在周末则从下周
    第一个交易日开始使用。lag 为 1 适合“收盘后才提交、次日才能交易”的设定。
    方向是向过去匹配，因此决策日看不到更晚的 filed_date。
    同一个可用日有多条记录时，留下 period_end 较晚的那条，同一报告期再比 filed_date。
    """
    if announcement_lag < 0:
        raise ValueError("announcement_lag 不能为负，财报只能在公告之后使用")
    factor_columns = list(ALL_FACTOR_COLUMNS if factor_columns is None else factor_columns)
    if "ticker" not in fundamentals.columns or "filed_date" not in fundamentals.columns:
        raise KeyError("基本面需要 ticker 和 filed_date")
    missing = [column for column in factor_columns if column not in fundamentals.columns]
    if missing:
        raise KeyError("基本面缺少列: " + ", ".join(missing))

    left = keys.loc[:, ["ticker", "date"]].copy()
    left["ticker"] = left["ticker"].astype(str)
    left["date"] = _as_naive_datetime(left["date"])
    if left.duplicated(["ticker", "date"]).any():
        raise ValueError("同一股票同一交易日有多行，请先去重")
    # merge_asof 要求连接键全局单调。按日期排序后，同一股票内部的日期仍然是递增的。
    left = left.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)

    empty = left.copy()
    empty["available_date"] = pd.Series(pd.NaT, index=empty.index, dtype="datetime64[ns]")
    empty["filed_date"] = pd.Series(pd.NaT, index=empty.index, dtype="datetime64[ns]")
    empty["period_end"] = pd.Series(pd.NaT, index=empty.index, dtype="datetime64[ns]")
    for column in factor_columns:
        empty[column] = np.nan

    fund = fundamentals.copy()
    fund["ticker"] = fund["ticker"].astype(str)
    fund["filed_date"] = _as_naive_datetime(fund["filed_date"])
    if "period_end" not in fund.columns:
        fund["period_end"] = pd.Series(pd.NaT, index=fund.index, dtype="datetime64[ns]")
    else:
        fund["period_end"] = _as_naive_datetime(fund["period_end"])
    fund = fund.dropna(subset=["filed_date"])
    known = set(left["ticker"])
    fund = fund[fund["ticker"].isin(known)]
    if fund.empty:
        return empty

    for column in factor_columns:
        if pd.api.types.is_bool_dtype(fund[column]):
            fund[column] = fund[column].astype("float64")
        else:
            fund[column] = pd.to_numeric(fund[column], errors="coerce")

    calendar = pd.DatetimeIndex(calendar).sort_values()
    cal_dt = calendar.to_numpy(dtype="datetime64[ns]")
    filed_ns = fund["filed_date"].to_numpy(dtype="int64")
    position = np.searchsorted(cal_dt.astype("int64"), filed_ns, side="left") + int(announcement_lag)
    usable = (position >= 0) & (position < len(cal_dt))
    available = np.full(len(fund), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    available[usable] = cal_dt[position[usable]]
    fund = fund.copy()
    fund["available_date"] = available
    fund = fund.dropna(subset=["available_date"])
    if fund.empty:
        return empty

    keep = ["ticker", "available_date", "filed_date", "period_end", *factor_columns]
    fund = fund.sort_values(["ticker", "available_date", "period_end", "filed_date"], kind="mergesort")
    fund = fund.drop_duplicates(["ticker", "available_date"], keep="last")
    fund = fund.loc[:, keep].sort_values(["available_date", "ticker"], kind="mergesort")

    aligned = pd.merge_asof(
        left,
        fund,
        left_on="date",
        right_on="available_date",
        by="ticker",
        direction="backward",
        allow_exact_matches=True,
    )
    return aligned


def _position_on_or_before(calendar: pd.DatetimeIndex, day: str) -> int:
    stamp = pd.Timestamp(day)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return int(calendar.searchsorted(stamp, side="right")) - 1


def split_indices(
    dataset: StockMultimodalDataset,
    train_end: str,
    val_end: str | None = None,
) -> dict[str, list[int]]:
    """按决策日切分，并留出长度等于 horizon 的隔离带。

    训练样本的标签收盘价必须落在 train_end 当天或之前。
    决策日还在训练区间、但未来收益已经用到验证区间的样本，训练和验证都不用。
    验证样本同样要求标签落在 val_end 之内。
    """
    train_pos = _position_on_or_before(dataset.calendar, train_end)
    if val_end is None:
        val_pos = len(dataset.calendar) - 1
    else:
        val_pos = _position_on_or_before(dataset.calendar, val_end)
    horizon = dataset.horizon
    train: list[int] = []
    val: list[int] = []
    for index, (_, end) in enumerate(dataset.samples):
        label_pos = end + horizon
        if end <= train_pos and label_pos <= train_pos:
            train.append(index)
        elif end > train_pos and label_pos <= val_pos:
            val.append(index)
    return {"train": train, "val": val}


class StockMultimodalDataset(Dataset):
    """滑动窗口数据集。一个样本是某只股票在决策日 t 的 T 日量价和当日可见基本面。

    标签是未来 horizon 个交易日的简单收益：close[t+horizon] / close[t] - 1。
    这里的交易日来自行情日期的并集，请只传入同一市场的交易日。
    基本面向量在模型里复制到每一天；复制的是决策日 t 已经公告的那一版，
    不是窗口内每一天各看一次，也不是 t 之后的新财报。

    返回字典：
        ohlcv [T, F]，ohlcv_valid [T]，ohlcv_obs [T, F]
        continuous [C]，continuous_mask [C]
        fscore 标量，binary [9]，label 标量，ticker，date
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame,
        window: int = 20,
        horizon: int = 5,
        announcement_lag: int = 0,
        feature_columns: Sequence[str] | None = None,
        min_valid_ratio: float = 0.8,
        z_clip: float | None = 5.0,
        require_fundamental: bool = False,
    ) -> None:
        if window < 2:
            raise ValueError("window 至少为 2")
        if horizon < 1:
            raise ValueError("horizon 至少为 1")
        if not 0 < min_valid_ratio <= 1:
            raise ValueError("min_valid_ratio 要落在 (0, 1] 内")

        self.window = int(window)
        self.horizon = int(horizon)
        self.announcement_lag = int(announcement_lag)
        self.require_fundamental = bool(require_fundamental)

        # 1. 清洗行情，生成因果衍生特征。标签留在 raw_close，不参与标准化。
        price_frame, columns = _prepare_prices(prices, feature_columns)
        self.feature_columns = columns
        self.ohlcv_dim = len(columns)
        self.calendar = _trading_calendar(price_frame["date"])
        panel = _on_calendar(price_frame, self.calendar)

        # 2. 量价按决策日所在的交易日做截面标准化，再记录哪些位置原来是空的。
        panel = cross_sectional_zscore(panel, columns, by="date", clip=z_clip)

        # 3. 财报只匹配 available_date <= 当天 的最近一条。
        fund = _prepare_fundamentals(fundamentals)
        aligned = align_fundamentals_asof(
            panel.loc[:, ["ticker", "date"]],
            fund,
            self.calendar,
            announcement_lag=self.announcement_lag,
            factor_columns=ALL_FACTOR_COLUMNS,
        )
        panel = panel.merge(aligned, on=["ticker", "date"], how="left", validate="one_to_one")

        # 4. 低优因子取负，然后只对连续基本面做截面标准化。评分保持原始离散值。
        panel = orient_factors(panel)
        panel = cross_sectional_zscore(panel, CONTINUOUS_COLUMNS, by="date", clip=z_clip)
        panel["FScore_code"] = encode_ordinal(panel[FSCORE_COLUMN], FSCORE_MAX)
        for column in BINARY_COLUMNS:
            panel[f"{column}_code"] = encode_ordinal(panel[column], 1)

        self._ohlcv: dict[str, np.ndarray] = {}
        self._obs: dict[str, np.ndarray] = {}
        self._valid: dict[str, np.ndarray] = {}
        self._continuous: dict[str, np.ndarray] = {}
        self._continuous_mask: dict[str, np.ndarray] = {}
        self._fscore: dict[str, np.ndarray] = {}
        self._binary: dict[str, np.ndarray] = {}
        self._label: dict[str, np.ndarray] = {}
        self._filed: dict[str, np.ndarray] = {}
        self._available: dict[str, np.ndarray] = {}

        calendar_np = self.calendar.to_numpy(dtype="datetime64[ns]")
        min_valid_bars = max(1, int(np.ceil(min_valid_ratio * self.window)))
        samples: list[tuple[str, int]] = []

        for ticker, block in panel.groupby("ticker", sort=False):
            block = block.sort_values("date")
            dates = block["date"].to_numpy(dtype="datetime64[ns]")
            if len(block) != len(calendar_np) or not np.array_equal(dates, calendar_np):
                raise RuntimeError(f"{ticker} 的交易日历没有对齐")

            ohlcv, ohlcv_obs = fill_missing_with_mask(block, columns)
            continuous, continuous_obs = fill_missing_with_mask(block, CONTINUOUS_COLUMNS)
            raw_close = pd.to_numeric(block["raw_close"], errors="coerce").to_numpy(dtype=np.float64)
            day_valid = np.isfinite(raw_close) & (raw_close > 0)
            label = np.full(len(block), np.nan, dtype=np.float32)
            current = raw_close[:-horizon]
            future = raw_close[horizon:]
            good = np.isfinite(current) & np.isfinite(future) & (current > 0) & (future > 0)
            change = np.full(current.shape[0], np.nan, dtype=np.float64)
            change[good] = future[good] / current[good] - 1.0
            label[:-horizon] = change.astype(np.float32)

            name = str(ticker)
            self._ohlcv[name] = ohlcv
            self._obs[name] = ohlcv_obs
            self._valid[name] = day_valid
            self._continuous[name] = continuous
            self._continuous_mask[name] = continuous_obs.astype(np.float32)
            self._fscore[name] = block["FScore_code"].to_numpy(dtype=np.int64)
            self._binary[name] = block[[f"{column}_code" for column in BINARY_COLUMNS]].to_numpy(dtype=np.int64)
            self._label[name] = label
            self._filed[name] = pd.to_datetime(block["filed_date"]).to_numpy(dtype="datetime64[ns]")
            self._available[name] = pd.to_datetime(block["available_date"]).to_numpy(dtype="datetime64[ns]")

            last_end = len(block) - horizon
            for end in range(self.window - 1, last_end):
                start = end - self.window + 1
                if not bool(day_valid[end]):
                    continue
                if int(day_valid[start : end + 1].sum()) < min_valid_bars:
                    continue
                if not np.isfinite(label[end]):
                    continue
                filed = self._filed[name][end]
                available = self._available[name][end]
                decision = calendar_np[end]
                if (not np.isnat(filed) and filed > decision) or (not np.isnat(available) and available > decision):
                    raise RuntimeError(f"{name} 在 {decision} 用到了尚未公告的财报")
                has_fundamental = (
                    int(self._fscore[name][end]) != 0
                    or bool(continuous_obs[end].any())
                    or bool((self._binary[name][end] != 0).any())
                )
                if self.require_fundamental and not has_fundamental:
                    continue
                samples.append((name, end))

        self.samples = samples
        self.decision_dates = pd.DatetimeIndex([self.calendar[end] for _, end in self.samples])
        if not self.samples:
            warnings.warn(
                "没有可用样本。请检查窗口、预测期、公告日和有效交易日比例。",
                RuntimeWarning,
                stacklevel=2,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        ticker, end = self.samples[index]
        start = end - self.window + 1
        return {
            "ohlcv": torch.tensor(self._ohlcv[ticker][start : end + 1]),
            "ohlcv_valid": torch.tensor(self._valid[ticker][start : end + 1]),
            "ohlcv_obs": torch.tensor(self._obs[ticker][start : end + 1]),
            "continuous": torch.tensor(self._continuous[ticker][end]),
            "continuous_mask": torch.tensor(self._continuous_mask[ticker][end]),
            "fscore": torch.tensor(int(self._fscore[ticker][end]), dtype=torch.long),
            "binary": torch.tensor(self._binary[ticker][end].copy(), dtype=torch.long),
            "label": torch.tensor(float(self._label[ticker][end]), dtype=torch.float32),
            "ticker": ticker,
            "date": self.decision_dates[index].strftime("%Y-%m-%d"),
        }

    def pit_audit(self) -> pd.DataFrame:
        """列出每个样本的决策日、所用财报的公告日和可用日。leak 为 True 表示用到了未来公告。"""
        rows = []
        for index, (ticker, end) in enumerate(self.samples):
            rows.append(
                {
                    "ticker": ticker,
                    "decision_date": self.decision_dates[index],
                    "filed_date": self._filed[ticker][end],
                    "available_date": self._available[ticker][end],
                }
            )
        audit = pd.DataFrame(rows)
        if audit.empty:
            audit["leak"] = pd.Series(dtype=bool)
            return audit
        filed = pd.to_datetime(audit["filed_date"])
        available = pd.to_datetime(audit["available_date"])
        decision = pd.to_datetime(audit["decision_date"])
        audit["leak"] = (filed.notna() & (filed > decision)) | (available.notna() & (available > decision))
        return audit


@dataclass
class ModelConfig:
    """d_model 必须能被 nhead 整除。head_dropout 默认 0.3，用在预测头。"""

    ohlcv_dim: int
    window: int = 20
    n_continuous: int = len(CONTINUOUS_COLUMNS)
    n_binary: int = len(BINARY_COLUMNS)
    fscore_classes: int = FSCORE_MAX + 2
    binary_classes: int = 3
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    head_dropout: float = 0.3
    embedding_dim: int = 8
    continuous_hidden: int = 32
    pool: str = "cls"

    def __post_init__(self) -> None:
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model 必须能被注意力头数整除")
        if self.pool not in {"cls", "last"}:
            raise ValueError("pool 只能是 cls 或 last")
        if self.ohlcv_dim <= 0 or self.window < 2:
            raise ValueError("特征维数或窗口长度不合法")
        if not 0 <= self.head_dropout < 1:
            raise ValueError("head_dropout 要落在 [0, 1) 内")
        if self.fscore_classes < 2 or self.binary_classes < 2:
            raise ValueError("Embedding 至少要包含缺失类和一个有效类别")


def config_for_dataset(dataset: StockMultimodalDataset, **overrides: object) -> ModelConfig:
    params: dict[str, object] = {
        "ohlcv_dim": dataset.ohlcv_dim,
        "window": dataset.window,
        "n_continuous": len(CONTINUOUS_COLUMNS),
        "n_binary": len(BINARY_COLUMNS),
    }
    params.update(overrides)
    return ModelConfig(**params)  # type: ignore[arg-type]


class MultimodalReturnTransformer(nn.Module):
    """量价序列与低频基本面融合后的 Transformer 回归模型。

    形状：
        ohlcv [B, T, F]                  截面标准化后的日频特征，缺失处为 0
        ohlcv_valid [B, T]               True 表示当天有正的收盘价，False 不参与注意力
        ohlcv_obs [B, T, F]              单个特征是否有观测
        continuous [B, C]                决策日的连续基本面
        continuous_mask [B, C]
        fscore [B]                       0 为缺失，1..10 对应 0..9 分
        binary [B, 9]                    0 为缺失，1 为 0 分，2 为 1 分

    基本面先映射到 d_model，再复制 T 份，和每一天的量价投影拼接。
    拼接后的宽度是 2 * d_model，经一层线性层回到 d_model，才能送进标准 Encoder。
    pool='cls' 时在序列前面加可学习的 [CLS]，并读取它的输出；
    pool='last' 时读取最后一天。预测头是两层 MLP，中间 Dropout 默认为 0.3。
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.d_model

        # 连续变量：数值和缺失掩码一起进线性层。掩码避免“填 0”被当成截面均值。
        self.continuous_proj = nn.Linear(config.n_continuous * 2, config.continuous_hidden)
        self.fscore_embedding = nn.Embedding(config.fscore_classes, config.embedding_dim)
        self.binary_embeddings = nn.ModuleList(
            [nn.Embedding(config.binary_classes, config.embedding_dim) for _ in range(config.n_binary)]
        )
        fund_input = config.continuous_hidden + config.embedding_dim * (1 + config.n_binary)
        self.fund_proj = nn.Linear(fund_input, d_model)
        self.fund_norm = nn.LayerNorm(d_model)

        # 量价同样把观测掩码拼在特征后面再投影。
        self.ohlcv_proj = nn.Linear(config.ohlcv_dim * 2, d_model)
        self.price_norm = nn.LayerNorm(d_model)
        self.fusion = nn.Linear(d_model * 2, d_model)

        token_count = config.window + (1 if config.pool == "cls" else 0)
        self.position = nn.Parameter(torch.zeros(1, token_count, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if config.pool == "cls" else None

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        encoder_kwargs: dict[str, object] = {
            "num_layers": config.num_layers,
            "norm": nn.LayerNorm(d_model),
        }
        if "enable_nested_tensor" in inspect.signature(nn.TransformerEncoder.__init__).parameters:
            # 带 padding mask 时关闭嵌套张量快速路径，避免不同 PyTorch 版本的警告和分支差异。
            encoder_kwargs["enable_nested_tensor"] = False
        self.encoder = nn.TransformerEncoder(layer, **encoder_kwargs)  # type: ignore[arg-type]

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(d_model, 1),
        )
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.position, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        ohlcv: torch.Tensor,
        ohlcv_valid: torch.Tensor,
        ohlcv_obs: torch.Tensor,
        continuous: torch.Tensor,
        continuous_mask: torch.Tensor,
        fscore: torch.Tensor,
        binary: torch.Tensor,
    ) -> torch.Tensor:
        if ohlcv.ndim != 3:
            raise ValueError("ohlcv 的形状应为 [B, T, F]")
        batch_size, time_steps, _ = ohlcv.shape
        if time_steps != self.config.window:
            raise ValueError(f"输入序列长度是 {time_steps}，模型窗口是 {self.config.window}")
        if binary.shape[-1] != self.config.n_binary:
            raise ValueError("二元分项数量和模型配置不一致")

        ohlcv = torch.nan_to_num(ohlcv, nan=0.0, posinf=0.0, neginf=0.0)
        continuous = torch.nan_to_num(continuous, nan=0.0, posinf=0.0, neginf=0.0)
        observed = ohlcv_obs.to(dtype=ohlcv.dtype)
        price = self.price_norm(self.ohlcv_proj(torch.cat([ohlcv, observed], dim=-1)))

        continuous_in = torch.cat([continuous, continuous_mask.to(dtype=continuous.dtype)], dim=-1)
        parts = [self.continuous_proj(continuous_in), self.fscore_embedding(fscore.long())]
        binary = binary.long()
        for index, embedding in enumerate(self.binary_embeddings):
            parts.append(embedding(binary[:, index]))
        # 决策日的基本面向量复制到窗口中的每一天，再和当天的量价投影拼接。
        fund = self.fund_norm(self.fund_proj(torch.cat(parts, dim=-1)))
        fund_seq = fund.unsqueeze(1).repeat(1, time_steps, 1)
        sequence = self.fusion(torch.cat([price, fund_seq], dim=-1))

        valid = ohlcv_valid.bool()
        if self.cls_token is not None:
            cls = self.cls_token.expand(batch_size, -1, -1)
            sequence = torch.cat([cls, sequence], dim=1)
            ones = torch.ones(batch_size, 1, dtype=torch.bool, device=sequence.device)
            valid = torch.cat([ones, valid], dim=1)
        sequence = sequence + self.position[:, : sequence.size(1)]
        # PyTorch 的 key padding mask 里，True 表示这个位置不参与注意力。
        encoded = self.encoder(sequence, src_key_padding_mask=~valid)
        token = encoded[:, 0] if self.config.pool == "cls" else encoded[:, -1]
        return self.head(token).squeeze(-1)

    def forward_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward(
            batch["ohlcv"],
            batch["ohlcv_valid"],
            batch["ohlcv_obs"],
            batch["continuous"],
            batch["continuous_mask"],
            batch["fscore"],
            batch["binary"],
        )


def _connected_zero(reference: torch.Tensor) -> torch.Tensor:
    """返回挂在 reference 计算图上的 0，空批次或常数批次也能反向传播。"""
    return reference.reshape(-1).sum() * 0.0


def _pearson(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = left - left.mean()
    y = right - right.mean()
    denom = x.norm(p=2) * y.norm(p=2)
    if float(denom.detach()) < eps:
        return _connected_zero(left)
    return (x * y).sum() / denom


def _soft_rank(values: torch.Tensor, temperature: float) -> torch.Tensor:
    """可导的名次。数值越大，名次越高。

    先用自身的标准差缩放差值，使温度和收益率的数量级无关。
    对角项是自己和自己比较，不计入名次。
    """
    scale = values.detach().std(unbiased=False).clamp_min(1e-6)
    logits = (values.unsqueeze(1) - values.unsqueeze(0)) / (temperature * scale)
    weights = torch.sigmoid(logits)
    eye = torch.eye(values.shape[0], device=values.device, dtype=values.dtype)
    return (weights * (1.0 - eye)).sum(dim=1)


def _average_rank(values: torch.Tensor) -> torch.Tensor:
    """精确 Spearman 用的平均秩，从 0 起算。并列观测平分名次。"""
    # [i, j] 为 True 表示 v_i < v_j。沿 i 求和后，结果 j 是小于 v_j 的个数。
    less = (values.unsqueeze(1) < values.unsqueeze(0)).sum(dim=0).to(dtype=torch.float32)
    equal = (values.unsqueeze(1) == values.unsqueeze(0)).sum(dim=0).to(dtype=torch.float32)
    return less + (equal - 1.0) * 0.5


class RankICLoss(nn.Module):
    """一个 batch 内的 Spearman 秩相关取负，作为回归排序损失。

    硬 argsort 的 Spearman 对预测值的梯度几乎处处为 0，不能直接训练。
    这里对预测和标签都做温度软排序，再计算皮尔逊相关，loss = -rho。
    rho 接近 1 时损失接近 -1。一个 batch 应当是同一决策日的横截面，
    请和 CrossSectionBatchSampler 一起用。有效股票少于 min_samples，
    或标签没有区分度时，损失为 0。
    """

    def __init__(self, temperature: float = 0.25, min_samples: int = 5) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature 必须为正")
        if min_samples < 3:
            raise ValueError("min_samples 至少为 3")
        self.temperature = float(temperature)
        self.min_samples = int(min_samples)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.reshape(-1)
        target = target.detach().reshape(-1)
        if pred.shape != target.shape:
            raise ValueError("预测和标签的数量要一致")
        valid = torch.isfinite(pred) & torch.isfinite(target)
        if int(valid.sum().detach()) < self.min_samples:
            return _connected_zero(pred)
        pred = pred[valid]
        target = target[valid]
        if torch.unique(target).numel() < 2:
            return _connected_zero(pred)
        pred_rank = _soft_rank(pred, self.temperature)
        target_rank = _soft_rank(target, self.temperature)
        return -_pearson(pred_rank, target_rank)


@torch.no_grad()
def spearman_ic(pred: torch.Tensor, target: torch.Tensor) -> float:
    """评估用的精确 Spearman。样本少于 3 个或某一列是常数时返回 nan。"""
    pred = pred.detach().reshape(-1).float()
    target = target.detach().reshape(-1).float()
    valid = torch.isfinite(pred) & torch.isfinite(target)
    pred = pred[valid]
    target = target[valid]
    if pred.numel() < 3:
        return float("nan")
    if torch.unique(pred).numel() < 2 or torch.unique(target).numel() < 2:
        return float("nan")
    value = float(_pearson(_average_rank(pred), _average_rank(target)).detach())
    if not math.isfinite(value):
        return float("nan")
    return value


class CrossSectionBatchSampler(Sampler[list[int]]):
    """每个 batch 是同一个决策日、股票数达到 min_names 的横截面。"""

    def __init__(
        self,
        decision_dates: Sequence[pd.Timestamp],
        indices: Sequence[int] | None = None,
        shuffle: bool = True,
        min_names: int = 8,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if min_names < 2:
            raise ValueError("min_names 至少为 2")
        chosen = range(len(decision_dates)) if indices is None else indices
        groups: dict[pd.Timestamp, list[int]] = {}
        for index in chosen:
            day = pd.Timestamp(decision_dates[index])
            groups.setdefault(day, []).append(int(index))
        ordered = sorted(groups.items(), key=lambda item: item[0])
        self.batches = tuple(tuple(items) for _, items in ordered if len(items) >= min_names)
        self.shuffle = bool(shuffle)
        self._generator = None
        if seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(int(seed))
        if not self.batches:
            warnings.warn(
                "没有一个交易日的股票数达到 min_names，DataLoader 将是空的。",
                RuntimeWarning,
                stacklevel=2,
            )

    def __iter__(self) -> Iterator[list[int]]:
        count = len(self.batches)
        if self.shuffle:
            order = torch.randperm(count, generator=self._generator).tolist()
        else:
            order = list(range(count))
        for position in order:
            yield list(self.batches[position])

    def __len__(self) -> int:
        return len(self.batches)


def make_loader(
    dataset: StockMultimodalDataset,
    indices: Sequence[int] | None = None,
    shuffle: bool = True,
    min_names: int = 8,
    seed: int | None = None,
    num_workers: int = 0,
) -> DataLoader:
    """按决策日组 batch。Windows 上 num_workers 建议保持 0。"""
    sampler = CrossSectionBatchSampler(
        dataset.decision_dates,
        indices=indices,
        shuffle=shuffle,
        min_names=min_names,
        seed=seed,
    )
    return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers)


def _move_batch(batch: dict, device: torch.device | str) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def run_epoch(
    model: MultimodalReturnTransformer,
    loader: DataLoader,
    criterion: RankICLoss,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
    grad_clip: float | None = 1.0,
    log_every: int = 0,
) -> dict[str, float | int]:
    """optimizer 为空时只评估。训练时按 batch 的 Rank IC 损失反传，并裁剪梯度。"""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_ic = 0.0
    ic_count = 0
    batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = _move_batch(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            pred = model.forward_batch(batch)
            loss = criterion(pred, batch["label"])
            if training:
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            ic = spearman_ic(pred, batch["label"])
            total_loss += float(loss.detach())
            if math.isfinite(ic):
                total_ic += ic
                ic_count += 1
            batches += 1
            if log_every and batches % log_every == 0:
                phase = "train" if training else "eval"
                print(f"  {phase} batch {batches} loss={float(loss.detach()):.4f}", flush=True)
    if batches == 0:
        return {"loss": float("nan"), "rank_ic": float("nan"), "batches": 0}
    rank_ic = total_ic / ic_count if ic_count else float("nan")
    return {"loss": total_loss / batches, "rank_ic": rank_ic, "batches": batches}


def make_synthetic_market(
    n_stocks: int = 16,
    n_days: int = 60,
    start: str = "2020-01-02",
    seed: int = 7,
    filing_index: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成一份可运行的合成市场，只用于检查代码，不代表真实收益率。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    if not 0 <= filing_index < n_days:
        raise ValueError("filing_index 超出交易日范围")
    price_frames = []
    factor_rows = []
    for i in range(n_stocks):
        ticker = f"S{i:02d}"
        shock = rng.normal(0.0004 * (i - n_stocks / 2.0), 0.012, n_days)
        close = 40.0 * np.cumprod(1.0 + shock)
        opened = close * (1.0 + rng.normal(0.0, 0.003, n_days))
        high = np.maximum(opened, close) * (1.0 + rng.uniform(0.0, 0.01, n_days))
        low = np.minimum(opened, close) * (1.0 - rng.uniform(0.0, 0.01, n_days))
        volume = rng.integers(1_000, 20_000, n_days)
        price_frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "open": opened,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
        )
        factor_rows.append(
            {
                "ticker": ticker,
                "filed_date": dates[filing_index],
                "period_end": dates[filing_index] - pd.Timedelta(days=30),
                "ROE": 0.02 * i,
                "ROA": 0.01 * i,
                "GrossProfitability": 0.2 + 0.01 * i,
                "BM": 0.5 - 0.01 * i,
                "AssetTurnover": 0.4 + 0.02 * i,
                "Accruals": 0.05 - 0.005 * i,
                "AssetGrowth": 0.15 - 0.01 * i,
                "NetStockIssues": 0.04 - 0.002 * i,
                "FScore": i % 10,
                **{name: (i + offset) % 2 for offset, name in enumerate(BINARY_COLUMNS)},
            }
        )
    return pd.concat(price_frames, ignore_index=True), pd.DataFrame(factor_rows)


def demo() -> None:
    """用合成数据走一遍公告日检查、按日切分和三个训练 epoch。"""
    torch.manual_seed(7)
    prices, fundamentals = make_synthetic_market()
    dataset = StockMultimodalDataset(prices, fundamentals, window=20, horizon=5, announcement_lag=0)
    audit = dataset.pit_audit()
    if not audit.empty and bool(audit["leak"].any()):
        raise RuntimeError("公告日对齐检查失败")
    midpoint = dataset.calendar[len(dataset.calendar) // 2]
    parts = split_indices(dataset, train_end=str(midpoint.date()))
    print(f"样本 {len(dataset)}，训练 {len(parts['train'])}，验证 {len(parts['val'])}")
    loader = make_loader(dataset, parts["train"], shuffle=True, min_names=8, seed=7)
    model = MultimodalReturnTransformer(config_for_dataset(dataset))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = RankICLoss()
    for epoch in range(1, 4):
        metrics = run_epoch(model, loader, criterion, optimizer)
        print(
            f"epoch {epoch}: loss={metrics['loss']:.4f} "
            f"rank_ic={metrics['rank_ic']:.4f} batches={metrics['batches']}"
        )


if __name__ == "__main__":
    demo()
