"""量价、价格和 K 线因子。只返回原始值，不去极值、不标准化、不翻转方向。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from services.errors import DataGapError

FACTOR_HELP = {
    "CCV_N": "N 日收盘价与成交量的相关系数",
    "CCOIV_N": "N 日日内涨跌与成交量的相关系数，预期反转",
    "COV_N": "N 日隔夜跳空与昨日成交量的相关系数，预期动量",
    "VPD_N": "N 日放量方向",
    "VPR_N": "N 日成交量加权平均收益",
    "ROC_N": "N 日价格动量",
    "VOL_N": "N 日收益标准差",
    "MA_DEV_N": "收盘价相对 N 日均线的偏离",
    "KMID": "实体涨跌幅",
    "KLEN": "全日振幅",
    "UPPER_SHADOW": "上影线",
    "LOWER_SHADOW": "下影线",
    "CLOSE_POS": "收盘在全日区间中的位置",
    "RISK_ADJ_MOM": "20 日动量除以 20 日波动",
    "VOL_PRICE_RES": "20 日动量乘以相对成交量",
}


def _close(frame: pd.DataFrame) -> pd.Series:
    column = "adjusted_close" if "adjusted_close" in frame.columns else "close"
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(frame[name], errors="coerce").astype(float)


def _ret(close: pd.Series) -> pd.Series:
    return close / close.shift(1) - 1.0


def _roll_corr(left: pd.Series, right: pd.Series, window: int) -> pd.Series:
    """窗口不足或任一方差为 0 时为 NaN，不填充。"""
    return left.rolling(window, min_periods=window).corr(right)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.where(den != 0)


def ccv(frame: pd.DataFrame, window: int) -> pd.Series:
    return _roll_corr(_close(frame), _column(frame, "volume"), window)


def ccoiv(frame: pd.DataFrame, window: int) -> pd.Series:
    body = _close(frame) - _column(frame, "open")
    return _roll_corr(body, _column(frame, "volume"), window)


def cov(frame: pd.DataFrame, window: int) -> pd.Series:
    overnight = _column(frame, "open") - _close(frame).shift(1)
    return _roll_corr(overnight, _column(frame, "volume").shift(1), window)


def vpd(frame: pd.DataFrame, window: int) -> pd.Series:
    close = _close(frame)
    volume = _column(frame, "volume")
    base = volume.rolling(window, min_periods=window).mean()
    signed = np.sign(close - _column(frame, "open")) * _safe_div(volume, base)
    return signed.rolling(window, min_periods=window).mean()


def vpr(frame: pd.DataFrame, window: int) -> pd.Series:
    close = _close(frame)
    volume = _column(frame, "volume")
    weighted = (_ret(close) * volume).rolling(window, min_periods=window).sum()
    base = volume.rolling(window, min_periods=window).sum()
    return _safe_div(weighted, base)


def roc(frame: pd.DataFrame, window: int) -> pd.Series:
    close = _close(frame)
    return close / close.shift(window) - 1.0


def vol(frame: pd.DataFrame, window: int) -> pd.Series:
    return _ret(_close(frame)).rolling(window, min_periods=window).std(ddof=1)


def ma_dev(frame: pd.DataFrame, window: int) -> pd.Series:
    close = _close(frame)
    return close / close.rolling(window, min_periods=window).mean() - 1.0


def kmid(frame: pd.DataFrame, window: int = 0) -> pd.Series:
    del window
    opened = _column(frame, "open")
    return _safe_div(_close(frame) - opened, opened)


def klen(frame: pd.DataFrame, window: int = 0) -> pd.Series:
    del window
    opened = _column(frame, "open")
    return _safe_div(_column(frame, "high") - _column(frame, "low"), opened)


def upper_shadow(frame: pd.DataFrame, window: int = 0) -> pd.Series:
    del window
    opened = _column(frame, "open")
    close = _close(frame)
    return _safe_div(_column(frame, "high") - np.maximum(opened, close), opened)


def lower_shadow(frame: pd.DataFrame, window: int = 0) -> pd.Series:
    del window
    opened = _column(frame, "open")
    close = _close(frame)
    return _safe_div(np.minimum(opened, close) - _column(frame, "low"), opened)


def close_pos(frame: pd.DataFrame, window: int = 0) -> pd.Series:
    del window
    high = _column(frame, "high")
    low = _column(frame, "low")
    return _safe_div(_close(frame) - low, high - low)


def risk_adj_mom(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    del window
    momentum = roc(frame, 20)
    sigma = vol(frame, 20)
    return _safe_div(momentum, sigma)


def vol_price_res(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    del window
    volume = _column(frame, "volume")
    base = volume.rolling(20, min_periods=20).mean()
    return roc(frame, 20) * _safe_div(volume, base)


FACTOR_REGISTRY = {
    "CCV_N": ccv,
    "CCOIV_N": ccoiv,
    "COV_N": cov,
    "VPD_N": vpd,
    "VPR_N": vpr,
    "ROC_N": roc,
    "VOL_N": vol,
    "MA_DEV_N": ma_dev,
    "KMID": kmid,
    "KLEN": klen,
    "UPPER_SHADOW": upper_shadow,
    "LOWER_SHADOW": lower_shadow,
    "CLOSE_POS": close_pos,
    "RISK_ADJ_MOM": risk_adj_mom,
    "VOL_PRICE_RES": vol_price_res,
}


def compute_factor(frame: pd.DataFrame, factor_name: str, params: dict | None = None) -> pd.Series:
    """单只股票的因子原始值，索引与输入日期对齐。"""
    if factor_name not in FACTOR_REGISTRY:
        raise DataGapError(f"没有这个因子：{factor_name}")
    window = int((params or {}).get("N", 20))
    if window < 2 and factor_name.endswith("_N"):
        raise DataGapError("滚动窗口至少 2 日")
    values = FACTOR_REGISTRY[factor_name](frame, window)
    values = values.reindex(frame.index)
    values.name = factor_name
    return values


def compute_factor_frame(frame: pd.DataFrame, factor_name: str, params: dict | None = None) -> pd.DataFrame:
    values = compute_factor(frame, factor_name, params)
    return values.to_frame(factor_name)
