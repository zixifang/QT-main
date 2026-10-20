"""价格来源。本地 CSV 用 Close，不读 Adj Close。其他源只说明为什么不能用。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from services.errors import DataGapError, SourceUnavailable


def read_symbol_file(path: Path) -> list[str]:
    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip().upper()
        if text:
            symbols.append(text)
    return symbols


def read_price_csv(path: Path, start: str, end: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "Date" not in frame.columns or "Close" not in frame.columns:
        raise DataGapError(f"{path.name} 缺少 Date 或 Close")
    dates = pd.to_datetime(frame["Date"])
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert(None)
    volume = pd.to_numeric(frame["Volume"], errors="coerce") if "Volume" in frame.columns else np.nan
    out = pd.DataFrame(
        {
            "date": dates.dt.normalize(),
            "close": pd.to_numeric(frame["Close"], errors="coerce"),
            "volume": volume,
        }
    )
    out = out.dropna(subset=["date", "close"])
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    out = out[(out["date"] >= start_ts) & (out["date"] <= end_ts)]
    out = out.sort_values("date").drop_duplicates("date", keep="last")
    if out.empty:
        raise DataGapError(f"{path.stem} 在 {start} 到 {end} 没有行情")
    return out.set_index("date")


class LocalCsvSource:
    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        path = self.root / "data" / "nasdaq100" / f"{symbol}.csv"
        if not path.exists():
            raise DataGapError(f"没有本地文件 {symbol}.csv")
        return read_price_csv(path, start, end)


class BlockedSource:
    def __init__(self, message: str) -> None:
        self.message = message

    def load(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        raise SourceUnavailable(self.message)


class NasdaqDataLinkSource:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key.strip()

    def load(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        if not self.api_key:
            raise SourceUnavailable("没有填写 Nasdaq Data Link 的 API Key")
        raise SourceUnavailable("Key 只留在本地配置里，接口尚未接入，不会发出请求")
