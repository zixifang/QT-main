"""下载约 40 只主要科技美股近 10 年的前复权日线。

运行：python -m data_fetch.tech_us

数据来自 yfinance（Yahoo Finance 的非官方接口，无需 API Key）。
单只股票失败会重试，不会中断其余股票。
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


def _use_ascii_ca_bundle() -> None:
    """libcurl 打不开路径里含中文的 CA 证书，先复制到纯英文临时目录。"""
    for key in ("SSL_CERT_FILE", "CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE"):
        path = os.environ.get(key)
        if not path or not os.path.exists(path):
            continue
        try:
            path.encode("ascii")
        except UnicodeEncodeError:
            break
        return
    import certifi

    source = certifi.where()
    try:
        source.encode("ascii")
        return
    except UnicodeEncodeError:
        pass
    target = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".") / "certifi-cacert.pem"
    shutil.copyfile(source, target)
    location = str(target)
    os.environ["SSL_CERT_FILE"] = location
    os.environ["CURL_CA_BUNDLE"] = location
    os.environ["REQUESTS_CA_BUNDLE"] = location


_use_ascii_ca_bundle()

import yfinance as yf

# 改这里即可，不必动下面的函数。
# START / END 留空时，按运行当天向前推 YEARS 年。yfinance 的 end 是开区间，脚本会自动加一天，使 END 当天包含在结果里。
START = ""
END = ""
YEARS = 10
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "tech-us-10y"
COMBINED_NAME = "tech_us_10y.csv"
INTERVAL = "1d"
AUTO_ADJUST = True
MAX_RETRIES = 3
SLEEP_SECONDS = 1.0
# 实际跨度短于这个年数时只记录，不中断。
MIN_HISTORY_YEARS = 9.5

# Block 在 2025-01-21 把代码从 SQ 改成 XYZ，这里用现行代码。
TICKERS = [
    # 大型科技
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    # 半导体
    "AMD", "INTC", "AVGO", "QCOM", "MU", "TSM", "ASML",
    # 软件 / 云
    "CRM", "ADBE", "ORCL", "NOW", "SNOW", "PLTR",
    # 互联网 / 平台
    "NFLX", "UBER", "SHOP", "XYZ", "PYPL",
    # 设备、半导体设备、网络安全与设计软件
    "CSCO", "IBM", "TXN", "AMAT", "LRCX", "KLAC", "ARM", "SMCI",
    "PANW", "CRWD", "SNPS", "CDNS", "ADI", "MRVL", "ANET",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tech_us")


def resolve_dates(start: str = START, end: str = END, years: int = YEARS) -> tuple[str, str]:
    """得到闭区间上的开始日、结束日。结束日本身要被下载到。"""
    end_day = date.fromisoformat(end) if end else date.today()
    start_day = date.fromisoformat(start) if start else end_day.replace(year=end_day.year - years)
    if start_day >= end_day:
        raise ValueError(f"开始日 {start_day} 必须早于结束日 {end_day}")
    return start_day.isoformat(), end_day.isoformat()


def download_raw(ticker: str, start: str, end: str) -> pd.DataFrame:
    """下载单只股票。end 传入 yfinance 前加一天，因为该参数不包含结束日。"""
    end_exclusive = (date.fromisoformat(end) + timedelta(days=1)).isoformat()
    frame = yf.download(
        ticker,
        start=start,
        end=end_exclusive,
        interval=INTERVAL,
        auto_adjust=AUTO_ADJUST,
        progress=False,
        threads=False,
        group_by="ticker",
    )
    if frame is None:
        return pd.DataFrame()
    return frame


def flatten_columns(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """把 yfinance 的多层列收成单层小写列名。"""
    if isinstance(frame.columns, pd.MultiIndex):
        levels = [frame.columns.get_level_values(i) for i in range(frame.columns.nlevels)]
        if ticker in levels[0]:
            frame = frame[ticker].copy()
        elif ticker in levels[-1]:
            frame = frame.xs(ticker, axis=1, level=-1).copy()
        else:
            frame = frame.copy()
            frame.columns = levels[0]
    else:
        frame = frame.copy()
    frame.columns = [str(name).strip().lower().replace(" ", "_") for name in frame.columns]
    return frame


def clean_prices(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """整理成回测用的日线表：日期升序，去掉空行和没有收盘价的行。"""
    frame = flatten_columns(frame, ticker)
    if frame.empty:
        return frame
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    price_columns = [name for name in ("open", "high", "low", "close", "adj_close", "volume") if name in frame.columns]
    if not price_columns:
        return pd.DataFrame()
    before = len(frame)
    frame = frame.dropna(how="all", subset=price_columns)
    if "close" not in frame.columns:
        return pd.DataFrame()
    frame = frame.dropna(subset=["close"])
    dropped = before - len(frame)
    if dropped:
        logger.info("%s 删除 %s 行缺失或空白记录", ticker, dropped)
    # auto_adjust=True 时收盘价已经是前复权，Adj Close 往往不再单独给出。
    if "adj_close" not in frame.columns:
        frame["adj_close"] = frame["close"]
    keep = [name for name in ("open", "high", "low", "close", "adj_close", "volume") if name in frame.columns]
    return frame.loc[:, keep]


def to_storage_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """列名与现有 CSV 对齐，日期写成 Date 列。"""
    renamed = frame.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adj_close": "Adj Close",
            "volume": "Volume",
        }
    )
    renamed = renamed.reset_index()
    date_column = "Date" if "Date" in renamed.columns else renamed.columns[0]
    renamed = renamed.rename(columns={date_column: "Date"})
    renamed["Date"] = pd.to_datetime(renamed["Date"]).dt.strftime("%Y-%m-%d")
    return renamed


def save_ticker(frame: pd.DataFrame, ticker: str, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{ticker}.csv"
    to_storage_frame(frame).to_csv(path, index=False)
    return path


def save_combined(frames: dict[str, pd.DataFrame], data_dir: Path) -> Path:
    pieces = []
    for ticker, frame in frames.items():
        stored = to_storage_frame(frame)
        stored.insert(0, "Ticker", ticker)
        pieces.append(stored)
    combined = pd.concat(pieces, ignore_index=True)
    path = data_dir / COMBINED_NAME
    combined.to_csv(path, index=False)
    return path


def history_is_short(frame: pd.DataFrame, start: str) -> bool:
    if frame.empty:
        return True
    first = pd.Timestamp(frame.index.min())
    last = pd.Timestamp(frame.index.max())
    span_years = (last - first).days / 365.25
    requested = pd.Timestamp(start)
    # 上市晚于请求起点一个多月，视为这段历史不足 10 年。
    return span_years < MIN_HISTORY_YEARS or first > requested + pd.Timedelta(days=40)


def fetch_one(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """下载并清洗一只股票。网络或空数据都会重试。"""
    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = download_raw(ticker, start, end)
            cleaned = clean_prices(raw, ticker)
            if cleaned.empty:
                last_error = "返回空表"
                logger.warning("%s 第 %s 次%s", ticker, attempt, last_error)
            else:
                return cleaned
        except Exception as exc:
            last_error = str(exc)
            logger.warning("%s 第 %s 次下载失败：%s", ticker, attempt, last_error)
        if attempt < MAX_RETRIES:
            time.sleep(SLEEP_SECONDS)
    logger.error("%s 在 %s 次尝试后放弃：%s", ticker, MAX_RETRIES, last_error)
    return None


def download_all(
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    data_dir: Path = DATA_DIR,
) -> dict[str, pd.DataFrame]:
    start, end = resolve_dates(start or START, end or END)
    tickers = tickers or list(TICKERS)
    logger.info("下载 %s 只股票，%s 至 %s，前复权日线", len(tickers), start, end)
    saved: dict[str, pd.DataFrame] = {}
    short: list[str] = []
    failed: list[str] = []
    for index, ticker in enumerate(tickers, start=1):
        frame = fetch_one(ticker, start, end)
        if frame is None:
            failed.append(ticker)
        else:
            path = save_ticker(frame, ticker, data_dir)
            if history_is_short(frame, start):
                short.append(ticker)
                logger.warning(
                    "%s 历史不足 %.1f 年：%s 条，%s 至 %s，已保存 %s",
                    ticker,
                    MIN_HISTORY_YEARS,
                    len(frame),
                    frame.index.min().date(),
                    frame.index.max().date(),
                    path,
                )
            else:
                logger.info(
                    "%s 成功：%s 条，%s 至 %s，已保存 %s",
                    ticker,
                    len(frame),
                    frame.index.min().date(),
                    frame.index.max().date(),
                    path,
                )
            saved[ticker] = frame
        if index < len(tickers):
            time.sleep(SLEEP_SECONDS)
    if saved:
        combined = save_combined(saved, data_dir)
        logger.info("合并文件 %s，共 %s 只", combined, len(saved))
    logger.info("完成：成功 %s，历史较短 %s，失败 %s", len(saved), short or "无", failed or "无")
    return saved


def main() -> None:
    download_all()


if __name__ == "__main__":
    main()
