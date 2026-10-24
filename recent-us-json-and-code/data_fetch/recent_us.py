"""下载近期美股大盘日线，供三个技术算法做样本外回测。

数据源：新浪美股日线接口。返回未复权成交价，因此 `Adj Close` 与 `Close` 相同。
区间：2024-01-02 至 2026-10-23，每只股票约 706 个交易日。

运行：python -m data_fetch.recent_us
输出：data/recent-us-2024-2026/{代码}.csv
       data/recent-us-2024-2026/recent_us_2024_2026.csv（带 Ticker 列的长表）
       data/recent-us-2024-2026/quality.csv
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "recent-us-2024-2026"
START = "2024-01-02"
END = "2026-10-23"
SINA_KLINE = "https://stock.finance.sina.com.cn/usstock/api/json.php/US_MinKService.getDailyK?symbol="
MAX_RETRIES = 4

# 新浪返回未复权价；这些是区间内发生的拆股，按比例回溯调整，避免被当成 -90% 收益。
SPLITS = {
    "NVDA": (("2024-06-10", 10.0),),
    "AVGO": (("2024-07-15", 10.0),),
    "NFLX": (("2025-11-17", 10.0),),
    "WMT": (("2024-02-26", 3.0),),
}

# 20 只大盘股，覆盖科技、金融、医疗、消费、工业、能源。
TICKERS = (
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "AVGO", "JPM", "BAC",
    "JNJ", "UNH", "WMT", "COST", "KO",
    "CAT", "GE", "XOM", "CVX", "NFLX",
)
COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]


def fetch_ticker(ticker: str) -> pd.DataFrame:
    """下载单只股票，筛到 START..END，返回按日期升序的行情表。"""
    url = SINA_KLINE + urllib.parse.quote(ticker.upper())
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                rows = json.loads(response.read().decode("utf-8"))
            if not isinstance(rows, list):
                raise ValueError(f"{ticker} 返回不是列表")
            records = []
            for row in rows:
                day = str(row.get("d", ""))
                if START <= day <= END:
                    records.append(
                        {
                            "Date": day,
                            "Open": row.get("o"),
                            "High": row.get("h"),
                            "Low": row.get("l"),
                            "Close": row.get("c"),
                            "Adj Close": row.get("c"),
                            "Volume": row.get("v"),
                        }
                    )
            frame = pd.DataFrame(records, columns=COLUMNS)
            if frame.empty:
                raise ValueError(f"{ticker} 在 {START}..{END} 没有数据")
            frame["Date"] = pd.to_datetime(frame["Date"])
            for column in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame = frame.dropna(subset=["Date", "Close"]).sort_values("Date")
            frame = frame.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
            return frame
        except Exception as exc:  # 网络抖动或接口临时断开，重试
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"{ticker} 下载失败：{last_error}")


def adjust_splits(ticker: str, frame: pd.DataFrame) -> pd.DataFrame:
    """把拆股日之前的价格除以拆股比例，成交量乘以比例，保持收益连续。"""
    adjusted = frame.copy()
    for day, ratio in SPLITS.get(ticker, ()):
        cutoff = pd.Timestamp(day)
        mask = adjusted["Date"] < cutoff
        if not mask.any():
            continue
        for column in ("Open", "High", "Low", "Close", "Adj Close"):
            adjusted.loc[mask, column] = adjusted.loc[mask, column] / ratio
        adjusted.loc[mask, "Volume"] = adjusted.loc[mask, "Volume"] * ratio
    returns = adjusted["Close"].pct_change()
    leftover = adjusted[returns.abs() > 0.5]
    if not leftover.empty:
        days = ", ".join(leftover["Date"].dt.strftime("%Y-%m-%d"))
        print(f"警告 {ticker} 仍有超过 50% 的单日跳变：{days}", flush=True)
    return adjusted


def write_symbol(ticker: str, frame: pd.DataFrame) -> Path:
    path = DATA_DIR / f"{ticker}.csv"
    output = frame.copy()
    output["Date"] = output["Date"].dt.strftime("%Y-%m-%d")
    output.to_csv(path, index=False, float_format="%.6f")
    return path


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    quality: list[dict] = []
    failures: list[str] = []

    for ticker in TICKERS:
        try:
            frame = fetch_ticker(ticker)
            frame = adjust_splits(ticker, frame)
        except Exception as exc:
            print(f"失败 {ticker}: {exc}", flush=True)
            failures.append(ticker)
            continue
        write_symbol(ticker, frame)
        frames[ticker] = frame
        quality.append(
            {
                "ticker": ticker,
                "rows": len(frame),
                "first": frame["Date"].iloc[0].strftime("%Y-%m-%d"),
                "last": frame["Date"].iloc[-1].strftime("%Y-%m-%d"),
            }
        )
        print(f"已写入 {ticker}.csv：{len(frame)} 行，{quality[-1]['first']} 至 {quality[-1]['last']}", flush=True)

    if not frames:
        raise RuntimeError("没有任何股票下载成功")

    merged = []
    for ticker, frame in frames.items():
        piece = frame.copy()
        piece.insert(0, "Ticker", ticker)
        merged.append(piece)
    long_frame = pd.concat(merged, ignore_index=True).sort_values(["Ticker", "Date"])
    long_frame["Date"] = long_frame["Date"].dt.strftime("%Y-%m-%d")
    long_frame.to_csv(DATA_DIR / "recent_us_2024_2026.csv", index=False, float_format="%.6f")

    pd.DataFrame(quality).to_csv(DATA_DIR / "quality.csv", index=False)
    print(f"完成：{len(frames)} 只股票，长表 {len(long_frame)} 行，目录 {DATA_DIR}", flush=True)
    if failures:
        raise RuntimeError(f"以下股票失败：{', '.join(failures)}")


if __name__ == "__main__":
    main()
