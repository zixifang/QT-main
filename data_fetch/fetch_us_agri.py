"""抓美国农业相关股票的日线，写入 data/us-agri/<TICKER>.csv。

为什么不用 yfinance
-------------------
2026-10 这台机器上 `yfinance` 对全部 30 只都返回 `YFRateLimitError`，
直连 `query1/query2.finance.yahoo.com` 的 chart 接口也是 403。所以改用
stockanalysis 的公开 history 接口（无需 key）。

代价：那个接口 **最长只给 5 年**（`range=10Y` 实际返回 1255 根），拿不到 10 年。
本脚本不编造数据：抓不到就跳过并打印，不会用模拟值填充。

用法：
    python data_fetch/fetch_us_agri.py
"""

from __future__ import annotations

import csv
import pathlib
import time

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "us-agri"

SECTOR_MAP = {
    "种植养殖加工": ["ADM", "BG", "TSN", "CALM", "DOLE", "ALCO", "LMNR",
                     "VITL", "AVO", "LND", "VFF", "SDOT", "AFRI"],
    "农机": ["DE", "AGCO", "CNH", "TTC"],
    "化肥农化": ["CTVA", "NTR", "MOS", "CF", "FMC", "IPI", "ICL", "SMG"],
    "动保饲料": ["ZTS", "ELAN", "DAR", "NEOG", "PPC"],
}

# SDOT 在 stockanalysis 上的数据明显异常：单日 +247%、20 次涨跌超 40%、
# 2021 年高点 2340 而 2026 年只剩 7 块、成交量出现过 105 股。
# 已从数据集剔除；要恢复就从这里删掉它。
QUARANTINE = {"SDOT"}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Accept": "application/json",
}
API = "https://stockanalysis.com/api/symbol/s/{ticker}/history?range=10Y"


def fetch(ticker: str) -> list[dict]:
    for _ in range(3):
        try:
            resp = requests.get(API.format(ticker=ticker.lower()), headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                rows = resp.json().get("data")
                if isinstance(rows, list) and rows:
                    return rows
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.2)
    return []


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tickers = [ticker for names in SECTOR_MAP.values() for ticker in names]

    for ticker in tickers:
        if ticker in QUARANTINE:
            print(f"{ticker}: 数据异常，已隔离，跳过")
            continue
        rows = fetch(ticker)
        if not rows:
            print(f"{ticker}: 无数据，跳过")
            continue
        rows.reverse()   # 接口是最新在前

        path = OUT / f"{ticker}.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])
            for row in rows:
                close = row["c"]
                adj = row.get("a") or close
                factor = (adj / close) if close else 1.0    # 复权折算进 OHLC
                writer.writerow([
                    row["t"],
                    f"{row['o'] * factor:.4f}",
                    f"{row['h'] * factor:.4f}",
                    f"{row['l'] * factor:.4f}",
                    f"{close * factor:.4f}",
                    f"{adj:.4f}",
                    int(float(row.get("v") or 0)),
                ])
        print(f"{ticker:6} {len(rows):5} 行  {rows[0]['t']} .. {rows[-1]['t']}")
        time.sleep(0.4)

    print(f"\n写入 {OUT.relative_to(ROOT).as_posix()}，{len(list(OUT.glob('*.csv')))} 个文件")


if __name__ == "__main__":
    main()
