"""
下载美国农业相关股票过去 10 年的日线行情，整理成长格式并保存为 CSV。

依赖（本脚本不执行安装，请先自行安装）：
    pip install yfinance pandas

运行：
    python fetch_us_agri_stocks.py
"""

from __future__ import annotations

import yfinance as yf
import pandas as pd

# ---------------------------------------------------------------- 板块映射
SECTOR_MAP = {
    "种植养殖加工": [
        "ADM", "BG", "TSN", "CALM", "DOLE", "ALCO", "LMNR",
        "VITL", "AVO", "LND", "VFF", "SDOT", "AFRI",
    ],
    "农机": ["DE", "AGCO", "CNH", "TTC"],
    "化肥农化": ["CTVA", "NTR", "MOS", "CF", "FMC", "IPI", "ICL", "SMG"],
    "动保饲料": ["ZTS", "ELAN", "DAR", "NEOG", "PPC"],
}

OUTPUT = "us_agri_stocks_10y.csv"
FIELDS = ["ticker", "sector", "date", "open", "high", "low", "close", "volume"]


def main() -> None:
    tickers = [ticker for names in SECTOR_MAP.values() for ticker in names]
    sector_of = {ticker: sector for sector, names in SECTOR_MAP.items() for ticker in names}

    print(f"开始下载 {len(tickers)} 只标的的日线（period=10y, interval=1d, auto_adjust=True）…")
    data = yf.download(
        tickers=tickers,
        period="10y",
        interval="1d",
        auto_adjust=True,      # 新版 yfinance 默认就是 True，列名不含 adj close
        group_by="ticker",     # 列是 MultiIndex：(ticker, field)
        threads=True,
        progress=False,
    )

    if data is None or data.empty:
        print("下载结果为空：可能是网络不通、Yahoo Finance 限流或请求超时。未生成 CSV。")
        return

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        try:
            sub = data[ticker]
        except KeyError:
            print(f"{ticker}: 无数据，跳过")
            continue

        if sub is None or sub.empty:
            print(f"{ticker}: 无数据，跳过")
            continue

        sub = sub.dropna(how="all")
        if sub.empty:
            print(f"{ticker}: 无数据，跳过")
            continue

        # 日期从索引变成普通列
        sub = sub.reset_index()
        sub.columns = [str(column).lower() for column in sub.columns]
        if "date" not in sub.columns:                      # 索引没有名字时兜底
            sub = sub.rename(columns={sub.columns[0]: "date"})

        sub["ticker"] = ticker
        sub["sector"] = sector_of[ticker]

        for column in ("open", "high", "low", "close", "volume"):
            if column not in sub.columns:
                sub[column] = pd.NA

        frames.append(sub.loc[:, FIELDS])

    if not frames:
        print("没有任何标的取到数据，未生成 CSV。")
        return

    merged = pd.concat(frames, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d")
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
    merged.to_csv(OUTPUT, index=False)

    print(f"\n已保存 {OUTPUT}：{len(merged)} 行，{merged['ticker'].nunique()} 只标的\n")

    # 起止日期与行数：历史不足 10 年的公司（CTVA、NTR、VITL 等）一眼可见
    summary = merged.groupby("ticker")["date"].agg(["min", "max", "count"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
