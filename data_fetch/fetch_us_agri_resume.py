"""抓美国农业股日线，带断点续爬：失败/过短的标的记进状态文件，之后重跑只补这些。

为什么需要断点
--------------
Yahoo 这个出口 IP 被限流（yfinance 报 YFRateLimitError，直连 chart 接口 403），
一次跑不完 30 只。所以把每只的结果写进 `_fetch_state.json`：

- `ok`    行数够，下次直接跳过
- `short` 抓到了但行数不够（例如 CTVA 只有 17 根）
- `failed` 没抓到

重跑时只处理 `short` / `failed`，抓到了就覆盖写盘并改成 `ok`。
多轮之间会等待，避免继续撞限流。

用法：
    python data_fetch/fetch_us_agri_resume.py                 # 默认 6 轮，每轮间隔 120 秒
    python data_fetch/fetch_us_agri_resume.py --rounds 2 --sleep 20
    python data_fetch/fetch_us_agri_resume.py --only CTVA     # 只补一只
    python data_fetch/fetch_us_agri_resume.py --force         # 全部重抓
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import time
from datetime import datetime

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "us-agri"
STATE = OUT / "_fetch_state.json"
LONG = ROOT / "data" / "us-agri" / "us_agri_stocks_10y.csv"

SECTOR_MAP = {
    "种植养殖加工": ["ADM", "BG", "TSN", "CALM", "DOLE", "ALCO", "LMNR",
                     "VITL", "AVO", "LND", "VFF", "SDOT", "AFRI"],
    "农机": ["DE", "AGCO", "CNH", "TTC"],
    "化肥农化": ["CTVA", "NTR", "MOS", "CF", "FMC", "IPI", "ICL", "SMG"],
    "动保饲料": ["ZTS", "ELAN", "DAR", "NEOG", "PPC"],
}
SECTOR_OF = {ticker: sector for sector, names in SECTOR_MAP.items() for ticker in names}

# SDOT 在 stockanalysis 上的数据明显异常：单日 +247%、20 次涨跌超 40%、
# 2021 年高点 2340 而 2026 年只剩 7 块、成交量出现过 105 股。
# 已从数据集剔除；要恢复就从这里删掉它。
QUARANTINE = {"SDOT"}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Accept": "application/json",
}
SA_API = "https://stockanalysis.com/api/symbol/s/{ticker}/history?range=10Y"


# ---------------------------------------------------------------- 数据源

def from_yfinance(ticker: str) -> list[tuple]:
    """返回 [(date, open, high, low, close, adj, volume)]；被限流就抛异常。"""
    try:
        import yfinance as yf
    except ImportError:
        return []
    frame = yf.Ticker(ticker).history(period="10y", interval="1d", auto_adjust=True)
    if frame is None or frame.empty:
        return []
    rows = []
    for index, row in frame.iterrows():
        rows.append((index.strftime("%Y-%m-%d"), float(row["Open"]), float(row["High"]),
                     float(row["Low"]), float(row["Close"]), float(row["Close"]),
                     int(float(row["Volume"]))))
    return rows


def from_stockanalysis(ticker: str) -> list[tuple]:
    for _ in range(2):
        try:
            resp = requests.get(SA_API.format(ticker=ticker.lower()), headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                payload = resp.json().get("data")
                if isinstance(payload, list) and payload:
                    payload.reverse()          # 接口最新在前
                    rows = []
                    for item in payload:
                        close = item["c"]
                        adj = item.get("a") or close
                        k = (adj / close) if close else 1.0
                        rows.append((item["t"], item["o"] * k, item["h"] * k,
                                     item["l"] * k, close * k, adj,
                                     int(float(item.get("v") or 0))))
                    return rows
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.0)
    return []


def fetch_one(ticker: str) -> tuple[list[tuple], str]:
    """先试 yfinance（能给 10 年），不行退回 stockanalysis（只有 5 年）。"""
    try:
        rows = from_yfinance(ticker)
        if rows:
            return rows, "yfinance"
    except Exception as exc:  # noqa: BLE001
        print(f"    yfinance: {type(exc).__name__}")
    rows = from_stockanalysis(ticker)
    return rows, ("stockanalysis" if rows else "")


# ---------------------------------------------------------------- 落盘

def write_csv(ticker: str, rows: list[tuple]) -> None:
    path = OUT / f"{ticker}.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])
        for day, o, h, l, c, a, v in rows:
            writer.writerow([day, f"{o:.4f}", f"{h:.4f}", f"{l:.4f}", f"{c:.4f}", f"{a:.4f}", v])


def write_long() -> int:
    """把现有 CSV 拼成长表，供直接分析用。"""
    total = 0
    with LONG.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ticker", "sector", "date", "open", "high", "low", "close", "volume"])
        for ticker in SECTOR_OF:
            path = OUT / f"{ticker}.csv"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as src:
                reader = csv.DictReader(src)
                for row in reader:
                    writer.writerow([ticker, SECTOR_OF[ticker], row["Date"], row["Open"],
                                     row["High"], row["Low"], row["Close"], row["Volume"]])
                    total += 1
    return total


def load_state() -> dict:
    if STATE.exists():
        # utf-8-sig：记事本存过的 JSON 会带 BOM，这里一并兼容
        return json.loads(STATE.read_text(encoding="utf-8-sig"))
    return {"updated": None, "tickers": {}}


def save_state(state: dict) -> None:
    state["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- 主流程

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=6, help="最多重试几轮")
    parser.add_argument("--sleep", type=float, default=120.0, help="每轮之间等多少秒")
    parser.add_argument("--min-rows", type=int, default=500, help="少于此行数视为不足")
    parser.add_argument("--only", default="", help="只处理这些代码，逗号分隔")
    parser.add_argument("--force", action="store_true", help="忽略断点，全部重抓")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    state = load_state()
    tickers = [t.strip().upper() for t in args.only.split(",") if t.strip()] or list(SECTOR_OF)
    tickers = [t for t in tickers if t not in QUARANTINE]

    for ticker in tickers:
        state["tickers"].setdefault(ticker, {"status": "pending", "attempts": 0})

    pending = [t for t in tickers
               if args.force or state["tickers"][t]["status"] != "ok"]
    done_before = len(tickers) - len(pending)
    print(f"共 {len(tickers)} 只，断点里已完成 {done_before} 只，本次要处理 {len(pending)} 只\n")

    for round_index in range(1, args.rounds + 1):
        if not pending:
            break
        print(f"— 第 {round_index}/{args.rounds} 轮，待处理 {len(pending)} 只 —")
        still: list[str] = []

        for ticker in pending:
            entry = state["tickers"][ticker]
            entry["attempts"] = entry.get("attempts", 0) + 1
            rows, source = fetch_one(ticker)

            if rows and len(rows) >= args.min_rows:
                write_csv(ticker, rows)
                entry.update({"status": "ok", "source": source, "rows": len(rows),
                              "first": rows[0][0], "last": rows[-1][0]})
                print(f"  {ticker:6} ✓ {len(rows):5} 行  {rows[0][0]} .. {rows[-1][0]}  ({source})")
            elif rows:
                # 行数不够就先不落盘：半截数据混进 data/ 会把整个数据集的时间交集拉短
                entry.update({"status": "short", "source": source, "rows": len(rows),
                              "first": rows[0][0], "last": rows[-1][0]})
                print(f"  {ticker:6} ! 只有 {len(rows)} 行（<{args.min_rows}），不写盘，下轮再试")
                still.append(ticker)
            else:
                entry.update({"status": "failed"})
                print(f"  {ticker:6} ✗ 无数据（第 {entry['attempts']} 次）")
                still.append(ticker)
            save_state(state)          # 每只都落盘，随时中断都不丢
            time.sleep(0.4)

        pending = still
        if pending and round_index < args.rounds:
            wait = args.sleep * round_index
            print(f"  等 {wait:.0f} 秒后重试剩下的 {len(pending)} 只…\n")
            time.sleep(wait)

    total = write_long()
    print(f"\n长表已更新：{LONG.relative_to(ROOT).as_posix()}（{total} 行）")

    ok = [t for t, e in state["tickers"].items() if e["status"] == "ok"]
    short = [t for t, e in state["tickers"].items() if e["status"] == "short"]
    failed = [t for t, e in state["tickers"].items() if e["status"] == "failed"]
    print(f"完成 {len(ok)} 只；不足 {len(short)} 只 {short}；失败 {len(failed)} 只 {failed}")
    if short or failed:
        print("断点已保存，稍后重跑同一条命令即可只补这些（或 --only 指定）。")


if __name__ == "__main__":
    main()
