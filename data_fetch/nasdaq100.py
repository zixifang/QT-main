"""下载纳斯达克100指数和当前成分股自 2015-01-01 起的前复权日线。

运行：python -m data_fetch.nasdaq100
重抓：python -m data_fetch.nasdaq100 --force

数据来自 yfinance（Yahoo Finance 的非官方接口，无需 API Key）。
已有且非空的 CSV 会跳过。单只失败会重试，不会中断其余股票。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import time
import urllib.parse
import urllib.request
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
START = "2015-01-01"
END = ""
# local：读本地名单。wiki：抓维基百科 Nasdaq-100 表格。
CONSTITUENTS_SOURCE = "local"
CONSTITUENTS_FILE = Path(__file__).resolve().parent / "nasdaq100_constituents.txt"
WIKI_URLS = (
    "https://en.wikipedia.org/wiki/Nasdaq-100",
    "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
)
INDEX_TICKER = "^NDX"
INDEX_STEM = "NDX"
PROXY_TICKER = "QQQ"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "nasdaq100"
COMBINED_NAME = "nasdaq100.csv"
QUALITY_NAME = "quality.csv"
FAILED_NAME = "failed.txt"
INTERVAL = "1d"
AUTO_ADJUST = True
MAX_RETRIES = 3
SLEEP_SECONDS = 1.0
BACKOFF_BASE = 2.0
# yfinance 易限流，eastmoney 批量时会断开。sina 返回未复权成交价，Adj Close 与 Close 相同。
PRICE_SOURCE = "sina"
CHECKPOINT_NAME = "checkpoint.txt"
EASTMONEY_SEARCH = "https://searchapi.eastmoney.com/api/suggest/get"
EASTMONEY_KLINE = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
EASTMONEY_TOKEN = "D43BF722C8E33BDC906FB84D85E326E8"
SINA_KLINE = "https://stock.finance.sina.com.cn/usstock/api/json.php/US_MinKService.getDailyK?symbol="
_SECIDS = {"NDX": "100.NDX100", "QQQ": "105.QQQ"}
TICKER_TOKEN = re.compile(r"^[A-Z0-9.\-]+")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("nasdaq100")


def resolve_dates(start: str = START, end: str = END) -> tuple[str, str]:
    """得到闭区间上的开始日、结束日。结束日本身要被下载到。"""
    end_day = date.fromisoformat(end) if end else date.today()
    start_day = date.fromisoformat(start)
    if start_day >= end_day:
        raise ValueError(f"开始日 {start_day} 必须早于结束日 {end_day}")
    return start_day.isoformat(), end_day.isoformat()


def http_body(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def http_json(url: str) -> dict:
    payload = http_body(url)
    if not isinstance(payload, dict):
        raise ValueError(f"返回不是 JSON 对象：{url}")
    return payload


def eastmoney_secid(ticker: str) -> str:
    """把美股代码换成东方财富 QuoteID。指数 NDX 固定为纳斯达克100。"""
    symbol = ticker.upper().lstrip("^")
    cached = _SECIDS.get(symbol)
    if cached:
        return cached
    query = urllib.parse.urlencode({"input": symbol, "type": "14", "token": EASTMONEY_TOKEN, "count": "8"})
    rows = (http_json(f"{EASTMONEY_SEARCH}?{query}").get("QuotationCodeTable") or {}).get("Data") or []
    for prefer_stock in (True, False):
        for row in rows:
            if str(row.get("Code", "")).upper() != symbol or not row.get("QuoteID"):
                continue
            if prefer_stock and str(row.get("Classify")) != "UsStock":
                continue
            quote = str(row["QuoteID"])
            _SECIDS[symbol] = quote
            return quote
    raise LookupError(f"东方财富没有 {ticker} 的行情代码")


def bars_from_klines(klines: list[str]) -> pd.DataFrame:
    """东方财富日线收成与 yfinance 清洗结果相同的列。前复权收盘价记入 adj_close。"""
    records = []
    for line in klines:
        parts = line.split(",")
        if len(parts) < 6:
            continue
        records.append((parts[0], parts[1], parts[3], parts[4], parts[2], parts[5]))
    frame = pd.DataFrame(records, columns=["date", "open", "high", "low", "close", "volume"])
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["adj_close"] = frame["close"]
    return frame.set_index("date")


def sina_symbol(ticker: str) -> str:
    symbol = ticker.upper().lstrip("^")
    if symbol == "NDX":
        return ".NDX"
    return symbol


def frame_from_ohlcv(records: list[tuple]) -> pd.DataFrame:
    frame = pd.DataFrame(records, columns=["date", "open", "high", "low", "close", "volume"])
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["adj_close"] = frame["close"]
    return frame.set_index("date")


def sina_raw(ticker: str, start: str, end: str) -> pd.DataFrame:
    """新浪日线。日期筛到请求区间，成交价同时写入 adj_close。"""
    payload = http_body(SINA_KLINE + urllib.parse.quote(sina_symbol(ticker)))
    if not isinstance(payload, list):
        return pd.DataFrame()
    records = []
    for row in payload:
        day = str(row.get("d", ""))
        if start <= day <= end:
            records.append((day, row.get("o"), row.get("h"), row.get("l"), row.get("c"), row.get("v")))
    return frame_from_ohlcv(records)


def eastmoney_raw(ticker: str, start: str, end: str) -> pd.DataFrame:
    """参数保持逗号原样。urlencode 会把逗号转成 %2C，接口会直接断开。"""
    secid = eastmoney_secid(ticker)
    url = (
        f"{EASTMONEY_KLINE}?secid={secid}&klt=101&fqt=1"
        f"&beg={start.replace('-', '')}&end={end.replace('-', '')}"
        "&fields1=f1,f2,f3,f4,f5,f6"
        "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
    )
    klines = (http_json(url).get("data") or {}).get("klines") or []
    return bars_from_klines(klines)


def download_raw(ticker: str, start: str, end: str) -> pd.DataFrame:
    """下载单只代码。yfinance 的 end 是开区间，会自动加一天。"""
    if PRICE_SOURCE == "sina":
        return sina_raw(ticker, start, end)
    if PRICE_SOURCE == "eastmoney":
        return eastmoney_raw(ticker, start, end)
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
    if not isinstance(frame, pd.DataFrame):
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


def clean_prices(frame: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, int]:
    """整理成回测用的日线表，并返回含缺失价格的行数。"""
    frame = flatten_columns(frame, ticker)
    if frame.empty:
        return frame, 0
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    price_columns = [name for name in ("open", "high", "low", "close", "adj_close", "volume") if name in frame.columns]
    if "close" not in frame.columns:
        return pd.DataFrame(), 0
    missing = int(frame.loc[:, price_columns].isna().any(axis=1).sum())
    before = len(frame)
    frame = frame.dropna(how="all", subset=price_columns).dropna(subset=["close"])
    if before - len(frame):
        logger.info("%s 删除 %s 行缺失或空白记录", ticker, before - len(frame))
    if "adj_close" not in frame.columns:
        frame["adj_close"] = frame["close"]
    keep = [name for name in ("open", "high", "low", "close", "adj_close", "volume") if name in frame.columns]
    return frame.loc[:, keep], missing


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
    order = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    return renamed.loc[:, [name for name in order if name in renamed.columns]]


def save_ticker(frame: pd.DataFrame, stem: str, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{stem}.csv"
    to_storage_frame(frame).to_csv(path, index=False)
    return path


def file_is_ready(path: Path) -> bool:
    """已存在且能读出至少一行时，断点续跑可以跳过。"""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        sample = pd.read_csv(path, nrows=1)
    except Exception:
        logger.exception("无法读取已有文件 %s", path)
        return False
    return not sample.empty


def backoff_sleep(attempt: int) -> None:
    time.sleep(SLEEP_SECONDS * (BACKOFF_BASE ** (attempt - 1)))


def fetch_one(ticker: str, start: str, end: str) -> tuple[pd.DataFrame | None, int]:
    """下载并清洗一只代码。网络或空数据都会重试，最后一次失败不中断整批。"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            cleaned, missing = clean_prices(download_raw(ticker, start, end), ticker)
        except Exception:
            logger.exception("%s 第 %s 次下载失败", ticker, attempt)
            if attempt < MAX_RETRIES:
                backoff_sleep(attempt)
            continue
        if not cleaned.empty:
            return cleaned, missing
        logger.warning("%s 第 %s 次返回空表", ticker, attempt)
        if attempt < MAX_RETRIES:
            backoff_sleep(attempt)
    logger.error("%s 在 %s 次尝试后放弃", ticker, MAX_RETRIES)
    return None, 0


def unique_tickers(tickers: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for ticker in tickers:
        if ticker not in seen:
            seen.add(ticker)
            ordered.append(ticker)
    return ordered


def load_local_constituents(path: Path = CONSTITUENTS_FILE) -> list[str]:
    tickers: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip().upper()
        if text:
            tickers.append(text)
    if not tickers:
        raise ValueError(f"成分股名单为空：{path}")
    return unique_tickers(tickers)


def normalize_ticker(value: object) -> str:
    match = TICKER_TOKEN.match(str(value).strip().upper())
    return match.group(0) if match else ""


def ticker_column(table: pd.DataFrame) -> str | None:
    columns = {str(name).strip().lower(): name for name in table.columns}
    for key in ("ticker", "symbol"):
        if key in columns:
            return str(columns[key])
    return None


def tickers_from_html(html: bytes) -> list[str]:
    best: list[str] = []
    for table in pd.read_html(html):
        column = ticker_column(table)
        if column is None:
            continue
        found = [normalize_ticker(value) for value in table[column]]
        found = [ticker for ticker in found if ticker]
        if len(found) > len(best):
            best = found
    return unique_tickers(best)


def load_wiki_constituents() -> list[str]:
    """抓维基百科成分股表。页面失败会记日志并继续试下一个地址。"""
    last_error: Exception | None = None
    for url in WIKI_URLS:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "QT-main nasdaq100 research"})
            with urllib.request.urlopen(request, timeout=30) as response:
                tickers = tickers_from_html(response.read())
        except Exception as exc:
            last_error = exc
            logger.exception("读取维基百科失败：%s", url)
            continue
        if len(tickers) >= 100:
            return tickers
        logger.warning("维基百科成分股不足 100 只：%s 得到 %s", url, len(tickers))
    if last_error is not None:
        raise RuntimeError("维基百科成分股抓取失败") from last_error
    raise RuntimeError("维基百科成分股不足 100 只")


def load_constituents() -> list[str]:
    if CONSTITUENTS_SOURCE == "wiki":
        tickers = load_wiki_constituents()
        logger.info("成分股来自维基百科，共 %s 只", len(tickers))
        return tickers
    tickers = load_local_constituents()
    logger.info("成分股来自 %s，共 %s 只", CONSTITUENTS_FILE.name, len(tickers))
    return tickers


def fetch_index(start: str, end: str) -> tuple[pd.DataFrame | None, int, str]:
    """优先拉 ^NDX。失败时改用 QQQ 的调整收盘价，并在日志里标明代理。"""
    frame, missing = fetch_one(INDEX_TICKER, start, end)
    if frame is not None:
        return frame, missing, PRICE_SOURCE
    logger.warning("指数 %s 拉取失败，改用 %s 的调整收盘价作为代理", INDEX_TICKER, PROXY_TICKER)
    proxy, missing = fetch_one(PROXY_TICKER, start, end)
    if proxy is None:
        return None, 0, PROXY_TICKER
    proxy = proxy.copy()
    proxy["close"] = proxy["adj_close"]
    logger.warning("指数文件 %s.csv 使用 %s 行情，收盘价替换为调整收盘价", INDEX_STEM, PROXY_TICKER)
    return proxy, missing, "QQQ代理"


def read_stored(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame.sort_values("Date")


def missing_in_stored(frame: pd.DataFrame) -> int:
    columns = [name for name in ("Open", "High", "Low", "Close", "Adj Close", "Volume") if name in frame.columns]
    if not columns:
        return 0
    return int(frame.loc[:, columns].isna().any(axis=1).sum())


def quality_row(stem: str, frame: pd.DataFrame, missing: int, source: str, end: str) -> dict:
    first = pd.Timestamp(frame["Date"].iloc[0])
    last = pd.Timestamp(frame["Date"].iloc[-1])
    halted = int(last < pd.Timestamp(end) - pd.Timedelta(days=10))
    return {
        "ticker": stem,
        "first_date": first.strftime("%Y-%m-%d"),
        "last_date": last.strftime("%Y-%m-%d"),
        "rows": int(len(frame)),
        "missing": int(missing),
        "halted_or_delisted": halted,
        "source": source,
    }


def download_symbol(
    ticker: str,
    stem: str,
    start: str,
    end: str,
    force: bool,
    source: str,
    data_dir: Path,
    pending: list[str],
) -> tuple[pd.DataFrame | None, dict | None, bool]:
    """返回行情、质量行，以及这次是否真的访问了网络。"""
    path = data_dir / f"{stem}.csv"
    if not force and file_is_ready(path):
        stored = read_stored(path)
        logger.info("跳过已有文件 %s", path.name)
        mark_done(pending, stem, data_dir)
        return stored, quality_row(stem, stored, missing_in_stored(stored), "cached", end), False
    frame, missing = fetch_one(ticker, start, end)
    if frame is None:
        return None, None, True
    save_ticker(frame, stem, data_dir)
    mark_done(pending, stem, data_dir)
    stored = to_storage_frame(frame)
    logger.info("%s 成功：%s 条，%s 至 %s", stem, len(stored), stored["Date"].iloc[0], stored["Date"].iloc[-1])
    return stored, quality_row(stem, stored, missing, source, end), True


def write_checkpoint(pending: list[str], data_dir: Path) -> None:
    """还没落成 CSV 的代码。中断后重跑会跳过已有文件，断点文件只保留剩余项。"""
    data_dir.mkdir(parents=True, exist_ok=True)
    body = "\n".join(pending)
    text = f"# source={PRICE_SOURCE}\n# pending={len(pending)}\n"
    if body:
        text += body + "\n"
    (data_dir / CHECKPOINT_NAME).write_text(text, encoding="utf-8")


def mark_done(pending: list[str], stem: str, data_dir: Path) -> None:
    if stem in pending:
        pending.remove(stem)
        write_checkpoint(pending, data_dir)


def pending_symbols(tickers: list[str], data_dir: Path, force: bool) -> list[str]:
    symbols = [INDEX_STEM, *tickers]
    if force:
        return symbols
    return [stem for stem in symbols if not file_is_ready(data_dir / f"{stem}.csv")]


def write_reports(rows: list[dict], failed: list[str], data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(data_dir / QUALITY_NAME, index=False)
    text = ("\n".join(failed) + "\n") if failed else ""
    (data_dir / FAILED_NAME).write_text(text, encoding="utf-8")
    logger.info("质量报告 %s 行，失败 %s", len(rows), failed or "无")


def save_combined(frames: dict[str, pd.DataFrame], data_dir: Path) -> Path:
    pieces: list[pd.DataFrame] = []
    for ticker, frame in frames.items():
        stored = frame.drop(columns=["Ticker"], errors="ignore").copy()
        stored["Date"] = pd.to_datetime(stored["Date"]).dt.strftime("%Y-%m-%d")
        stored.insert(0, "Ticker", ticker)
        pieces.append(stored)
    path = data_dir / COMBINED_NAME
    pd.concat(pieces, ignore_index=True).to_csv(path, index=False)
    logger.info("合并文件 %s，共 %s 只", path, len(frames))
    return path


def collect_index(
    start: str,
    end: str,
    force: bool,
    saved: dict[str, pd.DataFrame],
    rows: list[dict],
    failed: list[str],
    data_dir: Path,
    pending: list[str],
) -> None:
    path = data_dir / f"{INDEX_STEM}.csv"
    if not force and file_is_ready(path):
        stored = read_stored(path)
        logger.info("跳过已有文件 %s", path.name)
        saved[INDEX_STEM] = stored
        rows.append(quality_row(INDEX_STEM, stored, missing_in_stored(stored), "cached", end))
        mark_done(pending, INDEX_STEM, data_dir)
        return
    frame, missing, source = fetch_index(start, end)
    time.sleep(SLEEP_SECONDS)
    if frame is None:
        failed.append(INDEX_TICKER)
        return
    save_ticker(frame, INDEX_STEM, data_dir)
    mark_done(pending, INDEX_STEM, data_dir)
    stored = to_storage_frame(frame)
    first = pd.Timestamp(stored["Date"].iloc[0])
    if first > pd.Timestamp(start) + pd.Timedelta(days=40):
        logger.warning("%s 历史晚于请求起点，实际从 %s 开始", INDEX_STEM, first.strftime("%Y-%m-%d"))
    logger.info("%s 成功：%s 条，%s 至 %s，来源 %s", INDEX_STEM, len(stored), stored["Date"].iloc[0], stored["Date"].iloc[-1], source)
    saved[INDEX_STEM] = stored
    rows.append(quality_row(INDEX_STEM, stored, missing, source, end))


def collect_constituents(
    tickers: list[str],
    start: str,
    end: str,
    force: bool,
    saved: dict[str, pd.DataFrame],
    rows: list[dict],
    failed: list[str],
    data_dir: Path,
    pending: list[str],
) -> None:
    for ticker in tickers:
        stored, row, downloaded = download_symbol(ticker, ticker, start, end, force, PRICE_SOURCE, data_dir, pending)
        if stored is None or row is None:
            failed.append(ticker)
        else:
            saved[ticker] = stored
            rows.append(row)
        if downloaded:
            time.sleep(SLEEP_SECONDS)


def download_all(force: bool = False, data_dir: Path = DATA_DIR) -> None:
    start, end = resolve_dates()
    try:
        tickers = load_constituents()
    except Exception:
        logger.exception("成分股名单读取失败")
        raise
    logger.info("下载纳斯达克100，%s 至 %s，前复权日线，源 %s，force=%s", start, end, PRICE_SOURCE, force)
    saved: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    failed: list[str] = []
    pending = pending_symbols(tickers, data_dir, force)
    write_checkpoint(pending, data_dir)
    logger.info("断点已写入 %s，剩余 %s", data_dir / CHECKPOINT_NAME, len(pending))
    collect_index(start, end, force, saved, rows, failed, data_dir, pending)
    collect_constituents(tickers, start, end, force, saved, rows, failed, data_dir, pending)
    write_reports(rows, failed, data_dir)
    if saved:
        save_combined(saved, data_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="下载纳斯达克100指数和成分股日线")
    parser.add_argument("--force", action="store_true", help="忽略已有 CSV，全部重新下载")
    download_all(force=parser.parse_args().force)


if __name__ == "__main__":
    main()
