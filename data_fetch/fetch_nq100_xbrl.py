"""从 SEC EDGAR 拉取纳斯达克100的 XBRL 基本面，写成因子研究用的长表。

运行（在项目根目录）：
    python -m data_fetch.fetch_nq100_xbrl

SEC 要求 User-Agent 里有可联系的邮箱。请设置环境变量 SEC_USER_AGENT，
格式为：YourName you@example.com
不要加括号，也不要使用 GitHub noreply 地址，否则 SEC 返回 403。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "xbrl"
RAW_DIR = OUTPUT_DIR / "raw"
CIK_CACHE = OUTPUT_DIR / "company_tickers.json"
CONSTITUENTS_URL = "https://yfiua.github.io/index-constituents/constituents-nasdaq100.csv"
LOCAL_CONSTITUENTS = Path(__file__).resolve().parent / "nasdaq100_constituents.txt"
CIK_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
USER_AGENT = os.environ.get("SEC_USER_AGENT", "").strip()
RATE_LIMIT_DELAY = 0.15
FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A"}
MONEY_UNITS = ("USD",)
SHARE_UNITS = ("shares",)
EPS_UNITS = ("USD/shares",)

# 每个因子：依次尝试的 (分类法, 标签)。用到后面的标签时记入 missing_concepts。
CONCEPTS: dict[str, list[tuple[str, str]]] = {
    "Assets": [("us-gaap", "Assets"), ("us-gaap", "AssetsCurrent")],
    "Revenues": [
        ("us-gaap", "Revenues"),
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ("us-gaap", "SalesRevenueNet"),
    ],
    "NetIncomeLoss": [("us-gaap", "NetIncomeLoss"), ("us-gaap", "ProfitLoss")],
    "OperatingIncomeLoss": [("us-gaap", "OperatingIncomeLoss")],
    "GrossProfit": [("us-gaap", "GrossProfit")],
    "StockholdersEquity": [
        ("us-gaap", "StockholdersEquity"),
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    ],
    "Cash": [
        ("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
        ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
    ],
    "SharesOutstanding": [
        ("us-gaap", "CommonStockSharesOutstanding"),
        ("dei", "EntityCommonStockSharesOutstanding"),
    ],
    "EarningsPerShareDiluted": [
        ("us-gaap", "EarningsPerShareDiluted"),
        ("us-gaap", "EarningsPerShareBasicAndDiluted"),
    ],
    "ResearchAndDevelopmentExpense": [("us-gaap", "ResearchAndDevelopmentExpense")],
    "CapitalExpenditure": [
        ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
        ("us-gaap", "PaymentsForCapitalImprovements"),
    ],
    "OperatingCashFlow": [
        ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
        ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
    ],
    "AssetsCurrent": [("us-gaap", "AssetsCurrent")],
    "LiabilitiesCurrent": [("us-gaap", "LiabilitiesCurrent")],
}
DEBT_PARTS = (
    ("us-gaap", "LongTermDebt"),
    ("us-gaap", "ShortTermBorrowings"),
    ("us-gaap", "DebtCurrent"),
    ("us-gaap", "Liabilities"),
)
FALLBACK_NOTES = {
    "AssetsCurrent": "仅流动资产，不是总资产",
    "EntityCommonStockSharesOutstanding": "dei 分类法，口径可能与流通股不同",
    "Liabilities": "粗略备选，含经营负债",
}


def session() -> requests.Session:
    connection = requests.Session()
    connection.headers["User-Agent"] = USER_AGENT
    connection.headers["Accept"] = "application/json"
    return connection


def get_json(connection: requests.Session, url: str) -> dict | None:
    """最多重试两次。404 视为没有这份数据。"""
    delay = 0.5
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = connection.get(url, timeout=90)
            if response.status_code == 404:
                return None
            if response.status_code == 403:
                print(
                    f"[WARN] SEC 拒绝访问（403）{url}。"
                    "请把 SEC_USER_AGENT 改成「名字 空格 真实邮箱」，不要用括号或 noreply。"
                )
                return None
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt == 2:
                break
            time.sleep(delay)
            delay *= 2
    print(f"[WARN] 请求失败 {url}：{last_error}")
    return None


def get_nq100_tickers() -> list[str]:
    """当前纳斯达克100代码。先读公开 CSV，失败再用本地名单。"""
    try:
        frame = pd.read_csv(CONSTITUENTS_URL)
        column = "Symbol" if "Symbol" in frame.columns else frame.columns[0]
        tickers = sorted({str(value).strip().upper() for value in frame[column].dropna() if str(value).strip()})
        if tickers:
            print(f"成分股 {len(tickers)} 只，来自 {CONSTITUENTS_URL}")
            print("[NOTE] 这是当前成分，不是历史时点成分，有幸存者偏差。")
            return tickers
    except (OSError, ValueError, pd.errors.ParserError, requests.RequestException) as exc:
        print(f"[WARN] 在线成分股名单失败：{exc}。改用本地名单。")
    names = []
    for line in LOCAL_CONSTITUENTS.read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip().upper()
        if text:
            names.append(text)
    tickers = sorted(set(names))
    print(f"成分股 {len(tickers)} 只，来自本地名单。这是当前成分，不是历史时点成分。")
    return tickers


def get_cik_mapping(connection: requests.Session, refresh: bool) -> dict[str, str]:
    """SEC ticker 到 10 位 CIK。映射缓存在 data/xbrl/company_tickers.json。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if CIK_CACHE.exists() and not refresh:
        payload = json.loads(CIK_CACHE.read_text(encoding="utf-8"))
    else:
        time.sleep(RATE_LIMIT_DELAY)
        payload = get_json(connection, CIK_URL)
        if payload is None:
            raise RuntimeError("没有拿到 SEC ticker-CIK 映射")
        CIK_CACHE.write_text(json.dumps(payload), encoding="utf-8")
    mapping: dict[str, str] = {}
    for item in payload.values():
        ticker = str(item.get("ticker") or "").upper()
        if not ticker:
            continue
        mapping[ticker] = str(int(item["cik_str"])).zfill(10)
    return mapping


def lookup_cik(connection: requests.Session, ticker: str) -> str | None:
    """ticker 映射里没有时，用 EDGAR 公司页解析 CIK。Electronic Arts 就属于这种情况。"""
    time.sleep(RATE_LIMIT_DELAY)
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&count=1&output=atom"
    try:
        response = connection.get(url, timeout=60)
        if response.status_code != 200:
            print(f"[WARN] {ticker} 的公司页返回 {response.status_code}")
            return None
    except requests.RequestException as exc:
        print(f"[WARN] {ticker} 的公司页请求失败：{exc}")
        return None
    match = re.search(r"<cik>(\d+)</cik>", response.text, flags=re.IGNORECASE)
    if match is None:
        return None
    return match.group(1).zfill(10)


def fetch_companyfacts(connection: requests.Session, cik: str, refresh: bool) -> dict | None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{cik}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    time.sleep(RATE_LIMIT_DELAY)
    payload = get_json(connection, FACTS_URL.format(cik=cik))
    if payload is None:
        return None
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _units_for(tag: str) -> tuple[str, ...]:
    if tag in {"CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"}:
        return SHARE_UNITS
    if tag.startswith("EarningsPerShare"):
        return EPS_UNITS
    return MONEY_UNITS


def _rows_for_tag(raw: dict, ticker: str, cik: str, taxonomy: str, tag: str) -> list[dict]:
    node = raw.get("facts", {}).get(taxonomy, {}).get(tag, {})
    units = node.get("units") or {}
    rows = []
    for unit in _units_for(tag):
        for fact in units.get(unit) or []:
            form = str(fact.get("form") or "")
            if form not in FORMS or not fact.get("end") or fact.get("val") is None:
                continue
            start = fact.get("start")
            duration = None
            if start:
                duration = int((pd.Timestamp(fact["end"]) - pd.Timestamp(start)).days)
            rows.append(
                {
                    "ticker": ticker,
                    "cik": cik,
                    "concept": f"{taxonomy}_{tag}",
                    "value": float(fact["val"]),
                    "unit": unit,
                    "period_end": fact["end"],
                    "fiscal_year": fact.get("fy"),
                    "fiscal_period": fact.get("fp"),
                    "form": form,
                    "filed_date": fact.get("filed"),
                    "duration_days": duration,
                    "_framed": bool(fact.get("frame")),
                }
            )
    return rows


def extract_concepts(raw: dict, ticker: str, cik: str) -> tuple[list[dict], list[dict]]:
    """同一因子的主标签和备选标签都保留。主标签完全没有时才记备选。债务组成部分全部保留。"""
    rows: list[dict] = []
    coverage: list[dict] = []
    for name, options in CONCEPTS.items():
        found = False
        primary_found = False
        fallback_tag = None
        for taxonomy, tag in options:
            chosen = _rows_for_tag(raw, ticker, cik, taxonomy, tag)
            if not chosen:
                continue
            found = True
            if tag == options[0][1]:
                primary_found = True
            elif fallback_tag is None:
                fallback_tag = tag
            rows.extend(chosen)
        if not found:
            coverage.append({"ticker": ticker, "concept": name, "fallback_used_or_none": "none"})
            print(f"[WARN] {ticker} 没有 {name}")
            continue
        if not primary_found and fallback_tag is not None:
            note = FALLBACK_NOTES.get(fallback_tag, "")
            label = f"{fallback_tag}（{note}）" if note else fallback_tag
            coverage.append({"ticker": ticker, "concept": name, "fallback_used_or_none": label})
    for taxonomy, tag in DEBT_PARTS:
        rows.extend(_rows_for_tag(raw, ticker, cik, taxonomy, tag))
    return rows, coverage


def deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    """同一期间只留首次披露那个财年里的版本。以后年度报表中的比较期不覆盖财年和提交日。"""
    if frame.empty:
        return frame
    work = frame.copy()
    work["_duration_key"] = pd.to_numeric(work["duration_days"], errors="coerce").fillna(-1)
    keys = ["ticker", "concept", "period_end", "form", "_duration_key"]
    ordered = work.sort_values(["filed_date", "_framed"], kind="mergesort")
    origin = ordered.groupby(keys, dropna=False, sort=False).head(1)
    origin = origin[keys + ["fiscal_year", "fiscal_period"]].rename(
        columns={"fiscal_year": "_origin_year", "fiscal_period": "_origin_period"}
    )
    merged = ordered.merge(origin, on=keys, how="left")
    same_year = merged["_origin_year"].isna() | merged["fiscal_year"].eq(merged["_origin_year"])
    chosen = merged.loc[same_year].sort_values(["filed_date", "_framed"], kind="mergesort")
    chosen = chosen.groupby(keys, dropna=False, sort=False).tail(1).copy()
    has_origin = chosen["_origin_year"].notna()
    chosen.loc[has_origin, "fiscal_year"] = chosen.loc[has_origin, "_origin_year"]
    chosen.loc[has_origin, "fiscal_period"] = chosen.loc[has_origin, "_origin_period"]
    return chosen.drop(columns=["_framed", "_duration_key", "_origin_year", "_origin_period"]).reset_index(drop=True)


def add_total_debt(frame: pd.DataFrame, coverage: list[dict]) -> pd.DataFrame:
    """长期债务加短期借款；没有短期标签时只保留长期债务。整家公司都没有债务标签时，才用负债合计。"""
    if frame.empty:
        return frame
    debt_names = {
        "us-gaap_LongTermDebt",
        "us-gaap_ShortTermBorrowings",
        "us-gaap_DebtCurrent",
    }
    needed = debt_names | {"us-gaap_Liabilities"}
    parts = frame[frame["concept"].isin(needed)]
    debt_tickers = set(parts.loc[parts["concept"].isin(debt_names), "ticker"])
    made: list[dict] = []
    methods: dict[str, set[str]] = {}
    group_cols = ["ticker", "cik", "period_end", "form", "duration_days"]
    for key, group in parts.groupby(group_cols, dropna=False):
        ticker = key[0]
        picked = {row.concept: row for row in group.itertuples(index=False)}
        long_term = picked.get("us-gaap_LongTermDebt")
        short_term = picked.get("us-gaap_ShortTermBorrowings")
        current = picked.get("us-gaap_DebtCurrent")
        liabilities = picked.get("us-gaap_Liabilities")
        if ticker in debt_tickers:
            if long_term is not None and short_term is not None:
                base, extra, method = long_term, short_term, "LongTermDebt+ShortTermBorrowings"
            elif long_term is not None and current is not None:
                base, extra, method = long_term, current, "LongTermDebt+DebtCurrent"
            elif long_term is not None:
                base, extra, method = long_term, None, "LongTermDebt（短期标签缺失，未加总）"
            elif short_term is not None:
                base, extra, method = short_term, None, "ShortTermBorrowings（缺少长期债务）"
            elif current is not None:
                base, extra, method = current, None, "DebtCurrent（缺少长期债务）"
            else:
                continue
        elif liabilities is not None:
            base, extra, method = liabilities, None, "Liabilities（粗略，含经营负债）"
        else:
            continue
        filed_dates = [base.filed_date]
        value = float(base.value)
        if extra is not None:
            value += float(extra.value)
            filed_dates.append(extra.filed_date)
        known_filed = [str(item) for item in filed_dates if item]
        methods.setdefault(ticker, set()).add(method)
        made.append(
            {
                "ticker": ticker,
                "cik": key[1],
                "concept": "derived_TotalDebt",
                "value": value,
                "unit": "USD",
                "period_end": key[2],
                "fiscal_year": base.fiscal_year,
                "fiscal_period": base.fiscal_period,
                "form": key[3],
                "filed_date": max(known_filed) if known_filed else None,
                "duration_days": base.duration_days,
            }
        )
    tickers = set(frame["ticker"])
    for ticker in sorted(tickers):
        used = methods.get(ticker)
        if not used:
            coverage.append({"ticker": ticker, "concept": "TotalDebt", "fallback_used_or_none": "none"})
            print(f"[WARN] {ticker} 没有 TotalDebt")
        else:
            label = next(iter(used)) if len(used) == 1 else "mixed: " + ", ".join(sorted(used))
            coverage.append({"ticker": ticker, "concept": "TotalDebt", "fallback_used_or_none": label})
    if not made:
        return frame
    return pd.concat([frame, pd.DataFrame(made)], ignore_index=True)


def cast_types(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["period_end"] = pd.to_datetime(out["period_end"]).dt.date
    out["filed_date"] = pd.to_datetime(out["filed_date"]).dt.date
    out["fiscal_year"] = pd.to_numeric(out["fiscal_year"], errors="coerce").astype("Int64")
    out["duration_days"] = pd.to_numeric(out["duration_days"], errors="coerce").astype("Int64")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    columns = [
        "ticker",
        "cik",
        "concept",
        "value",
        "unit",
        "period_end",
        "fiscal_year",
        "fiscal_period",
        "form",
        "filed_date",
        "duration_days",
    ]
    return out[columns]


def main() -> None:
    parser = argparse.ArgumentParser(description="下载纳斯达克100的 SEC XBRL 基本面")
    parser.add_argument("--limit", type=int, default=0, help="只抓前 N 只，0 表示全部")
    parser.add_argument("--force", action="store_true", help="忽略本地缓存，重新请求 SEC")
    args = parser.parse_args()
    if not USER_AGENT or "@" not in USER_AGENT:
        raise SystemExit("请设置 SEC_USER_AGENT，格式为：YourName you@example.com。SEC 会拒绝括号形式和 noreply 邮箱。")
    print(f"User-Agent: {USER_AGENT}")
    connection = session()
    tickers = get_nq100_tickers()
    if args.limit > 0:
        tickers = tickers[: args.limit]
    cik_map = get_cik_mapping(connection, args.force)
    rows: list[dict] = []
    coverage: list[dict] = []
    skipped: list[dict] = []
    for index, ticker in enumerate(tickers, start=1):
        cik = cik_map.get(ticker)
        if not cik:
            cik = lookup_cik(connection, ticker)
            if cik:
                print(f"[NOTE] {ticker} 不在 ticker 映射里，公司页 CIK 为 {cik}")
        if not cik:
            print(f"[WARN] {ticker} 无 CIK，跳过。")
            skipped.append({"ticker": ticker, "reason": "无 CIK"})
            continue
        raw = fetch_companyfacts(connection, cik, args.force)
        if raw is None:
            print(f"[WARN] {ticker}（CIK {cik}）无 companyfacts 数据。")
            skipped.append({"ticker": ticker, "reason": "无 companyfacts 或 HTTP 错误"})
            continue
        extracted, report = extract_concepts(raw, ticker, cik)
        rows.extend(extracted)
        coverage.extend(report)
        print(f"[{index}/{len(tickers)}] {ticker}：{len(extracted)} 条事实")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = cast_types(add_total_debt(deduplicate(pd.DataFrame(rows)), coverage))
    parquet_path = OUTPUT_DIR / "nq100_xbrl_fundamentals.parquet"
    csv_path = OUTPUT_DIR / "nq100_xbrl_fundamentals.csv"
    frame.to_parquet(parquet_path, index=False)
    frame.to_csv(csv_path, index=False, encoding="utf-8")
    pd.DataFrame(coverage, columns=["ticker", "concept", "fallback_used_or_none"]).to_csv(
        OUTPUT_DIR / "missing_concepts.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame(skipped, columns=["ticker", "reason"]).to_csv(
        OUTPUT_DIR / "skipped_tickers.csv", index=False, encoding="utf-8"
    )
    print(f"已保存 {len(frame)} 行到 {parquet_path}")


if __name__ == "__main__":
    main()
