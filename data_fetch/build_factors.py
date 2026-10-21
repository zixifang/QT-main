"""从 XBRL 基本面长表计算盈利、估值、质量、F-Score 和增长因子。

运行（在项目根目录）：
    python -m data_fetch.build_factors
    python -m data_fetch.build_factors --market data/market.parquet

没有市场数据时，账面市值比为空。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_LONG = ROOT / "data" / "xbrl" / "nq100_xbrl_fundamentals.parquet"
OUTPUT_FACTORS = ROOT / "data" / "factors" / "factors_nq100.parquet"
OUTPUT_MISSING = ROOT / "data" / "factors" / "factors_missing_report.csv"

KEYS = ["ticker", "period_end", "form", "filed_date"]
FACTOR_COLUMNS = [
    "ROE",
    "ROA",
    "GrossProfitability",
    "BM",
    "Accruals",
    "AssetTurnover",
    "F_ROA",
    "F_CFO",
    "F_dROA",
    "F_ACCRUAL",
    "F_dLEVER",
    "F_dLIQUID",
    "F_EQ_ISSUE",
    "F_dGROSS_MARGIN",
    "F_dASSET_TURN",
    "FScore",
    "FScore_coverage",
    "AssetGrowth",
    "NetStockIssues",
]
F_ITEMS = [
    "F_ROA",
    "F_CFO",
    "F_dROA",
    "F_ACCRUAL",
    "F_dLEVER",
    "F_dLIQUID",
    "F_EQ_ISSUE",
    "F_dGROSS_MARGIN",
    "F_dASSET_TURN",
]
# 利润表和现金流量表用区间长度区分单季、年初至今和全年。
FLOW_TAGS = {
    "us-gaap_Revenues",
    "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap_SalesRevenueNet",
    "us-gaap_NetIncomeLoss",
    "us-gaap_ProfitLoss",
    "us-gaap_GrossProfit",
    "us-gaap_NetCashProvidedByUsedInOperatingActivities",
    "us-gaap_NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
}
STOCK_TAGS = {
    "us-gaap_Assets",
    "us-gaap_AssetsCurrent",
    "us-gaap_StockholdersEquity",
    "us-gaap_StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "us-gaap_LongTermDebt",
    "us-gaap_LiabilitiesCurrent",
    "us-gaap_CommonStockSharesOutstanding",
    "dei_EntityCommonStockSharesOutstanding",
}
ALIASES = {
    "NetIncomeLoss": ["us-gaap_NetIncomeLoss", "us-gaap_ProfitLoss"],
    "StockholdersEquity": [
        "us-gaap_StockholdersEquity",
        "us-gaap_StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "Assets": ["us-gaap_Assets", "us-gaap_AssetsCurrent"],
    "GrossProfit": ["us-gaap_GrossProfit"],
    "Revenues": [
        "us-gaap_Revenues",
        "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap_SalesRevenueNet",
    ],
    "CFO": [
        "us-gaap_NetCashProvidedByUsedInOperatingActivities",
        "us-gaap_NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "LongTermDebt": ["us-gaap_LongTermDebt"],
    "AssetsCurrent": ["us-gaap_AssetsCurrent"],
    "LiabilitiesCurrent": ["us-gaap_LiabilitiesCurrent"],
    "Shares": ["us-gaap_CommonStockSharesOutstanding", "dei_EntityCommonStockSharesOutstanding"],
}
CONCEPT_COLS = list(ALIASES)
ANNUAL_DAYS = (300, 400)
QUARTER_DAYS = (70, 120)
MARKET_GAP_DAYS = 10


def load_wide(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读入长表，返回年度宽表和季度宽表。修正稿并入对应的 10-K 或 10-Q。"""
    long = pd.read_parquet(path)
    long["form"] = long["form"].replace({"10-K/A": "10-K", "10-Q/A": "10-Q"})
    long["period_end"] = pd.to_datetime(long["period_end"])
    long["filed_date"] = pd.to_datetime(long["filed_date"])
    long["duration_days"] = pd.to_numeric(long["duration_days"], errors="coerce")
    annual = _to_wide(_select_statement(long, "10-K"))
    quarterly = _to_wide(_select_statement(long, "10-Q"))
    return annual, quarterly


def _select_statement(long: pd.DataFrame, form: str) -> pd.DataFrame:
    part = long[long["form"] == form].copy()
    if part.empty:
        return part
    window = ANNUAL_DAYS if form == "10-K" else QUARTER_DAYS
    flow = part["concept"].isin(FLOW_TAGS) & part["duration_days"].between(*window)
    stock = part["concept"].isin(STOCK_TAGS) & part["duration_days"].isna()
    kept = part[flow | stock]
    kept = kept.sort_values("filed_date").drop_duplicates(
        ["ticker", "period_end", "form", "concept"], keep="last"
    )
    return kept


ANCHOR_TAGS = {
    "us-gaap_Assets",
    "us-gaap_StockholdersEquity",
    "us-gaap_NetIncomeLoss",
    "us-gaap_ProfitLoss",
    "us-gaap_Revenues",
    "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap_SalesRevenueNet",
    "us-gaap_GrossProfit",
}


def _period_meta(part: pd.DataFrame) -> pd.DataFrame:
    """财年和提交日以资产负债表或利润表为准，避免后一年才出现的债务标签改写期间。"""
    anchor = part[part["concept"].isin(ANCHOR_TAGS)]
    if anchor.empty:
        anchor = part
    anchor = anchor.sort_values("filed_date")
    keys = ["ticker", "period_end", "form"]
    origin = anchor.groupby(keys, dropna=False, sort=False).head(1)[keys + ["fiscal_year", "fiscal_period"]]
    origin = origin.rename(columns={"fiscal_year": "_origin_year", "fiscal_period": "_origin_period"})
    matched = anchor.merge(origin, on=keys, how="left")
    matched = matched[matched["fiscal_year"].eq(matched["_origin_year"]) | matched["_origin_year"].isna()]
    filed = matched.groupby(keys, dropna=False, sort=False)["filed_date"].max().reset_index()
    meta = origin.merge(filed, on=keys, how="left")
    return meta.rename(columns={"_origin_year": "fiscal_year", "_origin_period": "fiscal_period"})


def _to_wide(part: pd.DataFrame) -> pd.DataFrame:
    if part.empty:
        return _empty_wide()
    meta = _period_meta(part)
    wide = part.pivot_table(
        index=["ticker", "period_end", "form"],
        columns="concept",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    wide = wide.merge(meta, on=["ticker", "period_end", "form"], how="left")
    wide["fiscal_year"] = pd.to_numeric(wide["fiscal_year"], errors="coerce").astype("Int64")
    wide["fiscal_period"] = wide["fiscal_period"].fillna("").astype(str).str.strip()
    annual_blank = (wide["form"] == "10-K") & (wide["fiscal_period"] == "")
    wide.loc[annual_blank, "fiscal_period"] = "FY"
    resolved = _resolve(wide)
    core = ["Assets", "NetIncomeLoss", "Revenues", "GrossProfit", "CFO", "StockholdersEquity"]
    return resolved.loc[resolved[core].notna().any(axis=1)].reset_index(drop=True)


def _empty_wide() -> pd.DataFrame:
    columns = KEYS + ["fiscal_year", "fiscal_period", *CONCEPT_COLS]
    columns += [f"{name}_source" for name in CONCEPT_COLS]
    return pd.DataFrame(columns=columns)


def _pick(wide: pd.DataFrame, names: list[str]) -> tuple[pd.Series, pd.Series]:
    value = pd.Series(np.nan, index=wide.index, dtype="float64")
    source = pd.Series(pd.NA, index=wide.index, dtype="object")
    for name in names:
        if name not in wide.columns:
            continue
        observed = pd.to_numeric(wide[name], errors="coerce")
        take = value.isna() & observed.notna()
        value = value.where(~take, observed)
        source = source.where(~take, name)
    return value, source


def _resolve(wide: pd.DataFrame) -> pd.DataFrame:
    resolved = wide[["ticker", "period_end", "form", "filed_date", "fiscal_year", "fiscal_period"]].copy()
    for logical, names in ALIASES.items():
        value, source = _pick(wide, names)
        resolved[logical] = value
        resolved[f"{logical}_source"] = source
    return resolved


def get_prior_period(df: pd.DataFrame, ticker: str, period_end, form: str) -> pd.Series | None:
    """10-K 对上一财年的 10-K；10-Q 对上一财年同一财季。"""
    current = df[(df["ticker"] == ticker) & (df["period_end"] == pd.Timestamp(period_end)) & (df["form"] == form)]
    if current.empty or pd.isna(current.iloc[0]["fiscal_year"]):
        return None
    year = int(current.iloc[0]["fiscal_year"]) - 1
    hit = df[(df["ticker"] == ticker) & (df["form"] == form) & (df["fiscal_year"] == year)]
    if form == "10-Q":
        hit = hit[hit["fiscal_period"] == current.iloc[0]["fiscal_period"]]
    if hit.empty:
        return None
    if len(hit) > 1:
        target = pd.Timestamp(period_end) - pd.DateOffset(years=1)
        distance = (pd.to_datetime(hit["period_end"]) - target).abs()
        hit = hit.loc[distance.sort_values().index]
    return hit.iloc[0]


def lookup_prior_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """按 get_prior_period 的规则，把上期字段并到当期行，列名加 _prior。"""
    if df.empty or not columns:
        return df.copy()
    pieces = []
    for form, extra in (("10-K", []), ("10-Q", ["fiscal_period"])):
        current = df[df["form"] == form].copy()
        if current.empty:
            continue
        keys = ["ticker", "fiscal_year", *extra]
        previous = current[["ticker", "fiscal_year", "fiscal_period", "period_end", *columns]].copy()
        previous["fiscal_year"] = previous["fiscal_year"] + 1
        previous = previous.rename(columns={column: f"{column}_prior" for column in columns} | {"period_end": "period_end__prior"})
        previous = previous[keys + ["period_end__prior"] + [f"{column}_prior" for column in columns]]
        merged = current.merge(previous, on=keys, how="left")
        identity = ["ticker", "period_end", "form"]
        if merged.duplicated(identity).any():
            target = pd.to_datetime(merged["period_end"]) - pd.DateOffset(years=1)
            merged = merged.assign(_dist=(pd.to_datetime(merged["period_end__prior"]) - target).abs())
            merged = merged.sort_values("_dist", na_position="last").drop_duplicates(identity, keep="first")
        pieces.append(merged.drop(columns=["period_end__prior", "_dist"], errors="ignore"))
    if not pieces:
        return df.copy()
    return pd.concat(pieces, ignore_index=True)


def _ensure_prior(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [column for column in columns if f"{column}_prior" not in df.columns]
    if not missing:
        return df
    return lookup_prior_columns(df, missing)


def _average(current: pd.Series, prior: pd.Series) -> tuple[pd.Series, pd.Series]:
    degraded = current.notna() & prior.isna()
    both = (current + prior) / 2
    return both.where(~degraded, current), degraded


def _div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    valid = numerator.notna() & denominator.notna() & (denominator != 0)
    out = pd.Series(np.nan, index=numerator.index, dtype="float64")
    out.loc[valid] = numerator.loc[valid].to_numpy() / denominator.loc[valid].to_numpy()
    return out


def _flag(condition: pd.Series, valid: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=condition.index, dtype="float64")
    out.loc[valid] = condition.loc[valid].astype(float)
    return out


def _with_averages(df: pd.DataFrame) -> pd.DataFrame:
    frame = _ensure_prior(df, CONCEPT_COLS)
    frame["equity_avg"], frame["equity_degraded"] = _average(frame["StockholdersEquity"], frame["StockholdersEquity_prior"])
    frame["assets_avg"], frame["assets_degraded"] = _average(frame["Assets"], frame["Assets_prior"])
    return frame


def _reasons_from(frame: pd.DataFrame, mask: pd.Series, factor: str, reason: str) -> pd.DataFrame:
    hit = frame.loc[mask, ["ticker", "form"]].copy()
    hit["factor"] = factor
    hit["reason"] = reason
    return hit


def _append_reason(bucket: list[pd.DataFrame], frame: pd.DataFrame, mask: pd.Series, factor: str, reason: str) -> None:
    if mask.any():
        bucket.append(_reasons_from(frame, mask, factor, reason))


def _ratio_reasons(
    bucket: list[pd.DataFrame],
    frame: pd.DataFrame,
    factor: str,
    value: pd.Series,
    pieces: list[tuple[str, pd.Series]],
    degraded: pd.Series | None = None,
) -> None:
    """pieces 按优先级列出输入列。值算出来之后，再记录退化和备选标签。"""
    unexplained = value.isna()
    for name, series in pieces:
        missing = unexplained & series.isna()
        _append_reason(bucket, frame, missing, factor, f"missing_concept:{name}")
        unexplained = unexplained & ~missing
    _append_reason(bucket, frame, unexplained, factor, "zero_denominator")
    if degraded is not None:
        _append_reason(bucket, frame, value.notna() & degraded, factor, "prior_missing_used_current")
    for name, _series in pieces:
        source = frame[f"{name}_source"]
        primary = ALIASES[name][0]
        used_fallback = value.notna() & source.notna() & (source != primary)
        _append_reason(bucket, frame, used_fallback, factor, "fallback_concept:" + source.fillna(""))
    if any(name == "StockholdersEquity" for name, _series in pieces):
        negative = value.notna() & frame["StockholdersEquity"].lt(0)
        _append_reason(bucket, frame, negative, factor, "negative_equity")


def compute_profitability(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ROE、ROA 用平均分母。毛利润因子用当期总资产。"""
    frame = _with_averages(wide)
    roe = _div(frame["NetIncomeLoss"], frame["equity_avg"])
    roa = _div(frame["NetIncomeLoss"], frame["assets_avg"])
    gross = _div(frame["GrossProfit"], frame["Assets"])
    out = frame[KEYS].copy()
    out["ROE"] = roe
    out["ROA"] = roa
    out["GrossProfitability"] = gross
    bucket: list[pd.DataFrame] = []
    _ratio_reasons(
        bucket,
        frame,
        "ROE",
        roe,
        [("NetIncomeLoss", frame["NetIncomeLoss"]), ("StockholdersEquity", frame["StockholdersEquity"])],
        frame["equity_degraded"],
    )
    _ratio_reasons(
        bucket,
        frame,
        "ROA",
        roa,
        [("NetIncomeLoss", frame["NetIncomeLoss"]), ("Assets", frame["Assets"])],
        frame["assets_degraded"],
    )
    _ratio_reasons(
        bucket,
        frame,
        "GrossProfitability",
        gross,
        [("GrossProfit", frame["GrossProfit"]), ("Assets", frame["Assets"])],
    )
    return out, _concat_reasons(bucket)


def _market_table(market_df: pd.DataFrame) -> pd.DataFrame:
    frame = market_df.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"])
    if "market_cap" in frame.columns:
        frame["MarketCap"] = pd.to_numeric(frame["market_cap"], errors="coerce")
    elif {"close", "shares_outstanding"} <= set(frame.columns):
        frame["MarketCap"] = pd.to_numeric(frame["close"], errors="coerce") * pd.to_numeric(
            frame["shares_outstanding"], errors="coerce"
        )
    else:
        raise ValueError("市场数据需要 market_cap，或同时提供 close 与 shares_outstanding")
    frame = frame.dropna(subset=["date", "MarketCap"])
    frame = frame.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    return frame[["ticker", "date", "MarketCap"]]


def _align_market(wide: pd.DataFrame, market_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    prices = _market_table(market_df)
    left = wide[["ticker", "filed_date"]].copy()
    left["ticker"] = left["ticker"].astype(str).str.upper()
    left["_row"] = np.arange(len(left))
    left = left.dropna(subset=["filed_date"]).sort_values(["ticker", "filed_date"])
    aligned = pd.merge_asof(
        left,
        prices,
        left_on="filed_date",
        right_on="date",
        by="ticker",
        direction="forward",
    )
    gap = (aligned["date"] - aligned["filed_date"]).dt.days
    aligned["usable"] = aligned["date"].notna() & gap.le(MARKET_GAP_DAYS)
    aligned = aligned.set_index("_row").reindex(np.arange(len(wide)))
    cap = pd.Series(aligned["MarketCap"].to_numpy(), index=wide.index, dtype="float64")
    cap = cap.where(aligned["usable"].fillna(False).to_numpy())
    missing = cap.isna()
    return cap, missing


def compute_valuation(wide: pd.DataFrame, market_df: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """账面市值比。市值取 filed_date 之后 10 天内的第一个交易日。"""
    out = wide[KEYS].copy()
    bucket: list[pd.DataFrame] = []
    equity = wide["StockholdersEquity"]
    if market_df is None or market_df.empty:
        out["BM"] = np.nan
        _append_reason(bucket, wide, pd.Series(True, index=wide.index), "BM", "market_data_missing")
        return out, _concat_reasons(bucket)
    cap, unaligned = _align_market(wide, market_df)
    out["BM"] = _div(equity, cap)
    missing_equity = out["BM"].isna() & equity.isna()
    zero = out["BM"].isna() & equity.notna() & cap.notna() & cap.eq(0)
    no_price = out["BM"].isna() & equity.notna() & unaligned
    _append_reason(bucket, wide, missing_equity, "BM", "missing_concept:StockholdersEquity")
    _append_reason(bucket, wide, zero, "BM", "zero_denominator")
    _append_reason(bucket, wide, no_price, "BM", "market_date_unaligned")
    source = wide["StockholdersEquity_source"]
    primary = ALIASES["StockholdersEquity"][0]
    _append_reason(
        bucket,
        wide,
        out["BM"].notna() & source.notna() & (source != primary),
        "BM",
        "fallback_concept:" + source.fillna(""),
    )
    _append_reason(bucket, wide, out["BM"].notna() & equity.lt(0), "BM", "negative_equity")
    return out, _concat_reasons(bucket)


def compute_quality(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """应计利润保留原始符号，越负通常质量越好。"""
    frame = _with_averages(wide)
    accruals = _div(frame["NetIncomeLoss"] - frame["CFO"], frame["assets_avg"])
    turnover = _div(frame["Revenues"], frame["assets_avg"])
    out = frame[KEYS].copy()
    out["Accruals"] = accruals
    out["AssetTurnover"] = turnover
    bucket: list[pd.DataFrame] = []
    _ratio_reasons(
        bucket,
        frame,
        "Accruals",
        accruals,
        [
            ("NetIncomeLoss", frame["NetIncomeLoss"]),
            ("CFO", frame["CFO"]),
            ("Assets", frame["Assets"]),
        ],
        frame["assets_degraded"],
    )
    _ratio_reasons(
        bucket,
        frame,
        "AssetTurnover",
        turnover,
        [("Revenues", frame["Revenues"]), ("Assets", frame["Assets"])],
        frame["assets_degraded"],
    )
    return out, _concat_reasons(bucket)


def _level_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = frame.copy()
    metrics["ROA_level"] = _div(metrics["NetIncomeLoss"], metrics["assets_avg"])
    metrics["leverage"] = _div(metrics["LongTermDebt"], metrics["assets_avg"])
    metrics["current_ratio"] = _div(metrics["AssetsCurrent"], metrics["LiabilitiesCurrent"])
    metrics["gross_margin"] = _div(metrics["GrossProfit"], metrics["Revenues"])
    metrics["turnover_level"] = _div(metrics["Revenues"], metrics["assets_avg"])
    return metrics


def compute_piotroski(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """九个分项。差分比较的是上年同期那一行自己的指标，而不是把两期资产再平均一次。"""
    frame = _level_metrics(_with_averages(wide))
    frame = _ensure_prior(frame, ["ROA_level", "leverage", "current_ratio", "gross_margin", "turnover_level"])
    out = frame[KEYS].copy()
    out["F_ROA"] = _flag(frame["ROA_level"] > 0, frame["ROA_level"].notna())
    out["F_CFO"] = _flag(frame["CFO"] > 0, frame["CFO"].notna())
    out["F_dROA"] = _flag(frame["ROA_level"] > frame["ROA_level_prior"], frame["ROA_level"].notna() & frame["ROA_level_prior"].notna())
    out["F_ACCRUAL"] = _flag(frame["CFO"] > frame["NetIncomeLoss"], frame["CFO"].notna() & frame["NetIncomeLoss"].notna())
    out["F_dLEVER"] = _flag(frame["leverage"] < frame["leverage_prior"], frame["leverage"].notna() & frame["leverage_prior"].notna())
    out["F_dLIQUID"] = _flag(
        frame["current_ratio"] > frame["current_ratio_prior"],
        frame["current_ratio"].notna() & frame["current_ratio_prior"].notna(),
    )
    out["F_EQ_ISSUE"] = _flag(frame["Shares"] <= frame["Shares_prior"], frame["Shares"].notna() & frame["Shares_prior"].notna())
    out["F_dGROSS_MARGIN"] = _flag(
        frame["gross_margin"] > frame["gross_margin_prior"],
        frame["gross_margin"].notna() & frame["gross_margin_prior"].notna(),
    )
    out["F_dASSET_TURN"] = _flag(
        frame["turnover_level"] > frame["turnover_level_prior"],
        frame["turnover_level"].notna() & frame["turnover_level_prior"].notna(),
    )
    out["FScore"] = out[F_ITEMS].sum(axis=1, min_count=1)
    out["FScore_coverage"] = out[F_ITEMS].notna().sum(axis=1) / len(F_ITEMS)
    bucket: list[pd.DataFrame] = []
    _piotroski_reasons(bucket, frame, out)
    return out, _concat_reasons(bucket)


def _piotroski_reasons(bucket: list[pd.DataFrame], frame: pd.DataFrame, out: pd.DataFrame) -> None:
    _ratio_reasons(
        bucket,
        frame,
        "F_ROA",
        out["F_ROA"],
        [("NetIncomeLoss", frame["NetIncomeLoss"]), ("Assets", frame["Assets"])],
    )
    _append_reason(bucket, frame, out["F_CFO"].isna(), "F_CFO", "missing_concept:CFO")
    prior_gap = out["F_dROA"].isna() & out["F_ROA"].notna()
    _append_reason(bucket, frame, out["F_dROA"].isna() & out["F_ROA"].isna(), "F_dROA", "missing_concept:ROA")
    _append_reason(bucket, frame, prior_gap, "F_dROA", "missing_prior")
    _append_reason(bucket, frame, out["F_ACCRUAL"].isna() & frame["CFO"].isna(), "F_ACCRUAL", "missing_concept:CFO")
    _append_reason(
        bucket,
        frame,
        out["F_ACCRUAL"].isna() & frame["CFO"].notna() & frame["NetIncomeLoss"].isna(),
        "F_ACCRUAL",
        "missing_concept:NetIncomeLoss",
    )
    _append_reason(bucket, frame, out["F_dLEVER"].isna() & frame["LongTermDebt"].isna(), "F_dLEVER", "missing_concept:LongTermDebt")
    _append_reason(
        bucket,
        frame,
        out["F_dLEVER"].isna() & frame["LongTermDebt"].notna() & frame["leverage"].isna(),
        "F_dLEVER",
        "missing_concept:Assets",
    )
    _append_reason(bucket, frame, out["F_dLEVER"].isna() & frame["leverage"].notna(), "F_dLEVER", "missing_prior")
    liquid_gap = out["F_dLIQUID"].isna() & frame["current_ratio"].isna()
    _append_reason(bucket, frame, liquid_gap & frame["AssetsCurrent"].isna(), "F_dLIQUID", "missing_concept:AssetsCurrent")
    _append_reason(
        bucket,
        frame,
        liquid_gap & frame["AssetsCurrent"].notna(),
        "F_dLIQUID",
        "missing_concept:LiabilitiesCurrent",
    )
    _append_reason(bucket, frame, out["F_dLIQUID"].isna() & frame["current_ratio"].notna(), "F_dLIQUID", "missing_prior")
    _append_reason(bucket, frame, out["F_EQ_ISSUE"].isna() & frame["Shares"].isna(), "F_EQ_ISSUE", "missing_concept:Shares")
    _append_reason(bucket, frame, out["F_EQ_ISSUE"].isna() & frame["Shares"].notna(), "F_EQ_ISSUE", "missing_prior")
    _append_reason(bucket, frame, out["F_dGROSS_MARGIN"].isna() & frame["gross_margin"].isna(), "F_dGROSS_MARGIN", "missing_concept:GrossProfit")
    _append_reason(bucket, frame, out["F_dGROSS_MARGIN"].isna() & frame["gross_margin"].notna(), "F_dGROSS_MARGIN", "missing_prior")
    _append_reason(bucket, frame, out["F_dASSET_TURN"].isna() & frame["turnover_level"].isna(), "F_dASSET_TURN", "missing_concept:Revenues")
    _append_reason(bucket, frame, out["F_dASSET_TURN"].isna() & frame["turnover_level"].notna(), "F_dASSET_TURN", "missing_prior")
    _append_reason(bucket, frame, out["FScore"].isna(), "FScore", "no_components")


def compute_growth(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """资产增长和净股票发行。上期缺失时留空，不用当期值代替。"""
    frame = _ensure_prior(wide, ["Assets", "Shares"])
    out = frame[KEYS].copy()
    out["AssetGrowth"] = _div(frame["Assets"] - frame["Assets_prior"], frame["Assets_prior"])
    out["NetStockIssues"] = _div(frame["Shares"] - frame["Shares_prior"], frame["Shares_prior"])
    bucket: list[pd.DataFrame] = []
    _growth_reasons(bucket, frame, out["AssetGrowth"], "AssetGrowth", "Assets")
    _growth_reasons(bucket, frame, out["NetStockIssues"], "NetStockIssues", "Shares")
    return out, _concat_reasons(bucket)


def _growth_reasons(bucket: list[pd.DataFrame], frame: pd.DataFrame, value: pd.Series, factor: str, column: str) -> None:
    prior = frame[f"{column}_prior"]
    current = frame[column]
    _append_reason(bucket, frame, value.isna() & current.isna(), factor, f"missing_concept:{column}")
    _append_reason(bucket, frame, value.isna() & current.notna() & prior.isna(), factor, "missing_prior")
    _append_reason(bucket, frame, value.isna() & current.notna() & prior.notna() & prior.eq(0), factor, "zero_denominator")
    source = frame[f"{column}_source"]
    primary = ALIASES[column][0]
    _append_reason(
        bucket,
        frame,
        value.notna() & source.notna() & (source != primary),
        factor,
        "fallback_concept:" + source.fillna(""),
    )


def _concat_reasons(bucket: list[pd.DataFrame]) -> pd.DataFrame:
    columns = ["ticker", "form", "factor", "reason"]
    if not bucket:
        return pd.DataFrame(columns=columns)
    return pd.concat(bucket, ignore_index=True)


def _combine(wide: pd.DataFrame, market_df: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if wide.empty:
        empty = pd.DataFrame(columns=KEYS + FACTOR_COLUMNS)
        return empty, _concat_reasons([])
    blocks = []
    reports = []
    prepared = _with_averages(wide)
    for function in (compute_profitability, compute_quality, compute_piotroski, compute_growth):
        values, report = function(prepared)
        blocks.append(values)
        reports.append(report)
    values, report = compute_valuation(prepared, market_df)
    blocks.append(values)
    reports.append(report)
    factors = blocks[0]
    for block in blocks[1:]:
        factors = factors.merge(block, on=KEYS, how="left")
    return factors[KEYS + FACTOR_COLUMNS], pd.concat(reports, ignore_index=True)


def _summarize_missing(report: pd.DataFrame) -> pd.DataFrame:
    columns = ["ticker", "form", "factor", "reason", "n_periods"]
    if report.empty:
        return pd.DataFrame(columns=columns)
    counted = report.groupby(["ticker", "form", "factor", "reason"], as_index=False).size()
    return counted.rename(columns={"size": "n_periods"}).sort_values(["ticker", "form", "factor", "reason"])


def load_market(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="从 XBRL 长表计算纳斯达克100基本面因子")
    parser.add_argument("--input", type=Path, default=INPUT_LONG)
    parser.add_argument("--market", type=Path, default=None, help="含 ticker, date, market_cap 或 close, shares_outstanding")
    parser.add_argument("--output", type=Path, default=OUTPUT_FACTORS)
    parser.add_argument("--missing", type=Path, default=OUTPUT_MISSING)
    args = parser.parse_args()
    annual, quarterly = load_wide(args.input)
    market = load_market(args.market)
    if market is None:
        print("[NOTE] 未提供市场数据，账面市值比全部为空。")
    factors_annual, missing_annual = _combine(annual, market)
    factors_quarterly, missing_quarterly = _combine(quarterly, market)
    factors = pd.concat([factors_annual, factors_quarterly], ignore_index=True)
    factors = factors.sort_values(["ticker", "period_end", "form"]).reset_index(drop=True)
    factors["period_end"] = pd.to_datetime(factors["period_end"]).dt.date
    factors["filed_date"] = pd.to_datetime(factors["filed_date"]).dt.date
    for column in FACTOR_COLUMNS:
        factors[column] = pd.to_numeric(factors[column], errors="coerce")
    missing = _summarize_missing(pd.concat([missing_annual, missing_quarterly], ignore_index=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    factors.to_parquet(args.output, index=False)
    missing.to_csv(args.missing, index=False, encoding="utf-8")
    print(f"年度 {len(factors_annual)} 行，季度 {len(factors_quarterly)} 行，合计 {len(factors)} 行")
    coverage = factors[FACTOR_COLUMNS].notna().mean().sort_values(ascending=False)
    print(coverage.map(lambda value: f"{value:.1%}").to_string())
    print(f"已保存 {args.output}")


if __name__ == "__main__":
    main()
