import pandas as pd

from data_fetch.fetch_nq100_xbrl import deduplicate
from data_fetch.build_factors import (
    compute_growth,
    compute_piotroski,
    compute_profitability,
    compute_valuation,
    get_prior_period,
    lookup_prior_columns,
)


def _frame() -> pd.DataFrame:
    rows = [
        _row("2022-09-30", 2022, "FY", equity=80, assets=200, income=10, shares=100),
        _row("2023-09-30", 2023, "FY", equity=100, assets=220, income=20, shares=110),
        _row("2023-12-30", 2024, "Q1", equity=90, assets=210, income=4, shares=110, form="10-Q"),
        _row("2024-12-28", 2025, "Q1", equity=120, assets=0, income=6, shares=110, form="10-Q"),
    ]
    return pd.DataFrame(rows)


def _row(period, year, fiscal_period, equity, assets, income, shares, form="10-K") -> dict:
    item = {
        "ticker": "AAA",
        "period_end": pd.Timestamp(period),
        "form": form,
        "filed_date": pd.Timestamp(period) + pd.Timedelta(days=40),
        "fiscal_year": year,
        "fiscal_period": fiscal_period,
        "NetIncomeLoss": income,
        "StockholdersEquity": equity,
        "Assets": assets,
        "GrossProfit": income,
        "Revenues": income * 4,
        "CFO": income / 2,
        "LongTermDebt": 10,
        "AssetsCurrent": 50,
        "LiabilitiesCurrent": 25,
        "Shares": shares,
    }
    for name in (
        "NetIncomeLoss",
        "StockholdersEquity",
        "Assets",
        "GrossProfit",
        "Revenues",
        "CFO",
        "LongTermDebt",
        "AssetsCurrent",
        "LiabilitiesCurrent",
        "Shares",
    ):
        item[f"{name}_source"] = f"us-gaap_{name}"
    item["Shares_source"] = "us-gaap_CommonStockSharesOutstanding"
    item["CFO_source"] = "us-gaap_NetCashProvidedByUsedInOperatingActivities"
    return item


def test_later_filing_does_not_rewrite_fiscal_year():
    rows = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "concept": "us-gaap_Assets",
                "period_end": "2022-09-24",
                "form": "10-K",
                "duration_days": pd.NA,
                "filed_date": "2022-10-28",
                "fiscal_year": 2022,
                "fiscal_period": "FY",
                "value": 10.0,
                "_framed": False,
            },
            {
                "ticker": "AAA",
                "concept": "us-gaap_Assets",
                "period_end": "2022-09-24",
                "form": "10-K",
                "duration_days": pd.NA,
                "filed_date": "2025-10-31",
                "fiscal_year": 2025,
                "fiscal_period": "FY",
                "value": 12.0,
                "_framed": True,
            },
        ]
    )
    kept = deduplicate(rows)
    assert len(kept) == 1
    assert kept.iloc[0]["fiscal_year"] == 2022
    assert kept.iloc[0]["value"] == 10.0
    assert kept.iloc[0]["filed_date"] == "2022-10-28"


def test_prior_period_is_same_quarter_last_year():
    frame = _frame()
    prior = get_prior_period(frame, "AAA", "2024-12-28", "10-Q")
    assert prior is not None
    assert prior["period_end"] == pd.Timestamp("2023-12-30")
    linked = lookup_prior_columns(frame, ["Assets"])
    current = linked[(linked["form"] == "10-Q") & (linked["fiscal_year"] == 2025)].iloc[0]
    assert current["Assets_prior"] == 210


def test_roe_averages_equity_and_growth_needs_prior():
    frame = lookup_prior_columns(_frame(), ["NetIncomeLoss", "StockholdersEquity", "Assets", "Shares"])
    profitability, notes = compute_profitability(frame)
    annual = profitability[profitability["period_end"] == pd.Timestamp("2023-09-30")].iloc[0]
    assert annual["ROE"] == 20 / ((100 + 80) / 2)
    growth, growth_notes = compute_growth(frame)
    latest = growth[growth["period_end"] == pd.Timestamp("2023-09-30")].iloc[0]
    assert latest["AssetGrowth"] == (220 - 200) / 200
    assert latest["NetStockIssues"] == (110 - 100) / 100
    first = growth[growth["period_end"] == pd.Timestamp("2022-09-30")].iloc[0]
    assert pd.isna(first["AssetGrowth"])
    assert "missing_prior" in set(growth_notes.loc[growth_notes["factor"] == "AssetGrowth", "reason"])
    assert notes is not None


def test_zero_assets_do_not_produce_roa():
    frame = _frame().iloc[:1].copy()
    frame["Assets"] = 0
    profitability, notes = compute_profitability(frame)
    assert pd.isna(profitability.iloc[0]["ROA"])
    assert "zero_denominator" in set(notes.loc[notes["factor"] == "ROA", "reason"])


def test_book_to_market_uses_price_after_filing():
    frame = _frame()
    frame = frame[frame["period_end"] == pd.Timestamp("2023-09-30")].copy()
    market = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "date": ["2023-09-30", "2023-11-10", "2023-12-01"],
            "close": [10.0, 20.0, 30.0],
            "shares_outstanding": [10.0, 10.0, 10.0],
        }
    )
    valuation, notes = compute_valuation(frame, market)
    filed = pd.Timestamp("2023-09-30") + pd.Timedelta(days=40)
    row = valuation[valuation["period_end"] == pd.Timestamp("2023-09-30")].iloc[0]
    assert row["filed_date"] == filed
    assert row["BM"] == 100 / 200
    assert "market_date_unaligned" not in set(notes["reason"])


def test_fscore_sums_only_available_items():
    frame = _frame().iloc[:1].copy()
    scored, _notes = compute_piotroski(frame)
    row = scored.iloc[0]
    assert row["F_ROA"] == 1
    assert row["F_CFO"] == 1
    assert pd.isna(row["F_dROA"])
    assert row["FScore"] == row[["F_ROA", "F_CFO", "F_dROA", "F_ACCRUAL", "F_dLEVER", "F_dLIQUID", "F_EQ_ISSUE", "F_dGROSS_MARGIN", "F_dASSET_TURN"]].sum(min_count=1)
    assert row["FScore_coverage"] < 1
