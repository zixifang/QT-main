import numpy as np
import pandas as pd

from algorithms.fscore_long_short import (
    F_COLUMNS,
    F_SIGN,
    align_fundamentals,
    average_dollar_volume,
    compute_fscore,
    quarter_rebalance_dates,
    run_fscore_backtest,
    simulate_open_execution,
    target_weights,
)


def test_rebalance_dates_are_the_tenth_session_of_quarter_months():
    january = pd.bdate_range("2020-01-01", periods=12)
    april = pd.bdate_range("2020-04-01", periods=9)
    july = pd.bdate_range("2020-07-01", periods=10)
    calendar = january.append(april).append(july)
    dates = quarter_rebalance_dates(calendar)
    assert list(dates) == [january[9], july[9]]


def test_merge_asof_uses_only_filings_known_on_the_rebalance_date():
    fundamentals = pd.DataFrame(
        [
            _fundamental("AAA", "2020-01-10", "2019-12-31", 8),
            _fundamental("AAA", "2020-01-11", "2019-12-31", 0),
        ]
    )
    aligned = align_fundamentals(fundamentals, pd.to_datetime(["2020-01-10", "2020-01-15"]))
    on_day = aligned[aligned["rebalance_date"] == pd.Timestamp("2020-01-10")].iloc[0]
    later = aligned[aligned["rebalance_date"] == pd.Timestamp("2020-01-15")].iloc[0]
    assert on_day["filing_date"] == pd.Timestamp("2020-01-10")
    assert compute_fscore(pd.DataFrame([on_day])).iloc[0] == 8
    assert later["filing_date"] == pd.Timestamp("2020-01-11")
    assert compute_fscore(pd.DataFrame([later])).iloc[0] == 0


def test_fscore_scales_seven_items_and_drops_six():
    kept = _fundamental("AAA", "2020-01-01", "2019-12-31", 9)
    for column in F_COLUMNS[:2]:
        kept[column] = np.nan
    dropped = _fundamental("BBB", "2020-01-01", "2019-12-31", 9)
    for column in F_COLUMNS[:3]:
        dropped[column] = np.nan
    scores = compute_fscore(pd.DataFrame([kept, dropped]))
    assert scores.iloc[0] == 9
    assert np.isnan(scores.iloc[1])

    partial = _fundamental("CCC", "2020-01-01", "2019-12-31", 5)
    partial[F_COLUMNS[-1]] = np.nan
    partial[F_COLUMNS[-2]] = np.nan
    assert compute_fscore(pd.DataFrame([partial])).iloc[0] == 5 / 7 * 9


def test_tied_scores_at_the_cutoff_all_stay_in_the_book():
    scores = pd.Series({"A": 9.0, "B": 9.0, "C": 5.0, "D": 1.0, "E": 0.0, "F": 0.0})
    weights = target_weights(scores, 0.10)
    assert set(weights[weights > 0].index) == {"A", "B"}
    assert float(weights.min()) >= 0.0
    assert abs(weights.sum() - 1) < 1e-12


def test_next_open_does_not_earn_the_gap_and_keeps_the_position():
    days = pd.bdate_range("2020-01-02", periods=4)
    signal = pd.DataFrame(np.nan, index=days, columns=["AAA", "BBB"])
    signal.loc[days[0], ["AAA", "BBB"]] = [1.0, 0.0]
    prices = _flat_prices(days, ["AAA", "BBB"])
    prices.loc[(prices["date"] == days[1]) & (prices["ticker"] == "AAA"), ["open", "close"]] = [15.0, 15.0]
    prices.loc[(prices["date"] == days[2]) & (prices["ticker"] == "AAA"), ["open", "close"]] = [15.0, 18.0]
    daily = simulate_open_execution(signal, prices)
    assert daily.loc[days[0], "net_return"] == 0
    assert abs(daily.loc[days[1], "gross_return"]) < 1e-12
    assert abs(daily.loc[days[1], "borrow_cost"]) < 1e-12
    assert abs(daily.loc[days[2], "gross_return"] - 0.20) < 1e-12
    assert abs(daily.loc[days[2], "turnover"]) < 1e-12
    assert abs(daily.loc[days[1], "turnover"] - 0.5) < 1e-12


def test_split_sized_jump_does_not_enter_the_return():
    days = pd.bdate_range("2020-01-02", periods=3)
    signal = pd.DataFrame(np.nan, index=days, columns=["AAA", "BBB"])
    signal.loc[days[0]] = [1.0, 0.0]
    prices = _flat_prices(days, ["AAA", "BBB"])
    prices.loc[(prices["date"] == days[2]) & (prices["ticker"] == "AAA"), ["open", "close"]] = [20.0, 20.0]
    daily = simulate_open_execution(signal, prices)
    assert abs(daily.loc[days[2], "gross_return"]) < 1e-12


def test_vwap_fill_uses_the_vwap_column():
    days = pd.bdate_range("2020-01-02", periods=3)
    signal = pd.DataFrame(np.nan, index=days, columns=["AAA", "BBB"])
    signal.loc[days[0]] = [1.0, 0.0]
    prices = _flat_prices(days, ["AAA", "BBB"])
    prices["vwap"] = 10.0
    execution = prices["date"] == days[1]
    prices.loc[execution & (prices["ticker"] == "AAA"), ["open", "vwap", "close"]] = [99.0, 10.0, 12.0]
    daily = simulate_open_execution(signal, prices, execution="vwap")
    assert abs(daily.loc[days[1], "gross_return"] - 0.20) < 1e-12


def test_full_backtest_filters_liquidity_and_future_filings():
    calendar = pd.bdate_range("2019-11-01", "2020-01-31")
    rebalance = quarter_rebalance_dates(calendar)
    january = rebalance[rebalance.year == 2020][0]
    names = [f"S{i}" for i in range(10)]
    thin = "S9"
    prices = _flat_prices(calendar, names + ["SPY"])
    prices.loc[prices["ticker"] == thin, "volume"] = 100_000
    prices.loc[prices["ticker"] != thin, "volume"] = 200_000
    fundamentals = []
    for rank, ticker in enumerate(names):
        score = 9 - rank
        row = _fundamental(ticker, "2019-03-01", "2018-12-31", score)
        fundamentals.append(row)
        leaked = _fundamental(ticker, january + pd.Timedelta(days=1), "2019-12-31", 0 if ticker == "S0" else 9)
        fundamentals.append(leaked)
    result = run_fscore_backtest(prices, pd.DataFrame(fundamentals))
    january_scores = result["scores"]
    january_scores = january_scores[january_scores["rebalance_date"] == january]
    assert january_scores.loc[january_scores["ticker"] == "S0", "fscore"].iloc[0] == 9
    assert not bool(january_scores.loc[january_scores["ticker"] == thin, "liquid"].iloc[0])
    assert january_scores.loc[january_scores["ticker"] == "S0", "weight"].iloc[0] == 1
    assert january_scores.loc[january_scores["ticker"] == "S8", "weight"].iloc[0] == 0
    assert january_scores.loc[january_scores["ticker"] == thin, "weight"].iloc[0] == 0
    assert float(january_scores["weight"].min()) >= 0.0
    execution = result["rebalance"]["execution_date"].iloc[0]
    assert abs(result["daily"].loc[execution, "net_return"]) < 1e-12
    assert abs(result["daily"]["borrow_cost"].sum()) < 1e-12
    assert "标普500" in result["stats"].index
    assert result["stats"].loc["策略", "days"] == len(result["daily"])
    adv = average_dollar_volume(prices)
    assert adv.loc[january, thin] == 1_000_000
    assert adv.loc[january, "S0"] > 1_000_000


def _fundamental(ticker: str, filing_date: str, report_date: str, good_items: int) -> dict:
    row = {"ticker": ticker, "filing_date": filing_date, "report_date": report_date}
    for index, (column, sign) in enumerate(F_SIGN.items()):
        good = index < good_items
        if sign > 0:
            row[column] = 1.0 if good else -1.0
        else:
            row[column] = -1.0 if good else 1.0
    return row


def _flat_prices(days: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    rows = []
    for day in days:
        for ticker in tickers:
            rows.append(
                {
                    "date": day,
                    "ticker": ticker,
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 10.0,
                    "volume": 1_000_000.0,
                    "vwap": 10.0,
                }
            )
    return pd.DataFrame(rows)
