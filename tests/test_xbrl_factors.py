import pandas as pd

from algorithms.xbrl_factors import event_panel, select_weights


def test_filing_is_invisible_before_filed_date():
    events = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "filed_date": ["2020-03-01", "2021-03-01"],
            "period_end": ["2019-12-31", "2020-12-31"],
            "value": [1.0, 5.0],
        }
    )
    calendar = pd.to_datetime(["2020-02-28", "2020-03-02", "2021-03-02"])
    panel = event_panel(events, calendar, "value")
    assert pd.isna(panel.loc[pd.Timestamp("2020-02-28"), "AAA"])
    assert panel.loc[pd.Timestamp("2020-03-02"), "AAA"] == 1.0
    assert panel.loc[pd.Timestamp("2021-03-02"), "AAA"] == 5.0


def test_tied_scores_above_the_cutoff_all_stay():
    scores = pd.Series({"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.0, "E": 0.0})
    weights = select_weights(scores)
    assert set(weights[weights > 0].index) == {"A", "B", "C"}
    assert abs(float(weights.sum()) - 1.0) < 1e-9
