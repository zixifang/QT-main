import numpy as np

from algorithms.paper_account import CAPITAL, FEE_PER_SHARE, simulate


def test_one_share_lot_pays_fee_and_sells_before_cash_goes_negative():
    dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    closes = {"AAA": np.array([100.0, 100.0, 100.0])}
    weights = {"AAA": np.array([0.0, 1.0, 0.0])}
    result = simulate(dates, closes, weights)
    shares = int(np.floor(CAPITAL / (100.0 + FEE_PER_SHARE)))
    assert result["shares"][1, 0] == shares
    assert result["shares"][2, 0] == 0
    assert result["cash"][1] >= 0
    fee_twice = 2 * shares * FEE_PER_SHARE
    assert abs(result["nav"][2] - (CAPITAL - fee_twice)) < 1e-6
    assert result["buy_dates"]["AAA"] == ["2024-01-03"]
    assert result["sell_dates"]["AAA"] == ["2024-01-04"]


def test_weights_above_one_are_scaled_and_buys_shrink_to_cash():
    dates = ["2024-01-02", "2024-01-03"]
    closes = {
        "AAA": np.array([10.0, 10.0]),
        "BBB": np.array([10.0, 10.0]),
    }
    weights = {
        "AAA": np.array([0.0, 1.0]),
        "BBB": np.array([0.0, 1.0]),
    }
    result = simulate(dates, closes, weights)
    assert result["shares"][1].sum() > 0
    assert abs(result["shares"][1, 0] - result["shares"][1, 1]) <= 1
    marked = float(result["shares"][1].sum() * 10.0 + result["cash"][1])
    assert abs(marked - result["nav"][1]) < 1e-6
    assert result["cash"][1] >= 0
    assert result["nav"][1] < CAPITAL
