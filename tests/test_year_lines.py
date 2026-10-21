import pytest

from gui.app import path_from_level, year_table, yearly_return_lines


def test_year_lines_keep_saved_returns_in_year_order():
    lines = yearly_return_lines(
        {
            "2021": {
                "strategy": {"cumulative_return": 0.2},
                "benchmark": {"cumulative_return": 0.1},
                "holding": {"cumulative_return": 0.15},
            },
            "2020": {
                "strategy": {"cumulative_return": -0.05},
                "benchmark": {"cumulative_return": None},
                "holding": {"cumulative_return": 0.03},
            },
        }
    )
    assert lines["years"] == ["2020", "2021"]
    assert lines["strategy"] == [-0.05, 0.2]
    assert lines["benchmark"] == [None, 0.1]
    assert lines["holding"] == [0.03, 0.15]


def test_year_lines_omit_holding_when_the_dataset_has_none():
    lines = yearly_return_lines(
        {
            "2018": {
                "strategy": {"cumulative_return": 0.04},
                "benchmark": {"cumulative_return": 0.02},
            }
        }
    )
    assert lines["holding"] == []


def test_path_from_level_starts_at_zero():
    path = path_from_level([1.0, 1.1, 1.21])
    assert path[0] == pytest.approx(0.0)
    assert path[1] == pytest.approx(0.1)
    assert path[2] == pytest.approx(0.21)


def test_year_table_uses_only_the_selected_year():
    dates = ["2023-12-29", "2024-01-02", "2024-01-03"]
    algo = {
        "equity": [1.0, 1.1, 1.21],
        "benchmark_equity": [1.0, 1.0, 1.0],
        "position": [0.0, 1.0, 1.0],
    }
    block = year_table(dates, algo, "2024")
    assert block["strategy"]["cumulative_return"] == pytest.approx(0.21)
    assert block["strategy"]["trades"] == 1
    assert block["strategy"]["win_rate"] == 1.0
    assert block["benchmark"]["cumulative_return"] == 0.0
    assert block["benchmark"]["win_rate"] is None
    earlier = year_table(dates, algo, "2023")
    assert earlier["strategy"]["cumulative_return"] == 0.0
    assert earlier["strategy"]["trades"] == 0
