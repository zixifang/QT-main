"""行情窗口只读取 outdata 中的 JSON。

第三页统计表按选中年份切已保存的净值，不读 data 里的 CSV。

指数增强页面只调用 services，不读 data 里的 CSV。
运行前先生成结果：python -m algorithms.rsi
然后：python -m gui
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from pathlib import Path

from gui.outdata import list_datasets, load_paper, load_symbol

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PAGES = ("原始数据", "算法点位", "统计指标", "全局统计", "历史走势")
INDEX_PAGES = ("指数行情", "指数指标")
STAT_ROWS = (
    ("cumulative_return", "累计收益", "pct"),
    ("annualized_return", "年化收益", "pct"),
    ("volatility", "年化波动", "pct_plain"),
    ("sharpe", "夏普比率", "num"),
    ("max_drawdown", "最大回撤", "pct_plain"),
    ("calmar", "卡玛比率", "num"),
    ("trades", "交易次数", "int"),
    ("win_rate", "胜率", "pct_plain"),
)
INDEX_STAT_ROWS = (
    ("cumulative_return", "累计收益", "pct"),
    ("annualized_return", "年化收益", "pct"),
    ("wealth_ratio", "资金比", "times"),
    ("beta", "贝塔", "num"),
    ("volatility", "年化波动", "pct_plain"),
    ("sharpe", "夏普比率", "num"),
    ("max_drawdown", "最大回撤", "pct_plain"),
    ("calmar", "卡玛比率", "num"),
)


def to_dates(values: list[str]) -> np.ndarray:
    ordered = pd.to_datetime(values)
    return mdates.date2num(pd.DatetimeIndex(ordered).to_pydatetime())


def as_float(values: list) -> np.ndarray:
    return np.array([np.nan if value is None else float(value) for value in values], dtype=float)


def year_table(dates: list[str], algo: dict, year: str) -> dict | None:
    """按自然年切已保存的净值和持仓。表跟选中年份走，不读全样本 stats。"""
    equity = algo.get("equity") or []
    benchmark = algo.get("benchmark_equity") or []
    if not year or len(equity) != len(dates) or len(benchmark) != len(dates) or not dates:
        return None
    chosen = np.array([len(day) >= 4 and day[:4] == year for day in dates])
    if not chosen.any():
        return None
    strategy_returns = _daily_returns(equity)[chosen]
    benchmark_returns = _daily_returns(benchmark)[chosen]
    trades = _year_round_trips(dates, algo.get("position"), equity, int(year))
    return {
        "strategy": _path_table(strategy_returns, trades, hide_win_rate=False),
        "benchmark": _path_table(benchmark_returns, (0, 0), hide_win_rate=True),
    }


def _daily_returns(equity: list) -> np.ndarray:
    values = as_float(equity)
    returns = np.zeros(len(values), dtype=float)
    previous = np.nan
    for index, value in enumerate(values):
        if np.isfinite(previous) and previous != 0.0 and np.isfinite(value):
            returns[index] = value / previous - 1.0
        if np.isfinite(value):
            previous = value
    return returns


def _year_round_trips(dates: list[str], position, equity: list, year: int) -> tuple[int, int]:
    """这一年开的仓算一笔。跨年仍持有的，用该年最后一根净值盯市。"""
    if not isinstance(position, list) or len(position) != len(dates):
        return (0, 0)
    held = np.array([0.0 if value is None else float(value) for value in position], dtype=float)
    wealth = as_float(equity)
    trades = 0
    wins = 0
    entry = None
    entry_year = None
    previous = 0.0
    for index, current in enumerate(held):
        this_year = int(dates[index][:4]) if len(dates[index]) >= 4 else 0
        if previous <= 1e-8 and current > 1e-8:
            entry = float(wealth[index - 1]) if index and np.isfinite(wealth[index - 1]) else 1.0
            entry_year = this_year
        elif previous > 1e-8 and current <= 1e-8 and entry is not None:
            if entry_year == year:
                trades += 1
                if np.isfinite(wealth[index]) and wealth[index] > entry:
                    wins += 1
            entry = None
            entry_year = None
        elif entry_year == year and this_year != year:
            trades += 1
            if index and np.isfinite(wealth[index - 1]) and wealth[index - 1] > entry:
                wins += 1
            entry = None
            entry_year = None
        previous = float(current)
    if entry_year == year and entry is not None and len(wealth):
        trades += 1
        if np.isfinite(wealth[-1]) and wealth[-1] > entry:
            wins += 1
    return trades, wins


def _path_table(returns: np.ndarray, trades: tuple[int, int], hide_win_rate: bool) -> dict:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        values = np.zeros(1, dtype=float)
    cumulative = float(np.prod(1.0 + values) - 1.0)
    periods = max(int(values.size), 1)
    growth = 1.0 + cumulative
    annualized = -1.0 if growth <= 0.0 else float(growth ** (252 / periods) - 1.0)
    volatility = float(np.std(values, ddof=1) * np.sqrt(252)) if values.size > 1 else 0.0
    curve = np.cumprod(1.0 + values)
    drawdown = float(np.min(curve / np.maximum.accumulate(curve) - 1.0))
    if values.size < 2:
        sharpe = 0.0
    else:
        std = float(np.std(values, ddof=1))
        sharpe = 0.0 if std == 0.0 else float(np.mean(values) / std * np.sqrt(252))
    calmar = float(annualized / abs(drawdown)) if drawdown < 0 else 0.0
    trade_count, win_count = trades
    rate = None if hide_win_rate else ((win_count / trade_count) if trade_count else 0.0)
    return {
        "cumulative_return": cumulative,
        "annualized_return": annualized,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "calmar": calmar,
        "trades": int(trade_count),
        "win_rate": rate,
    }


def path_from_level(values: list) -> np.ndarray:
    """窗口内净值改成从 0 起的累计涨跌。只缩放已保存的水平，不另算收益。"""
    series = as_float(values)
    base = np.nan
    for value in series:
        if np.isfinite(value) and value != 0.0:
            base = value
            break
    if not np.isfinite(base):
        return series
    return series / base - 1.0


def rebase(values: list) -> np.ndarray:
    """窗口内两条净值都从 1 起算，买卖带来的涨跌才看得清。"""
    series = as_float(values)
    base = np.nan
    for value in series:
        if np.isfinite(value) and value != 0.0:
            base = value
            break
    if not np.isfinite(base):
        return series
    return series / base


def apply_date_axis(axis, dates: np.ndarray, mode: str) -> None:
    if mode == "month":
        axis.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        pad = 0.8
    elif mode == "decade":
        axis.xaxis.set_major_locator(mdates.YearLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        pad = 20
    else:
        axis.xaxis.set_major_locator(mdates.MonthLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        pad = 3
    if len(dates):
        axis.set_xlim(float(dates[0]) - pad, float(dates[-1]) + pad)
    axis.tick_params(axis="x", labelrotation=30)
    for label in axis.get_xticklabels():
        label.set_ha("right")


def yearly_return_lines(years: dict) -> dict[str, list]:
    """按年份取出已经保存的综合盈利率。缺少年份留空，不在这里连乘净值。"""
    labels: list[str] = []
    strategy: list[float | None] = []
    benchmark: list[float | None] = []
    holding: list[float | None] = []
    has_holding = False
    for year in sorted(years):
        block = years.get(year) or {}
        labels.append(str(year))
        strategy.append(_stored_cumulative(block.get("strategy")))
        benchmark.append(_stored_cumulative(block.get("benchmark")))
        if "holding" in block:
            has_holding = True
            holding.append(_stored_cumulative(block.get("holding")))
        else:
            holding.append(None)
    return {
        "years": labels,
        "strategy": strategy,
        "benchmark": benchmark,
        "holding": holding if has_holding else [],
    }


def _stored_cumulative(side) -> float | None:
    if not isinstance(side, dict):
        return None
    value = side.get("cumulative_return")
    if value is None:
        return None
    return float(value)


def _line_values(values: list[float | None]) -> list[float]:
    return [np.nan if value is None else value for value in values]


def place_legend(axis, ncol: int) -> None:
    """图例横排在坐标轴上沿，不压住折线。"""
    axis.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=ncol,
        fontsize=9,
        borderaxespad=0.15,
        handlelength=1.5,
        columnspacing=1.0,
    )


def slice_record(
    record: dict,
    year: str,
    month: str | None = None,
    span_years: int | None = None,
    start: str | None = None,
    end: str | None = None,
    entire: bool = False,
) -> dict | None:
    """按 JSON 里的日期截取展示区间，不改统计指标。"""
    dates = record.get("dates") or []
    if not dates or (not year and not span_years and not (start and end) and not entire):
        return None
    stamps = pd.to_datetime(dates)
    if entire:
        mask = np.ones(len(dates), dtype=bool)
    elif start and end:
        mask = (stamps >= pd.Timestamp(start)) & (stamps <= pd.Timestamp(end))
    elif span_years:
        if year:
            in_year = stamps.year == int(year)
            if not bool(in_year.any()):
                return None
            end = stamps[in_year].max()
        else:
            end = stamps.max()
        start = end - pd.DateOffset(years=int(span_years))
        mask = (stamps >= start) & (stamps <= end)
    else:
        mask = stamps.year == int(year)
        if month:
            mask = mask & (stamps.month == int(month))
    keep = np.asarray(mask)
    if not keep.any():
        return None
    out = dict(record)
    for key in ("dates", "open", "high", "low", "close", "volume"):
        values = record.get(key)
        if isinstance(values, list):
            out[key] = [value for value, shown in zip(values, keep) if shown]
    shown_dates = set(out["dates"])
    algorithms = {}
    for algo_id, algo in (record.get("algorithms") or {}).items():
        copied = dict(algo)
        for key in ("indicator", "signal", "position", "equity", "benchmark_equity"):
            values = algo.get(key)
            if isinstance(values, list) and len(values) == len(dates):
                copied[key] = [value for value, shown in zip(values, keep) if shown]
        copied["buy_dates"] = [day for day in algo.get("buy_dates") or [] if day in shown_dates]
        copied["sell_dates"] = [day for day in algo.get("sell_dates") or [] if day in shown_dates]
        algorithms[algo_id] = copied
    out["algorithms"] = algorithms
    return out


def mask_dates(
    dates: list[str],
    year: str = "",
    month: str | None = None,
    span_years: int | None = None,
    start: str | None = None,
    end: str | None = None,
) -> np.ndarray | None:
    if not dates or (not year and not span_years and not (start and end)):
        return None
    stamps = pd.to_datetime(dates)
    if start and end:
        mask = (stamps >= pd.Timestamp(start)) & (stamps <= pd.Timestamp(end))
    elif span_years:
        if year:
            in_year = stamps.year == int(year)
            if not bool(in_year.any()):
                return None
            end_stamp = stamps[in_year].max()
        else:
            end_stamp = stamps.max()
        start_stamp = end_stamp - pd.DateOffset(years=int(span_years))
        mask = (stamps >= start_stamp) & (stamps <= end_stamp)
    else:
        mask = stamps.year == int(year)
        if month:
            mask = mask & (stamps.month == int(month))
    keep = np.asarray(mask)
    if not bool(keep.any()):
        return None
    return keep


def slice_columns(dates: list[str], columns: dict[str, list], keep: np.ndarray) -> dict[str, list]:
    out = {"dates": [day for day, shown in zip(dates, keep) if shown]}
    for key, values in columns.items():
        if isinstance(values, list) and len(values) == len(dates):
            out[key] = [value for value, shown in zip(values, keep) if shown]
        else:
            out[key] = []
    return out


def _tagged(change: float | None) -> str:
    if change is None:
        return "flat"
    return "up" if change >= 0 else "down"


def _price_pairs(record: dict) -> list[tuple[str, float]]:
    dates = record.get("dates") or []
    closes = record.get("close") or []
    pairs = []
    for day, price in zip(dates, closes):
        if price is None or len(day) < 10:
            continue
        pairs.append((day, float(price)))
    return pairs


def _series_pairs(record: dict, key: str) -> list[tuple[str, float]]:
    dates = record.get("dates") or []
    values = record.get(key) or []
    if len(values) != len(dates):
        return []
    pairs = []
    for day, value in zip(dates, values):
        if value is None or len(day) < 10:
            continue
        pairs.append((day, float(value)))
    return pairs


def _board_fields(record: dict, year: str, decade: bool = False, entire: bool = False) -> tuple[float | None, float | None, str]:
    """按年时，整体涨跌相对上一年末，年末价格是该年最后收盘价。按10年时，用最近10年的整体涨跌和期末价格。

    有 return_index 时，涨跌用这条已去掉拆股跳变的净值；年末价格仍是原始收盘价。
    """
    prices = _price_pairs(record)
    levels = _series_pairs(record, "return_index") or prices
    if not prices:
        return None, None, "flat"
    if entire:
        end_price = prices[-1][1]
        base = levels[0][1]
        change = None if base in (None, 0.0) else levels[-1][1] / base - 1.0
        return change, end_price, _tagged(change)
    if decade:
        end_day, end_price = prices[-1]
        cutoff = (pd.Timestamp(end_day) - pd.DateOffset(years=10)).strftime("%Y-%m-%d")
        base = _level_on_or_before(levels, cutoff)
        end_level = _level_on_or_before(levels, end_day)
        if base in (None, 0.0):
            base = levels[0][1]
        change = None if base in (None, 0.0) or end_level is None else end_level / base - 1.0
        return change, end_price, _tagged(change)
    previous = None
    year_end = None
    end_price = None
    for day, price in prices:
        if day[:4] == year:
            end_price = price
    for day, level in levels:
        if day[:4] < year:
            previous = level
        elif day[:4] == year:
            year_end = level
    if not year or end_price is None or year_end is None:
        return None, None, "flat"
    base = previous if previous not in (None, 0.0) else _first_level_in_year(levels, year)
    change = None if base in (None, 0.0) else year_end / base - 1.0
    return change, end_price, _tagged(change)


def _level_on_or_before(pairs: list[tuple[str, float]], day: str) -> float | None:
    found = None
    for stamp, level in pairs:
        if stamp <= day:
            found = level
        else:
            break
    return found


def _first_level_in_year(pairs: list[tuple[str, float]], year: str) -> float | None:
    for stamp, level in pairs:
        if stamp[:4] == year:
            return level
    return None


def _format_price(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:,.2f}"


def format_stat(value, kind: str) -> str:
    if value is None:
        return "—"
    if kind == "pct":
        return f"{float(value) * 100:+.2f}%"
    if kind == "pct_plain":
        return f"{float(value) * 100:.2f}%"
    if kind == "int":
        return str(int(value))
    if kind == "times":
        return f"{float(value):.2f}倍"
    return f"{float(value):.2f}"


ROOT = Path(__file__).resolve().parents[1]


class MarketApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("行情演示")
        self.geometry("1200x760")
        self.minsize(980, 640)
        self.datasets = list_datasets()
        self.dataset: dict | None = None
        self.symbol_cache: dict[tuple[str, str], dict] = {}
        self.paper_cache: dict[tuple[str, str], dict | None] = {}
        self.desk_var = tk.StringVar(value="行情演示")
        self._suspend = False
        self.research_mode = False
        self.lab_mode = False
        self.enhance = None
        self.lab = None
        self.page = 0
        self.page_text = tk.StringVar(value=self._page_caption())
        self.view_mode = tk.StringVar(value="按年展示")
        self.year_var = tk.StringVar()
        self.month_var = tk.StringVar(value="1")
        self._years: tuple[str, ...] = ()
        self.month_buttons: dict[int, ttk.Radiobutton] = {}
        self._build()
        self._load_dataset(0)
        self.redraw()

    def _pages(self) -> tuple[str, ...]:
        return INDEX_PAGES if self._is_benchmark() else PAGES

    def _page_caption(self) -> str:
        pages = self._pages()
        page = min(self.page, len(pages) - 1)
        return f"{page + 1} / {len(pages)}  {pages[page]}"

    def _build(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        side = ttk.Frame(self, padding=12)
        side.grid(row=0, column=0, sticky="ns")
        side.rowconfigure(5, weight=3)
        side.rowconfigure(7, weight=2)

        ttk.Label(side, text="数据集").grid(row=0, column=0, sticky="w")
        self.dataset_picker = ttk.Combobox(side, state="readonly", width=28)
        self.dataset_picker.grid(row=1, column=0, sticky="ew", pady=(4, 8))
        self.dataset_picker["values"] = [item["name"] for item in self.datasets]
        self.dataset_picker.bind("<<ComboboxSelected>>", lambda _event: self._on_dataset())

        ttk.Label(side, text="模式").grid(row=2, column=0, sticky="w")
        self.desk_picker = ttk.Combobox(
            side,
            textvariable=self.desk_var,
            state="readonly",
            width=28,
            values=("行情演示", "实盘模拟"),
        )
        self.desk_picker.grid(row=3, column=0, sticky="ew", pady=(4, 8))
        self.desk_picker.bind("<<ComboboxSelected>>", lambda _event: self._on_desk())

        ttk.Label(side, text="股票").grid(row=4, column=0, sticky="w")
        columns = ("symbol", "return", "end")
        self.board = ttk.Treeview(side, columns=columns, show="headings", height=12, selectmode="browse")
        self.board.heading("symbol", text="代码")
        self.board.heading("return", text="全年涨跌")
        self.board.heading("end", text="年末价格")
        self.board.column("symbol", width=120, anchor="w")
        self.board.column("return", width=78, anchor="e")
        self.board.column("end", width=84, anchor="e")
        self.board.grid(row=5, column=0, sticky="nsew", pady=(4, 8))
        self.board.tag_configure("up", foreground="#0b6a2b")
        self.board.tag_configure("down", foreground="#9b1c1c")
        self.board.bind("<<TreeviewSelect>>", lambda _event: self._on_symbol())

        ttk.Label(side, text="算法结果集").grid(row=6, column=0, sticky="w")
        algo_box = ttk.Frame(side)
        algo_box.grid(row=7, column=0, sticky="nsew", pady=(4, 8))
        algo_box.rowconfigure(0, weight=1)
        algo_box.columnconfigure(0, weight=1)
        self.algorithms = ttk.Treeview(algo_box, columns=("name",), show="headings", height=7, selectmode="browse")
        self.algorithms.heading("name", text="名称")
        self.algorithms.column("name", width=220, anchor="w")
        self.algorithms.grid(row=0, column=0, sticky="nsew")
        algo_scroll = ttk.Scrollbar(algo_box, orient="vertical", command=self.algorithms.yview)
        self.algorithms.configure(yscrollcommand=algo_scroll.set)
        algo_scroll.grid(row=0, column=1, sticky="ns")
        self.algorithms.bind("<<TreeviewSelect>>", lambda _event: self.redraw())
        self.algorithm_note = ttk.Label(side, wraplength=300, justify="left")
        self.algorithm_note.grid(row=8, column=0, sticky="ew")

        main = ttk.Frame(self, padding=(0, 12, 12, 12))
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(2, weight=1)
        self._chart_host = main
        pager = ttk.Frame(main)
        pager.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.mode_var = tk.StringVar(value="行情演示")
        self.mode_picker = ttk.Combobox(
            pager,
            textvariable=self.mode_var,
            state="readonly",
            width=12,
            values=("行情演示", "指数增强", "因子测试"),
        )
        self.mode_picker.pack(side="left", padx=(0, 8))
        self.mode_picker.bind("<<ComboboxSelected>>", lambda _event: self._on_mode())
        self.prev_button = ttk.Button(pager, text="上一页", command=self.prev_page)
        self.prev_button.pack(side="left")
        ttk.Label(pager, textvariable=self.page_text).pack(side="left", padx=12)
        self.next_button = ttk.Button(pager, text="下一页", command=self.next_page)
        self.next_button.pack(side="left")

        period = ttk.Frame(main)
        self.period = period
        period.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        period.columnconfigure(1, weight=1)
        year_row = ttk.Frame(period)
        year_row.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(year_row, text="年").pack(side="left")
        self.year_box = ttk.Frame(year_row)
        self.year_box.pack(side="left", padx=(6, 0))
        self.view_picker = ttk.Combobox(
            year_row,
            textvariable=self.view_mode,
            state="readonly",
            width=14,
            values=("按月展示", "按年展示", "按整体展示", "按10年展示"),
        )
        self.view_picker.pack(side="right")
        self.view_picker.bind("<<ComboboxSelected>>", lambda _event: self.redraw())

        month_row = ttk.Frame(period)
        month_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Label(month_row, text="月").pack(side="left")
        self.month_box = ttk.Frame(month_row)
        self.month_box.pack(side="left", padx=(6, 0))
        for month in range(1, 13):
            button = ttk.Radiobutton(
                self.month_box,
                text=f"{month}月",
                value=str(month),
                variable=self.month_var,
                command=self.redraw,
                width=4,
            )
            button.pack(side="left", padx=(0, 2))
            self.month_buttons[month] = button

        self.figure = Figure(figsize=(8.4, 6.2), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=main)
        self.canvas.get_tk_widget().grid(row=2, column=0, sticky="nsew")
        toolbar = NavigationToolbar2Tk(self.canvas, main, pack_toolbar=False)
        self.toolbar = toolbar
        toolbar.grid(row=3, column=0, sticky="ew")
        toolbar.update()

    def _on_dataset(self) -> None:
        index = self.dataset_picker.current()
        self._load_dataset(index)
        if self.lab_mode:
            if not self._sync_lab():
                self._return_to_market("这个数据集没有因子测试用的行情，已回到行情演示。")
            return
        if self.research_mode:
            if not self._sync_enhance():
                self._return_to_market("这个数据集没有指数增强用的行情，已回到行情演示。")
            return
        self.redraw()

    def _source_dir(self) -> Path | None:
        dataset = self.dataset
        if not dataset:
            return None
        source = str(dataset.get("source") or "").strip()
        if not source:
            return None
        path = Path(source)
        if not path.is_absolute():
            path = ROOT / path
        return path if path.is_dir() else None

    def _price_symbols(self) -> list[str]:
        folder = self._source_dir()
        if folder is None or not self.dataset:
            return []
        names = []
        for symbol in self.dataset.get("symbols") or []:
            if symbol == "NDX":
                continue
            if (folder / f"{symbol}.csv").is_file():
                names.append(str(symbol))
        return names

    def _dataset_span(self) -> tuple[str, str] | None:
        names = self._price_symbols()
        if not names:
            return None
        try:
            record = self._cached_symbol(names[0])
        except FileNotFoundError:
            return None
        dates = record.get("dates") or []
        if len(dates) < 2:
            return None
        return str(dates[0]), str(dates[-1])

    def _focus_symbol(self) -> str | None:
        return None if self._is_benchmark() else self.selected_symbol()

    def _sync_lab(self) -> bool:
        names = self._price_symbols()
        folder = self._source_dir()
        if len(names) < 2 or folder is None or not self._ensure_lab():
            return False
        has_index = (folder / "NDX.csv").is_file()
        self.lab.apply_dataset(self.dataset.get("name") or "", names, self._dataset_span(), str(folder), has_index, self._focus_symbol())
        return True

    def _sync_enhance(self) -> bool:
        folder = self._source_dir()
        names = self._price_symbols()
        if folder is None or not (folder / "NDX.csv").is_file() or len(names) < 2 or not self._ensure_enhance():
            return False
        self.enhance.apply_dataset(self.dataset.get("name") or "", names, self._dataset_span(), self._focus_symbol())
        return True

    def _return_to_market(self, message: str) -> None:
        self.mode_var.set("行情演示")
        self._show_market()
        messagebox.showinfo("行情演示", message)

    def _load_dataset(self, index: int) -> None:
        self._suspend = True
        self.dataset = self.datasets[index] if 0 <= index < len(self.datasets) else None
        if self.dataset is not None and not self.dataset_picker.get():
            self.dataset_picker.current(index)
        self._fill_board()
        self._fill_algorithms()
        self._suspend = False

    def _fill_board(self) -> None:
        selected = self.selected_symbol()
        self.board.delete(*self.board.get_children())
        mode = self._view()
        decade = mode == "decade"
        entire = mode == "all"
        self.board.heading("return", text="整体涨跌")
        self.board.heading("end", text="期末价格" if decade or entire else "年末价格")
        symbols = self._board_ids()
        for symbol in symbols:
            record = self._cached_symbol(symbol)
            change, end_price, tag = _board_fields(record, self.year_var.get(), decade=decade, entire=entire)
            label = record.get("name") if self._is_benchmark(symbol) else symbol
            self.board.insert(
                "",
                "end",
                iid=symbol,
                values=(label, format_stat(change, "pct"), _format_price(end_price)),
                tags=(tag,),
            )
        stocks = [symbol for symbol in symbols if not self._is_benchmark(symbol)]
        fallback = stocks[0] if stocks else (symbols[0] if symbols else None)
        if fallback:
            self.board.selection_set(selected if selected in symbols else fallback)
        dataset_id = self.dataset["dataset_id"] if self.dataset else ""
        self._board_marker = (dataset_id, self.year_var.get(), self._view())

    def _cached_symbol(self, symbol: str) -> dict:
        key = (self.dataset["dataset_id"], symbol)
        if key not in self.symbol_cache:
            self.symbol_cache[key] = load_symbol(self.dataset, symbol)
        return self.symbol_cache[key]

    def _trained_algorithms(self) -> list[dict]:
        """只保留当前数据集股票 JSON 里已经写入结果的算法。"""
        if not self.dataset:
            return []
        registered = {item["id"]: item for item in self.dataset.get("algorithms") or [] if item.get("id")}
        found: list[dict] = []
        seen: set[str] = set()
        for symbol in self.dataset.get("symbols") or []:
            try:
                record = self._cached_symbol(symbol)
            except FileNotFoundError:
                continue
            for algo_id, algo in (record.get("algorithms") or {}).items():
                if not algo_id or algo_id in seen:
                    continue
                seen.add(algo_id)
                meta = registered.get(algo_id) or {}
                payload = algo or {}
                found.append(
                    {
                        "id": algo_id,
                        "name": meta.get("name") or payload.get("name") or algo_id,
                        "note": meta.get("note") or payload.get("note") or "",
                    }
                )
        order = {item["id"]: index for index, item in enumerate(self.dataset.get("algorithms") or []) if item.get("id")}
        found.sort(key=lambda item: (order.get(item["id"], len(order)), item["id"]))
        return found

    def _fill_algorithms(self) -> None:
        selected = self.selected_algorithm()
        items = self._trained_algorithms()
        self.visible_algorithms = items
        chosen = self.algorithms.selection()
        if chosen:
            self.algorithms.selection_remove(*chosen)
        self.algorithms.delete(*self.algorithms.get_children())
        for item in items:
            self.algorithms.insert("", "end", iid=item["id"], values=(item["name"],))
        if items:
            target = selected if any(item["id"] == selected for item in items) else items[0]["id"]
            self.algorithms.selection_set(target)
        else:
            self.algorithm_note.configure(text="")

    def _benchmark_id(self) -> str | None:
        item = (self.dataset or {}).get("benchmark") if self.dataset else None
        if isinstance(item, dict) and item.get("id"):
            return str(item["id"])
        return None

    def _is_benchmark(self, symbol: str | None = None) -> bool:
        bench = self._benchmark_id()
        if not bench:
            return False
        chosen = self.selected_symbol() if symbol is None else symbol
        return chosen == bench

    def _board_ids(self) -> list[str]:
        symbols = list(self.dataset.get("symbols") or []) if self.dataset else []
        bench = self._benchmark_id()
        if bench and bench not in symbols:
            return [bench, *symbols]
        return symbols

    def _on_symbol(self) -> None:
        if self._suspend:
            return
        if self.page >= len(self._pages()):
            self.page = 0
        symbol = None if self._is_benchmark() else self.selected_symbol()
        if self.lab_mode and self.lab is not None:
            self.lab.on_show(symbol)
            return
        if self.research_mode and self.enhance is not None:
            self.enhance.on_show(symbol)
            return
        self.redraw()

    def _ensure_enhance(self) -> bool:
        if self.enhance is not None:
            return True
        try:
            from gui.enhance import EnhancePanel

            self.enhance = EnhancePanel(self._chart_host, get_symbol=self.selected_symbol)
        except Exception as exc:
            messagebox.showerror("指数增强", f"页面没有建起来：{exc}")
            self.mode_var.set("行情演示")
            self._show_market()
            return False
        self.enhance.grid(row=1, column=0, rowspan=3, sticky="nsew")
        return True

    def _ensure_lab(self) -> bool:
        if self.lab is not None:
            return True
        try:
            from gui.factor_lab import FactorLabPanel

            self.lab = FactorLabPanel(self._chart_host, get_symbol=self.selected_symbol)
        except Exception as exc:
            messagebox.showerror("因子测试", f"页面没有建起来：{exc}")
            self.mode_var.set("行情演示")
            self._show_market()
            return False
        self.lab.grid(row=1, column=0, rowspan=3, sticky="nsew")
        return True

    def _park_panels(self) -> None:
        if self.enhance is not None:
            self.enhance.grid_remove()
        if self.lab is not None:
            self.lab.grid_remove()

    def _is_paper(self) -> bool:
        return self.desk_var.get() == "实盘模拟"

    def _desk_title(self) -> str:
        return "实盘模拟" if self._is_paper() else "行情演示"

    def _on_desk(self) -> None:
        if self._is_paper():
            self.mode_picker.configure(state="disabled")
            if self.mode_var.get() != "行情演示" or self.research_mode or self.lab_mode:
                self.mode_var.set("行情演示")
                self._show_market()
                return
            self.title("实盘模拟")
            self.redraw()
            return
        self.mode_picker.configure(state="readonly")
        self.title("行情演示")
        self.redraw()

    def _paper_book(self) -> dict | None:
        algorithm_id = self.selected_algorithm()
        if self.dataset is None or not algorithm_id:
            return None
        key = (self.dataset["dataset_id"], algorithm_id)
        if key not in self.paper_cache:
            self.paper_cache[key] = load_paper(self.dataset, algorithm_id)
        return self.paper_cache[key]

    def _paper_slice(self, paper: dict) -> dict | None:
        dates = paper.get("dates") or []
        bounds = self._algorithm_bounds(self.current_record())
        if bounds:
            keep = mask_dates(dates, start=bounds[0], end=bounds[1])
        elif self._view() == "decade":
            keep = mask_dates(dates, self.year_var.get(), span_years=10)
        else:
            month = self.month_var.get() if self._view() == "month" else None
            keep = mask_dates(dates, self.year_var.get(), month)
        if keep is None:
            return None
        symbol = self.selected_symbol() or ""
        book = (paper.get("symbols") or {}).get(symbol) or {}
        return slice_columns(
            dates,
            {
                "nav": paper.get("nav") or [],
                "benchmark_nav": paper.get("benchmark_nav") or [],
                "shares": book.get("shares") or [],
            },
            keep,
        )

    def _show_market(self) -> None:
        self.research_mode = False
        self.lab_mode = False
        self._park_panels()
        self.period.grid()
        self.canvas.get_tk_widget().grid()
        self.toolbar.grid()
        self.title(self._desk_title())
        self.redraw()

    def _on_mode(self) -> None:
        if self._is_paper() and self.mode_var.get() != "行情演示":
            self.mode_var.set("行情演示")
            self._show_market()
            return
        mode = self.mode_var.get()
        symbol = self._focus_symbol()
        if mode == "指数增强":
            if not self._sync_enhance():
                self.mode_var.set("行情演示")
                self._show_market()
                messagebox.showinfo("行情演示", "这个数据集没有指数增强用的行情。")
                return
            self._park_panels()
            self.research_mode = True
            self.lab_mode = False
            self.period.grid_remove()
            self.canvas.get_tk_widget().grid_remove()
            self.toolbar.grid_remove()
            self.enhance.grid()
            self.prev_button.state(["disabled"])
            self.next_button.state(["disabled"])
            self.page_text.set("指数增强")
            self.title("指数增强")
            self.enhance.on_show(symbol)
            return
        if mode == "因子测试":
            if not self._sync_lab():
                self.mode_var.set("行情演示")
                self._show_market()
                messagebox.showinfo("行情演示", "这个数据集没有因子测试用的行情。")
                return
            self._park_panels()
            self.research_mode = False
            self.lab_mode = True
            self.period.grid_remove()
            self.canvas.get_tk_widget().grid_remove()
            self.toolbar.grid_remove()
            self.lab.grid()
            self.prev_button.state(["disabled"])
            self.next_button.state(["disabled"])
            self.page_text.set("因子测试")
            self.title("因子测试")
            self.lab.on_show(symbol)
            return
        self._show_market()

    def selected_symbol(self) -> str | None:
        if not hasattr(self, "board"):
            return None
        chosen = self.board.selection()
        return chosen[0] if chosen else None

    def selected_algorithm(self) -> str | None:
        if not hasattr(self, "algorithms"):
            return None
        chosen = self.algorithms.selection()
        return chosen[0] if chosen else None

    def current_record(self) -> dict | None:
        symbol = self.selected_symbol()
        if self.dataset is None or not symbol:
            return None
        return self._cached_symbol(symbol)

    def prev_page(self) -> None:
        if self.page == 0:
            return
        self.page -= 1
        self.redraw()

    def next_page(self) -> None:
        if self.page >= len(self._pages()) - 1:
            return
        self.page += 1
        self.redraw()

    def redraw(self) -> None:
        if self._suspend or self.research_mode or self.lab_mode:
            return
        pages = self._pages()
        if self.page >= len(pages):
            self.page = 0
        self.page_text.set(self._page_caption())
        self.prev_button.state(["!disabled"] if self.page else ["disabled"])
        self.next_button.state(["!disabled"] if self.page < len(pages) - 1 else ["disabled"])
        self.figure.clear()
        if not self.datasets:
            self._draw_message("outdata 里还没有数据集。\n请先运行：python -m algorithms.rsi")
        else:
            record = self.current_record()
            algorithm_id = self.selected_algorithm()
            note = ""
            if algorithm_id:
                match = [item for item in getattr(self, "visible_algorithms", []) if item["id"] == algorithm_id]
                note = match[0]["note"] if match else ""
            self.algorithm_note.configure(text=note)
            self._sync_period(record)
            self._clamp_year_to_view(record)
            self._sync_board()
            on_index = self._is_benchmark()
            view_only = self._algorithm_bounds(record) is not None
            if view_only:
                if self.page == 4:
                    self._set_period_enabled(True)
                    if self._view() not in ("all", "decade"):
                        self._enable_view_years(record)
                else:
                    self._set_period_locked()
                    if self.page == 2:
                        self._enable_view_years(record)
            elif on_index:
                self._set_period_enabled(True)
            elif self.page == 3:
                self._set_period_enabled(False)
            else:
                self._set_period_enabled(True)
            shown = self._window(record) if record else None
            paper = self._paper_book() if self._is_paper() else None
            if self._is_paper() and paper is None and self.page != 0:
                self._draw_message("还没有这个算法的实盘账户。\n请先运行：python -m algorithms.paper_account")
            elif on_index:
                if record is None:
                    self._draw_message("请选择纳斯达克指数。")
                elif shown is None:
                    self._draw_message(f"{self._span_text()}没有数据。")
                elif self.page == 0:
                    self._draw_raw(shown)
                elif self._is_paper():
                    self._draw_paper_account(paper, algorithm_id)
                else:
                    self._draw_index(shown, algorithm_id)
            elif self.page == 3:
                self._draw_overview(algorithm_id)
            elif self.page == 4:
                self._draw_year_lines(algorithm_id)
            elif record is None:
                self._draw_message("请选择数据集和股票。")
            elif shown is None:
                self._draw_message(f"{self._span_text()}没有数据。")
            elif self.page == 0:
                self._draw_raw(shown)
            elif self.page == 1:
                self._draw_signals(shown, algorithm_id, paper)
            else:
                if self._is_paper():
                    self._draw_paper_account(paper, algorithm_id)
                else:
                    self._draw_stats(shown, algorithm_id)
        if self.page == 3 and self.datasets and not self._is_benchmark():
            self.figure.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.04, hspace=0.32)
        else:
            self.figure.tight_layout()
        self.canvas.draw_idle()

    def _view(self) -> str:
        mode = self.view_mode.get()
        if mode == "按月展示":
            return "month"
        if mode == "按10年展示":
            return "decade"
        if mode == "按整体展示":
            return "all"
        return "year"

    def _algorithm_bounds(self, record: dict | None) -> tuple[str, str] | None:
        """带 view_start 的算法，右侧只画这一段，用来把验证集和训练年份分开。"""
        algorithm_id = self.selected_algorithm()
        algo = ((record or {}).get("algorithms") or {}).get(algorithm_id) or {}
        start, end = algo.get("view_start"), algo.get("view_end")
        if start and end:
            return str(start), str(end)
        return None

    def _clamp_year_to_view(self, record: dict | None) -> None:
        """验证集算法的年份停在这段区间里，统计表不会落到训练年份。"""
        bounds = self._algorithm_bounds(record)
        if not bounds:
            return
        start_y, end_y = bounds[0][:4], bounds[1][:4]
        year = self.year_var.get()
        if year and start_y <= year <= end_y:
            return
        inside = [item for item in self._years if start_y <= item <= end_y]
        if inside:
            self.year_var.set(inside[-1])

    def _enable_view_years(self, record: dict | None) -> None:
        bounds = self._algorithm_bounds(record)
        if not bounds:
            return
        start_y, end_y = bounds[0][:4], bounds[1][:4]
        for child in self.year_box.winfo_children():
            text = str(child.cget("text"))
            child.state(["!disabled"] if start_y <= text <= end_y else ["disabled"])

    def _span_text(self) -> str:
        bounds = self._algorithm_bounds(self.current_record())
        if bounds:
            return f"验证集 {bounds[0]} 至 {bounds[1]}"
        year = self.year_var.get() or "—"
        if self._view() == "month" and self.month_var.get():
            return f"{year}年{int(self.month_var.get())}月"
        if self._view() == "decade":
            return "近10年"
        if self._view() == "all":
            return "全部"
        return f"{year}年"

    def _window(self, record: dict) -> dict | None:
        bounds = self._algorithm_bounds(record)
        if bounds:
            return slice_record(record, "", start=bounds[0], end=bounds[1])
        if self._view() == "all":
            return slice_record(record, "", entire=True)
        if self._view() == "decade":
            return slice_record(record, "", span_years=10)
        month = self.month_var.get() if self._view() == "month" else None
        return slice_record(record, self.year_var.get(), month)

    def _sync_period(self, record: dict | None) -> None:
        dates = (record or {}).get("dates") or []
        years = tuple(sorted({item[:4] for item in dates if len(item) >= 4}))
        if years != self._years:
            for child in self.year_box.winfo_children():
                child.destroy()
            for year in years:
                ttk.Radiobutton(
                    self.year_box,
                    text=year,
                    value=year,
                    variable=self.year_var,
                    command=self.redraw,
                ).pack(side="left", padx=(0, 4))
            self._years = years
        if years and self.year_var.get() not in years:
            self.year_var.set(years[-1])
        self._refresh_months(dates)

    def _sync_board(self) -> None:
        dataset_id = self.dataset["dataset_id"] if self.dataset else ""
        marker = (dataset_id, self.year_var.get(), self._view())
        if marker == getattr(self, "_board_marker", None):
            return
        self._suspend = True
        self._fill_board()
        self._suspend = False

    def _set_period_enabled(self, enabled: bool) -> None:
        """前三页按10年时年份变灰。全局统计页仍按选中年份读取。"""
        self.view_picker.state(["!disabled"] if enabled else ["disabled"])
        lock_year = enabled and self._view() in ("decade", "all")
        for child in self.year_box.winfo_children():
            child.state(["disabled"] if lock_year else ["!disabled"])
        if enabled and self._view() not in ("decade", "all"):
            dates = (self.current_record() or {}).get("dates") or []
            self._refresh_months(dates)
        else:
            for button in self.month_buttons.values():
                button.state(["disabled"])

    def _set_period_locked(self) -> None:
        """历年走势使用全部年份，年份、月份和展示方式都停用。"""
        self.view_picker.state(["disabled"])
        for child in self.year_box.winfo_children():
            child.state(["disabled"])
        for button in self.month_buttons.values():
            button.state(["disabled"])

    def _refresh_months(self, dates: list[str]) -> None:
        year = self.year_var.get()
        available = {int(item[5:7]) for item in dates if item[:4] == year and len(item) >= 7}
        for month, button in self.month_buttons.items():
            enabled = self._view() == "month" and month in available
            button.state(["!disabled"] if enabled else ["disabled"])
        current = int(self.month_var.get() or "0")
        if available and current not in available:
            self.month_var.set(str(min(available)))

    def _draw_message(self, text: str) -> None:
        axis = self.figure.add_subplot(1, 1, 1)
        axis.axis("off")
        axis.text(0.5, 0.5, text, ha="center", va="center", fontsize=12)

    def _draw_raw(self, record: dict) -> None:
        dates = to_dates(record["dates"])
        close = as_float(record["close"])
        volume = as_float(record["volume"])
        price_axis = self.figure.add_subplot(2, 1, 1)
        volume_axis = self.figure.add_subplot(2, 1, 2, sharex=price_axis)
        price_axis.plot(dates, close, color="#1f4b99", linewidth=1.4, label=record["price_field"])
        title_name = record.get("name") or record["symbol"]
        price_axis.set_title(f"{title_name} 原始价格 {self._span_text()}", pad=18)
        price_axis.set_ylabel("价格")
        place_legend(price_axis, 1)
        price_axis.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        price_axis.tick_params(labelbottom=False)
        volume_axis.bar(dates, volume, width=0.8, color="#8aa0b8", align="center")
        volume_name = record.get("name") or record["symbol"]
        volume_axis.set_title(f"{volume_name} 成交量")
        volume_axis.set_ylabel("成交量" if record.get("kind") == "index" else "股")
        volume_axis.set_xlabel("日期")
        volume_axis.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        apply_date_axis(volume_axis, dates, self._view())

    def _draw_signals(self, record: dict, algorithm_id: str | None, paper: dict | None = None) -> None:
        algo = (record.get("algorithms") or {}).get(algorithm_id or "")
        if not algo:
            self._draw_message("当前股票的 JSON 里没有这个算法。")
            return
        dates = to_dates(record["dates"])
        close = as_float(record["close"])
        by_date = dict(zip(record["dates"], close))
        price_axis = self.figure.add_subplot(2, 1, 1)
        indicator_axis = self.figure.add_subplot(2, 1, 2, sharex=price_axis)
        price_axis.plot(dates, close, color="#1f4b99", linewidth=1.3, label=record["price_field"])
        buys = algo.get("buy_dates", [])
        sells = algo.get("sell_dates", [])
        if paper is not None:
            book = (paper.get("symbols") or {}).get(record.get("symbol") or "") or {}
            buys = book.get("buy_dates") or []
            sells = book.get("sell_dates") or []
        self._mark(price_axis, buys, by_date, "^", "#0b6a2b", "买入")
        self._mark(price_axis, sells, by_date, "v", "#9b1c1c", "卖出")
        price_axis.set_title(f"{record['symbol']} 价格与{algo['name']}买卖点 {self._span_text()}", pad=18)
        price_axis.set_ylabel("价格")
        place_legend(price_axis, 3)
        price_axis.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        price_axis.tick_params(labelbottom=False)

        indicator_axis.plot(dates, as_float(algo["indicator"]), color="#1f4b99", linewidth=1.2, label=algo["id"].upper())
        params = algo.get("params") or {}
        if "oversold" in params:
            indicator_axis.axhline(params["oversold"], color="#0b6a2b", linewidth=0.8, linestyle="--", label=f"超卖 {params['oversold']:.0f}")
        if "overbought" in params:
            indicator_axis.axhline(params["overbought"], color="#9b1c1c", linewidth=0.8, linestyle="--", label=f"超买 {params['overbought']:.0f}")
        indicator_axis.set_title(f"{record['symbol']} {algo['id'].upper()}", pad=18)
        indicator_axis.set_ylabel(algo["id"].upper())
        indicator_axis.set_xlabel("日期")
        place_legend(indicator_axis, 3)
        indicator_axis.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        apply_date_axis(indicator_axis, dates, self._view())

    def _mark(self, axis, days: list[str], by_date: dict, marker: str, color: str, label: str) -> None:
        points = [(day, by_date[day]) for day in days if day in by_date]
        if not points:
            return
        when = to_dates([day for day, _price in points])
        prices = [price for _day, price in points]
        axis.scatter(when, prices, marker=marker, s=36, color=color, label=label, zorder=3)

    def _draw_stats(self, record: dict, algorithm_id: str | None) -> None:
        algo = (record.get("algorithms") or {}).get(algorithm_id or "")
        if not algo:
            self._draw_message("当前股票的 JSON 里没有这个算法。")
            return
        dates = to_dates(record["dates"])
        curve = self.figure.add_subplot(2, 1, 1)
        curve.plot(dates, rebase(algo["equity"]), color="#1f4b99", linewidth=1.4, label=algo["name"])
        curve.plot(dates, rebase(algo["benchmark_equity"]), color="#888888", linewidth=1.2, label="买入持有")
        curve.set_title(f"{record['symbol']} 净值（{self._span_text()}）", pad=18)
        curve.set_ylabel("净值")
        curve.set_xlabel("日期")
        place_legend(curve, 2)
        curve.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        apply_date_axis(curve, dates, self._view())

        table_axis = self.figure.add_subplot(2, 1, 2)
        table_axis.axis("off")
        fee = (algo.get("params") or {}).get("fee")
        fee_text = f"，策略手续费 {fee:.1%}" if isinstance(fee, (int, float)) else ""
        full = self.current_record() or record
        full_algo = ((full.get("algorithms") or {}).get(algorithm_id or "")) or algo
        year = self.year_var.get() or ""
        table_axis.set_title(f"统计指标（{year}年）{fee_text}", loc="left", fontsize=11, pad=8)
        block = year_table(full.get("dates") or [], full_algo, year)
        if not block:
            table_axis.text(0.0, 0.5, f"{year}年没有统计", fontsize=11)
            return
        strategy = block["strategy"]
        benchmark = block["benchmark"]
        rows = [
            [label, format_stat(strategy.get(key), kind), format_stat(benchmark.get(key), kind)]
            for key, label, kind in STAT_ROWS
        ]
        table = table_axis.table(cellText=rows, colLabels=["指标", "策略", "买入持有"], loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.35)

    def _draw_paper_account(self, paper: dict | None, algorithm_id: str | None) -> None:
        """整账户美元净值，不把第一条缩放到 1。"""
        if not paper:
            self._draw_message("还没有这个算法的实盘账户。\n请先运行：python -m algorithms.paper_account")
            return
        shown = self._paper_slice(paper)
        if shown is None:
            self._draw_message(f"{self._span_text()}没有数据。")
            return
        dates = to_dates(shown["dates"])
        curve = self.figure.add_subplot(2, 1, 1)
        bench_name = str(paper.get("benchmark_name") or "买入持有")
        curve.plot(dates, as_float(shown["nav"]), color="#1f4b99", linewidth=1.4, label=self._algorithm_name(algorithm_id))
        curve.plot(dates, as_float(shown["benchmark_nav"]), color="#888888", linewidth=1.2, label=bench_name)
        symbol = self.selected_symbol() or ""
        shares = shown.get("shares") or []
        if symbol in (paper.get("symbols") or {}):
            held = int(shares[-1]) if shares else 0
            curve.set_title(f"{symbol} 持股 {held}（{self._span_text()}）", pad=18)
        else:
            curve.set_title(f"组合账户（{self._span_text()}）", pad=18)
        curve.set_ylabel("美元")
        curve.set_xlabel("日期")
        curve.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{value:,.0f}"))
        place_legend(curve, 2)
        curve.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        apply_date_axis(curve, dates, self._view())

        year = self.year_var.get() or ""
        block = (paper.get("years") or {}).get(year) or {}
        strategy = block.get("strategy") or {}
        benchmark = block.get("benchmark") or {}
        table_axis = self.figure.add_subplot(2, 1, 2)
        table_axis.axis("off")
        table_axis.set_title(f"统计指标（{year}年），本金 100000 美元，每股手续费 0.005 美元", loc="left", fontsize=11, pad=8)
        if not strategy:
            table_axis.text(0.0, 0.5, f"{year}年没有统计", fontsize=11)
            return
        rows = [
            [label, format_stat(strategy.get(key), kind), format_stat(benchmark.get(key), kind)]
            for key, label, kind in STAT_ROWS
        ]
        table = table_axis.table(cellText=rows, colLabels=["指标", "账户", bench_name], loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.35)

    def _draw_index(self, record: dict, algorithm_id: str | None) -> None:
        """只画组合净值和纳斯达克指数，下面是同一段区间的指标。"""
        algo = (record.get("algorithms") or {}).get(algorithm_id or "")
        if not algo:
            self._draw_message("当前指数 JSON 里没有这个算法。")
            return
        dates = to_dates(record["dates"])
        curve = self.figure.add_subplot(2, 1, 1)
        curve.plot(dates, rebase(algo["equity"]), color="#1f4b99", linewidth=1.4, label=algo["name"])
        curve.plot(dates, rebase(algo["benchmark_equity"]), color="#888888", linewidth=1.2, label=record.get("name") or "纳斯达克指数")
        curve.set_title(f"{algo['name']} 与{record.get('name') or '纳斯达克指数'}（{self._span_text()}）", pad=18)
        curve.set_ylabel("净值")
        curve.set_xlabel("日期")
        place_legend(curve, 2)
        curve.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        apply_date_axis(curve, dates, self._view())

        block, span = self._index_stats(algo)
        strategy = block.get("strategy") or {}
        benchmark = block.get("benchmark") or {}
        table_axis = self.figure.add_subplot(2, 1, 2)
        table_axis.axis("off")
        note = record.get("universe_note") or "没有历史成分名单，不能称为指数增强。"
        table_axis.set_title(f"指数指标（{span}）\n{note}", loc="left", fontsize=9, pad=8)
        rows = []
        for key, label, kind in INDEX_STAT_ROWS:
            rows.append([label, format_stat(strategy.get(key), kind), format_stat(benchmark.get(key), kind)])
        reported = (algo.get("stats_since_2021") or {}).get("strategy") or {}
        rows.append(["2021年以来资金比", format_stat(reported.get("wealth_ratio"), "times"), "1.00倍" if reported.get("wealth_ratio") is not None else "—"])
        rows.append(["2021年以来贝塔", format_stat(reported.get("beta"), "num"), "1.00" if reported.get("beta") is not None else "—"])
        table = table_axis.table(
            cellText=rows,
            colLabels=["指标", "组合", record.get("name") or "纳斯达克指数"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.35)

    def _index_stats(self, algo: dict) -> tuple[dict, str]:
        if self._view() == "decade":
            return algo.get("stats_10y") or algo.get("stats") or {}, "近10年"
        year = self.year_var.get()
        found = (algo.get("years") or {}).get(year)
        if found:
            return found, f"{year}年"
        return algo.get("stats") or {}, "全样本"

    def _stored_overview(self, algorithm_id: str | None) -> dict | None:
        year = self.year_var.get()
        if self.dataset is None or not algorithm_id or not year:
            return None
        for item in self.dataset.get("algorithms") or []:
            if item.get("id") == algorithm_id:
                return (item.get("years") or {}).get(year)
        return None

    def _draw_overview(self, algorithm_id: str | None) -> None:
        if self.dataset is None:
            self._draw_message("outdata 里还没有数据集。\n请先运行：python -m algorithms.rsi")
            return
        year = self.year_var.get() or "—"
        stored = self._stored_years(algorithm_id)
        if stored and year not in stored:
            year = sorted(stored)[-1]
        block = (stored or {}).get(year) if stored else self._stored_overview(algorithm_id)
        if not block:
            hint = "python -m algorithms.paper_account" if self._is_paper() else "python -m algorithms.rsi"
            self._draw_message(f"{year}年还没有保存全局统计。\n请重新运行：{hint}")
            return
        symbols = [row for row in block.get("symbols") or [] if row.get("cumulative_return") is not None]
        ordered = sorted(symbols, key=lambda row: float(row["cumulative_return"]))
        returns = [float(row["cumulative_return"]) for row in ordered]
        labels = [row["symbol"] for row in ordered]
        colors = ["#0b6a2b" if value >= 0 else "#9b1c1c" for value in returns]
        grid = self.figure.add_gridspec(2, 1, height_ratios=[3.05, 1.4], hspace=0.32)
        bar_axis = self.figure.add_subplot(grid[0, 0])
        metric_axis = self.figure.add_subplot(grid[1, 0])
        bar_axis.barh(labels, returns, color=colors, height=0.72)
        bar_axis.axvline(0, color="#666666", linewidth=0.6)
        bar_axis.set_title(f"{year}年各股策略累计收益")
        bar_axis.set_xlabel("累计收益")
        bar_axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{value * 100:.0f}%"))
        bar_axis.tick_params(axis="y", labelsize=8 if len(labels) <= 16 else 7)
        bar_axis.grid(True, axis="x", linewidth=0.4, alpha=0.5)
        self._draw_metric_bars(metric_axis, block, algorithm_id, year)

    def _history_span(self) -> str:
        view = self._view()
        year = self.year_var.get() or "—"
        if view == "month" and self.month_var.get():
            return f"{year}年{int(self.month_var.get())}月"
        if view == "year":
            return f"{year}年"
        if view == "decade":
            return "近10年"
        bounds = self._algorithm_bounds(self.current_record())
        if bounds:
            return f"验证集 {bounds[0]} 至 {bounds[1]}"
        return "全部"

    def _history_keep(self, dates: list[str]) -> np.ndarray | None:
        if not dates:
            return None
        bounds = self._algorithm_bounds(self.current_record())
        view = self._view()
        if view == "month":
            keep = mask_dates(dates, self.year_var.get(), self.month_var.get())
        elif view == "year":
            keep = mask_dates(dates, self.year_var.get())
        elif view == "decade":
            keep = mask_dates(dates, self.year_var.get(), span_years=10)
        elif bounds:
            keep = mask_dates(dates, start=bounds[0], end=bounds[1])
        else:
            keep = np.ones(len(dates), dtype=bool)
        if keep is None:
            return None
        if bounds and view != "all":
            limited = mask_dates(dates, start=bounds[0], end=bounds[1])
            if limited is None:
                return None
            keep = keep & limited
        if not bool(np.asarray(keep).any()):
            return None
        return np.asarray(keep)

    def _history_frame(self, algorithm_id: str | None) -> dict | None:
        if self._is_paper():
            paper = self._paper_book()
            if not paper or not paper.get("dates"):
                return None
            return {
                "dates": paper.get("dates") or [],
                "strategy": paper.get("nav") or [],
                "benchmark": paper.get("benchmark_nav") or [],
                "holding": [],
            }
        if self.dataset is None or not algorithm_id:
            return None
        for item in self.dataset.get("algorithms") or []:
            if item.get("id") == algorithm_id and item.get("history"):
                return item["history"]
        bench = self._benchmark_id()
        if not bench:
            return None
        try:
            record = self._cached_symbol(bench)
        except FileNotFoundError:
            return None
        algo = (record.get("algorithms") or {}).get(algorithm_id) or {}
        if not algo.get("equity"):
            return None
        return {
            "dates": record.get("dates") or [],
            "strategy": algo.get("equity") or [],
            "benchmark": algo.get("benchmark_equity") or [],
            "holding": [],
        }

    def _draw_year_lines(self, algorithm_id: str | None) -> None:
        """历史走势。有日度净值时按整体、年、月截取；否则退回各年综合盈利率。"""
        frame = self._history_frame(algorithm_id)
        if frame and frame.get("dates"):
            keep = self._history_keep(frame["dates"])
            if keep is None:
                self._draw_message(f"{self._history_span()}没有走势。")
                return
            shown = slice_columns(
                frame["dates"],
                {
                    "strategy": frame.get("strategy") or [],
                    "benchmark": frame.get("benchmark") or [],
                    "holding": frame.get("holding") or [],
                },
                keep,
            )
            axis = self.figure.add_subplot(1, 1, 1)
            stamps = to_dates(shown["dates"])
            axis.plot(stamps, path_from_level(shown["strategy"]), color="#1f4b99", linewidth=1.6, label="策略")
            if shown["benchmark"]:
                axis.plot(stamps, path_from_level(shown["benchmark"]), color="#888888", linewidth=1.4, label=self._benchmark_label())
            if any(value is not None for value in shown["holding"]):
                axis.plot(stamps, path_from_level(shown["holding"]), color="#c47b2b", linewidth=1.4, label="持有")
            axis.axhline(0, color="#666666", linewidth=0.6)
            axis.set_title(f"{self._algorithm_name(algorithm_id)} 历史走势（{self._history_span()}）", pad=18)
            axis.set_xlabel("日期")
            axis.set_ylabel("累计涨跌")
            axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{value * 100:.0f}%"))
            place_legend(axis, 3)
            axis.grid(True, axis="y", linewidth=0.4, alpha=0.5)
            axis_mode = "decade" if self._view() == "all" else self._view()
            apply_date_axis(axis, stamps, axis_mode)
            return
        if self._view() == "month":
            self._draw_message("这一页还没有按月的走势。\n请重新运行对应的算法。")
            return
        years = self._stored_years(algorithm_id)
        if self._view() == "year" and self.year_var.get():
            years = {self.year_var.get(): years.get(self.year_var.get())} if self.year_var.get() in years else {}
        lines = yearly_return_lines(years)
        if not lines["years"] or all(value is None for value in lines["strategy"]):
            self._draw_message("还没有分年统计。\n请重新运行对应的算法。")
            return
        axis = self.figure.add_subplot(1, 1, 1)
        stamps = [int(year) for year in lines["years"]]
        axis.plot(stamps, _line_values(lines["strategy"]), color="#1f4b99", linewidth=1.6, marker="o", label="策略")
        axis.plot(stamps, _line_values(lines["benchmark"]), color="#888888", linewidth=1.4, marker="s", label=self._benchmark_label())
        if lines["holding"]:
            axis.plot(stamps, _line_values(lines["holding"]), color="#c47b2b", linewidth=1.4, marker="^", label="持有")
        axis.axhline(0, color="#666666", linewidth=0.6)
        axis.set_xticks(stamps)
        axis.set_title(f"{self._algorithm_name(algorithm_id)} 历史走势（{self._history_span()}）", pad=18)
        axis.set_xlabel("年份")
        axis.set_ylabel("综合盈利率")
        axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{value * 100:.0f}%"))
        place_legend(axis, 3)
        axis.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    def _stored_years(self, algorithm_id: str | None) -> dict:
        if self._is_paper():
            paper = self._paper_book() or {}
            return paper.get("years") or {}
        if self.dataset is None or not algorithm_id:
            return {}
        for item in self.dataset.get("algorithms") or []:
            if item.get("id") == algorithm_id:
                return item.get("years") or {}
        return {}

    def _benchmark_label(self) -> str:
        if self._is_paper():
            paper = self._paper_book() or {}
            if paper.get("benchmark_name"):
                return str(paper["benchmark_name"])
        named = self.dataset.get("benchmark") if self.dataset else None
        if isinstance(named, dict) and named.get("name"):
            return str(named["name"])
        return "买入持有"

    def _algorithm_name(self, algorithm_id: str | None) -> str:
        for item in (self.dataset or {}).get("algorithms") or []:
            if item.get("id") == algorithm_id:
                return str(item.get("name") or algorithm_id)
        return algorithm_id or ""

    def _draw_metric_bars(self, axis, block: dict, algorithm_id: str | None, year: str) -> None:
        """指标横排成两行，避免竖表挡住上方的收益图。"""
        strategy = block.get("strategy") or {}
        benchmark = block.get("benchmark") or {}
        count = int(strategy.get("count") or 0)
        beat = strategy.get("beat_benchmark")
        columns = [
            ("count", "股票数", "int"),
            ("profit_ratio", "盈利占比", "pct_plain"),
            ("win_rate", "综合胜率", "pct_plain"),
            ("cumulative_return", "综合盈利率", "pct"),
            ("median_return", "收益中位数", "pct"),
            ("annualized_return", "平均年化", "pct"),
            ("sharpe", "平均夏普", "num"),
            ("max_drawdown", "平均回撤", "pct_plain"),
            ("trades", "交易次数", "int"),
        ]
        headers = [label for _key, label, _kind in columns]
        holding = block.get("holding") or None
        sides = [strategy, benchmark]
        bench_label = self._benchmark_label()
        row_labels = ["策略", bench_label]
        if holding:
            sides.append(holding)
            row_labels.append("持有")
        cells = [[self._overview_cell(side, key, kind) for key, _label, kind in columns] for side in sides]
        cells[0].append(f"{int(beat)}/{count}" if beat is not None and count else "—")
        for row in cells[1:]:
            row.append("—")
        headers.append("跑赢")
        axis.axis("off")
        name = next((item["name"] for item in self.dataset["algorithms"] if item["id"] == algorithm_id), algorithm_id or "")
        axis.set_title(f"{name} {year}年指标", loc="left", fontsize=11, pad=6)
        table = axis.table(cellText=cells, rowLabels=row_labels, colLabels=headers, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.28)

    def _overview_cell(self, side: dict, key: str, kind: str) -> str:
        value = side.get(key)
        if kind == "int":
            return "—" if value is None else str(int(value))
        return format_stat(value, kind)


def main() -> None:
    app = MarketApp()
    app.mainloop()


if __name__ == "__main__":
    main()
