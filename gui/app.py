"""行情窗口只读取 outdata 中的 JSON，不在界面里计算指标。

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

from gui.outdata import list_datasets, load_symbol

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PAGES = ("原始数据", "算法点位", "统计指标", "收益对比", "全局统计")
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
    ("excess_return", "超额收益", "pct"),
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


def slice_record(record: dict, year: str, month: str | None = None, span_years: int | None = None) -> dict | None:
    """按 JSON 里的日期截取展示区间，不改统计指标。"""
    dates = record.get("dates") or []
    if not dates or (not year and not span_years):
        return None
    stamps = pd.to_datetime(dates)
    if span_years:
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


def _board_fields(record: dict, year: str, decade: bool = False) -> tuple[float | None, float | None, str]:
    """按年时，整体涨跌相对上一年末，年末价格是该年最后收盘价。按10年时，用最近10年的整体涨跌和期末价格。

    有 return_index 时，涨跌用这条已去掉拆股跳变的净值；年末价格仍是原始收盘价。
    """
    prices = _price_pairs(record)
    levels = _series_pairs(record, "return_index") or prices
    if not prices:
        return None, None, "flat"
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
    return f"{float(value):.2f}"


class MarketApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("行情")
        self.geometry("1200x760")
        self.minsize(980, 640)
        self.datasets = list_datasets()
        self.dataset: dict | None = None
        self.symbol_cache: dict[tuple[str, str], dict] = {}
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
        side.rowconfigure(3, weight=3)
        side.rowconfigure(5, weight=2)

        ttk.Label(side, text="数据集").grid(row=0, column=0, sticky="w")
        self.dataset_picker = ttk.Combobox(side, state="readonly", width=28)
        self.dataset_picker.grid(row=1, column=0, sticky="ew", pady=(4, 8))
        self.dataset_picker["values"] = [item["name"] for item in self.datasets]
        self.dataset_picker.bind("<<ComboboxSelected>>", lambda _event: self._on_dataset())

        ttk.Label(side, text="股票").grid(row=2, column=0, sticky="w")
        columns = ("symbol", "return", "end")
        self.board = ttk.Treeview(side, columns=columns, show="headings", height=12, selectmode="browse")
        self.board.heading("symbol", text="代码")
        self.board.heading("return", text="全年涨跌")
        self.board.heading("end", text="年末价格")
        self.board.column("symbol", width=120, anchor="w")
        self.board.column("return", width=78, anchor="e")
        self.board.column("end", width=84, anchor="e")
        self.board.grid(row=3, column=0, sticky="nsew", pady=(4, 8))
        self.board.tag_configure("up", foreground="#0b6a2b")
        self.board.tag_configure("down", foreground="#9b1c1c")
        self.board.bind("<<TreeviewSelect>>", lambda _event: self._on_symbol())

        ttk.Label(side, text="算法").grid(row=4, column=0, sticky="w")
        self.algorithms = ttk.Treeview(side, columns=("name",), show="headings", height=9, selectmode="browse")
        self.algorithms.heading("name", text="名称")
        self.algorithms.column("name", width=220, anchor="w")
        self.algorithms.grid(row=5, column=0, sticky="nsew", pady=(4, 8))
        self.algorithms.bind("<<TreeviewSelect>>", lambda _event: self.redraw())
        self.algorithm_note = ttk.Label(side, wraplength=240)
        self.algorithm_note.grid(row=6, column=0, sticky="w")

        main = ttk.Frame(self, padding=(0, 12, 12, 12))
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(2, weight=1)
        self._chart_host = main
        pager = ttk.Frame(main)
        pager.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.mode_var = tk.StringVar(value="行情")
        self.mode_picker = ttk.Combobox(
            pager,
            textvariable=self.mode_var,
            state="readonly",
            width=12,
            values=("行情", "指数增强", "因子测试"),
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
            width=12,
            values=("按月展示", "按年展示", "按10年展示"),
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
        symbol = None if self._is_benchmark() else self.selected_symbol()
        if self.lab_mode and self.lab is not None:
            self.lab.on_show(symbol)
            return
        if self.research_mode and self.enhance is not None:
            self.enhance.on_show(symbol)
            return
        self.redraw()

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
        decade = self._view() == "decade"
        self.board.heading("return", text="整体涨跌")
        self.board.heading("end", text="期末价格" if decade else "年末价格")
        symbols = self._board_ids()
        for symbol in symbols:
            record = self._cached_symbol(symbol)
            change, end_price, tag = _board_fields(record, self.year_var.get(), decade=decade)
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
            self.mode_var.set("行情")
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
            self.mode_var.set("行情")
            self._show_market()
            return False
        self.lab.grid(row=1, column=0, rowspan=3, sticky="nsew")
        return True

    def _park_panels(self) -> None:
        if self.enhance is not None:
            self.enhance.grid_remove()
        if self.lab is not None:
            self.lab.grid_remove()

    def _show_market(self) -> None:
        self.research_mode = False
        self.lab_mode = False
        self._park_panels()
        self.period.grid()
        self.canvas.get_tk_widget().grid()
        self.toolbar.grid()
        self.title("行情")
        self.redraw()

    def _on_mode(self) -> None:
        mode = self.mode_var.get()
        symbol = None if self._is_benchmark() else self.selected_symbol()
        if mode == "指数增强":
            if not self._ensure_enhance():
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
            if not self._ensure_lab():
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
            self._sync_board()
            on_index = self._is_benchmark()
            self._set_period_enabled(on_index or self.page != len(pages) - 1)
            shown = self._window(record) if record else None
            if on_index:
                if record is None:
                    self._draw_message("请选择纳斯达克指数。")
                elif shown is None:
                    self._draw_message(f"{self._span_text()}没有数据。")
                elif self.page == 0:
                    self._draw_raw(shown)
                else:
                    self._draw_index(shown, algorithm_id)
            elif self.page == len(PAGES) - 1:
                self._draw_overview(algorithm_id)
            elif record is None:
                self._draw_message("请选择数据集和股票。")
            elif shown is None:
                self._draw_message(f"{self._span_text()}没有数据。")
            elif self.page == 0:
                self._draw_raw(shown)
            elif self.page == 1:
                self._draw_signals(shown, algorithm_id)
            elif self.page == 2:
                self._draw_stats(shown, algorithm_id)
            else:
                self._draw_return_compare(shown, algorithm_id)
        if self.page == len(self._pages()) - 1 and self.datasets and not self._is_benchmark():
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
        return "year"

    def _span_text(self) -> str:
        year = self.year_var.get() or "—"
        if self._view() == "month" and self.month_var.get():
            return f"{year}年{int(self.month_var.get())}月"
        if self._view() == "decade":
            return "近10年"
        return f"{year}年"

    def _window(self, record: dict) -> dict | None:
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
        """前四页按10年时年份变灰。全局统计页仍按选中年份读取。"""
        self.view_picker.state(["!disabled"] if enabled else ["disabled"])
        lock_year = enabled and self._view() == "decade"
        for child in self.year_box.winfo_children():
            child.state(["disabled"] if lock_year else ["!disabled"])
        if enabled and self._view() != "decade":
            dates = (self.current_record() or {}).get("dates") or []
            self._refresh_months(dates)
        else:
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

    def _draw_signals(self, record: dict, algorithm_id: str | None) -> None:
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
        self._mark(price_axis, algo.get("buy_dates", []), by_date, "^", "#0b6a2b", "买入")
        self._mark(price_axis, algo.get("sell_dates", []), by_date, "v", "#9b1c1c", "卖出")
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
        table_axis.set_title(f"统计指标（全样本）{fee_text}", loc="left", fontsize=11, pad=8)
        strategy = algo["stats"]["strategy"]
        benchmark = algo["stats"]["benchmark"]
        rows = [
            [label, format_stat(strategy.get(key), kind), format_stat(benchmark.get(key), kind)]
            for key, label, kind in STAT_ROWS
        ]
        table = table_axis.table(cellText=rows, colLabels=["指标", "策略", "买入持有"], loc="center", cellLoc="center")
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
        table_axis.set_title(f"指数指标（{span}）", loc="left", fontsize=11, pad=8)
        rows = []
        for key, label, kind in INDEX_STAT_ROWS:
            bench_text = "—" if key == "excess_return" else format_stat(benchmark.get(key), kind)
            rows.append([label, format_stat(strategy.get(key), kind), bench_text])
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

    def _draw_return_compare(self, record: dict, algorithm_id: str | None) -> None:
        """只画策略净值和买入持有，不放统计表。"""
        algo = (record.get("algorithms") or {}).get(algorithm_id or "")
        if not algo:
            self._draw_message("当前股票的 JSON 里没有这个算法。")
            return
        dates = to_dates(record["dates"])
        curve = self.figure.add_subplot(1, 1, 1)
        curve.plot(dates, rebase(algo["equity"]), color="#1f4b99", linewidth=1.6, label=algo["name"])
        curve.plot(dates, rebase(algo["benchmark_equity"]), color="#888888", linewidth=1.4, label="买入持有")
        curve.set_title(f"{record['symbol']} 收益与买入持有（{self._span_text()}）", pad=18)
        curve.set_ylabel("净值")
        curve.set_xlabel("日期")
        place_legend(curve, 2)
        curve.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        apply_date_axis(curve, dates, self._view())

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
        block = self._stored_overview(algorithm_id)
        if not block:
            self._draw_message(f"{year}年还没有保存全局统计。\n请重新运行：python -m algorithms.rsi")
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
        bench_label = "买入持有"
        named = self.dataset.get("benchmark") if self.dataset else None
        if isinstance(named, dict) and named.get("name"):
            bench_label = named["name"]
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
