"""行情窗口只读取 outdata 中的 JSON，不在界面里计算指标。

运行前先生成结果：python -m algorithms.rsi
然后：python -m gui
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from gui.outdata import list_datasets, load_symbol

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PAGES = ("原始数据", "算法点位", "统计指标")
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


def to_dates(values: list[str]) -> np.ndarray:
    ordered = pd.to_datetime(values)
    return mdates.date2num(pd.DatetimeIndex(ordered).to_pydatetime())


def as_float(values: list) -> np.ndarray:
    return np.array([np.nan if value is None else float(value) for value in values], dtype=float)


def apply_date_axis(axis, dates: np.ndarray) -> None:
    axis.xaxis.set_major_locator(mdates.MonthLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    if len(dates):
        axis.set_xlim(float(dates[0]) - 3, float(dates[-1]) + 3)
    axis.tick_params(axis="x", labelrotation=30)
    for label in axis.get_xticklabels():
        label.set_ha("right")


def _board_fields(record: dict) -> tuple[float | None, float | None, str]:
    closes = [value for value in record.get("close") or [] if value is not None]
    end_price = float(closes[-1]) if closes else None
    change = None
    for algo in (record.get("algorithms") or {}).values():
        change = (algo.get("stats") or {}).get("benchmark", {}).get("cumulative_return")
        break
    tag = "up" if change is not None and change >= 0 else "down"
    return change, end_price, tag


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
        self.page = 0
        self.page_text = tk.StringVar(value=self._page_caption())
        self._build()
        self._load_dataset(0)
        self.redraw()

    def _page_caption(self) -> str:
        return f"{self.page + 1} / {len(PAGES)}  {PAGES[self.page]}"

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
        self.board.column("symbol", width=72, anchor="w")
        self.board.column("return", width=78, anchor="e")
        self.board.column("end", width=84, anchor="e")
        self.board.grid(row=3, column=0, sticky="nsew", pady=(4, 8))
        self.board.tag_configure("up", foreground="#0b6a2b")
        self.board.tag_configure("down", foreground="#9b1c1c")
        self.board.bind("<<TreeviewSelect>>", lambda _event: self._on_symbol())

        ttk.Label(side, text="算法").grid(row=4, column=0, sticky="w")
        self.algorithms = ttk.Treeview(side, columns=("name",), show="headings", height=6, selectmode="browse")
        self.algorithms.heading("name", text="名称")
        self.algorithms.column("name", width=220, anchor="w")
        self.algorithms.grid(row=5, column=0, sticky="nsew", pady=(4, 8))
        self.algorithms.bind("<<TreeviewSelect>>", lambda _event: self.redraw())
        self.algorithm_note = ttk.Label(side, wraplength=240)
        self.algorithm_note.grid(row=6, column=0, sticky="w")

        main = ttk.Frame(self, padding=(0, 12, 12, 12))
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)
        pager = ttk.Frame(main)
        pager.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.prev_button = ttk.Button(pager, text="上一页", command=self.prev_page)
        self.prev_button.pack(side="left")
        ttk.Label(pager, textvariable=self.page_text).pack(side="left", padx=12)
        self.next_button = ttk.Button(pager, text="下一页", command=self.next_page)
        self.next_button.pack(side="left")

        self.figure = Figure(figsize=(8.4, 6.2), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=main)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        toolbar = NavigationToolbar2Tk(self.canvas, main, pack_toolbar=False)
        toolbar.grid(row=2, column=0, sticky="ew")
        toolbar.update()

    def _on_dataset(self) -> None:
        index = self.dataset_picker.current()
        self._load_dataset(index)
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
        symbols = self.dataset["symbols"] if self.dataset else []
        for symbol in symbols:
            record = self._cached_symbol(symbol)
            change, end_price, tag = _board_fields(record)
            self.board.insert(
                "",
                "end",
                iid=symbol,
                values=(symbol, format_stat(change, "pct"), _format_price(end_price)),
                tags=(tag,),
            )
        if symbols:
            self.board.selection_set(selected if selected in symbols else symbols[0])

    def _cached_symbol(self, symbol: str) -> dict:
        key = (self.dataset["dataset_id"], symbol)
        if key not in self.symbol_cache:
            self.symbol_cache[key] = load_symbol(self.dataset, symbol)
        return self.symbol_cache[key]

    def _fill_algorithms(self) -> None:
        selected = self.selected_algorithm()
        self.algorithms.delete(*self.algorithms.get_children())
        items = self.dataset["algorithms"] if self.dataset else []
        for item in items:
            self.algorithms.insert("", "end", iid=item["id"], values=(item["name"],))
        if items:
            target = selected if any(item["id"] == selected for item in items) else items[0]["id"]
            self.algorithms.selection_set(target)

    def _on_symbol(self) -> None:
        if self._suspend:
            return
        self.redraw()

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
        if self.page >= len(PAGES) - 1:
            return
        self.page += 1
        self.redraw()

    def redraw(self) -> None:
        if self._suspend:
            return
        self.page_text.set(self._page_caption())
        self.prev_button.state(["!disabled"] if self.page else ["disabled"])
        self.next_button.state(["!disabled"] if self.page < len(PAGES) - 1 else ["disabled"])
        self.figure.clear()
        if not self.datasets:
            self._draw_message("outdata 里还没有数据集。\n请先运行：python -m algorithms.rsi")
        else:
            record = self.current_record()
            algorithm_id = self.selected_algorithm()
            note = ""
            if self.dataset and algorithm_id:
                match = [item for item in self.dataset["algorithms"] if item["id"] == algorithm_id]
                note = match[0]["note"] if match else ""
            self.algorithm_note.configure(text=note)
            if record is None:
                self._draw_message("请选择数据集和股票。")
            elif self.page == 0:
                self._draw_raw(record)
            elif self.page == 1:
                self._draw_signals(record, algorithm_id)
            else:
                self._draw_stats(record, algorithm_id)
        self.figure.tight_layout()
        self.canvas.draw_idle()

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
        price_axis.set_title(f"{record['symbol']} 原始价格")
        price_axis.set_ylabel("价格")
        price_axis.legend(frameon=False, loc="upper left")
        price_axis.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        price_axis.tick_params(labelbottom=False)
        volume_axis.bar(dates, volume, width=0.8, color="#8aa0b8", align="center")
        volume_axis.set_title(f"{record['symbol']} 成交量")
        volume_axis.set_ylabel("股")
        volume_axis.set_xlabel("日期")
        volume_axis.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        apply_date_axis(volume_axis, dates)

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
        price_axis.set_title(f"{record['symbol']} 价格与{algo['name']}买卖点")
        price_axis.set_ylabel("价格")
        price_axis.legend(frameon=False, loc="upper left")
        price_axis.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        price_axis.tick_params(labelbottom=False)

        indicator_axis.plot(dates, as_float(algo["indicator"]), color="#1f4b99", linewidth=1.2, label=algo["id"].upper())
        params = algo.get("params") or {}
        if "oversold" in params:
            indicator_axis.axhline(params["oversold"], color="#0b6a2b", linewidth=0.8, linestyle="--", label=f"超卖 {params['oversold']:.0f}")
        if "overbought" in params:
            indicator_axis.axhline(params["overbought"], color="#9b1c1c", linewidth=0.8, linestyle="--", label=f"超买 {params['overbought']:.0f}")
        indicator_axis.set_title(f"{record['symbol']} {algo['id'].upper()}")
        indicator_axis.set_ylabel(algo["id"].upper())
        indicator_axis.set_xlabel("日期")
        indicator_axis.legend(frameon=False, loc="upper left", ncol=3)
        indicator_axis.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        apply_date_axis(indicator_axis, dates)

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
        curve.plot(dates, as_float(algo["equity"]), color="#1f4b99", linewidth=1.4, label=algo["name"])
        curve.plot(dates, as_float(algo["benchmark_equity"]), color="#888888", linewidth=1.2, label="买入持有")
        curve.set_title(f"{record['symbol']} 净值（起点 = 1）")
        curve.set_ylabel("净值")
        curve.set_xlabel("日期")
        curve.legend(frameon=False, loc="upper left")
        curve.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        apply_date_axis(curve, dates)

        table_axis = self.figure.add_subplot(2, 1, 2)
        table_axis.axis("off")
        fee = (algo.get("params") or {}).get("fee")
        fee_text = f"，策略手续费 {fee:.1%}" if isinstance(fee, (int, float)) else ""
        table_axis.set_title(f"统计指标{fee_text}", loc="left", fontsize=11, pad=8)
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


def main() -> None:
    app = MarketApp()
    app.mainloop()


if __name__ == "__main__":
    main()
