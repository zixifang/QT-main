"""因子测试和策略回测。只调用 services，不读 CSV。"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from services.factor_lab_service import list_constituents, list_factors, run_factor_tests, run_strategy
from services.task_service import TaskService


def _pct(value) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.2f}%"


def _num(value) -> str:
    if value is None:
        return "—"
    return f"{float(value):.2f}"


class FactorLabPanel(ttk.Frame):
    def __init__(self, master, get_symbol) -> None:
        super().__init__(master)
        self.get_symbol = get_symbol
        self.tasks = TaskService()
        self.task_id = ""
        self.factors = list_factors()
        self.test_result = None
        self.backtest_result = None
        self._build()

    def on_show(self, symbol: str | None) -> None:
        self.focus_var.set(symbol or "未选择")

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        book = ttk.Notebook(self)
        book.grid(row=0, column=0, sticky="nsew")
        test = ttk.Frame(book, padding=8)
        trade = ttk.Frame(book, padding=8)
        book.add(test, text="因子测试")
        book.add(trade, text="策略回测")
        self._build_test(test)
        self._build_trade(trade)

    def _build_test(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)
        form = ttk.Frame(parent)
        form.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        self.focus_var = tk.StringVar(value="未选择")
        self.symbols_var = tk.StringVar(value="AAPL,MSFT,NVDA,AMZN,GOOGL,META,AVGO,COST,NFLX,TSLA")
        self.start_var = tk.StringVar(value="2018-01-01")
        self.end_var = tk.StringVar(value="2024-12-31")
        self.window_var = tk.StringVar(value="20")
        self.horizon_var = tk.StringVar(value="20")
        self.groups_var = tk.StringVar(value="5")
        self.winsor_var = tk.BooleanVar(value=True)
        self.zscore_var = tk.BooleanVar(value=True)
        self.industry_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="尚未运行")
        self.chart_var = tk.StringVar(value="IC 折线")
        self.which_var = tk.StringVar()
        ttk.Label(form, text="左侧焦点").grid(row=0, column=0, sticky="w")
        ttk.Label(form, textvariable=self.focus_var).grid(row=0, column=1, sticky="w")
        ttk.Button(form, text="加入左侧选中", command=self._add_symbol).grid(row=1, column=0, columnspan=2, sticky="ew", pady=4)
        self._entry(form, "股票池", self.symbols_var, 2)
        self._entry(form, "开始", self.start_var, 3)
        self._entry(form, "结束", self.end_var, 4)
        self._entry(form, "窗口 N", self.window_var, 5)
        self._entry(form, "未来收益天数", self.horizon_var, 6)
        self._entry(form, "分组数", self.groups_var, 7)
        ttk.Checkbutton(form, text="去极值", variable=self.winsor_var).grid(row=8, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(form, text="截面标准化", variable=self.zscore_var).grid(row=9, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(form, text="行业中性", variable=self.industry_var).grid(row=10, column=0, columnspan=2, sticky="w")
        ttk.Label(form, text="因子").grid(row=11, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.factor_list = tk.Listbox(form, selectmode="extended", height=8, exportselection=False, width=28)
        self.factor_list.grid(row=12, column=0, columnspan=2, sticky="ew")
        for item in self.factors:
            self.factor_list.insert("end", item["id"])
        self.factor_list.selection_set(0)
        if self.factor_list.size() > 5:
            self.factor_list.selection_set(5)
        ttk.Button(form, text="填入纳斯达克100", command=self._fill_universe).grid(row=13, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(form, text="运行因子测试", command=self._run_tests).grid(row=14, column=0, columnspan=2, sticky="ew")
        ttk.Label(form, textvariable=self.status_var, wraplength=220).grid(row=15, column=0, columnspan=2, sticky="w", pady=4)

        result = ttk.Frame(parent)
        result.grid(row=0, column=1, sticky="nsew")
        result.columnconfigure(0, weight=1)
        result.rowconfigure(2, weight=1)
        bar = ttk.Frame(result)
        bar.grid(row=0, column=0, sticky="ew")
        chart = ttk.Combobox(
            bar,
            textvariable=self.chart_var,
            state="readonly",
            width=12,
            values=("IC 折线", "IC 分布", "分组净值", "分组年化", "多空净值", "相关热力图"),
        )
        chart.pack(side="left")
        chart.bind("<<ComboboxSelected>>", lambda _event: self._draw_test())
        self.which = ttk.Combobox(bar, textvariable=self.which_var, state="readonly", width=16)
        self.which.pack(side="left", padx=6)
        self.which.bind("<<ComboboxSelected>>", lambda _event: self._draw_test())
        self.summary = tk.Text(result, height=6, wrap="word")
        self.summary.grid(row=1, column=0, sticky="ew", pady=4)
        self.summary.configure(state="disabled")
        self.test_figure = Figure(figsize=(7.2, 4.6), dpi=100)
        self.test_canvas = FigureCanvasTkAgg(self.test_figure, master=result)
        self.test_canvas.get_tk_widget().grid(row=2, column=0, sticky="nsew")

    def _build_trade(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)
        form = ttk.Frame(parent)
        form.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        self.trade_factor = tk.StringVar(value=self.factors[0]["id"] if self.factors else "")
        self.direction_var = tk.StringVar(value="自动")
        self.rebalance_var = tk.StringVar(value="每月")
        self.top_var = tk.StringVar(value="5")
        self.weight_var = tk.StringVar(value="等权")
        self.cap_var = tk.StringVar(value="0.10")
        self.commission_var = tk.StringVar(value="0.0003")
        self.slippage_var = tk.StringVar(value="0.0005")
        self.tax_var = tk.StringVar(value="0")
        self.lot_var = tk.StringVar(value="1")
        self.fill_var = tk.StringVar(value="收盘价")
        self.benchmark_var = tk.StringVar(value="纳斯达克指数")
        self.short_var = tk.BooleanVar(value=False)
        self.trade_status = tk.StringVar(value="尚未运行")
        self._entry(form, "持仓数量", self.top_var, 0)
        self._entry(form, "单票上限", self.cap_var, 1)
        self._entry(form, "佣金", self.commission_var, 2)
        self._entry(form, "滑点", self.slippage_var, 3)
        self._entry(form, "卖出税", self.tax_var, 4)
        self._entry(form, "最小股数", self.lot_var, 5)
        self._combo(form, "因子", self.trade_factor, 6, [item["id"] for item in self.factors])
        self._combo(form, "方向", self.direction_var, 7, ("自动", "做多高分", "做多低分"))
        self._combo(form, "调仓", self.rebalance_var, 8, ("每日", "每周", "每月"))
        self._combo(form, "权重", self.weight_var, 9, ("等权", "因子值加权", "市值加权"))
        self._combo(form, "成交价", self.fill_var, 10, ("收盘价", "次日开盘"))
        self._combo(form, "基准", self.benchmark_var, 11, ("纳斯达克指数", "股票池等权"))
        ttk.Checkbutton(form, text="同时持有空头组", variable=self.short_var).grid(row=12, column=0, columnspan=2, sticky="w")
        ttk.Button(form, text="运行回测", command=self._run_backtest).grid(row=13, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(form, text="导出成交 CSV", command=self._export).grid(row=14, column=0, columnspan=2, sticky="ew")
        ttk.Label(form, textvariable=self.trade_status, wraplength=220).grid(row=15, column=0, columnspan=2, sticky="w", pady=4)

        result = ttk.Frame(parent)
        result.grid(row=0, column=1, sticky="nsew")
        result.columnconfigure(0, weight=1)
        result.rowconfigure(1, weight=1)
        result.rowconfigure(2, weight=1)
        self.metric_var = tk.StringVar(value="运行后在这里显示绩效。")
        ttk.Label(result, textvariable=self.metric_var, justify="left", wraplength=640).grid(row=0, column=0, sticky="w")
        self.trade_figure = Figure(figsize=(7.2, 4.2), dpi=100)
        self.trade_canvas = FigureCanvasTkAgg(self.trade_figure, master=result)
        self.trade_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        columns = ("date", "ticker", "side", "shares", "price", "commission", "slippage", "cash_after")
        self.trades = ttk.Treeview(result, columns=columns, show="headings", height=6)
        headers = ("日期", "代码", "方向", "股数", "价格", "佣金", "滑点", "现金")
        for key, title, width in zip(columns, headers, (90, 70, 50, 70, 70, 70, 70, 90)):
            self.trades.heading(key, text=title)
            self.trades.column(key, width=width, anchor="e")
        self.trades.grid(row=2, column=0, sticky="nsew", pady=(4, 0))

    def _entry(self, form: ttk.Frame, label: str, variable: tk.StringVar, row: int) -> None:
        ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(form, textvariable=variable, width=22).grid(row=row, column=1, sticky="ew", pady=2)

    def _combo(self, form: ttk.Frame, label: str, variable: tk.StringVar, row: int, values) -> None:
        ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Combobox(form, textvariable=variable, values=values, state="readonly", width=20).grid(row=row, column=1, sticky="ew", pady=2)

    def _symbols(self) -> list[str]:
        raw = self.symbols_var.get().replace("，", ",").replace("\n", ",").replace(" ", ",")
        return [part.strip().upper() for part in raw.split(",") if part.strip()]

    def _add_symbol(self) -> None:
        symbol = self.get_symbol()
        if not symbol or symbol == "NDX":
            return
        current = self._symbols()
        if symbol not in current:
            current.append(symbol)
            self.symbols_var.set(",".join(current))

    def _fill_universe(self) -> None:
        names = list_constituents()
        if names:
            self.symbols_var.set(",".join(names))

    def _selected_factors(self) -> list[str]:
        return [self.factor_list.get(index) for index in self.factor_list.curselection()]

    def _base_config(self) -> dict:
        return {
            "symbols": self._symbols(),
            "start": self.start_var.get().strip(),
            "end": self.end_var.get().strip(),
            "N": int(self.window_var.get()),
            "horizon": int(self.horizon_var.get()),
            "winsorize": bool(self.winsor_var.get()),
            "standardize": bool(self.zscore_var.get()),
            "industry_neutral": bool(self.industry_var.get()),
        }

    def _run_tests(self) -> None:
        try:
            config = self._base_config()
            config["factors"] = self._selected_factors()
            config["groups"] = int(self.groups_var.get())
        except ValueError:
            messagebox.showerror("因子测试", "窗口、未来收益天数和分组数要填整数。")
            return
        self._start("因子测试", lambda progress, cancel: run_factor_tests(config, progress, cancel), self._show_tests, self.status_var)

    def _run_backtest(self) -> None:
        try:
            config = self._base_config()
            config.update(
                {
                    "factor": self.trade_factor.get(),
                    "direction": {"自动": "auto", "做多高分": 1, "做多低分": -1}[self.direction_var.get()],
                    "rebalance": {"每日": "D", "每周": "W", "每月": "M"}[self.rebalance_var.get()],
                    "top_n": int(self.top_var.get()),
                    "weight": {"等权": "equal", "因子值加权": "factor", "市值加权": "cap"}[self.weight_var.get()],
                    "max_weight": float(self.cap_var.get()),
                    "commission": float(self.commission_var.get()),
                    "slippage": float(self.slippage_var.get()),
                    "sell_tax": float(self.tax_var.get()),
                    "lot": int(self.lot_var.get()),
                    "fill": "next_open" if self.fill_var.get() == "次日开盘" else "close",
                    "benchmark": "equal" if self.benchmark_var.get() == "股票池等权" else "NDX",
                    "short": bool(self.short_var.get()),
                }
            )
        except ValueError:
            messagebox.showerror("策略回测", "持仓数量、成本和股数要填数字。")
            return
        self._start("策略回测", lambda progress, cancel: run_strategy(config, progress, cancel), self._show_backtest, self.trade_status)

    def _start(self, name: str, fn, done, status: tk.StringVar) -> None:
        if self.task_id:
            state = self.tasks.poll(self.task_id)
            if state["status"] == "running":
                status.set("已有任务在跑")
                return
        status.set(f"正在{name}")
        self.task_id = self.tasks.start(name, fn)
        self.after(200, lambda: self._poll(done, status))

    def _poll(self, done, status: tk.StringVar) -> None:
        state = self.tasks.poll(self.task_id)
        logs = state.get("logs") or []
        status.set(f"{state.get('progress', 0)}% {logs[-1] if logs else ''}")
        if state["status"] == "running":
            self.after(200, lambda: self._poll(done, status))
            return
        if state["status"] == "error":
            messagebox.showerror("因子测试", state.get("error") or "计算失败")
            return
        if state["status"] == "cancelled":
            status.set("已取消")
            return
        done(state.get("result") or {})

    def _show_tests(self, result: dict) -> None:
        self.test_result = result
        names = list((result.get("factors") or {}).keys())
        self.which["values"] = names
        if names:
            self.which_var.set(names[0])
        lines = list(result.get("warnings") or [])
        for name in names:
            report = result["factors"][name]
            pearson = report["ic"]["pearson"]
            spearman = report["ic"]["spearman"]
            groups = report["groups"]
            lines.append(
                f"{name}  IC {_pct(pearson['mean'])}  IR {_num(pearson['ir'])}  胜率 {_pct(pearson['win_rate'])}"
                f"  |  RankIC {_pct(spearman['mean'])}  IR {_num(spearman['ir'])}  胜率 {_pct(spearman['win_rate'])}"
            )
            lines.append(
                f"    分组 {groups.get('groups')}  换手 {_pct(groups.get('turnover'))}"
                f"  单调性 {_num(groups.get('monotonicity'))}  {'递增' if groups.get('increasing') else '未严格递增'}"
            )
        high = (result.get("correlation") or {}).get("high_pairs") or []
        if high:
            lines.append("高相关：" + "；".join(f"{item['left']}/{item['right']} {item['corr']:.2f}" for item in high))
        self._write(self.summary, "\n".join(lines) or "没有可展示的结果")
        self._draw_test()

    def _draw_test(self) -> None:
        result = self.test_result or {}
        self.test_figure.clear()
        kind = self.chart_var.get()
        if kind == "相关热力图":
            self._draw_heatmap(result.get("correlation") or {})
        else:
            report = (result.get("factors") or {}).get(self.which_var.get())
            if not report:
                self._blank(self.test_figure, "运行后在这里显示图表")
            elif kind == "IC 折线":
                self._draw_ic(report["ic"])
            elif kind == "IC 分布":
                self._draw_hist(report["ic"])
            elif kind == "分组净值":
                self._draw_groups(report["groups"])
            elif kind == "分组年化":
                self._draw_annual(report["groups"])
            else:
                self._draw_spread(report["groups"])
        self.test_figure.tight_layout()
        self.test_canvas.draw_idle()

    def _draw_ic(self, ic: dict) -> None:
        axis = self.test_figure.add_subplot(1, 1, 1)
        for key, label, color in (("pearson", "IC", "#1f4b99"), ("spearman", "RankIC", "#9b1c1c")):
            series = ic[key]["series"]
            if len(series):
                axis.plot(series.index, series.to_numpy(), label=label, color=color, linewidth=1.0)
        axis.axhline(0, color="#666666", linewidth=0.6)
        axis.legend(frameon=False)
        axis.set_title("IC 与 RankIC")
        axis.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    def _draw_hist(self, ic: dict) -> None:
        axis = self.test_figure.add_subplot(1, 1, 1)
        values = ic["spearman"]["series"].to_numpy()
        if len(values):
            axis.hist(values, bins=20, color="#1f4b99", alpha=0.85)
        axis.set_title("RankIC 分布")
        axis.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    def _draw_groups(self, groups: dict) -> None:
        axis = self.test_figure.add_subplot(1, 1, 1)
        dates = groups.get("dates") or []
        for group, path in sorted((groups.get("nav") or {}).items()):
            if len(path) == len(dates) and len(path):
                axis.plot(dates, path, label=f"第 {group + 1} 组", linewidth=1.1)
        axis.legend(frameon=False, ncol=2, fontsize=8)
        axis.set_title("分组累计净值")
        axis.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    def _draw_annual(self, groups: dict) -> None:
        axis = self.test_figure.add_subplot(1, 1, 1)
        annual = groups.get("annual") or []
        values = [0.0 if item is None else float(item) for item in annual]
        colors = ["#0b6a2b" if item >= 0 else "#9b1c1c" for item in values]
        axis.bar([str(index + 1) for index in range(len(annual))], values, color=colors)
        axis.set_title("分组年化收益")
        axis.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    def _draw_spread(self, groups: dict) -> None:
        axis = self.test_figure.add_subplot(1, 1, 1)
        dates = groups.get("dates") or []
        path = groups.get("long_short") or []
        if len(path) == len(dates) and len(path):
            axis.plot(dates, path, color="#1f4b99", linewidth=1.3)
        axis.set_title("高分组减低分组")
        axis.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    def _draw_heatmap(self, correlation: dict) -> None:
        matrix = correlation.get("pearson")
        if matrix is None or getattr(matrix, "empty", True):
            self._blank(self.test_figure, "至少两个因子才有相关矩阵")
            return
        axis = self.test_figure.add_subplot(1, 1, 1)
        image = axis.imshow(matrix.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1)
        names = list(correlation.get("names") or [])
        axis.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
        axis.set_yticks(range(len(names)), names, fontsize=8)
        self.test_figure.colorbar(image, ax=axis, fraction=0.046)
        axis.set_title("Pearson 相关")

    def _show_backtest(self, result: dict) -> None:
        self.backtest_result = result
        metrics = result.get("metrics") or {}
        text = "  ".join(
            [
                f"方向 {'高分' if result.get('direction') == 1 else '低分'}",
                f"累计 {_pct(metrics.get('cumulative_return'))}",
                f"年化 {_pct(metrics.get('annualized_return'))}",
                f"波动 {_pct(metrics.get('volatility'))}",
                f"夏普 {_num(metrics.get('sharpe'))}",
                f"回撤 {_pct(metrics.get('max_drawdown'))}",
                f"卡玛 {_num(metrics.get('calmar'))}",
                f"换手 {_num(metrics.get('turnover'))}",
                f"胜率 {_pct(metrics.get('win_rate'))}",
                f"盈亏比 {_num(metrics.get('payoff'))}",
                f"超额 {_pct(metrics.get('excess_return'))}",
            ]
        )
        notes = list(result.get("warnings") or [])
        self.metric_var.set(text + ("\n" + "\n".join(notes) if notes else ""))
        self._fill_trades(result.get("trades"))
        self._draw_backtest(result)

    def _fill_trades(self, trades) -> None:
        self.trades.delete(*self.trades.get_children())
        if trades is None or trades.empty:
            return
        for row in trades.tail(200).itertuples(index=False):
            self.trades.insert(
                "",
                "end",
                values=(
                    row.date,
                    row.ticker,
                    "买入" if row.side == "buy" else "卖出",
                    f"{row.shares:.0f}",
                    f"{row.price:.2f}",
                    f"{row.commission:.2f}",
                    f"{row.slippage:.2f}",
                    f"{row.cash_after:.0f}",
                ),
            )

    def _draw_backtest(self, result: dict) -> None:
        self.trade_figure.clear()
        nav = result.get("nav")
        if nav is None or len(nav) == 0:
            self._blank(self.trade_figure, "没有净值")
            self.trade_canvas.draw_idle()
            return
        curve = self.trade_figure.add_subplot(3, 1, 1)
        drawdown = self.trade_figure.add_subplot(3, 1, 2, sharex=curve)
        weights = self.trade_figure.add_subplot(3, 1, 3)
        base = float(nav.iloc[0]) or 1.0
        curve.plot(nav.index, nav.to_numpy() / base, color="#1f4b99", label="组合", linewidth=1.2)
        benchmark = result.get("benchmark")
        if benchmark is not None and len(benchmark.dropna()):
            aligned = benchmark.reindex(nav.index).dropna()
            if len(aligned):
                curve.plot(aligned.index, aligned.to_numpy() / float(aligned.iloc[0]), color="#888888", label="基准", linewidth=1.0)
        curve.legend(frameon=False)
        curve.set_title("净值")
        curve.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        ratio = nav / nav.cummax() - 1.0
        drawdown.fill_between(nav.index, ratio.to_numpy(), 0, color="#9b1c1c", alpha=0.35)
        drawdown.set_title("回撤")
        drawdown.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        self._draw_holdings(weights, result)
        self.trade_figure.tight_layout()
        self.trade_canvas.draw_idle()

    def _draw_holdings(self, axis, result: dict) -> None:
        positions = result.get("positions")
        close = result.get("close")
        nav = result.get("nav")
        if positions is None or close is None or nav is None or positions.empty or not float(nav.iloc[-1]):
            axis.axis("off")
            return
        day = positions.index[-1]
        prices = close.reindex(index=positions.index, columns=positions.columns).loc[day]
        values = (positions.loc[day] * prices).dropna()
        values = values[values.abs() > 0].sort_values()
        if values.empty:
            axis.axis("off")
            return
        weights = values / float(nav.iloc[-1])
        axis.barh(weights.index.astype(str), weights.to_numpy(), color="#1f4b99")
        axis.set_title("期末持仓权重")
        axis.grid(True, axis="x", linewidth=0.4, alpha=0.5)

    def _export(self) -> None:
        result = self.backtest_result or {}
        trades = result.get("trades")
        if trades is None or trades.empty:
            messagebox.showinfo("策略回测", "还没有成交记录。")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        trades.to_csv(path, index=False, encoding="utf-8-sig")
        self.trade_status.set(f"已导出 {path}")

    def _write(self, widget: tk.Text, content: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", content)
        widget.configure(state="disabled")

    def _blank(self, figure: Figure, text: str) -> None:
        axis = figure.add_subplot(1, 1, 1)
        axis.axis("off")
        axis.text(0.5, 0.5, text, ha="center", va="center")
