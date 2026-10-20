"""指数增强页面。只调用 services，不读 data 里的 CSV，也不算因子。"""

from __future__ import annotations

import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from factors.catalog import catalog
from services.config_store import ConfigStore, validate_config
from services.data_service import DataService
from services.errors import ServiceError
from services.eval_service import EvalService
from services.factor_service import FactorService
from services.composite_service import CompositeService
from services.backtest_service import BacktestService
from services.report_service import ReportService
from services.store import Store
from services.task_service import TaskService

SOURCE_LABELS = {
    "本地未复权 CSV": "local_csv",
    "Yahoo（关闭）": "yahoo",
    "Nasdaq Data Link": "nasdaq_data_link",
}
SCHEME_LABELS = {"等权": "equal", "IC 加权": "ic", "最大 ICIR": "max_icir"}
REBALANCE_LABELS = {"每月": "M", "每季": "Q"}
ADJUST_LABELS = {"不复权": "raw", "后复权": "hfq", "前复权": "qfq"}
POOL_LABELS = {"手填名单": "manual", "历史快照": "snapshot"}
DIRECTION = {1: "越高越好", -1: "越低越好", 0: "不作为选股方向"}


def _label_of(mapping: dict[str, str], value: str) -> str:
    for label, stored in mapping.items():
        if stored == value:
            return label
    return next(iter(mapping))


def _symbols(text: str) -> list[str]:
    raw = text.replace("，", ",").replace("\n", ",").replace(" ", ",")
    return [part.strip().upper() for part in raw.split(",") if part.strip()]


def _write(widget: tk.Text, content: str) -> None:
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget.insert("end", content)
    widget.configure(state="disabled")


def _pct(value) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.2f}%"


def _num(value) -> str:
    if value is None:
        return "—"
    return f"{float(value):.2f}"


def describe(item: dict) -> str:
    fields = "、".join(item["fields"]) or "无"
    options = "、".join(item["configurable"]) or "无"
    return "\n".join(
        [
            item["name"],
            item["formula"],
            f"字段：{fields}",
            f"窗口：{item['window']}",
            f"方向：{DIRECTION[item['direction']]}",
            f"角色：{item['role']}",
            f"可配置：{options}",
            "有数据，可以计算。" if item["implemented"] else "当前没有数据，勾选后结果为空，并在日志里告警。",
        ]
    )


class EnhancePanel(ttk.Frame):
    """六个步骤。计算都放在后台线程，窗口只收进度和结果。"""

    def __init__(self, master, get_symbol) -> None:
        super().__init__(master)
        self.get_symbol = get_symbol
        self.tasks = TaskService()
        self.config = ConfigStore().load()
        self.task_id = ""
        self._log_from = 0
        self._poll_gen = 0
        self.factor_vars: dict[str, tk.BooleanVar] = {}
        self.risk_vars: dict[str, tk.BooleanVar] = {}
        self._vars()
        self._build()
        self.write_form(self.config)

    def on_show(self, symbol: str | None) -> None:
        self.focus_var.set(symbol or "未选择")

    def _vars(self) -> None:
        self.focus_var = tk.StringVar()
        self.source_var = tk.StringVar()
        self.api_key_var = tk.StringVar()
        self.universe_var = tk.StringVar()
        self.start_var = tk.StringVar()
        self.end_var = tk.StringVar()
        self.pool_var = tk.StringVar()
        self.lookback_var = tk.StringVar()
        self.skip_var = tk.StringVar()
        self.vol_var = tk.StringVar()
        self.explain_var = tk.StringVar()
        self.factor_choice = tk.StringVar()
        self.horizon_var = tk.StringVar()
        self.scheme_var = tk.StringVar()
        self.rebalance_var = tk.StringVar()
        self.cost_var = tk.StringVar()
        self.slip_var = tk.StringVar()
        self.cap_var = tk.StringVar()
        self.penalty_var = tk.StringVar()
        self.adjust_var = tk.StringVar()
        self.jump_var = tk.StringVar()
        self.train_var = tk.StringVar()
        self.valid_var = tk.StringVar()
        self.beta_var = tk.StringVar()
        self.liq_var = tk.StringVar()
        self.benchmark_var = tk.StringVar()
        self.alert_var = tk.StringVar(value="左侧代码只用来加入股票池。行情始终读 data/nasdaq100 的未复权文件。")

    def _build(self) -> None:
        ttk.Label(self, textvariable=self.alert_var, wraplength=860, foreground="#8a4b08").pack(fill="x", pady=(0, 6))
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)
        self.notebook.add(self._data_page(), text="数据管理")
        self.notebook.add(self._factor_page(), text="因子配置")
        self.notebook.add(self._eval_page(), text="单因子检验")
        self.notebook.add(self._composite_page(), text="多因子合成")
        self.notebook.add(self._backtest_page(), text="回测")
        self.notebook.add(self._report_page(), text="报告")
        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=(6, 0))
        ttk.Button(bar, text="高级选项", command=self._toggle_advanced).pack(side="left")
        self.advanced = ttk.LabelFrame(self, text="高级选项", padding=6)
        self._advanced_fields(self.advanced)
        self.advanced.pack(fill="x", pady=(4, 0))
        self.advanced.pack_forget()
        self._advanced_open = False
        status = ttk.Frame(self)
        status.pack(fill="x", pady=(6, 0))
        self.progress_bar = ttk.Progressbar(status, maximum=100)
        self.progress_bar.pack(side="left", fill="x", expand=True)
        ttk.Button(status, text="取消", command=self.cancel_task).pack(side="left", padx=(8, 0))
        self.log = ScrolledText(self, height=6, font=("Microsoft YaHei UI", 9))
        self.log.pack(fill="x", pady=(6, 0))

    def _data_page(self) -> ttk.Frame:
        page = ttk.Frame(self.notebook, padding=8)
        form = ttk.Frame(page)
        form.pack(fill="x")
        self._entry(form, "主数据源", self.source_var, 0, values=tuple(SOURCE_LABELS), width=18)
        ttk.Label(form, text="API Key").grid(row=0, column=2, sticky="w", padx=(8, 4))
        ttk.Entry(form, textvariable=self.api_key_var, show="*", width=24).grid(row=0, column=3, sticky="w")
        self._entry(form, "股票池来源", self.pool_var, 1, values=tuple(POOL_LABELS), width=18)
        ttk.Label(form, text="左侧选中").grid(row=1, column=2, sticky="w", padx=(8, 4))
        ttk.Label(form, textvariable=self.focus_var).grid(row=1, column=3, sticky="w")
        ttk.Label(form, text="股票池").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(form, textvariable=self.universe_var, width=42).grid(row=2, column=1, columnspan=2, sticky="we", pady=(6, 0))
        ttk.Button(form, text="加入左侧选中", command=self.add_focus).grid(row=2, column=3, sticky="w", padx=(8, 0), pady=(6, 0))
        self._entry(form, "开始", self.start_var, 3, width=16)
        self._entry(form, "结束", self.end_var, 3, column=2, width=16)
        actions = ttk.Frame(page)
        actions.pack(fill="x", pady=(8, 4))
        ttk.Button(actions, text="载入并查看覆盖", command=self.run_coverage).pack(side="left")
        ttk.Button(actions, text="保存配置", command=self.save_config).pack(side="left", padx=8)
        ttk.Button(actions, text="清除缓存", command=self.clear_cache).pack(side="left")
        columns = ("symbol", "rows", "first", "last", "jumps", "source", "warning")
        self.coverage_tree = ttk.Treeview(page, columns=columns, show="headings", height=8)
        headers = ("代码", "行数", "起始", "结束", "跳空", "来源", "说明")
        widths = (70, 60, 100, 100, 50, 90, 320)
        for column, header, width in zip(columns, headers, widths):
            self.coverage_tree.heading(column, text=header)
            self.coverage_tree.column(column, width=width, anchor="w")
        self.coverage_tree.pack(fill="both", expand=True)
        return page

    def _factor_page(self) -> ttk.Frame:
        page = ttk.Frame(self.notebook, padding=8)
        checks = ttk.Frame(page)
        checks.pack(fill="x")
        alphas = [item for item in catalog() if item["role"] != "risk"]
        for index, item in enumerate(alphas):
            var = tk.BooleanVar()
            self.factor_vars[item["id"]] = var
            text = item["name"] if item["implemented"] else f"{item['name']}（无数据）"
            ttk.Checkbutton(checks, text=text, variable=var).grid(row=index // 2, column=index % 2, sticky="w", padx=(0, 16))
        ttk.Label(page, text="中性化（样本太少时会跳过并告警）").pack(anchor="w", pady=(8, 0))
        risks = ttk.Frame(page)
        risks.pack(fill="x")
        for index, item in enumerate(item for item in catalog() if item["role"] == "risk"):
            var = tk.BooleanVar()
            self.risk_vars[item["id"]] = var
            text = item["name"] if item["implemented"] else f"{item['name']}（无数据）"
            ttk.Checkbutton(risks, text=text, variable=var).grid(row=0, column=index, sticky="w", padx=(0, 12))
        windows = ttk.Frame(page)
        windows.pack(fill="x", pady=(8, 4))
        self._entry(windows, "动量回看", self.lookback_var, 0, width=8)
        self._entry(windows, "动量跳过", self.skip_var, 0, column=2, width=8)
        self._entry(windows, "波动回看", self.vol_var, 0, column=4, width=8)
        names = [item["name"] for item in catalog()]
        explain = ttk.Frame(page)
        explain.pack(fill="x", pady=(6, 0))
        ttk.Label(explain, text="公式说明").pack(side="left")
        self.explain_box = ttk.Combobox(explain, textvariable=self.explain_var, state="readonly", width=18, values=tuple(names))
        self.explain_box.pack(side="left", padx=6)
        self.explain_box.bind("<<ComboboxSelected>>", lambda _event: self.show_formula())
        self.formula = tk.Text(page, height=8, wrap="word", font=("Microsoft YaHei UI", 9))
        self.formula.pack(fill="both", expand=True, pady=(6, 0))
        ttk.Button(page, text="按当前配置计算已选因子", command=self.run_factors).pack(anchor="w", pady=(6, 0))
        self.factor_text = tk.Text(page, height=5, wrap="word", font=("Microsoft YaHei UI", 9))
        self.factor_text.pack(fill="x", pady=(6, 0))
        return page

    def _eval_page(self) -> ttk.Frame:
        page = ttk.Frame(self.notebook, padding=8)
        names = [item["name"] for item in catalog()]
        row = ttk.Frame(page)
        row.pack(fill="x")
        ttk.Label(row, text="因子").pack(side="left")
        self.eval_box = ttk.Combobox(row, textvariable=self.factor_choice, state="readonly", width=18, values=tuple(names))
        self.eval_box.pack(side="left", padx=6)
        ttk.Label(row, text="预测期（交易日）").pack(side="left")
        ttk.Entry(row, textvariable=self.horizon_var, width=6).pack(side="left", padx=6)
        ttk.Button(row, text="运行检验", command=self.run_eval).pack(side="left")
        self.eval_text = tk.Text(page, height=5, wrap="word", font=("Microsoft YaHei UI", 9))
        self.eval_text.pack(fill="x", pady=(6, 0))
        self.eval_fig = Figure(figsize=(7.2, 3.2), dpi=100)
        self.eval_canvas = FigureCanvasTkAgg(self.eval_fig, master=page)
        self.eval_canvas.get_tk_widget().pack(fill="both", expand=True)
        return page

    def _composite_page(self) -> ttk.Frame:
        page = ttk.Frame(self.notebook, padding=8)
        row = ttk.Frame(page)
        row.pack(fill="x")
        ttk.Label(row, text="权重方案").pack(side="left")
        ttk.Combobox(row, textvariable=self.scheme_var, state="readonly", width=12, values=tuple(SCHEME_LABELS)).pack(side="left", padx=6)
        ttk.Button(row, text="合成", command=self.run_composite).pack(side="left")
        self.composite_text = tk.Text(page, height=14, wrap="word", font=("Microsoft YaHei UI", 9))
        self.composite_text.pack(fill="both", expand=True, pady=(6, 0))
        return page

    def _backtest_page(self) -> ttk.Frame:
        page = ttk.Frame(self.notebook, padding=8)
        row = ttk.Frame(page)
        row.pack(fill="x")
        ttk.Label(row, text="再平衡").pack(side="left")
        ttk.Combobox(row, textvariable=self.rebalance_var, state="readonly", width=8, values=tuple(REBALANCE_LABELS)).pack(side="left", padx=4)
        ttk.Label(row, text="成本bps").pack(side="left")
        ttk.Entry(row, textvariable=self.cost_var, width=6).pack(side="left", padx=4)
        ttk.Label(row, text="滑点bps").pack(side="left")
        ttk.Entry(row, textvariable=self.slip_var, width=6).pack(side="left", padx=4)
        ttk.Label(row, text="单名上限").pack(side="left")
        ttk.Entry(row, textvariable=self.cap_var, width=6).pack(side="left", padx=4)
        ttk.Label(row, text="分散惩罚").pack(side="left")
        ttk.Entry(row, textvariable=self.penalty_var, width=6).pack(side="left", padx=4)
        ttk.Button(row, text="运行回测", command=self.run_backtest).pack(side="left", padx=8)
        self.backtest_text = tk.Text(page, height=8, wrap="word", font=("Microsoft YaHei UI", 9))
        self.backtest_text.pack(fill="x", pady=(6, 0))
        self.backtest_fig = Figure(figsize=(7.2, 3.2), dpi=100)
        self.backtest_canvas = FigureCanvasTkAgg(self.backtest_fig, master=page)
        self.backtest_canvas.get_tk_widget().pack(fill="both", expand=True)
        return page

    def _report_page(self) -> ttk.Frame:
        page = ttk.Frame(self.notebook, padding=8)
        row = ttk.Frame(page)
        row.pack(fill="x")
        ttk.Button(row, text="导出 HTML", command=lambda: self.run_report("html")).pack(side="left")
        ttk.Button(row, text="导出 PDF", command=lambda: self.run_report("pdf")).pack(side="left", padx=8)
        self.report_text = tk.Text(page, height=16, wrap="word", font=("Microsoft YaHei UI", 9))
        self.report_text.pack(fill="both", expand=True, pady=(6, 0))
        return page

    def _advanced_fields(self, frame: ttk.LabelFrame) -> None:
        self._entry(frame, "复权", self.adjust_var, 0, values=tuple(ADJUST_LABELS), width=10)
        self._entry(frame, "跳空阈值", self.jump_var, 0, column=2, width=8)
        self._entry(frame, "训练年数", self.train_var, 0, column=4, width=6)
        self._entry(frame, "验证年数", self.valid_var, 0, column=6, width=6)
        self._entry(frame, "贝塔窗口", self.beta_var, 1, width=8)
        self._entry(frame, "流动性窗口", self.liq_var, 1, column=2, width=8)
        self._entry(frame, "基准", self.benchmark_var, 1, column=4, width=8)

    def _entry(self, parent, label: str, variable: tk.StringVar, row: int, column: int = 0, values=None, width: int = 14):
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 4), pady=2)
        if values is None:
            widget = ttk.Entry(parent, textvariable=variable, width=width)
        else:
            widget = ttk.Combobox(parent, textvariable=variable, state="readonly", width=width, values=values)
        widget.grid(row=row, column=column + 1, sticky="w", pady=2)
        return widget

    def _toggle_advanced(self) -> None:
        if self._advanced_open:
            self.advanced.pack_forget()
            self._advanced_open = False
            return
        self.advanced.pack(fill="x", pady=(4, 0), before=self.progress_bar.master)
        self._advanced_open = True

    def write_form(self, config: dict) -> None:
        self.source_var.set(_label_of(SOURCE_LABELS, config["source_order"][0]))
        self.api_key_var.set(str((config.get("api_keys") or {}).get("nasdaq_data_link", "")))
        self.pool_var.set(_label_of(POOL_LABELS, config.get("universe_mode", "manual")))
        self.universe_var.set(", ".join(config["universe"]))
        self.start_var.set(config["start"])
        self.end_var.set(config["end"])
        mom = config["factors"]["residual_momentum"]
        vol = config["factors"]["low_volatility"]
        self.lookback_var.set(str(mom.get("lookback", 252)))
        self.skip_var.set(str(mom.get("skip", 21)))
        self.vol_var.set(str(vol.get("lookback", 60)))
        for item in catalog():
            if item["role"] == "risk":
                self.risk_vars[item["id"]].set(item["id"] in config.get("neutralize", []))
            else:
                enabled = bool((config.get("factors") or {}).get(item["id"], {}).get("enabled", item["defaults"].get("enabled", False)))
                self.factor_vars[item["id"]].set(enabled)
        self.explain_var.set("残差动量")
        self.factor_choice.set("残差动量")
        self.show_formula()
        self.horizon_var.set(str(config["horizon"]))
        self.scheme_var.set(_label_of(SCHEME_LABELS, config["weight_scheme"]))
        self.rebalance_var.set(_label_of(REBALANCE_LABELS, config["rebalance"]))
        self.cost_var.set(str(config["cost_bps"]))
        self.slip_var.set(str(config["slippage_bps"]))
        self.cap_var.set(str(config["max_weight"]))
        self.penalty_var.set(str(config["diversify_penalty"]))
        self.adjust_var.set(_label_of(ADJUST_LABELS, config["adjust"]))
        self.jump_var.set(str(config["jump_limit"]))
        self.train_var.set(str(config["train_years"]))
        self.valid_var.set(str(config["valid_years"]))
        self.beta_var.set(str(config["beta_lookback"]))
        self.liq_var.set(str(config["liquidity_lookback"]))
        self.benchmark_var.set(config["benchmark"])

    def read_form(self) -> dict:
        config = ConfigStore().load()
        primary = SOURCE_LABELS[self.source_var.get()]
        config["source_order"] = [primary] + [item for item in ("local_csv", "yahoo", "nasdaq_data_link") if item != primary]
        config["api_keys"] = {"nasdaq_data_link": self.api_key_var.get().strip()}
        config["universe_mode"] = POOL_LABELS[self.pool_var.get()]
        config["universe"] = _symbols(self.universe_var.get())
        config["start"] = self.start_var.get().strip()
        config["end"] = self.end_var.get().strip()
        config["horizon"] = self._int(self.horizon_var, "预测期")
        config["weight_scheme"] = SCHEME_LABELS[self.scheme_var.get()]
        config["rebalance"] = REBALANCE_LABELS[self.rebalance_var.get()]
        config["cost_bps"] = self._float(self.cost_var, "成本")
        config["slippage_bps"] = self._float(self.slip_var, "滑点")
        config["max_weight"] = self._float(self.cap_var, "单名上限")
        config["diversify_penalty"] = self._float(self.penalty_var, "分散惩罚")
        config["adjust"] = ADJUST_LABELS[self.adjust_var.get()]
        config["jump_limit"] = self._float(self.jump_var, "跳空阈值")
        config["train_years"] = self._int(self.train_var, "训练年数")
        config["valid_years"] = self._int(self.valid_var, "验证年数")
        config["beta_lookback"] = self._int(self.beta_var, "贝塔窗口")
        config["liquidity_lookback"] = self._int(self.liq_var, "流动性窗口")
        config["benchmark"] = self.benchmark_var.get().strip().upper() or "NDX"
        config["neutralize"] = [key for key, var in self.risk_vars.items() if var.get()]
        config["factors"] = dict(config.get("factors") or {})
        for item in catalog():
            if item["role"] == "risk":
                continue
            params = dict(config["factors"].get(item["id"], {}))
            params["enabled"] = bool(self.factor_vars[item["id"]].get())
            if item["id"] == "residual_momentum":
                params["lookback"] = self._int(self.lookback_var, "动量回看")
                params["skip"] = self._int(self.skip_var, "动量跳过")
            if item["id"] == "low_volatility":
                params["lookback"] = self._int(self.vol_var, "波动回看")
            config["factors"][item["id"]] = params
        return validate_config(config)

    def show_formula(self) -> None:
        name = self.explain_var.get()
        for item in catalog():
            if item["name"] == name:
                _write(self.formula, describe(item))
                return

    def add_focus(self) -> None:
        symbol = self.get_symbol()
        if not symbol:
            return
        current = _symbols(self.universe_var.get())
        if symbol.upper() not in current:
            current.append(symbol.upper())
        self.universe_var.set(", ".join(current))

    def save_config(self) -> None:
        try:
            config = self.read_form()
        except ServiceError as exc:
            messagebox.showerror("指数增强", str(exc))
            return
        ConfigStore().save(config)
        self.config = config
        self._append_log("配置已写入 config/ndx_enhance.json")

    def clear_cache(self) -> None:
        Store().clear()
        self._append_log("已清除 cache/ndx_enhance")

    def run_coverage(self) -> None:
        self._start("覆盖度", lambda cfg, progress, cancel: DataService().coverage(cfg, progress=progress, cancel=cancel), self._show_coverage)

    def run_factors(self) -> None:
        def work(cfg, progress, cancel):
            service = FactorService()
            rows = []
            for item in catalog():
                if item["role"] == "risk" or not (cfg.get("factors") or {}).get(item["id"], {}).get("enabled"):
                    continue
                rows.append(service.compute(cfg, item["id"], progress=progress, cancel=cancel))
            if not rows:
                from services.errors import DataGapError

                raise DataGapError("没有勾选可计算的因子")
            return rows

        self._start("因子", work, self._show_factors)

    def run_eval(self) -> None:
        def work(cfg, progress, cancel):
            factor_id = next(item["id"] for item in catalog() if item["name"] == self.factor_choice.get())
            return EvalService().single(cfg, factor_id, progress=progress, cancel=cancel)

        self._start("单因子检验", work, self._show_eval)

    def run_composite(self) -> None:
        self._start("合成", lambda cfg, progress, cancel: CompositeService().combine(cfg, progress=progress, cancel=cancel), self._show_composite)

    def run_backtest(self) -> None:
        self._start("回测", lambda cfg, progress, cancel: BacktestService().run(cfg, progress=progress, cancel=cancel), self._show_backtest)

    def run_report(self, kind: str) -> None:
        self._start("报告", lambda cfg, progress, cancel: ReportService().export(cfg, kind, progress=progress, cancel=cancel), self._show_report)

    def cancel_task(self) -> None:
        if self.task_id:
            self.tasks.cancel(self.task_id)

    def _start(self, name: str, fn, on_done) -> None:
        if self.task_id:
            state = self.tasks.poll(self.task_id)
            if state["status"] == "running":
                self._append_log("已有任务在跑，可先取消")
                return
        try:
            config = self.read_form()
        except ServiceError as exc:
            messagebox.showerror("指数增强", str(exc))
            return
        ConfigStore().save(config)
        self.config = config
        self._append_log(f"—— {name} ——")
        self._log_from = 0
        self._poll_gen += 1
        generation = self._poll_gen

        def wrapped(progress, cancel):
            return fn(config, progress, cancel)

        self.task_id = self.tasks.start(name, wrapped)
        self.after(150, lambda: self._poll(generation, on_done))

    def _poll(self, generation: int, on_done) -> None:
        if generation != self._poll_gen:
            return
        state = self.tasks.poll(self.task_id)
        self.progress_bar["value"] = state["progress"]
        for line in state["logs"][self._log_from :]:
            self._append_log(line)
        self._log_from = len(state["logs"])
        if state["status"] == "running":
            self.after(150, lambda: self._poll(generation, on_done))
            return
        if state["status"] == "done":
            self.progress_bar["value"] = 100
            on_done(state["result"])
            return
        if state["status"] == "error":
            self.progress_bar["value"] = 0
            messagebox.showerror("指数增强", (state["error"] or "失败").strip().splitlines()[-1])

    def _show_coverage(self, result: dict) -> None:
        self.coverage_tree.delete(*self.coverage_tree.get_children())
        for row in result["rows"]:
            self.coverage_tree.insert(
                "",
                "end",
                values=(row["symbol"], row["rows"], row["first"], row["last"], row["jumps"], row["source"], row["warning"]),
            )
        self.alert_var.set(" ".join(result["warnings"]))

    def _show_factors(self, rows: list[dict]) -> None:
        lines = []
        for row in rows:
            latest = "、".join(
                f"{item['symbol']} {item['score']:.3f}" if item["score"] is not None else item["symbol"]
                for item in row["latest"]
            )
            lines.append(f"{row['name']}  缓存 {row['cache']}  {latest or '空'}")
            lines.extend(row["warnings"])
        _write(self.factor_text, "\n".join(lines))

    def _show_eval(self, result: dict) -> None:
        text = "\n".join(
            [
                f"IC 均值 {_num(result['ic_mean'])}    ICIR {_num(result['icir'])}    t {_num(result['ic_t'])}",
                f"月末样本 {result['n_ic']}    分组 {result['n_buckets']}    换手 {_pct(result['turnover'])}",
                *result["warnings"],
                "IC 用了未来收益，只出现在这一页，不会进入组合权重。",
            ]
        )
        _write(self.eval_text, text)
        self.eval_fig.clear()
        left = self.eval_fig.add_subplot(211)
        right = self.eval_fig.add_subplot(212)
        if result["ic"]:
            left.bar([item["date"][2:7] for item in result["ic"]], [item["ic"] for item in result["ic"]])
            left.set_title("月末 IC")
            left.tick_params(axis="x", labelrotation=60, labelsize=7)
        if result["spread"]:
            right.plot([item["date"] for item in result["spread"]], [item["high"] for item in result["spread"]], label="高分组")
            right.plot([item["date"] for item in result["spread"]], [item["low"] for item in result["spread"]], label="低分组")
            right.legend(fontsize=8)
            right.set_title("分层净值")
        self.eval_fig.tight_layout()
        self.eval_canvas.draw_idle()

    def _show_composite(self, result: dict) -> None:
        weights = "、".join(f"{name} {value:.2f}" for name, value in result["last_weights"].items() if value is not None)
        corr = "、".join(f"{item['left']}/{item['right']} {item['corr']:.2f}" if item["corr"] is not None else f"{item['left']}/{item['right']} —" for item in result["correlation"])
        preview = "、".join(
            f"{item['symbol']} {item['score']:.3f}" if item["score"] is not None else item["symbol"] for item in result["score_preview"]
        )
        _write(self.composite_text, "\n".join([f"方案 {result['scheme']}", f"最新权重 {weights}", f"相关性 {corr or '—'}", f"最新得分 {preview}", *result["warnings"]]))

    def _show_backtest(self, result: dict) -> None:
        lines = list(result["warnings"])
        lines.append(f"训练截至 {result['splits']['train_end']}，验证从 {result['splits']['valid_start']} 开始")
        for label, key in (("全样本", "full"), ("训练", "train"), ("验证", "valid")):
            stats = result[key]
            lines.append(
                f"{label} 累计 {_pct(stats['cumulative'])} 夏普 {_num(stats['sharpe'])} 回撤 {_pct(stats['max_drawdown'])} 跟踪误差 {_pct(stats['tracking_error'])}"
            )
        lines.append("分年")
        for row in result["yearly"]:
            lines.append(f"{row['year']} 策略 {_pct(row['strategy'])} 基准 {_pct(row['benchmark'])}")
        _write(self.backtest_text, "\n".join(lines))
        self._plot_nav(result["nav"])

    def _show_report(self, result: dict) -> None:
        path = result.get("html") or result.get("pdf") or "没有写出文件"
        _write(self.report_text, path + "\n\n" + result.get("summary", ""))

    def _plot_nav(self, nav: list[dict]) -> None:
        self.backtest_fig.clear()
        if not nav:
            self.backtest_canvas.draw_idle()
            return
        dates = [datetime.fromisoformat(item["date"]) for item in nav]
        top = self.backtest_fig.add_subplot(211)
        bottom = self.backtest_fig.add_subplot(212)
        top.plot(dates, [item["strategy"] for item in nav], label="组合")
        top.plot(dates, [item["benchmark"] for item in nav], label="NDX")
        top.legend(fontsize=8)
        top.set_title("净值")
        bottom.plot(dates, [item["strategy_drawdown"] for item in nav], label="组合回撤")
        bottom.plot(dates, [item["benchmark_drawdown"] for item in nav], label="NDX 回撤")
        bottom.legend(fontsize=8)
        self.backtest_fig.tight_layout()
        self.backtest_canvas.draw_idle()

    def _append_log(self, line: str) -> None:
        self.log.insert("end", line + "\n")
        self.log.see("end")

    def _float(self, variable: tk.StringVar, name: str) -> float:
        try:
            return float(variable.get())
        except ValueError as exc:
            from services.errors import DataGapError

            raise DataGapError(f"{name} 不是数字") from exc

    def _int(self, variable: tk.StringVar, name: str) -> int:
        return int(self._float(variable, name))
