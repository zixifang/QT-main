"""组合回测。跟踪误差只作为结果，不作为硬约束。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest.portfolio import performance, simulate, yearly_table
from preprocess.prepare import period_ends
from services.composite_service import CompositeService
from services.config_store import validate_config
from services.data_service import MANUAL_UNIVERSE_WARNING, DataService
from services.errors import DataGapError
from services.paths import ROOT
from services.runtime import check_cancel, num, report


class BacktestService:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else ROOT
        self.data = DataService(self.root)
        self.composite = CompositeService(self.root)

    def run(self, config: dict, *, refresh: bool = False, cancel=None, progress=None) -> dict:
        cfg = validate_config(config)
        check_cancel(cancel)
        report(progress, 70, "确定股票池")
        bench_only = self.data.load_market(
            [],
            cfg["benchmark"],
            cfg["start"],
            cfg["end"],
            cfg["adjust"],
            cfg["jump_limit"],
            cfg["source_order"],
            cfg.get("api_keys") or {},
            refresh=refresh,
            cancel=cancel,
            progress=progress,
        )
        calendar = bench_only["close"][cfg["benchmark"]].dropna().index
        ends = period_ends(calendar, cfg["rebalance"])
        membership = self.data.universe_on(cfg, [pd.Timestamp(day).strftime("%Y-%m-%d") for day in ends])
        symbols = membership["symbols"] or [symbol for symbol in cfg["universe"] if symbol != cfg["benchmark"]]
        if not symbols:
            raise DataGapError("没有可交易的股票")
        run_cfg = dict(cfg)
        run_cfg["universe"] = symbols + [cfg["benchmark"]]
        check_cancel(cancel)
        score, comp = self.composite.score_frame(run_cfg, refresh=refresh, cancel=cancel, progress=progress)
        market = self.data.load_market(
            symbols,
            cfg["benchmark"],
            cfg["start"],
            cfg["end"],
            cfg["adjust"],
            cfg["jump_limit"],
            cfg["source_order"],
            cfg.get("api_keys") or {},
            cancel=cancel,
            progress=progress,
        )
        prices = market["close"].reindex(columns=symbols).dropna(how="all")
        result = simulate(
            prices,
            score.reindex(columns=symbols),
            market["close"][cfg["benchmark"]],
            cfg["rebalance"],
            cfg["cost_bps"],
            cfg["slippage_bps"],
            cfg["max_weight"],
            cfg["diversify_penalty"],
            cfg["jump_limit"],
            eligible=membership["by_date"],
            cancel=cancel,
            progress=progress,
        )
        warnings = list(membership["warnings"]) + list(comp.get("warnings") or []) + list(result["warnings"])
        if cfg["benchmark"] in cfg["universe"]:
            warnings.append("基准不进入组合")
        warnings.append("跟踪误差是组合减去 NDX 之后的结果，优化时没有把它当成硬约束。")
        warnings.append("组合是所选股票的多头，不是用成分股权重复制出来的指数增强。")
        active = result["active"]
        net = result["net"].loc[active]
        bench = result["benchmark"].loc[active]
        one_way = result["one_way"].loc[active]
        cost = result["cost"].loc[active]
        if net.empty:
            raise DataGapError("因子预热之后没有持仓，回测没有开始")
        valid_start = pd.Timestamp(cfg["start"]) + pd.DateOffset(years=int(cfg["train_years"]))
        train_net = net.loc[net.index < valid_start]
        valid_net = net.loc[net.index >= valid_start]
        if valid_net.empty:
            warnings.append("验证段是空的")
        payload = {
            "warnings": _unique(warnings),
            "full": performance(net, bench, one_way, cost),
            "train": performance(train_net, bench, one_way, cost),
            "valid": performance(valid_net, bench, one_way, cost),
            "yearly": yearly_table(net, bench),
            "nav": _nav(net, bench),
            "splits": {
                "train_end": (valid_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "valid_start": valid_start.strftime("%Y-%m-%d"),
            },
            "scheme": comp.get("scheme"),
            "last_weights": comp.get("last_weights", {}),
            "survivor_warning": MANUAL_UNIVERSE_WARNING if cfg["universe_mode"] != "snapshot" else "",
        }
        report(progress, 90, "回测完成")
        return payload


def _nav(net: pd.Series, bench: pd.Series) -> list[dict]:
    bench = bench.reindex(net.index).fillna(0.0)
    strategy = (1.0 + net).cumprod()
    base = (1.0 + bench).cumprod()
    strategy_dd = strategy / strategy.cummax() - 1.0
    base_dd = base / base.cummax() - 1.0
    rows = []
    for day in strategy.index:
        rows.append(
            {
                "date": pd.Timestamp(day).strftime("%Y-%m-%d"),
                "strategy": num(strategy.loc[day]),
                "benchmark": num(base.loc[day]),
                "strategy_drawdown": num(strategy_dd.loc[day]),
                "benchmark_drawdown": num(base_dd.loc[day]),
            }
        )
    return rows


def _unique(items: list[str]) -> list[str]:
    out = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out
