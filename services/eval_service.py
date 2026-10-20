"""月末 IC 和到下一次再平衡的分层。IC 用到未来收益，只用于检验。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from factors.catalog import get_factor
from preprocess.prepare import period_ends
from services.config_store import validate_config
from services.data_service import DataService
from services.factor_service import FactorService
from services.paths import ROOT
from services.runtime import check_cancel, digest, num, report
from services.store import CACHE_VERSION, Store


def ic_table(score: pd.DataFrame, close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    aligned = score.reindex(close.index)
    forward = close.shift(-int(horizon)) / close - 1.0
    rows = []
    for day in period_ends(close.index, "M"):
        if day not in aligned.index or day not in forward.index:
            continue
        pair = pd.concat([aligned.loc[day], forward.loc[day]], axis=1).dropna()
        if len(pair) < 2:
            continue
        corr = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
        if pd.notna(corr):
            rows.append((pd.Timestamp(day), float(corr)))
    if not rows:
        return pd.DataFrame(columns=["ic"])
    frame = pd.DataFrame(rows, columns=["date", "ic"]).set_index("date")
    frame.index = pd.DatetimeIndex(frame.index)
    return frame


def ic_matrix(panels: dict[str, pd.DataFrame], close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    columns = {}
    for name, score in panels.items():
        table = ic_table(score, close, horizon)
        columns[name] = table["ic"] if "ic" in table.columns else pd.Series(dtype=float)
    return pd.DataFrame(columns)


def quantile_paths(score: pd.DataFrame, close: pd.DataFrame, rebalance: str) -> tuple[list[dict], float | None, int, list[str]]:
    warnings = []
    count = int(score.shape[1])
    if count >= 5:
        buckets = 5
    elif count >= 2:
        buckets = 2
        warnings.append(f"只有 {count} 只股票，分层不用五组，改成高低两组")
    else:
        return [], None, 1, ["股票不足两组，没有分层"]
    ends = period_ends(close.index, rebalance)
    ends = ends[ends.isin(close.index)]
    levels = close.reindex(ends)
    forward = levels.shift(-1) / levels - 1.0
    high_ret, low_ret, dates = [], [], []
    previous = None
    changes = []
    used = buckets
    for day in ends[:-1]:
        if day not in score.index:
            continue
        values = score.loc[day].dropna()
        future = forward.loc[day]
        if len(values) < 2 or future.dropna().empty:
            continue
        try:
            labels = pd.qcut(values, min(buckets, len(values)), labels=False, duplicates="drop")
        except ValueError:
            continue
        if labels.nunique() < 2:
            continue
        used = int(labels.nunique())
        high = labels[labels == labels.max()].index
        low = labels[labels == labels.min()].index
        high_ret.append(float(future.reindex(high).mean()))
        low_ret.append(float(future.reindex(low).mean()))
        dates.append(pd.Timestamp(day))
        if previous is not None:
            both = previous.index.intersection(labels.index)
            if len(both):
                changes.append(float((previous.loc[both] != labels.loc[both]).mean()))
        previous = labels
    if not dates:
        return [], None, used, warnings + ["分层没有形成持有期"]
    high_nav = np.cumprod(1.0 + np.asarray(high_ret, dtype=float))
    low_nav = np.cumprod(1.0 + np.asarray(low_ret, dtype=float))
    spread = [
        {"date": day.strftime("%Y-%m-%d"), "high": float(high_nav[i]), "low": float(low_nav[i])}
        for i, day in enumerate(dates)
    ]
    turnover = float(np.mean(changes)) if changes else None
    return spread, turnover, used, warnings


class EvalService:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else ROOT
        self.store = Store(self.root)
        self.factors = FactorService(self.root)
        self.data = DataService(self.root)

    def single(self, config: dict, factor_id: str, *, refresh: bool = False, cancel=None, progress=None) -> dict:
        cfg = validate_config(config)
        item = get_factor(factor_id)
        tradable = [symbol for symbol in cfg["universe"] if symbol != cfg["benchmark"]]
        key = digest(
            CACHE_VERSION,
            "eval",
            factor_id,
            str(cfg["horizon"]),
            cfg["rebalance"],
            cfg["start"],
            cfg["end"],
            ",".join(tradable),
            self.data.fingerprint(tradable + [cfg["benchmark"]]),
            json.dumps(cfg["factors"].get(factor_id, {}), sort_keys=True, default=str),
            json.dumps(cfg["neutralize"], sort_keys=True),
            str(cfg["jump_limit"]),
        )
        if refresh:
            self.store.forget(key)
        elif not refresh and (cached := self.store.get_json(key)) is not None:
            cached["cache"] = "hit"
            report(progress, 60, f"{item['name']} 的检验使用缓存")
            return cached
        check_cancel(cancel)
        report(progress, 45, f"检验{item['name']}")
        score, meta = self.factors.panel(cfg, factor_id, refresh=refresh, cancel=cancel, progress=progress)
        if meta["degraded"] or score.empty:
            result = _empty_eval(item, list(meta["warnings"]))
            self.store.put_json(key, "eval", result)
            return result
        market = self.data.load_market(
            tradable,
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
        close = market["close"][tradable]
        table = ic_table(score, close, cfg["horizon"])
        values = table["ic"].to_numpy(dtype=float) if not table.empty else np.array([])
        spread, turnover, buckets, extra = quantile_paths(score, close, cfg["rebalance"])
        warnings = list(meta["warnings"]) + extra
        if len(values) == 0:
            warnings.append("没有足够的截面计算 IC")
        mean = float(np.mean(values)) if len(values) else None
        std = float(np.std(values, ddof=1)) if len(values) > 1 else None
        icir = mean / std if mean is not None and std else None
        tstat = mean / (std / np.sqrt(len(values))) if mean is not None and std and len(values) > 1 else None
        result = {
            "factor": item["id"],
            "name": item["name"],
            "degraded": False,
            "warnings": warnings,
            "ic_mean": num(mean),
            "icir": num(icir),
            "ic_t": num(tstat),
            "n_ic": int(len(values)),
            "n_buckets": int(buckets),
            "turnover": num(turnover),
            "ic": [{"date": pd.Timestamp(day).strftime("%Y-%m-%d"), "ic": num(table.loc[day, "ic"])} for day in table.index],
            "spread": spread,
            "cache": "miss",
        }
        self.store.put_json(key, "eval", result)
        report(progress, 70, f"{item['name']} 检验完成")
        return result


def _empty_eval(item: dict, warnings: list[str]) -> dict:
    return {
        "factor": item["id"],
        "name": item["name"],
        "degraded": True,
        "warnings": warnings,
        "ic_mean": None,
        "icir": None,
        "ic_t": None,
        "n_ic": 0,
        "n_buckets": 0,
        "turnover": None,
        "ic": [],
        "spread": [],
        "cache": "miss",
    }
