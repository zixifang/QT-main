"""三种合成权重。IC 权重只用预测期已经走完的历史。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from factors.catalog import catalog
from preprocess.prepare import known_history, period_ends
from services.config_store import validate_config
from services.data_service import DataService
from services.errors import DataGapError
from services.eval_service import ic_matrix
from services.factor_service import FactorService, factor_settings
from services.paths import ROOT
from services.runtime import check_cancel, digest, num, report
from services.store import CACHE_VERSION, Store


def weights_from_ic(history: pd.DataFrame, scheme: str, min_obs: int = 12) -> tuple[pd.Series, str | None]:
    columns = list(history.columns)
    equal = pd.Series(1.0 / len(columns), index=columns, dtype=float)
    if scheme == "equal":
        return equal, None
    usable = history.dropna(how="all")
    if len(usable) < min_obs:
        return equal, "IC 历史不足 12 期，权重退回等权"
    if scheme == "ic":
        score = usable.mean().clip(lower=0).fillna(0.0)
    elif scheme == "max_icir":
        std = usable.std(ddof=1).replace(0, pd.NA)
        score = (usable.mean() / std).clip(lower=0).fillna(0.0)
    else:
        raise DataGapError(f"未知权重方案：{scheme}")
    total = float(score.sum())
    if total <= 0:
        return equal, "IC 权重全小于或等于 0，退回等权"
    return score / total, None


class CompositeService:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else ROOT
        self.store = Store(self.root)
        self.factors = FactorService(self.root)
        self.data = DataService(self.root)

    def combine(self, config: dict, *, refresh: bool = False, cancel=None, progress=None) -> dict:
        _score, meta = self.score_frame(config, refresh=refresh, cancel=cancel, progress=progress)
        return meta

    def score_frame(self, config: dict, *, refresh: bool = False, cancel=None, progress=None):
        cfg = validate_config(config)
        tradable = [symbol for symbol in cfg["universe"] if symbol != cfg["benchmark"]]
        key = digest(
            CACHE_VERSION,
            "composite",
            cfg["weight_scheme"],
            str(cfg["horizon"]),
            cfg["start"],
            cfg["end"],
            ",".join(tradable),
            json.dumps(cfg["factors"], sort_keys=True, default=str),
            json.dumps(cfg["neutralize"], sort_keys=True),
            str(cfg["jump_limit"]),
            self.data.fingerprint(tradable + [cfg["benchmark"]]),
        )
        if refresh:
            self.store.forget(key)
        elif (cached := self.store.read_bundle(key)) is not None:
            frame, meta = cached
            meta["cache"] = "hit"
            report(progress, 75, "合成结果使用缓存")
            return frame, meta
        check_cancel(cancel)
        report(progress, 60, "合成因子")
        panels, warnings = {}, []
        for item in catalog():
            if item["role"] == "risk":
                continue
            settings = factor_settings(cfg, item)
            if not settings.get("enabled"):
                continue
            check_cancel(cancel)
            frame, meta = self.factors.panel(cfg, item["id"], refresh=refresh, cancel=cancel, progress=progress)
            if meta["degraded"] or frame.empty or frame.dropna(how="all").empty:
                warnings.append(f"{item['name']} 没有参与合成")
                warnings.extend(meta.get("warnings") or [])
                continue
            if item["role"] == "reference":
                warnings.append(f"{item['name']} 是排序参考，仍按勾选进入合成")
            panels[item["id"]] = frame
            warnings.extend(meta.get("warnings") or [])
        if not panels:
            raise DataGapError("没有可合成的因子")
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
        close = market["close"][list(panels[next(iter(panels))].columns)]
        history = ic_matrix(panels, close, cfg["horizon"])
        index = next(iter(panels.values())).index
        weights, weight_notes = self._weights(index, history, cfg["weight_scheme"], cfg["horizon"])
        warnings.extend(weight_notes)
        score = None
        for name, panel in panels.items():
            weighted = panel.reindex(index).mul(weights[name], axis=0)
            score = weighted if score is None else score.add(weighted)
        assert score is not None
        stacked = pd.DataFrame({name: panel.stack() for name, panel in panels.items()})
        corr = stacked.corr(method="spearman")
        correlation = []
        names = list(corr.columns)
        for left in names:
            for right in names:
                if left < right:
                    correlation.append({"left": left, "right": right, "corr": num(corr.loc[left, right])})
        last_day = score.dropna(how="all").index.max() if not score.dropna(how="all").empty else None
        preview = []
        if last_day is not None:
            for symbol in score.columns:
                preview.append({"symbol": str(symbol), "date": pd.Timestamp(last_day).strftime("%Y-%m-%d"), "score": num(score.loc[last_day, symbol])})
        last_weights = weights.dropna(how="all").iloc[-1] if not weights.dropna(how="all").empty else pd.Series(dtype=float)
        meta = {
            "scheme": cfg["weight_scheme"],
            "warnings": _unique(warnings),
            "last_weights": {name: num(last_weights.get(name)) for name in panels},
            "mean_weights": {name: num(weights[name].mean()) for name in panels},
            "correlation": correlation,
            "score_preview": preview,
            "cache": "miss",
            "media": "",
        }
        media = self.store.write_bundle(key, "composite", score, meta)
        meta["media"] = media
        report(progress, 80, "合成完成")
        return score, meta

    def _weights(self, index: pd.DatetimeIndex, history: pd.DataFrame, scheme: str, horizon: int) -> tuple[pd.DataFrame, list[str]]:
        columns = list(history.columns)
        if scheme == "equal" or not columns:
            frame = pd.DataFrame(1.0 / max(len(columns), 1), index=index, columns=columns)
            return frame, []
        ends = [day for day in period_ends(index, "M") if day in index]
        rows = []
        notes: list[str] = []
        for day in ends:
            known = known_history(history, day, index, horizon).tail(36)
            known = known.reindex(columns=columns) if not known.empty else pd.DataFrame(columns=columns)
            chosen, note = weights_from_ic(known, scheme)
            if note and note not in notes:
                notes.append(note)
            rows.append(chosen.rename(day))
        if not rows:
            equal = pd.DataFrame(1.0 / len(columns), index=index, columns=columns)
            return equal, notes or ["没有可用的 IC，权重退回等权"]
        full = pd.DataFrame(rows).reindex(index).ffill()
        return full, notes


def _unique(items: list[str]) -> list[str]:
    out = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out
