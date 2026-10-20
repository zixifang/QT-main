"""按配置计算单个因子。财务因子没有公告日数据时返回空结果。"""

from __future__ import annotations

from pathlib import Path

import json
import pandas as pd

from factors.catalog import catalog, get_factor
from factors.price import annualized_vol, liquidity_proxy, residual_momentum, rolling_beta
from preprocess.prepare import neutralize_frame, winsorize_frame, zscore_frame
from services.config_store import validate_config
from services.data_service import DataService
from services.paths import ROOT
from services.runtime import check_cancel, digest, num, report
from services.store import CACHE_VERSION, Store


def factor_settings(config: dict, item: dict) -> dict:
    params = dict(item.get("defaults") or {})
    params.update((config.get("factors") or {}).get(item["id"], {}))
    return params


def _unique(items: list[str]) -> list[str]:
    out = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def _latest(score: pd.DataFrame) -> list[dict]:
    if score.empty or score.dropna(how="all").empty:
        return []
    day = score.dropna(how="all").index.max()
    rows = []
    for symbol in score.columns:
        rows.append(
            {
                "symbol": str(symbol),
                "date": pd.Timestamp(day).strftime("%Y-%m-%d"),
                "score": num(score.loc[day, symbol]),
            }
        )
    return rows


def _empty_reason(item: dict) -> str:
    if item["id"] == "fwd_eps_growth":
        return "没有预期 EPS，因子为空并降级"
    if item["id"] == "peg":
        return "没有预期增速，PEG 为空并降级"
    if item["id"] == "size":
        return "没有股本，市值因子为空"
    if item["id"] == "industry":
        return "没有行业分类，行业中性化跳过"
    return "没有公告日对齐的财务数据，因子为空"


class FactorService:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else ROOT
        self.store = Store(self.root)
        self.data = DataService(self.root)

    def catalog(self) -> list[dict]:
        return catalog()

    def compute(self, config: dict, factor_id: str, *, refresh: bool = False, cancel=None, progress=None) -> dict:
        _frame, meta = self.panel(config, factor_id, refresh=refresh, cancel=cancel, progress=progress)
        return meta

    def panel(self, config: dict, factor_id: str, *, refresh: bool = False, cancel=None, progress=None):
        cfg = validate_config(config)
        item = get_factor(factor_id)
        settings = factor_settings(cfg, item)
        tradable = [symbol for symbol in cfg["universe"] if symbol != cfg["benchmark"]]
        key = digest(
            CACHE_VERSION,
            "factor",
            factor_id,
            json.dumps(settings, sort_keys=True, default=str),
            json.dumps(cfg["neutralize"], sort_keys=True),
            str(cfg["beta_lookback"]),
            str(cfg["liquidity_lookback"]),
            str(cfg["jump_limit"]),
            cfg["start"],
            cfg["end"],
            cfg["adjust"],
            ",".join(tradable),
            cfg["benchmark"],
            self.data.fingerprint(tradable + [cfg["benchmark"]]),
        )
        if refresh:
            self.store.forget(key)
        elif (cached := self.store.read_bundle(key)) is not None:
            frame, meta = cached
            meta["cache"] = "hit"
            meta.setdefault("warnings", [])
            report(progress, 40, f"{item['name']} 使用缓存")
            return frame, meta
        check_cancel(cancel)
        report(progress, 25, f"计算{item['name']}")
        if not item["implemented"]:
            meta = self._meta(item, True, [_empty_reason(item)], [], "miss", "")
            media = self.store.write_bundle(key, "factor", pd.DataFrame(), meta)
            meta["media"] = media
            if media == "csv":
                meta["warnings"] = _unique(meta["warnings"] + ["没有 pyarrow，序列改存 csv"])
            return pd.DataFrame(), meta
        market = self.data.load_market(
            tradable,
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
        close = market["close"][tradable]
        volume = market["volume"].reindex(columns=tradable)
        bench = market["close"][cfg["benchmark"]]
        raw = self._raw(item, settings, close, volume, bench, cfg["jump_limit"], cancel)
        warnings: list[str] = []
        if item["role"] in {"alpha", "reference"}:
            prepared = zscore_frame(winsorize_frame(raw))
            risks = self._risks(cfg, close, volume, bench, cancel)
            neutral, extra = neutralize_frame(prepared, risks)
            warnings.extend(extra)
            score = neutral * item["direction"] if item["direction"] else neutral
        else:
            score = zscore_frame(winsorize_frame(raw))
        meta = self._meta(item, False, _unique(warnings), _latest(score), "miss", "")
        media = self.store.write_bundle(key, "factor", score, meta)
        meta["media"] = media
        if media == "csv":
            meta["warnings"] = _unique(list(meta["warnings"]) + ["没有 pyarrow，序列改存 csv"])
        report(progress, 55, f"{item['name']} 计算完成")
        return score, meta

    def _meta(self, item: dict, degraded: bool, warnings: list[str], latest: list[dict], cache: str, media: str) -> dict:
        return {
            "factor": item["id"],
            "name": item["name"],
            "role": item["role"],
            "direction": item["direction"],
            "formula": item["formula"],
            "degraded": degraded,
            "warnings": warnings,
            "latest": latest,
            "cache": cache,
            "media": media,
        }

    def _raw(self, item, settings, close, volume, bench, jump_limit, cancel) -> pd.DataFrame:
        columns = {}
        for symbol in close.columns:
            check_cancel(cancel)
            if item["id"] == "residual_momentum":
                columns[symbol] = residual_momentum(close[symbol], bench, int(settings["lookback"]), int(settings["skip"]), jump_limit)
            elif item["id"] == "low_volatility":
                columns[symbol] = annualized_vol(close[symbol], int(settings["lookback"]), jump_limit)
            elif item["id"] == "beta":
                columns[symbol] = rolling_beta(close[symbol], bench, int(settings.get("lookback", 252)), jump_limit)
            elif item["id"] == "liquidity":
                columns[symbol] = liquidity_proxy(volume[symbol], int(settings.get("lookback", 20)))
            else:
                columns[symbol] = pd.Series(float("nan"), index=close.index)
        return pd.DataFrame(columns, index=close.index)

    def _risks(self, config, close, volume, bench, cancel) -> dict[str, pd.DataFrame]:
        risks = {}
        for name in config.get("neutralize") or []:
            check_cancel(cancel)
            if name == "beta":
                risks[name] = pd.DataFrame(
                    {symbol: rolling_beta(close[symbol], bench, int(config["beta_lookback"]), config["jump_limit"]) for symbol in close.columns}
                )
            elif name == "liquidity":
                risks[name] = pd.DataFrame(
                    {symbol: liquidity_proxy(volume[symbol], int(config["liquidity_lookback"])) for symbol in close.columns}
                )
            else:
                risks[name] = pd.DataFrame()
        return risks
