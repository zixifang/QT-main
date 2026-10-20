"""行情、覆盖度和历史成分。只读 data/nasdaq100 的未复权 Close。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from preprocess.prepare import jump_dates
from services.config_store import validate_config
from services.errors import AdjustError, ConstituentGapError, DataGapError, SourceUnavailable
from services.paths import ROOT
from services.runtime import check_cancel, digest, report
from services.sources import (
    BlockedSource,
    LocalCsvSource,
    NasdaqDataLinkSource,
    read_symbol_file,
)
from services.store import CACHE_VERSION, Store

STATIC_LIST_DATE = "2026-10-01"
MANUAL_UNIVERSE_WARNING = "股票池是手填的当前名单，不是当时的成分股。回测有幸存者偏差，不能当作指数增强的历史结果。"
STATIC_LIST_WARNING = "这是 2026-10 的静态成分名单，不能代表更早的成分股。"


def reject_adjust(adjust: str) -> None:
    text = str(adjust).strip().lower()
    if text in {"qfq", "forward", "前复权"}:
        raise AdjustError("禁止使用前复权。请改用 raw。")
    if text in {"hfq", "后复权"}:
        raise DataGapError("没有分红和拆股因子，不能做后复权。请使用 raw。疑似拆股日会单独告警。")
    if text not in {"raw", "none", "未复权"}:
        raise AdjustError(f"无法识别的复权方式：{adjust}")


def pack_market(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    columns = {}
    for symbol in close.columns:
        columns[f"close:{symbol}"] = close[symbol]
        columns[f"volume:{symbol}"] = volume[symbol] if symbol in volume.columns else float("nan")
    return pd.DataFrame(columns, index=close.index)


def unpack_market(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    close, volume = {}, {}
    for column in frame.columns:
        kind, _, symbol = str(column).partition(":")
        if kind == "close":
            close[symbol] = frame[column]
        elif kind == "volume":
            volume[symbol] = frame[column]
    return pd.DataFrame(close), pd.DataFrame(volume)


class DataService:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else ROOT
        self.store = Store(self.root)

    def coverage(self, config: dict, *, refresh: bool = False, cancel=None, progress=None) -> dict:
        del refresh
        cfg = validate_config(config)
        reject_adjust(cfg["adjust"])
        symbols = _listed(cfg["universe"], cfg["benchmark"])
        sources = self._sources(cfg)
        rows = []
        for index, symbol in enumerate(symbols):
            check_cancel(cancel)
            report(progress, int(10 + 70 * index / max(len(symbols), 1)), f"检查 {symbol}")
            try:
                frame, used, errors = self._load_one(symbol, cfg["start"], cfg["end"], cfg["source_order"], sources)
            except DataGapError as exc:
                rows.append(_empty_row(symbol, str(exc)))
                continue
            jumps = jump_dates(frame["close"], cfg["jump_limit"])
            notes = []
            if errors:
                notes.append("已降级：" + "；".join(errors))
            if jumps:
                notes.append(f"疑似拆股或异常跳空 {len(jumps)} 天：" + "、".join(jumps[:5]))
            series = frame["close"].dropna()
            rows.append(
                {
                    "symbol": symbol,
                    "rows": int(len(series)),
                    "first": series.index[0].strftime("%Y-%m-%d"),
                    "last": series.index[-1].strftime("%Y-%m-%d"),
                    "jumps": len(jumps),
                    "source": used,
                    "warning": "；".join(notes),
                }
            )
        report(progress, 100, "覆盖度检查完成")
        return {"rows": rows, "warnings": self._membership_warnings(cfg)}

    def prices(self, symbols: list[str], start: str, end: str, adjust: str = "raw", *, refresh: bool = False, cancel=None, progress=None) -> dict:
        reject_adjust(adjust)
        if not symbols:
            raise DataGapError("没有股票")
        names = [str(item).strip().upper() for item in symbols if str(item).strip()]
        market = self.load_market(
            names,
            names[0],
            start,
            end,
            adjust,
            0.5,
            ["local_csv"],
            {},
            refresh=refresh,
            cancel=cancel,
            progress=progress,
        )
        columns = []
        for symbol in market["close"].columns:
            series = market["close"][symbol].dropna()
            columns.append(
                {
                    "symbol": symbol,
                    "rows": int(len(series)),
                    "first": "" if series.empty else series.index[0].strftime("%Y-%m-%d"),
                    "last": "" if series.empty else series.index[-1].strftime("%Y-%m-%d"),
                    "jumps": market["jumps"].get(symbol, []),
                }
            )
        return {"adjust": "raw", "columns": columns, "warnings": market["warnings"], "cache": market["cache"]}

    def fundamentals(self, symbols: list[str], start: str, end: str) -> dict:
        del symbols, start, end
        return {
            "degraded": True,
            "rows": [],
            "warnings": ["没有公告日对齐的财务数据，EDGAR 尚未接入。预期 EPS 同样为空。"],
        }

    def constituents(self, as_of: str) -> dict:
        day = pd.Timestamp(as_of).strftime("%Y-%m-%d")
        path = self.root / "data" / "nasdaq100" / "snapshots" / f"{day}.txt"
        if path.exists():
            symbols = read_symbol_file(path)
            if not symbols:
                raise ConstituentGapError(f"{day} 的快照是空的")
            return {"as_of": day, "symbols": symbols, "source": "snapshot", "warning": None}
        if day >= STATIC_LIST_DATE:
            static = self.root / "data_fetch" / "nasdaq100_constituents.txt"
            if static.exists():
                symbols = read_symbol_file(static)
                if symbols:
                    return {"as_of": day, "symbols": symbols, "source": "static_2026_10", "warning": STATIC_LIST_WARNING}
        raise ConstituentGapError(f"{day} 没有历史成分快照，也没有 QQQ 历史持仓。不能用现在的名单回填这一天。")

    def universe_on(self, config: dict, dates: list[str]) -> dict:
        cfg = validate_config(config)
        if cfg["universe_mode"] != "snapshot":
            symbols = [symbol for symbol in cfg["universe"] if symbol != cfg["benchmark"]]
            return {"mode": "manual", "symbols": symbols, "by_date": None, "warnings": [MANUAL_UNIVERSE_WARNING]}
        by_date: dict[str, list[str]] = {}
        warnings: list[str] = []
        for day in dates:
            snap = self.constituents(day)
            names = [symbol for symbol in snap["symbols"] if symbol != cfg["benchmark"]]
            by_date[pd.Timestamp(day).strftime("%Y-%m-%d")] = names
            if snap.get("warning") and snap["warning"] not in warnings:
                warnings.append(snap["warning"])
        symbols = sorted({symbol for names in by_date.values() for symbol in names})
        return {"mode": "snapshot", "symbols": symbols, "by_date": by_date, "warnings": warnings}

    def pull(self, config: dict, *, refresh: bool = False, cancel=None, progress=None) -> dict:
        report(progress, 5, "本地源只检查文件，不会联网下载")
        return self.coverage(config, refresh=refresh, cancel=cancel, progress=progress)

    def load_market(
        self,
        symbols: list[str],
        benchmark: str,
        start: str,
        end: str,
        adjust: str,
        jump_limit: float,
        source_order: list[str],
        api_keys: dict,
        *,
        refresh: bool = False,
        cancel=None,
        progress=None,
    ) -> dict:
        reject_adjust(adjust)
        names = _listed(symbols, benchmark)
        fingerprint = self.fingerprint(names)
        key = digest(
            CACHE_VERSION,
            "market",
            ",".join(names),
            start,
            end,
            adjust,
            str(jump_limit),
            ",".join(source_order),
            fingerprint,
        )
        if refresh:
            self.store.forget(key)
        elif (cached := self.store.read_bundle(key)) is not None:
            frame, meta = cached
            close, volume = unpack_market(frame)
            report(progress, 15, "行情使用缓存")
            return {
                "close": close,
                "volume": volume,
                "jumps": meta.get("jumps", {}),
                "warnings": list(meta.get("warnings") or []),
                "cache": "hit",
                "media": meta.get("media", ""),
            }
        sources = self._sources({"api_keys": api_keys})
        closes, volumes, jumps, warnings = {}, {}, {}, []
        for index, symbol in enumerate(names):
            check_cancel(cancel)
            report(progress, int(10 + 15 * index / max(len(names), 1)), f"读取 {symbol}")
            frame, _used, errors = self._load_one(symbol, start, end, source_order, sources)
            closes[symbol] = frame["close"]
            volumes[symbol] = frame["volume"]
            jumps[symbol] = jump_dates(frame["close"], jump_limit)
            if errors:
                warnings.append(f"{symbol} 已降级：" + "；".join(errors))
            if jumps[symbol]:
                warnings.append(f"{symbol} 疑似拆股或异常跳空 {len(jumps[symbol])} 天")
        close = pd.DataFrame(closes).sort_index()
        volume = pd.DataFrame(volumes).reindex(close.index)
        media = self.store.write_bundle(key, "market", pack_market(close, volume), {"jumps": jumps, "warnings": warnings})
        if media == "csv":
            warnings.append("没有 pyarrow，序列改存 csv")
        return {"close": close, "volume": volume, "jumps": jumps, "warnings": warnings, "cache": "miss", "media": media}

    def fingerprint(self, symbols: list[str]) -> str:
        parts = []
        for symbol in sorted(set(symbols)):
            path = self.root / "data" / "nasdaq100" / f"{symbol}.csv"
            if path.exists():
                stat = path.stat()
                parts.append(f"{symbol}:{stat.st_mtime_ns}:{stat.st_size}")
            else:
                parts.append(f"{symbol}:missing")
        return "|".join(parts)

    def _membership_warnings(self, config: dict) -> list[str]:
        snap_dir = self.root / "data" / "nasdaq100" / "snapshots"
        count = len(list(snap_dir.glob("*.txt"))) if snap_dir.exists() else 0
        warnings = [f"历史成分快照 {count} 个。"]
        if config["universe_mode"] != "snapshot":
            warnings.append(MANUAL_UNIVERSE_WARNING)
        try:
            self.constituents(str(pd.Timestamp(config["start"]).date()))
        except ConstituentGapError as exc:
            warnings.append(str(exc))
        return warnings

    def _sources(self, config: dict) -> dict:
        key = str((config.get("api_keys") or {}).get("nasdaq_data_link", ""))
        return {
            "local_csv": LocalCsvSource(self.root),
            "yahoo": BlockedSource("不接入 Yahoo：yfinance 默认复权，不能当作未复权源。"),
            "nasdaq_data_link": NasdaqDataLinkSource(key),
            "edgar": BlockedSource("EDGAR 公司事实尚未接入，财务因子保持为空。"),
            "fama_french": BlockedSource("Fama-French 尚未接入。"),
            "qqq_holdings": BlockedSource("没有 QQQ 历史持仓文件。"),
        }

    def _load_one(self, symbol: str, start: str, end: str, order: list[str], sources: dict):
        errors = []
        for name in order:
            source = sources.get(name)
            if source is None:
                errors.append(f"{name}: 未知数据源")
                continue
            try:
                return source.load(symbol, start, end), name, errors
            except (SourceUnavailable, DataGapError) as exc:
                errors.append(f"{name}: {exc}")
        raise DataGapError("；".join(errors) if errors else f"{symbol} 没有行情")


def _listed(symbols: list[str], benchmark: str) -> list[str]:
    names = []
    for symbol in list(symbols) + [benchmark]:
        text = str(symbol).strip().upper()
        if text and text not in names:
            names.append(text)
    return names


def _empty_row(symbol: str, warning: str) -> dict:
    return {"symbol": symbol, "rows": 0, "first": "", "last": "", "jumps": 0, "source": "", "warning": warning}
