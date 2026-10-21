"""因子检验和策略回测的服务。界面只调用这里，不读 CSV。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest.execution import _signal_days, run_backtest
from factors.price_volume import FACTOR_HELP, FACTOR_REGISTRY, compute_factor
from preprocess.adjust import adjust_ohlcv
from preprocess.prepare import winsorize_frame, zscore_frame
from services.errors import DataGapError
from services.paths import ROOT
from services.runtime import check_cancel, report


def list_factors() -> list[dict]:
    return [{"id": name, "help": FACTOR_HELP[name]} for name in FACTOR_REGISTRY]


def list_constituents(root: Path | None = None) -> list[str]:
    path = (root or ROOT) / "data_fetch" / "nasdaq100_constituents.txt"
    if not path.exists():
        return []
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip().upper()
        if text and text != "NDX":
            names.append(text)
    return names


def run_factor_tests(config: dict, progress=None, cancel=None) -> dict:
    """IC、分层、换手和因子相关。原始因子先算完，再在这里做截面处理。"""
    check_cancel(cancel)
    report(progress, 5, "读取行情")
    frames, warnings = _load_frames(config)
    close = _close_panel(frames)
    window = int(config.get("N", 20))
    horizon = int(config.get("horizon", 20))
    groups = int(config.get("groups", 5))
    names = [name for name in config.get("factors") or [] if name in FACTOR_REGISTRY]
    if not names:
        raise DataGapError("至少选择一个因子")
    if config.get("industry_neutral"):
        warnings.append("没有行业分类，行业中性化已跳过")
    forward = close.shift(-horizon) / close - 1.0
    panels = {}
    reports = {}
    for index, name in enumerate(names):
        check_cancel(cancel)
        report(progress, int(15 + 70 * index / max(len(names), 1)), f"计算 {name}")
        raw = _panel(frames, name, window)
        prepared = _prepare(raw, bool(config.get("winsorize")), bool(config.get("standardize")))
        panels[name] = prepared
        reports[name] = {
            "ic": _ic_report(prepared, forward),
            "groups": _group_report(prepared, close, groups, horizon),
        }
    report(progress, 92, "计算因子相关")
    correlation = _correlation(panels)
    report(progress, 100, "因子测试完成")
    return {"warnings": warnings, "factors": reports, "correlation": correlation, "close": close}


def run_strategy(config: dict, progress=None, cancel=None) -> dict:
    """用同一份复权收盘价做股数回测。方向为 auto 时由 RankIC 均值决定。"""
    check_cancel(cancel)
    report(progress, 5, "读取行情")
    frames, warnings = _load_frames(config)
    close = _close_panel(frames)
    opened = _field_panel(frames, "open")
    name = str(config.get("factor") or "")
    if name not in FACTOR_REGISTRY:
        raise DataGapError("请选择一个因子")
    if config.get("industry_neutral"):
        warnings.append("没有行业分类，行业中性化已跳过")
    window = int(config.get("N", 20))
    report(progress, 30, f"计算 {name}")
    raw = _panel(frames, name, window)
    prepared = _prepare(raw, bool(config.get("winsorize")), bool(config.get("standardize")))
    direction, direction_note = _direction(prepared, close, int(config.get("horizon", 20)), config.get("direction", "auto"))
    if direction_note:
        warnings.append(direction_note)
    benchmark, bench_note = _benchmark(config, close)
    if bench_note:
        warnings.append(bench_note)
    report(progress, 55, "回测成交")
    result = run_backtest(
        close,
        prepared,
        {
            "direction": direction,
            "rebalance": config.get("rebalance", "M"),
            "top_n": int(config.get("top_n", 20)),
            "weight": config.get("weight", "equal"),
            "max_weight": float(config.get("max_weight", 0.10)),
            "commission": float(config.get("commission", 0.0003)),
            "slippage": float(config.get("slippage", 0.0005)),
            "sell_tax": float(config.get("sell_tax", 0.0)),
            "lot": int(config.get("lot", 1)),
            "fill": config.get("fill", "close"),
            "initial_cash": float(config.get("initial_cash", 1_000_000)),
            "short": bool(config.get("short", False)),
            "open": opened,
            "benchmark": benchmark,
            "warnings": warnings,
            "cancel": cancel,
        },
    )
    report(progress, 100, "回测完成")
    metrics = dict(result.metrics)
    warnings = list(metrics.pop("warnings", warnings))
    return {
        "nav": result.nav_series,
        "trades": result.trades,
        "positions": result.positions,
        "metrics": metrics,
        "benchmark": benchmark,
        "warnings": warnings,
        "direction": direction,
        "close": close,
    }


def _load_frames(config: dict) -> tuple[dict[str, pd.DataFrame], list[str]]:
    symbols = [str(item).strip().upper() for item in config.get("symbols") or [] if str(item).strip()]
    if not symbols:
        raise DataGapError("股票池是空的")
    start = str(config.get("start") or "2015-01-01")
    end = str(config.get("end") or "2025-12-31")
    root = Path(config["root"]) if config.get("root") else ROOT
    folder = Path(config["data_dir"]) if config.get("data_dir") else root / "data" / "nasdaq100"
    frames: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []
    for symbol in symbols:
        path = folder / f"{symbol}.csv"
        if not path.exists():
            warnings.append(f"没有 {symbol}.csv，已跳过")
            continue
        raw = _read_ohlcv(path, start, end)
        adjusted, jumps = adjust_ohlcv(raw)
        frames[symbol] = adjusted
        if bool(jumps.any()):
            warnings.append(f"{symbol} 有 {int(jumps.sum())} 天疑似拆股，只做标记，因子仍按原序列计算")
    if len(frames) < 2:
        raise DataGapError("至少需要两只有行情的股票")
    return frames, warnings


def _read_ohlcv(path: Path, start: str, end: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "Date" not in frame.columns or "Close" not in frame.columns:
        raise DataGapError(f"{path.name} 缺少 Date 或 Close")
    dates = pd.to_datetime(frame["Date"])
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert(None)
    adjusted = frame["Adj Close"] if "Adj Close" in frame.columns else frame["Close"]
    out = pd.DataFrame(
        {
            "date": dates.dt.normalize(),
            "open": pd.to_numeric(frame["Open"], errors="coerce") if "Open" in frame.columns else pd.to_numeric(frame["Close"], errors="coerce"),
            "high": pd.to_numeric(frame["High"], errors="coerce") if "High" in frame.columns else pd.to_numeric(frame["Close"], errors="coerce"),
            "low": pd.to_numeric(frame["Low"], errors="coerce") if "Low" in frame.columns else pd.to_numeric(frame["Close"], errors="coerce"),
            "close": pd.to_numeric(frame["Close"], errors="coerce"),
            "adjusted_close": pd.to_numeric(adjusted, errors="coerce"),
            "volume": pd.to_numeric(frame["Volume"], errors="coerce") if "Volume" in frame.columns else np.nan,
        }
    )
    out = out.dropna(subset=["date", "close"])
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    out = out[(out["date"] >= start_ts) & (out["date"] <= end_ts)]
    out = out.sort_values("date").drop_duplicates("date", keep="last")
    if out.empty:
        raise DataGapError(f"{path.stem} 在 {start} 到 {end} 没有行情")
    return out.set_index("date")


def _close_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return _field_panel(frames, "adjusted_close")


def _field_panel(frames: dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
    return pd.DataFrame({symbol: frame[field] for symbol, frame in frames.items()}).sort_index()


def _panel(frames: dict[str, pd.DataFrame], name: str, window: int) -> pd.DataFrame:
    columns = {symbol: compute_factor(frame, name, {"N": window}) for symbol, frame in frames.items()}
    return pd.DataFrame(columns).sort_index()


def _prepare(panel: pd.DataFrame, winsorize: bool, standardize: bool) -> pd.DataFrame:
    out = panel.astype(float)
    if winsorize:
        out = winsorize_frame(out)
    if standardize:
        out = zscore_frame(out)
    return out


def _ic_report(factor: pd.DataFrame, forward: pd.DataFrame) -> dict:
    pearson = _ic_series(factor, forward, "pearson")
    spearman = _ic_series(factor, forward, "spearman")
    return {"pearson": _ic_stats(pearson), "spearman": _ic_stats(spearman)}


def _ic_series(factor: pd.DataFrame, forward: pd.DataFrame, method: str) -> pd.Series:
    shared = factor.index.intersection(forward.index)
    values = []
    dates = []
    for day in shared:
        pair = pd.concat([factor.loc[day], forward.loc[day]], axis=1).dropna()
        if len(pair) < 3:
            continue
        corr = pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method)
        if pd.notna(corr):
            dates.append(day)
            values.append(float(corr))
    return pd.Series(values, index=pd.DatetimeIndex(dates), name=method)


def _ic_stats(series: pd.Series) -> dict:
    if series.empty:
        return {"mean": None, "std": None, "ir": None, "win_rate": None, "series": series}
    mean = float(series.mean())
    std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    ir = None if std == 0 else mean / std
    return {
        "mean": mean,
        "std": std,
        "ir": ir,
        "win_rate": float((series > 0).mean()),
        "series": series,
    }


def _group_report(factor: pd.DataFrame, close: pd.DataFrame, groups: int, horizon: int) -> dict:
    ends = [pd.Timestamp(day) for day in _signal_days(close.index, "M") if day in close.index]
    forward = close.shift(-int(horizon)) / close - 1.0
    buckets = max(int(groups), 2)
    members: dict[int, list[set[str]]] = {}
    holding: dict[int, list[float]] = {}
    horizon_returns: dict[int, list[float]] = {}
    dates: list[pd.Timestamp] = []
    locked: int | None = None
    for pos, day in enumerate(ends[:-1]):
        nxt = ends[pos + 1]
        values = factor.loc[day].dropna() if day in factor.index else pd.Series(dtype=float)
        if len(values) < 2:
            continue
        try:
            labels = pd.qcut(values, min(buckets, len(values)), labels=False, duplicates="drop")
        except ValueError:
            continue
        count = int(labels.nunique())
        if count < 2:
            continue
        if locked is None:
            locked = count
        elif count != locked:
            continue
        dates.append(day)
        for group in range(locked):
            names = set(labels[labels == group].index)
            members.setdefault(group, []).append(names)
            holding.setdefault(group, []).append(_basket_return(close, names, day, nxt))
            horizon_returns.setdefault(group, []).append(_mean_forward(forward, names, day))
    used = locked or 0
    nav = {}
    annual = []
    turnovers = []
    for group in range(used):
        path = holding.get(group, [])
        level = np.cumprod([1.0 + (0.0 if item is None or not np.isfinite(item) else item) for item in path]) if path else []
        nav[group] = level
        samples = [item for item in horizon_returns.get(group, []) if item is not None and np.isfinite(item)]
        mean = float(np.mean(samples)) if samples else None
        annual.append(None if mean is None or mean <= -1 else float((1.0 + mean) ** (252.0 / max(horizon, 1)) - 1.0))
        history = members.get(group, [])
        rates = []
        for prev, curr in zip(history, history[1:]):
            if not prev:
                continue
            changed = len(prev.symmetric_difference(curr))
            rates.append(0.5 * changed / len(prev))
        if rates:
            turnovers.append(float(np.mean(rates)))
    top = holding.get(used - 1, [])
    bottom = holding.get(0, [])
    spread = []
    for high, low in zip(top, bottom):
        if high is None or low is None or not np.isfinite(high) or not np.isfinite(low):
            spread.append(0.0)
        else:
            spread.append(high - low)
    long_short = list(np.cumprod([1.0 + item for item in spread])) if spread else []
    means = []
    for group in range(used):
        samples = [item for item in horizon_returns.get(group, []) if item is not None and np.isfinite(item)]
        if samples:
            means.append(float(np.mean(samples)))
    monotonic = None
    if len(means) >= 2:
        monotonic = float(pd.Series(means).corr(pd.Series(range(len(means))), method="spearman"))
    increasing = bool(means) and all(later >= earlier for earlier, later in zip(means, means[1:]))
    return {
        "dates": dates,
        "nav": nav,
        "annual": annual,
        "long_short": long_short,
        "turnover": float(np.mean(turnovers)) if turnovers else None,
        "monotonicity": monotonic,
        "increasing": increasing,
        "groups": used,
    }


def _basket_return(close: pd.DataFrame, names: set[str], start, end) -> float | None:
    if start not in close.index or end not in close.index or not names:
        return None
    returns = []
    for name in names:
        first = close.at[start, name]
        second = close.at[end, name]
        if np.isfinite(first) and np.isfinite(second) and first:
            returns.append(float(second) / float(first) - 1.0)
    if not returns:
        return None
    return float(np.mean(returns))


def _mean_forward(forward: pd.DataFrame, names: set[str], day) -> float | None:
    if day not in forward.index or not names:
        return None
    values = forward.loc[day, list(names)].astype(float)
    values = values[np.isfinite(values.to_numpy(dtype=float))]
    if values.empty:
        return None
    return float(values.mean())


def _correlation(panels: dict[str, pd.DataFrame]) -> dict:
    if len(panels) < 2:
        return {"pearson": None, "spearman": None, "names": list(panels), "high_pairs": []}
    stacked = {name: panel.stack(future_stack=True) for name, panel in panels.items()}
    frame = pd.DataFrame(stacked)
    pearson = frame.corr(method="pearson")
    spearman = frame.corr(method="spearman")
    high = []
    names = list(pearson.columns)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            value = pearson.at[left, right]
            if pd.notna(value) and abs(float(value)) >= 0.7:
                high.append({"left": left, "right": right, "corr": float(value)})
    return {"pearson": pearson, "spearman": spearman, "names": names, "high_pairs": high}


def _direction(factor: pd.DataFrame, close: pd.DataFrame, horizon: int, choice) -> tuple[int, str]:
    if str(choice) in {"1", "+1"} or choice == 1:
        return 1, ""
    if str(choice) in {"-1"} or choice == -1:
        return -1, ""
    forward = close.shift(-horizon) / close - 1.0
    stats = _ic_stats(_ic_series(factor, forward, "spearman"))
    mean = stats["mean"]
    if mean is None:
        return 1, "RankIC 为空，方向按做多高因子值"
    if mean >= 0:
        return 1, f"RankIC 均值为 {mean:.3f}，做多高因子值"
    return -1, f"RankIC 均值为 {mean:.3f}，做多低因子值"


def _benchmark(config: dict, close: pd.DataFrame) -> tuple[pd.Series | None, str]:
    mode = str(config.get("benchmark") or "NDX")
    if mode == "equal":
        returns = close.pct_change(fill_method=None)
        nav = (1.0 + returns.mean(axis=1).fillna(0.0)).cumprod()
        return nav, ""
    root = Path(config["root"]) if config.get("root") else ROOT
    folder = Path(config["data_dir"]) if config.get("data_dir") else root / "data" / "nasdaq100"
    path = folder / "NDX.csv"
    if not path.exists():
        returns = close.pct_change(fill_method=None)
        nav = (1.0 + returns.mean(axis=1).fillna(0.0)).cumprod()
        return nav, "没有 NDX.csv，基准改为股票池等权"
    start = str(config.get("start") or "2015-01-01")
    end = str(config.get("end") or "2025-12-31")
    frame = _read_ohlcv(path, start, end)
    adjusted, _jumps = adjust_ohlcv(frame)
    return adjusted["adjusted_close"].reindex(close.index), ""
