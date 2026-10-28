"""按 `docs/界面数据契约.md` 写 outdata 的公共部分。

分年统计、买卖点、数值规整这些口径与 `algorithms/rsi.py` 完全一致，这里直接复用它的实现：
同一套算法口径，第四页各算法算出来的年表才可比。`rsi.py` 保持独立、不依赖本模块。

本模块补的是 rsi 没做的三件事：

1. 合并写入：股票 JSON 里已有的其他算法 `id` 原样保留，只替换自己这一条；
2. `index.json` 的算法条目原位替换或追加，不覆盖别人的条目；
3. 第五页要的日度 `history`（各股净值等权平均），以及 `years` 的组织。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd

from algorithms import rsi
from algorithms.index_enhance import _write_merged_json as write_merged
from engine import prepare_close

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
OUT_ROOT = ROOT / "outdata"

SCHEMA_VERSION = 1
PRICE_FIELD = "close"

# 复用 rsi.py 的口径，避免两套实现算出不同的年表。
load_frames = rsi.load_frames
describe = rsi._performance
yearly_stats = rsi._yearly_stats
overview_year = rsi._overview_year
stats_payload = rsi._stats
series = rsi._series
trade_dates = rsi._trade_dates
write_json = rsi._write_json

FrameBuilder = Callable[[pd.Series], tuple[dict, dict, pd.Series, pd.Series]]


def build_payload(
    algorithm: dict,
    close: pd.Series,
    signal: pd.Series,
    indicator,
    params: dict,
    fee: float,
) -> tuple[dict, dict, pd.Series, pd.Series]:
    """组装一只股票的算法条目。返回 (条目, 各年统计, 策略净值, 买入持有净值)。"""
    strategy = describe(close, signal, fee)
    benchmark = describe(close, pd.Series(1.0, index=close.index), 0.0)
    yearly = yearly_stats(strategy["frame"], benchmark["frame"])
    buy_dates, sell_dates = trade_dates(close.index, strategy["position"])
    payload = {
        "id": algorithm["id"],
        "name": algorithm["name"],
        "note": algorithm["note"],
        "params": dict(params),
        "indicator": series(indicator),
        "signal": series(signal),
        "position": series(strategy["position"]),
        "buy_dates": buy_dates,
        "sell_dates": sell_dates,
        "equity": series(strategy["equity"]),
        "benchmark_equity": series(benchmark["equity"]),
        "stats": {
            "strategy": stats_payload(strategy),
            "benchmark": stats_payload(benchmark, hide_win_rate=True),
        },
    }
    return payload, yearly, strategy["equity"], benchmark["equity"]


def symbol_header(symbol: str, dataset_id: str, frame: pd.DataFrame, close: pd.Series) -> dict:
    """股票 JSON 里与算法无关的行情部分。只在文件不存在时写。"""
    aligned = frame.set_index("date").reindex(close.index)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "symbol": symbol,
        "price_field": PRICE_FIELD,
        "dates": [timestamp.strftime("%Y-%m-%d") for timestamp in close.index],
        "open": series(aligned.get("open")),
        "high": series(aligned.get("high")),
        "low": series(aligned.get("low")),
        "close": series(close),
        "volume": series(aligned.get("volume")),
    }


def merge_symbol(
    path: Path,
    symbol: str,
    dataset_id: str,
    frame: pd.DataFrame,
    close: pd.Series,
    payload: dict,
) -> dict:
    """把一只股票的算法条目合并进已有 JSON，保留别的算法。

    合并规则用仓库已有的 `_write_merged_json`：同名 id 覆盖，其余条目原样保留。
    """
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        dates = record.get("dates") or []
        if len(dates) != len(close):
            raise ValueError(
                f"{path.name} 已有 {len(dates)} 个日期，与行情 {len(close)} 不一致，拒绝合并"
            )
    else:
        record = symbol_header(symbol, dataset_id, frame, close)
    merged = {**record, "algorithms": {payload["id"]: payload}}
    write_merged(path, merged)
    return merged


def merge_index(path: Path, meta: dict, entry: dict) -> dict:
    """把算法条目写进 index.json。同 id 原位替换，其余条目与顺序保留。"""
    index = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    index["schema_version"] = SCHEMA_VERSION
    for key in ("dataset_id", "name", "source", "price_field"):
        index[key] = meta[key]
    if "symbols" in meta:
        index["symbols"] = list(meta["symbols"])
    entries = list(index.get("algorithms") or [])
    for position, item in enumerate(entries):
        if item.get("id") == entry["id"]:
            entries[position] = entry
            break
    else:
        entries.append(entry)
    index["algorithms"] = entries
    write_merged(path, index, catalog=True)
    return index


def build_history(equities: dict[str, pd.Series], benchmarks: dict[str, pd.Series]) -> dict:
    """第五页的日度净值：各股净值等权平均。值是净值水平，不是收益率。

    默认用各股共同的交易日，避免美股休市的日子只有外汇在动、把组合曲线拽偏。
    成分股上市时间差太多（共同交易日不足六成）时退回并集并向前填充。
    """
    strategy = pd.concat(equities, axis=1).sort_index()
    benchmark = pd.concat(benchmarks, axis=1).sort_index()
    common = strategy.dropna(how="any").index
    if len(common) >= 0.6 * len(strategy.index):
        strategy = strategy.loc[common]
        benchmark = benchmark.loc[common]
    else:
        strategy = strategy.ffill()
        benchmark = benchmark.ffill()
    return {
        "dates": [timestamp.strftime("%Y-%m-%d") for timestamp in strategy.index],
        "strategy": series(strategy.mean(axis=1)),
        "benchmark": series(benchmark.mean(axis=1)),
    }


def export_dataset(
    algorithm: dict,
    build: FrameBuilder,
    data_dir: Path,
    dataset_id: str,
    dataset_name: str,
    symbol_order: Iterable[str] | None = None,
    with_history: bool = True,
) -> Path | None:
    """把一个数据集里所有股票的结果合并进 outdata/<dataset_id>/。"""
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    data_dir = data_dir.resolve()
    frames = load_frames(data_dir, list(symbol_order or ()))
    if not frames:
        return None
    out_dir = OUT_ROOT / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    by_year: dict[str, dict[str, dict]] = {}
    equities: dict[str, pd.Series] = {}
    benchmarks: dict[str, pd.Series] = {}
    for symbol, frame in frames.items():
        close = prepare_close(frame.loc[:, ["date", PRICE_FIELD]].rename(columns={PRICE_FIELD: "close"}))
        payload, yearly, equity, benchmark_equity = build(close)
        merge_symbol(out_dir / f"{symbol}.json", symbol, dataset_id, frame, close, payload)
        for year, stats in yearly.items():
            by_year.setdefault(year, {})[symbol] = stats
        equities[symbol] = equity
        benchmarks[symbol] = benchmark_equity

    entry = dict(algorithm)
    entry["years"] = {
        year: overview_year(list(frames), rows) for year, rows in sorted(by_year.items())
    }
    if with_history:
        entry["history"] = build_history(equities, benchmarks)

    meta = {
        "dataset_id": dataset_id,
        "name": dataset_name,
        "source": data_dir.relative_to(ROOT).as_posix(),
        "price_field": PRICE_FIELD,
        "symbols": list(frames),
    }
    merge_index(out_dir / "index.json", meta, entry)
    return out_dir
