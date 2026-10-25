#!/usr/bin/env python3
"""校验 outdata/recent-us-2024-2026 下的 JSON 是否符合仓库契约。

运行：python verify_outputs.py
不依赖 pandas；只用 Python 标准库。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outdata" / "recent-us-2024-2026"
ALGORITHMS = ("macd_trend", "momentum_12_1", "donchian_breakout")
STAT_KEYS = (
    "cumulative_return",
    "annualized_return",
    "volatility",
    "sharpe",
    "max_drawdown",
    "calmar",
    "trades",
    "win_rate",
)
PRICE_FIELDS = ("open", "high", "low", "close", "volume", "return_index")
ALGO_FIELDS = ("indicator", "signal", "position", "equity", "benchmark_equity")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def main() -> int:
    problems: list[str] = []
    if not DATA.is_dir():
        print(f"缺少目录：{DATA}")
        return 1

    index = load(DATA / "index.json")
    if index.get("schema_version") != 1:
        problems.append("index.json: schema_version 不是 1")
    if index.get("dataset_id") != "recent-us-2024-2026":
        problems.append("index.json: dataset_id 不正确")
    symbols = index.get("symbols") or []
    if len(symbols) != 20:
        problems.append(f"index.json: symbols 数量为 {len(symbols)}，预期 20")
    algorithms = [item.get("id") for item in index.get("algorithms") or []]
    if algorithms != list(ALGORITHMS):
        problems.append(f"index.json: algorithms 为 {algorithms}")

    rows = 0
    for symbol in symbols:
        path = DATA / f"{symbol}.json"
        if not path.exists():
            problems.append(f"{symbol}: 缺少 JSON")
            continue
        payload = load(path)
        dates = payload.get("dates") or []
        count = len(dates)
        rows = max(rows, count)
        if count == 0:
            problems.append(f"{symbol}: dates 为空")
            continue
        if dates != sorted(dates) or len(set(dates)) != count:
            problems.append(f"{symbol}: dates 未按时间升序或不唯一")
        for field in PRICE_FIELDS:
            values = payload.get(field) or []
            if len(values) != count:
                problems.append(f"{symbol}: {field} 长度 {len(values)} != {count}")
                continue
            if any(value is not None and not is_number(value) for value in values):
                problems.append(f"{symbol}: {field} 含非数字")
        close = payload.get("close") or []
        if any(value is None or value <= 0 for value in close):
            problems.append(f"{symbol}: close 含空值或非正数")
        for i in range(1, len(close)):
            if close[i - 1] and abs(close[i] / close[i - 1] - 1.0) > 0.5 + 1e-9:
                problems.append(f"{symbol}: {dates[i]} 单日涨跌超过 50%")
                break
        for algorithm_id in ALGORITHMS:
            block = (payload.get("algorithms") or {}).get(algorithm_id)
            if not block:
                problems.append(f"{symbol}: 缺少 {algorithm_id}")
                continue
            for field in ALGO_FIELDS:
                values = block.get(field) or []
                if len(values) != count:
                    problems.append(f"{symbol}/{algorithm_id}: {field} 长度 {len(values)} != {count}")
            signal = block.get("signal") or []
            position = block.get("position") or []
            if any(value not in (0, 1) for value in signal):
                problems.append(f"{symbol}/{algorithm_id}: signal 不是 0/1")
            if any(value not in (0, 1) for value in position):
                problems.append(f"{symbol}/{algorithm_id}: position 不是 0/1")
            for i in range(count):
                expected = signal[i - 1] if i else 0
                if position[i] != expected:
                    problems.append(f"{symbol}/{algorithm_id}: position 未后移一天（第 {i} 行）")
                    break
            day_set = set(dates)
            trades = (block.get("buy_dates") or []) + (block.get("sell_dates") or [])
            if any(day not in day_set for day in trades):
                problems.append(f"{symbol}/{algorithm_id}: 买卖点不在 dates 中")
            stats = block.get("stats") or {}
            for side in ("strategy", "benchmark"):
                block_stats = stats.get(side) or {}
                for key in STAT_KEYS:
                    if key not in block_stats:
                        problems.append(f"{symbol}/{algorithm_id}: stats.{side} 缺少 {key}")
                    elif key != "win_rate" and not is_number(block_stats[key]):
                        problems.append(f"{symbol}/{algorithm_id}: stats.{side}.{key} 不是有限数字")

    paper_dir = DATA / "paper"
    for algorithm_id in ALGORITHMS:
        path = paper_dir / f"{algorithm_id}.json"
        if not path.exists():
            problems.append(f"paper/{algorithm_id}.json 缺失")
            continue
        paper = load(path)
        dates = paper.get("dates") or []
        if paper.get("capital") != 100000:
            problems.append(f"paper/{algorithm_id}: capital 不是 100000")
        for field in ("nav", "benchmark_nav", "cash"):
            values = paper.get(field) or []
            if len(values) != len(dates):
                problems.append(f"paper/{algorithm_id}: {field} 长度不一致")

    print(f"数据集: {index.get('dataset_id')} / {index.get('name')}")
    print(f"股票: {len(symbols)} 只，最长 {rows} 个交易日")
    print(f"算法: {', '.join(algorithms)}")
    print(f"paper 文件: {len(list(paper_dir.glob('*.json'))) if paper_dir.exists() else 0} 个")
    if problems:
        print(f"\n发现 {len(problems)} 个问题：")
        for item in problems[:30]:
            print(" -", item)
        if len(problems) > 30:
            print(f" ... 另有 {len(problems) - 30} 个")
        return 1
    print("\n验证通过：JSON 长度、0/1 仓位、仓位后移、买卖点、统计字段和 paper 账户全部符合契约。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
