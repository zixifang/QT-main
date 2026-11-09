"""逐个把七个指数增强组合和 RSI 写进行情数据集。

不改基本面长表，也不改纳斯达克XBRL里已有的 F-Score 和 Transformer。
已有的其他算法保留。没有指数文件的数据集，七个组合对样本等权。

运行：python -m algorithms.run_listed
"""

from __future__ import annotations

import json
from pathlib import Path

from algorithms.index_enhance import export_dataset, export_universe
from algorithms.rsi import export_dataset as export_rsi

ROOT = Path(__file__).resolve().parents[1]

JOBS = (
    ("nasdaq100", "纳斯达克100", "ndx"),
    ("tech-us-10y", "科技美股近十年日线", "equal"),
    ("manufacturing-us-10y", "美国制造业近十年日线", "equal"),
    ("healthcare-us-10y", "美国医疗健康近十年日线", "equal"),
    ("us-agri", "美国农业股（28 只，5 年）", "equal"),
)

KEEP = {
    "healthcare-us-10y": {"macd", "random_forest"},
    "us-agri": {"mom_rotation"},
}

SERIES_KEYS = ("indicator", "signal", "position", "equity", "benchmark_equity")


def _check(dataset_id: str) -> None:
    folder = ROOT / "outdata" / dataset_id
    index = json.loads((folder / "index.json").read_text(encoding="utf-8"))
    ids = [item.get("id") for item in index.get("algorithms") or []]
    print("ALGOS", dataset_id, ",".join(str(item) for item in ids), flush=True)
    missing = KEEP.get(dataset_id, set()) - set(ids)
    if missing:
        raise SystemExit(f"CHECK_FAIL {dataset_id} dropped {', '.join(sorted(missing))}")
    wanted = {"rsi", *(f"enhance-g{number}" for number in range(1, 8))}
    absent = wanted - set(ids)
    if absent:
        raise SystemExit(f"CHECK_FAIL {dataset_id} missing {', '.join(sorted(absent))}")
    symbol = index["symbols"][0]
    payload = json.loads((folder / f"{symbol}.json").read_text(encoding="utf-8"))
    length = len(payload.get("dates") or [])
    for algo_id, algo in (payload.get("algorithms") or {}).items():
        for key in SERIES_KEYS:
            if key in algo and len(algo[key]) != length:
                raise SystemExit(f"CHECK_FAIL LENGTH {dataset_id} {symbol} {algo_id} {key}")
    print("LENGTH_OK", dataset_id, symbol, length, flush=True)


def main() -> None:
    for folder, name, kind in JOBS:
        print(f"START {folder}", flush=True)
        data_dir = ROOT / "data" / folder
        if kind == "ndx":
            export_dataset(data_dir, ROOT / "outdata" / folder)
        else:
            export_universe(data_dir, folder, name)
        written = export_rsi(data_dir, folder, name)
        if written is None:
            raise SystemExit(f"{folder} 没有写入 RSI")
        _check(folder)
        print(f"DONE {folder}", flush=True)


if __name__ == "__main__":
    main()
