"""只读取 outdata 中的 JSON。这里不做指标计算。"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDATA_DIR = ROOT / "outdata"


def list_datasets(root: Path = OUTDATA_DIR) -> list[dict]:
    if not root.exists():
        return []
    datasets = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        index_path = folder / "index.json"
        if not index_path.exists():
            continue
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        payload["_dir"] = folder
        datasets.append(payload)
    return datasets


def load_symbol(dataset: dict, symbol: str) -> dict:
    path = Path(dataset["_dir"]) / f"{symbol}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))
