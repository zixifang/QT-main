"""读写 config/ndx_enhance.json。界面表单和计算都走这里，不各自记一份。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from services.errors import DataGapError
from services.paths import ROOT

DEFAULT_CONFIG: dict = {
    "universe": ["AAPL", "MSFT", "NVDA"],
    "benchmark": "NDX",
    "start": "2016-01-01",
    "end": "2025-12-31",
    "adjust": "raw",
    "source_order": ["local_csv", "yahoo", "nasdaq_data_link"],
    "api_keys": {"nasdaq_data_link": ""},
    "universe_mode": "manual",
    "rebalance": "M",
    "cost_bps": 10,
    "slippage_bps": 5,
    "max_weight": 0.5,
    "diversify_penalty": 1.0,
    "jump_limit": 0.5,
    "horizon": 21,
    "neutralize": ["beta", "liquidity"],
    "weight_scheme": "equal",
    "train_years": 7,
    "valid_years": 3,
    "beta_lookback": 252,
    "liquidity_lookback": 20,
    "factors": {
        "gross_margin_sharpe": {"enabled": False, "quarters": 12},
        "rd_to_sales": {"enabled": False, "quarters": 12},
        "revenue_growth": {"enabled": False, "quarters": 12},
        "fwd_eps_growth": {"enabled": False},
        "cfo_to_sales": {"enabled": False, "quarters": 4},
        "roe_stability": {"enabled": False, "quarters": 12},
        "peg": {"enabled": False},
        "residual_momentum": {"enabled": True, "lookback": 252, "skip": 21},
        "low_volatility": {"enabled": True, "lookback": 60},
    },
}


def merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def public_config(config: dict) -> dict:
    data = json.loads(json.dumps(config))
    keys = data.get("api_keys") or {}
    data["api_keys"] = {name: "已填写" if str(value).strip() else "" for name, value in keys.items()}
    return data


def validate_config(config: dict) -> dict:
    cfg = merge(DEFAULT_CONFIG, config)
    cfg["universe"] = [str(item).strip().upper() for item in cfg.get("universe", []) if str(item).strip()]
    cfg["benchmark"] = str(cfg.get("benchmark", "NDX")).strip().upper()
    cfg["adjust"] = str(cfg.get("adjust", "raw")).strip()
    cfg["universe_mode"] = "snapshot" if cfg.get("universe_mode") == "snapshot" else "manual"
    cfg["rebalance"] = str(cfg.get("rebalance", "M"))
    cfg["weight_scheme"] = str(cfg.get("weight_scheme", "equal"))
    if not cfg["universe"]:
        raise DataGapError("股票池是空的")
    try:
        start = pd.Timestamp(cfg["start"])
        end = pd.Timestamp(cfg["end"])
    except (TypeError, ValueError) as exc:
        raise DataGapError("开始或结束日期无法识别") from exc
    if start > end:
        raise DataGapError("开始日期晚于结束日期")
    tradable = [symbol for symbol in cfg["universe"] if symbol != cfg["benchmark"]]
    if not tradable:
        raise DataGapError("股票池里只有基准，没有可交易的股票")
    try:
        cap = float(cfg["max_weight"])
        penalty = float(cfg["diversify_penalty"])
        horizon = int(cfg["horizon"])
        jump = float(cfg["jump_limit"])
    except (TypeError, ValueError) as exc:
        raise DataGapError("成本、权重或窗口不是数字") from exc
    if not 0 < cap <= 1:
        raise DataGapError("单名上限要在 0 和 1 之间")
    if cap * len(tradable) < 1 - 1e-9:
        raise DataGapError("单名上限乘以股票数小于 1，约束无解")
    if penalty < 0:
        raise DataGapError("分散惩罚不能为负")
    if horizon < 1:
        raise DataGapError("预测期至少 1 个交易日")
    if jump <= 0:
        raise DataGapError("跳空阈值要大于 0")
    if cfg["rebalance"] not in {"M", "Q"}:
        raise DataGapError("再平衡只支持每月或每季")
    if cfg["weight_scheme"] not in {"equal", "ic", "max_icir"}:
        raise DataGapError("权重方案只能是等权、IC 加权或最大 ICIR")
    mom = cfg["factors"].get("residual_momentum", {})
    if int(mom.get("skip", 0)) < 0:
        raise DataGapError("动量跳过天数不能为负")
    if int(mom.get("lookback", 252)) < 10 or int(cfg["factors"].get("low_volatility", {}).get("lookback", 60)) < 10:
        raise DataGapError("回看窗口至少 10 个交易日")
    cfg["max_weight"] = cap
    cfg["diversify_penalty"] = penalty
    cfg["horizon"] = horizon
    cfg["jump_limit"] = jump
    cfg["train_years"] = int(cfg["train_years"])
    cfg["valid_years"] = int(cfg["valid_years"])
    cfg["beta_lookback"] = int(cfg["beta_lookback"])
    cfg["liquidity_lookback"] = int(cfg["liquidity_lookback"])
    cfg["cost_bps"] = float(cfg["cost_bps"])
    cfg["slippage_bps"] = float(cfg["slippage_bps"])
    return cfg


class ConfigStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else ROOT

    @property
    def path(self) -> Path:
        return self.root / "config" / "ndx_enhance.json"

    def load(self) -> dict:
        if not self.path.exists():
            return json.loads(json.dumps(DEFAULT_CONFIG))
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return json.loads(json.dumps(DEFAULT_CONFIG))
        if not isinstance(raw, dict):
            return json.loads(json.dumps(DEFAULT_CONFIG))
        return merge(DEFAULT_CONFIG, raw)

    def save(self, config: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
