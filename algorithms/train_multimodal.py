"""用纳斯达克100行情和 10-K 因子训练多模态 Transformer。

运行：python -m algorithms.train_multimodal

时间切分按决策日，标签不能跨过切分点：

- 实验集：训练集加验证集。训练集标签收盘价落在 2022-12-30 及以前。
  验证集决策日晚于训练截止日，且标签收盘价落在 2024-12-31 及以前。早停只用验证集。
- 测试集：决策日晚于 2024-12-31，直到最新行情。不参与训练，也不参与早停。

只使用 10-K。10-Q 的利润和现金流是单季口径，不能和年报混在同一次 as-of 里。
公告日当天可用，与现有因子回测一致。已去掉 GOOG，只留 GOOGL。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import numpy as np
import pandas as pd

from algorithms.multimodal_transformer import (
    RankICLoss,
    StockMultimodalDataset,
    config_for_dataset,
    make_loader,
    run_epoch,
    split_indices,
    MultimodalReturnTransformer,
)

ROOT = Path(__file__).resolve().parents[1]
PRICE_PATH = ROOT / "data" / "nasdaq100" / "nasdaq100.csv"
FACTOR_PATH = ROOT / "data" / "factors" / "factors_nq100.parquet"
OUT_DIR = ROOT / "outdata" / "multimodal"

TRAIN_END = "2022-12-30"
VAL_END = "2024-12-31"
SKIP_TICKERS = frozenset({"NDX", "GOOG"})
JUMP_LIMIT = 0.5
WINDOW = 20
HORIZON = 5
MIN_NAMES = 20


def load_prices(path: Path = PRICE_PATH) -> pd.DataFrame:
    """读成长表。单日涨跌超过 50% 的 K 线整根去掉，避免拆股或坏点进入标签。"""
    frame = pd.read_csv(path)
    frame = frame.rename(columns={column: str(column).strip() for column in frame.columns})
    frame = frame.rename(columns={"Ticker": "ticker", "Date": "date"})
    frame["ticker"] = frame["ticker"].astype(str).str.strip()
    frame = frame[~frame["ticker"].isin(SKIP_TICKERS)].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ("Open", "High", "Low", "Close", "Volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    frame = frame.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    previous = frame.groupby("ticker", sort=False)["close"].shift(1)
    jump = (frame["close"] / previous - 1.0).abs() > JUMP_LIMIT
    frame.loc[jump.fillna(False), ["open", "high", "low", "close", "volume"]] = np.nan
    return frame.reset_index(drop=True)


def load_annual_factors(path: Path = FACTOR_PATH) -> pd.DataFrame:
    factors = pd.read_parquet(path)
    annual = factors[factors["form"] == "10-K"].copy()
    if annual.empty:
        raise ValueError("因子表里没有 10-K")
    return annual.reset_index(drop=True)


def _span(dates: pd.DatetimeIndex) -> dict[str, str | int]:
    if len(dates) == 0:
        return {"samples": 0, "start": "", "end": "", "days": 0}
    return {
        "samples": int(len(dates)),
        "start": str(dates.min().date()),
        "end": str(dates.max().date()),
        "days": int(dates.normalize().nunique()),
    }


def build_split(dataset: StockMultimodalDataset, train_end: str, val_end: str) -> dict:
    parts = split_indices(dataset, train_end=train_end, val_end=val_end)
    used = set(parts["train"]) | set(parts["val"]) | set(parts["test"])
    embargo = [index for index in range(len(dataset)) if index not in used]
    train_span = _span(dataset.decision_dates[parts["train"]])
    val_span = _span(dataset.decision_dates[parts["val"]])
    test_span = _span(dataset.decision_dates[parts["test"]])
    summary = {
        "train_end": train_end,
        "val_end": val_end,
        "window": dataset.window,
        "horizon": dataset.horizon,
        "announcement_lag": dataset.announcement_lag,
        "form": "10-K",
        "experiment": {"train": train_span, "val": val_span},
        "test": test_span,
        "embargo": _span(dataset.decision_dates[embargo]),
        "train": train_span,
        "val": val_span,
        "unused": _span(dataset.decision_dates[embargo + parts["test"]]),
        "train_indices": parts["train"],
        "val_indices": parts["val"],
        "test_indices": parts["test"],
    }
    return summary


def _public_summary(summary: dict) -> dict:
    hidden = {"train_indices", "val_indices", "test_indices"}
    return {key: value for key, value in summary.items() if key not in hidden}


def peel_early_stop(
    dataset: StockMultimodalDataset,
    train_indices: list[int],
    ratio: float = 0.2,
) -> tuple[list[int], list[int]]:
    """从训练决策日的末尾切出早停用的验证段。标签跨进这段的样本两边都不用。"""
    if not 0 < ratio < 1:
        raise ValueError("早停比例要落在 (0, 1) 内")
    dates = dataset.decision_dates
    days = sorted({pd.Timestamp(dates[index]).normalize() for index in train_indices})
    if len(days) < 2:
        raise RuntimeError("训练决策日太少，无法留出早停段")
    hold_count = min(max(int(round(len(days) * ratio)), 1), len(days) - 1)
    hold_start = days[-hold_count]
    fit: list[int] = []
    monitor: list[int] = []
    for index in train_indices:
        end = dataset.samples[index][1]
        day = pd.Timestamp(dates[index]).normalize()
        label_day = pd.Timestamp(dataset.calendar[end + dataset.horizon]).normalize()
        if day >= hold_start:
            monitor.append(index)
        elif label_day >= hold_start:
            continue
        else:
            fit.append(index)
    if not fit or not monitor:
        raise RuntimeError("训练段内部没有同时留下训练和早停样本")
    return fit, monitor


def output_dir(dataset_id: str) -> Path:
    if dataset_id == "nasdaq-xbrl":
        return ROOT / "outdata" / "multimodal"
    return ROOT / "outdata" / "multimodal" / dataset_id


def verify_nasdaq_xbrl() -> None:
    """已有权重的训练截止日和验证截止日必须和实验集配置一致。"""
    split_path = ROOT / "outdata" / "multimodal" / "split.json"
    weights = ROOT / "outdata" / "multimodal" / "best.pt"
    if not split_path.exists() or not weights.exists():
        raise RuntimeError("纳斯达克 XBRL 没有已保存的实验集权重")
    saved = json.loads(split_path.read_text(encoding="utf-8"))
    if saved["train"]["end"] != "2022-12-22" or saved["val"]["end"] != "2024-12-23":
        raise RuntimeError("纳斯达克 XBRL 已有权重的切分不是实验集，不能直接沿用")
    print("纳斯达克XBRL 已有实验集权重，训练到 2022-12-22，验证到 2024-12-23，不重训。", flush=True)


def train(args: argparse.Namespace) -> None:
    from algorithms.transformer_splits import DATASETS, empty_factors, listed_symbols, load_price_file, ratio_train_end

    spec = next((item for item in DATASETS if item["id"] == args.dataset), None)
    if spec is None:
        raise RuntimeError(f"未知数据集 {args.dataset}")
    if args.dataset == "healthcare-us-10y" and args.min_names == MIN_NAMES:
        args.min_names = 8
    out_dir = output_dir(args.dataset)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"数据集 {args.dataset}，设备 {device}", flush=True)

    prices = load_price_file(spec["path"], spec["skip"], listed_symbols(spec["id"]))
    factors = load_annual_factors() if spec["fundamentals"] else empty_factors()
    print(
        f"行情 {prices['ticker'].nunique()} 只，"
        f"{prices['date'].min().date()} 至 {prices['date'].max().date()}；"
        f"基本面 {'10-K' if spec['fundamentals'] else '缺失'}",
        flush=True,
    )
    dataset = StockMultimodalDataset(
        prices,
        factors,
        window=args.window,
        horizon=args.horizon,
        announcement_lag=0,
        require_fundamental=bool(spec["fundamentals"]),
    )
    audit = dataset.pit_audit()
    if not audit.empty and bool(audit["leak"].any()):
        raise RuntimeError("训练集构造时发现公告日之后的财报被用到了")

    if spec.get("train_ratio"):
        train_end = ratio_train_end(dataset.calendar, dataset.window, dataset.horizon, float(spec["train_ratio"]))
        summary = build_split(dataset, train_end, train_end)
        fit, monitor = peel_early_stop(dataset, summary["train_indices"])
        summary["train_indices"] = fit
        summary["val_indices"] = monitor
        summary["experiment"]["train"] = _span(dataset.decision_dates[fit])
        summary["experiment"]["val"] = _span(dataset.decision_dates[monitor])
        summary["train"] = summary["experiment"]["train"]
        summary["val"] = summary["experiment"]["val"]
    else:
        summary = build_split(dataset, args.train_end, args.val_end)
    test_ids = set(summary["test_indices"])
    if test_ids & set(summary["train_indices"]) or test_ids & set(summary["val_indices"]):
        raise RuntimeError("测试集被放进了训练或早停")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "split.json").write_text(
        json.dumps(_public_summary(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    train_info = summary["train"]
    val_info = summary["val"]
    test_info = summary["test"]
    embargo_info = summary["embargo"]
    print(
        f"训练 {train_info['samples']} 个样本，{train_info['days']} 个决策日，"
        f"{train_info['start']} 至 {train_info['end']}"
    )
    print(
        f"验证 {val_info['samples']} 个样本，{val_info['days']} 个决策日，"
        f"{val_info['start']} 至 {val_info['end']}。早停只用这一段。"
    )
    print(
        f"测试 {test_info['samples']} 个样本，{test_info['days']} 个决策日，"
        f"{test_info['start']} 至 {test_info['end']}。不参与训练，也不参与早停。"
    )
    print(
        f"隔离带 {embargo_info['samples']} 个样本，"
        f"{embargo_info['start']} 至 {embargo_info['end']}。"
    )
    if train_info["samples"] == 0 or val_info["samples"] == 0:
        raise RuntimeError("训练集或验证集是空的，请放宽 min_names 或检查日期")

    train_loader = make_loader(
        dataset,
        summary["train_indices"],
        shuffle=True,
        min_names=args.min_names,
        seed=args.seed,
    )
    val_loader = make_loader(
        dataset,
        summary["val_indices"],
        shuffle=False,
        min_names=args.min_names,
    )
    print(f"训练 batch {len(train_loader)}，验证 batch {len(val_loader)}，每日至少 {args.min_names} 只股票")

    model = MultimodalReturnTransformer(config_for_dataset(dataset)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = RankICLoss()
    history: list[dict[str, float | int | str]] = []
    best_ic = -math.inf
    stale = 0
    start_epoch = 1
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    config = config_for_dataset(dataset).__dict__

    if args.resume:
        resume_path = last_path if last_path.exists() else best_path
        if resume_path.exists():
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_ic = float(checkpoint.get("best_ic", checkpoint.get("val_rank_ic", -math.inf)))
            stale = int(checkpoint.get("stale", 0))
            if "history" in checkpoint:
                history = list(checkpoint["history"])
            elif (out_dir / "history.csv").exists():
                previous = pd.read_csv(out_dir / "history.csv")
                previous = previous[previous["epoch"] <= int(checkpoint["epoch"])]
                history = previous.to_dict("records")
            print(
                f"从 {resume_path.name} 的第 {checkpoint['epoch']} 轮接着训练，"
                f"当前最好的验证 Rank IC 是 {best_ic:.4f}",
                flush=True,
            )
        else:
            print("没有找到已保存的权重，从头训练", flush=True)

    if start_epoch > args.epochs:
        print(f"已经完成 {args.epochs} 轮，没有剩余轮次", flush=True)
        return

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device=device,
            log_every=args.log_every,
        )
        val_metrics = run_epoch(model, val_loader, criterion, optimizer=None, device=device)
        history.append({"epoch": epoch, "split": "train", **train_metrics})
        history.append({"epoch": epoch, "split": "val", **val_metrics})
        print(
            f"epoch {epoch}  train loss={train_metrics['loss']:.4f} "
            f"rank_ic={train_metrics['rank_ic']:.4f} batches={train_metrics['batches']}",
            flush=True,
        )
        print(
            f"epoch {epoch}  val   loss={val_metrics['loss']:.4f} "
            f"rank_ic={val_metrics['rank_ic']:.4f} batches={val_metrics['batches']}",
            flush=True,
        )
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        val_ic = float(val_metrics["rank_ic"])
        if math.isfinite(val_ic) and val_ic > best_ic:
            best_ic = val_ic
            stale = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "val_rank_ic": val_ic,
                    "best_ic": best_ic,
                    "stale": stale,
                    "history": history,
                    "split": _public_summary(summary),
                },
                best_path,
            )
            print(f"已保存验证 Rank IC {val_ic:.4f} 的权重到 {best_path}", flush=True)
        else:
            stale += 1
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "val_rank_ic": val_ic,
                "best_ic": best_ic,
                "stale": stale,
                "history": history,
                "split": _public_summary(summary),
            },
            last_path,
        )
        if stale >= args.patience:
            print(f"验证 Rank IC 连续 {args.patience} 轮没有提高，停止训练", flush=True)
            break

    print(
        f"最好的验证 Rank IC 是 {best_ic:.4f}" if math.isfinite(best_ic) else "验证 Rank IC 没有有效数值",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练多模态收益率 Transformer")
    parser.add_argument("--dataset", default="nasdaq-xbrl", help="config/transformer_splits.json 里的数据集编号")
    parser.add_argument("--queue", action="store_true", help="核对纳斯达克XBRL后，按顺序训练其余数据集")
    parser.add_argument("--train-end", default=TRAIN_END)
    parser.add_argument("--val-end", default=VAL_END)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--min-names", type=int, default=MIN_NAMES)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--resume", action="store_true", help="从 last.pt 或 best.pt 接着训练")
    return parser.parse_args()


QUEUE = (
    "nasdaq100",
    "tech-us-10y",
    "manufacturing-us-10y",
    "healthcare-us-10y",
    "us-agri",
)


def train_queue() -> None:
    verify_nasdaq_xbrl()
    for dataset_id in QUEUE:
        print(f"开始训练 {dataset_id}", flush=True)
        args = parse_args()
        args.dataset = dataset_id
        args.queue = False
        train(args)
        print(f"完成训练 {dataset_id}", flush=True)


if __name__ == "__main__":
    args = parse_args()
    try:
        if args.queue:
            train_queue()
        else:
            train(args)
    except Exception:
        import traceback

        log_dir = output_dir(args.dataset)
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "train_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
