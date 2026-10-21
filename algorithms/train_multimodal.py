"""用纳斯达克100行情和 10-K 因子训练多模态 Transformer。

运行：python -m algorithms.train_multimodal

时间切分按决策日，标签不能跨过切分点：

- 训练集：标签收盘价落在 2022-12-30 及以前。
- 验证集：决策日晚于训练截止日，且标签收盘价落在 2024-12-31 及以前。
- 2025-01-01 之后的样本不参与训练，也不参与早停。

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
    used = set(parts["train"]) | set(parts["val"])
    unused = [index for index in range(len(dataset)) if index not in used]
    summary = {
        "train_end": train_end,
        "val_end": val_end,
        "window": dataset.window,
        "horizon": dataset.horizon,
        "announcement_lag": dataset.announcement_lag,
        "form": "10-K",
        "train": _span(dataset.decision_dates[parts["train"]]),
        "val": _span(dataset.decision_dates[parts["val"]]),
        "unused": _span(dataset.decision_dates[unused]),
        "train_indices": parts["train"],
        "val_indices": parts["val"],
    }
    return summary


def _public_summary(summary: dict) -> dict:
    hidden = {"train_indices", "val_indices"}
    return {key: value for key, value in summary.items() if key not in hidden}


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备 {device}")

    prices = load_prices()
    factors = load_annual_factors()
    print(
        f"行情 {prices['ticker'].nunique()} 只，"
        f"{prices['date'].min().date()} 至 {prices['date'].max().date()}；"
        f"10-K {factors['ticker'].nunique()} 只"
    )
    dataset = StockMultimodalDataset(
        prices,
        factors,
        window=args.window,
        horizon=args.horizon,
        announcement_lag=0,
        require_fundamental=True,
    )
    audit = dataset.pit_audit()
    if not audit.empty and bool(audit["leak"].any()):
        raise RuntimeError("训练集构造时发现公告日之后的财报被用到了")

    summary = build_split(dataset, args.train_end, args.val_end)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "split.json").write_text(
        json.dumps(_public_summary(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    train_info = summary["train"]
    val_info = summary["val"]
    unused_info = summary["unused"]
    print(
        f"训练 {train_info['samples']} 个样本，{train_info['days']} 个决策日，"
        f"{train_info['start']} 至 {train_info['end']}"
    )
    print(
        f"验证 {val_info['samples']} 个样本，{val_info['days']} 个决策日，"
        f"{val_info['start']} 至 {val_info['end']}"
    )
    print(
        f"未使用 {unused_info['samples']} 个样本，"
        f"{unused_info['start']} 至 {unused_info['end']}。"
        "其中包含训练和验证之间的隔离带，以及验证截止日之后的行情。"
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
    best_path = OUT_DIR / "best.pt"
    last_path = OUT_DIR / "last.pt"
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
            elif (OUT_DIR / "history.csv").exists():
                previous = pd.read_csv(OUT_DIR / "history.csv")
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
        pd.DataFrame(history).to_csv(OUT_DIR / "history.csv", index=False)
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


if __name__ == "__main__":
    try:
        train(parse_args())
    except Exception:
        import traceback

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "train_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
