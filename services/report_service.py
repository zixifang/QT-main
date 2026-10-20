"""把覆盖度、检验和回测写进 HTML / PDF。PDF 中文用微软雅黑。"""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

from services.backtest_service import BacktestService
from services.config_store import public_config, validate_config
from services.data_service import DataService
from services.eval_service import EvalService
from services.factor_service import FactorService, factor_settings
from services.paths import ROOT
from services.runtime import check_cancel, report
from factors.catalog import catalog

CAVEATS = (
    "手填股票池有幸存者偏差。没有历史成分快照时，不能把结果当成指数增强。",
    "财务数据要在公告日的下一交易日才能用。当前没有这部分数据，相关因子为空。",
    "价格用未复权 Close。前复权会被拒绝。没有分红拆股因子，不能做后复权。",
    "单日涨跌超过阈值的收益记为 0，原始收盘价不改。这不是复权，真跌破阈值也会被记成 0。",
    "分红没有计入股票收益，也没有计入 NDX。",
    "训练段是起始日往后数的自然年，默认 7 年；剩下的是验证段。动量预热会吃掉训练段开头。",
    "这里没有按利率环境把因子拆开，利率变化时因子可能衰减。",
    "预期 EPS 没有覆盖，预期增长因子保持降级。",
    "缓存键包含文件修改时间和参数。换了公式后要清缓存，否则可能读到旧结果。",
    "样本只有几只股票时，IC 和分层只说明流程能跑，不是策略有效的证据。",
    "跟踪误差是事后描述。优化约束是单名上限和向等权的分散惩罚。",
)

SCHEME_NAME = {"equal": "等权", "ic": "IC 加权", "max_icir": "最大 ICIR"}


def _pct(value) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.2f}%"


def _num(value) -> str:
    if value is None:
        return "—"
    return f"{float(value):.2f}"


def _font() -> FontProperties | None:
    path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if path.exists():
        return FontProperties(fname=str(path))
    return None


class ReportService:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else ROOT

    def export(self, config: dict, kind: str = "html", *, refresh: bool = False, cancel=None, progress=None) -> dict:
        cfg = validate_config(config)
        check_cancel(cancel)
        report(progress, 8, "汇总覆盖度")
        coverage = DataService(self.root).coverage(cfg, cancel=cancel, progress=progress)
        check_cancel(cancel)
        factor_rows = []
        factors = FactorService(self.root)
        enabled = []
        for item in catalog():
            if item["role"] == "risk":
                continue
            if factor_settings(cfg, item).get("enabled"):
                enabled.append(item)
        for item in enabled:
            check_cancel(cancel)
            factor_rows.append(factors.compute(cfg, item["id"], refresh=refresh, cancel=cancel, progress=progress))
        evals = []
        evaluator = EvalService(self.root)
        for item in enabled:
            check_cancel(cancel)
            evals.append(evaluator.single(cfg, item["id"], refresh=refresh, cancel=cancel, progress=progress))
        check_cancel(cancel)
        report(progress, 88, "运行回测并写报告")
        backtest = BacktestService(self.root).run(cfg, refresh=refresh, cancel=cancel, progress=progress)
        lines = build_lines(cfg, coverage, factor_rows, evals, backtest)
        folder = self.root / "reports"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = ""
        pdf_path = ""
        warnings = list(backtest["warnings"])
        if kind in {"html", "both"}:
            html_file = folder / f"ndx_enhance_{stamp}.html"
            html_file.write_text(to_html(lines), encoding="utf-8")
            html_path = str(html_file)
        if kind in {"pdf", "both"}:
            pdf_file = folder / f"ndx_enhance_{stamp}.pdf"
            try:
                write_pdf(pdf_file, lines, backtest.get("nav") or [])
                pdf_path = str(pdf_file)
            except OSError as exc:
                warnings.append(f"PDF 没有写成：{exc}")
        report(progress, 100, "报告已写出")
        return {
            "html": html_path,
            "pdf": pdf_path,
            "summary": "\n".join(lines),
            "warnings": warnings,
            "backtest": backtest,
        }


def build_lines(config: dict, coverage: dict, factors: list[dict], evals: list[dict], backtest: dict) -> list[str]:
    shown = public_config(config)
    lines = [
        "纳斯达克100指数增强样本报告",
        f"区间 {shown['start']} 至 {shown['end']}，基准 {shown['benchmark']}，复权 {shown['adjust']}",
        f"股票池 {'、'.join(shown['universe'])}，来源 {shown['universe_mode']}，权重 {SCHEME_NAME.get(shown['weight_scheme'], shown['weight_scheme'])}",
        "",
        "覆盖度",
    ]
    for row in coverage["rows"]:
        lines.append(
            f"{row['symbol']}  {row['rows']} 行  {row['first']} → {row['last']}  跳空 {row['jumps']}  来源 {row['source'] or '—'}  {row['warning']}"
        )
    lines.append("")
    lines.extend(coverage["warnings"])
    lines.append("")
    lines.append("因子")
    for row in factors:
        latest = "、".join(
            f"{item['symbol']} {item['score']:.3f}" if item["score"] is not None else f"{item['symbol']} —"
            for item in row.get("latest") or []
        )
        state = "降级" if row.get("degraded") else "可用"
        lines.append(f"{row['name']}（{state}，缓存 {row.get('cache', '—')}） {latest}")
        for warning in row.get("warnings") or []:
            lines.append(f"  {warning}")
    lines.append("")
    lines.append("单因子检验（月末 IC，预测期见配置；分层持有到下一次再平衡）")
    for row in evals:
        lines.append(
            f"{row['name']}  IC均值 {_num(row['ic_mean'])}  ICIR {_num(row['icir'])}  t {_num(row['ic_t'])}  "
            f"样本 {row['n_ic']}  分组 {row['n_buckets']}  换手 {_pct(row['turnover'])}"
        )
        for warning in row.get("warnings") or []:
            lines.append(f"  {warning}")
    lines.append("")
    lines.append("回测")
    lines.append(f"训练截止 {backtest['splits']['train_end']}，验证从 {backtest['splits']['valid_start']} 开始")
    for label, key in (("全样本", "full"), ("训练", "train"), ("验证", "valid")):
        stats = backtest[key]
        lines.append(
            f"{label}  累计 {_pct(stats['cumulative'])}  年化 {_pct(stats['annualized'])}  "
            f"波动 {_pct(stats['volatility'])}  夏普 {_num(stats['sharpe'])}  回撤 {_pct(stats['max_drawdown'])}  "
            f"超额 {_pct(stats['excess_cumulative'])}  跟踪误差 {_pct(stats['tracking_error'])}  "
            f"单边换手 {_pct(stats['avg_turnover'])}  成本 {_pct(stats['total_cost'])}"
        )
    lines.append("分年")
    for row in backtest["yearly"]:
        lines.append(f"{row['year']}  策略 {_pct(row['strategy'])}  基准 {_pct(row['benchmark'])}  超额 {_pct(row['excess'])}")
    lines.append("")
    lines.append("已知限制")
    lines.extend(CAVEATS)
    return lines


def to_html(lines: list[str]) -> str:
    body = "\n".join(f"<p>{escape(line) if line else '&nbsp;'}</p>" for line in lines)
    return (
        "<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        "<title>纳斯达克100指数增强样本</title>"
        "<style>body{font-family:'Microsoft YaHei',sans-serif;margin:32px;line-height:1.55;}</style>"
        f"</head><body>{body}</body></html>"
    )


def write_pdf(path: Path, lines: list[str], nav: list[dict]) -> None:
    font = _font()
    with PdfPages(path) as pdf:
        if nav:
            figure = Figure(figsize=(11, 6))
            axis = figure.add_subplot(111)
            dates = [item["date"] for item in nav]
            axis.plot(dates, [item["strategy"] for item in nav], label="组合")
            axis.plot(dates, [item["benchmark"] for item in nav], label="NDX")
            if font is not None:
                axis.set_title("净值", fontproperties=font)
                axis.legend(prop=font)
            else:
                axis.set_title("nav")
                axis.legend()
            figure.autofmt_xdate()
            figure.tight_layout()
            pdf.savefig(figure)
        chunk = 36
        for start in range(0, len(lines), chunk):
            figure = Figure(figsize=(8.27, 11.69))
            axis = figure.add_subplot(111)
            axis.axis("off")
            text = "\n".join(lines[start : start + chunk])
            if font is not None:
                axis.text(0.02, 0.98, text, va="top", fontsize=9, fontproperties=font)
            else:
                axis.text(0.02, 0.98, text, va="top", fontsize=9)
            pdf.savefig(figure)
