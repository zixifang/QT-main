"""进度、取消和数值清洗。长任务由界面轮询，不在这里碰窗口控件。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from services.errors import Cancelled

Progress = Callable[[int, str], None]
Cancel = Callable[[], bool]


def report(progress: Progress | None, pct: int, message: str) -> None:
    if progress is not None:
        progress(int(pct), message)


def check_cancel(cancel: Cancel | None) -> None:
    if cancel is not None and cancel():
        raise Cancelled("已取消")


def num(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def digest(*parts: str) -> str:
    import hashlib

    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()
