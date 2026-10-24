"""检查 pandas 与 numpy 是否可用，并用一段合成价格跑通回测。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.core import prepare_close, run_backtest, summarize


def main() -> None:
    print(f"pandas {pd.__version__}")
    print(f"numpy {np.__version__}")
    dates = pd.bdate_range("2024-01-02", periods=6)
    frame = pd.DataFrame({"date": dates, "close": [10, 10, 11, 12, 11, 13]})
    close = prepare_close(frame)
    position = pd.Series(1.0, index=close.index)
    result = run_backtest(close, position, fee=0.0)
    metrics = summarize(result)
    print(
        "cumulative_return={cumulative_return:.4f} max_drawdown={max_drawdown:.4f} sharpe={sharpe:.4f}".format(
            **metrics
        )
    )


if __name__ == "__main__":
    main()
