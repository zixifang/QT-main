# Agent 开发须知

后续 agent 改这个仓库时，按下面的边界做。不要把计算塞回界面，也不要让界面直接读 `data` 里的 CSV。

## 必须遵守

1. `gui/` 只读取 `outdata/**/*.json` 并绘图。禁止在 `gui/` 里计算 RSI、收益、回撤、夏普，禁止 `gui/` import `algorithms` 或 `engine`，禁止 `gui/` 打开 `data/` 下的 CSV。
2. 每个算法是 `algorithms/` 下的独立程序。它读取 `data/`，调用 `engine` 做回测，然后把 JSON 写到 `outdata/<dataset_id>/`。
3. RSI 是 `python -m algorithms.rsi`。它扫描 `data/` 下每个含单只股票 CSV 的子目录，用同一套 RSI 规则计算，并写到 `outdata/<dataset_id>/`。`Equity-Portfolio-Management-master` 对应 `outdata/equity-2018/`，其余数据集的 id 与数据目录名一致。带 `Ticker` 列的合并文件（如 `tech_us_10y.csv`）不要写成一只股票。`data/nasdaq100` 跳过，指数增强由 `python -m algorithms.index_enhance` 写到 `outdata/nasdaq100/`。不要把这段计算搬回 `gui/app.py`。
4. 界面上的数据集来自 `outdata` 子目录里的 `index.json`。股票表的行来自该文件的 `symbols`，列是代码、整体涨跌、价格。按年或按月时，价格是选中年份的最后收盘价，整体涨跌是它相对上一年最后一个收盘价的涨跌；该年之前没有行情时，改用当年第一个收盘价。股票 JSON 里有 `return_index` 时，整体涨跌改用这条净值，价格仍是原始 `close`。`return_index` 由算法写入，单日涨跌超过 50% 的日子收益记 0，用来避开未复权拆股。按10年时年份选择变灰，整体涨跌是最近10年的涨跌，价格是期末收盘价。不要用全样本的 `cumulative_return`。算法列表只显示当前数据集股票 JSON 的 `algorithms` 里已经有结果的算法，名称和说明优先用 `index.json` 里的登记。没有写入结果的算法不要显示，也不要把其他数据集的算法带过来。新增数据集或算法时改 JSON，不要在界面里写死代码列表。
5. 股票区域保持左侧表格，不要改回标签页。数据集下拉框放在股票表上方，算法列表放在股票表下方。
6. 右侧共五页：原始数据、算法点位、统计指标、收益对比、全局统计。前四页的数都从当前股票 JSON 里取。图表上方保留年份、月份选项，右侧下拉框有「按月展示」「按年展示」和「按10年展示」。按年只画选中年份，按月只画选中年月，按10年画最近10年并且年份选择变灰。没有数据的月份要禁用。按10年时月份选项停用。前四页折线图的图例横排在图上方，不压住折线。单股统计表保持 JSON 里的全样本 `stats`，不要在界面里按月份重算。第四页只画策略净值和买入持有。第五页只读 `index.json` 里当前算法的 `years[<选中年份>]`。RSI 按自然年写入这一份：`symbols` 是各股当年策略累计收益，`strategy` / `benchmark` 是当年汇总。综合胜率按该年交易次数加权，综合盈利率是该年各股累计收益的等权平均。指数增强的单股 `equity` 是该股持仓收益，`benchmark_equity` 是该股买入持有。第五页策略行的综合盈利率、年化、夏普和回撤与指数页同一条组合净值；纳斯达克指数行只有指数本身；`holding` 是各股买入持有的等权平均和中位数。股票数、收益中位数、盈利占比和柱状图仍按各股。界面换年就换读取的年份，不要在界面里重算收益、回撤或夏普。全局统计页仍可切换年份，月份和展示方式停用。
7. 横轴使用 JSON 里的 `dates`，格式 `YYYY-MM-DD`，按时间从早到晚，再按上面的年或月截取。不要改回未对齐的日期坐标。

## 目录契约

```text
data/<原始数据集>/                 算法程序的输入
outdata/<dataset_id>/index.json    界面用来列数据集、股票、算法
outdata/<dataset_id>/<SYMBOL>.json 界面用来画一只股票
algorithms/<name>.py               只负责一种算法的计算和写 JSON
gui/                               只负责读 JSON 和展示
engine/                            只给算法程序调用
```

## index.json

```json
{
  "schema_version": 1,
  "dataset_id": "equity-2018",
  "name": "界面下拉框显示的名字",
  "source": "data/相对路径",
  "price_field": "close",
  "symbols": ["AAPL"],
  "algorithms": [
    {
      "id": "rsi",
      "name": "RSI 均值回归",
      "note": "一句话说明规则",
      "years": {
        "2018": {
          "symbols": [{"symbol": "AAPL", "cumulative_return": 0.0, "benchmark_cumulative_return": 0.0}],
          "strategy": {"count": 1, "winners": 0, "losers": 0, "profit_ratio": 0.0, "win_rate": 0.0, "cumulative_return": 0.0, "median_return": 0.0, "annualized_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "trades": 0, "beat_benchmark": 0},
          "benchmark": {"count": 1, "winners": 0, "losers": 0, "profit_ratio": 0.0, "win_rate": null, "cumulative_return": 0.0, "median_return": 0.0, "annualized_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "trades": 0}
        }
      }
    }
  ]
}
```

`dataset_id` 必须和目录名一致。`symbols` 的顺序就是股票标签从左到右的顺序。每个代码都要有同名 `{SYMBOL}.json`。

## 股票 JSON

`schema_version` 固定为 `1`。数组长度都等于 `dates` 的长度，按下标对齐。没有值的位置写 `null`，不要写 `NaN`。

```json
{
  "schema_version": 1,
  "dataset_id": "equity-2018",
  "symbol": "AAPL",
  "price_field": "close",
  "dates": ["2018-01-02"],
  "open": [170.16],
  "high": [172.3],
  "low": [169.26],
  "close": [172.26],
  "volume": [25555900],
  "algorithms": {
    "rsi": {
      "id": "rsi",
      "name": "RSI 均值回归",
      "note": "RSI(14) 低于 30 开多，高于 70 平仓",
      "params": {"period": 14, "oversold": 30, "overbought": 70, "fee": 0.001},
      "indicator": [null],
      "signal": [0],
      "position": [0],
      "buy_dates": ["2018-02-01"],
      "sell_dates": ["2018-05-10"],
      "equity": [1.0],
      "benchmark_equity": [1.0],
      "stats": {
        "strategy": {
          "cumulative_return": 0.0,
          "annualized_return": 0.0,
          "volatility": 0.0,
          "sharpe": 0.0,
          "max_drawdown": 0.0,
          "calmar": 0.0,
          "trades": 0,
          "win_rate": 0.0
        },
        "benchmark": {
          "cumulative_return": 0.0,
          "annualized_return": 0.0,
          "volatility": 0.0,
          "sharpe": 0.0,
          "max_drawdown": 0.0,
          "calmar": 0.0,
          "trades": 1,
          "win_rate": null
        }
      }
    }
  }
}
```

字段含义：

| 字段 | 含义 |
|---|---|
| `close` | 原始价格。第一页画这列，第二页也在这列上标买卖点 |
| `volume` | 第一页下方的成交量 |
| `indicator` | 第二页下方的指标值。RSI 的前 14 天是 `null` |
| `signal` | 当日收盘才知道的目标仓位，尚未后移 |
| `position` | 回测实际持仓，等于 `signal` 后移一天 |
| `buy_dates` / `sell_dates` | `position` 发生变化的日期，必须能在 `dates` 里找到 |
| `equity` | 策略净值，起点约为 1 |
| `benchmark_equity` | 买入持有净值 |
| `stats` | 第三页表格。比例用小数，例如 `-0.0182` 表示 -1.82% |
| `stats.benchmark.win_rate` | 买入持有不计算胜率，写 `null`，界面显示为 — |

`params.oversold` 和 `params.overbought` 如果存在，第二页会把它们画成水平线。没有这两个参数的算法不要编造这两条线。

## 新增算法时要改的文件

1. 新建 `algorithms/<算法id>.py`，提供可单独运行的入口。
2. 读取已有的 `outdata/<dataset_id>/*.json`，或自己从 `data` 重算后整文件重写。
3. 在每只股票的 `algorithms` 里加入同一个 `id`。
4. 把 `{"id","name","note"}` 追加到 `index.json` 的 `algorithms`，不要覆盖别人的算法条目。
5. 跑一遍自己的程序，确认 `python -m gui` 能在不改界面代码的情况下看到新算法。

## 不要做的事

- 不要在界面点击时临时算 RSI 或回测。
- 不要把 CSV 列名直接交给界面；列名转换放在算法程序里。
- 不要修改 `schema_version`，除非同时改 `gui/app.py`、本文件和已生成的 JSON。
- 不要把 `outdata` 的结果再写回 `data`。
