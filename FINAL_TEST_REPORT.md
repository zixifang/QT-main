# 最后一次测试报告

按桌面《算法测试任务说明》执行。仓库 `zixifang/QT-main`，基线 `a836efe`（2026-11-04 拉取）。

## 一、任务要求与结果对照

| 任务要求 | 落实 |
|---|---|
| 用**自己当时上传**的算法 | 跑 `algorithms/ma_cross.py`（双均线趋势）和 `algorithms/bollinger.py`（布林带均值回归），规则、参数、`note` 未改 |
| 六个数据集全部要有结果 | 六个 `outdata/<dataset_id>/` 都写入了 `ma_cross` 与 `bollinger` |
| 只追加或覆盖自己的 id，保留别人的 | 每只股票 JSON 与 `index.json` 里原有算法条目零删除；`rsi`、`enhance-g1..g7`、`fscore`、`transformer`、`macd`、`random_forest`、`mom_rotation` 都在 |
| `python -m gui` 能打开，六个数据集都能选到我的算法 | 实测六数据集 × 两算法 × 五页（含两个指数行）共 176 次绘制，无异常 |
| 规则说明仍用上传时的 `note` | 未改：`SMA(20) 上穿 SMA(60) 开多，下穿平仓` / `收盘价跌破下轨（z<-2）开多，回到中轨（z>=0）平仓` |

## 二、六个数据集的结果（全样本，等权）

`history` 的口径：各股策略净值的等权平均；基准同为各股买入持有等权平均。

| 数据集 | 交易日 | 双均线 | 布林带 | 买入持有 | 交易次数（双均线/布林带） |
|---|---|---|---|---|---|
| tech-us-10y（40 只） | 2512 | +272.84% | +134.26% | +1782.63% | 898 / 1559 |
| manufacturing-us-10y（46 只） | 2512 | +133.04% | +51.94% | +530.36% | 1037 / 1881 |
| healthcare-us-10y（26 只） | 2512 | +82.92% | +75.17% | +303.53% | 634 / 1117 |
| us-agri（28 只） | 1240 | −2.76% | **+24.16%** | −6.85% | 339 / 668 |
| nasdaq100（100 只） | 2964 | +147.91% | +93.56% | +536.40% | 2435 / 4449 |
| nasdaq-xbrl（94 只） | 2964 | +150.49% | +94.38% | +547.49% | 2333 / 4260 |

结论与之前的样本一致：在长期上涨的市场里，两个只做多、只部分时间持仓的择时策略都跑不赢买入持有；
唯一跑赢的是 **us-agri**——这五年农业股整体下跌（基准 −6.85%），布林带靠超跌反弹拿到 +24.16%。

## 三、契约核对（自动）

`python -m unittest tests.test_ma_bollinger -v` → 8 项全过，逐数据集检查：

- `dataset_id` 与目录名一致；`symbols` 数量 40 / 46 / 26 / 28 / 100 / 94；
- 两个算法都在 `index.json`，且都有分年 `years` 与日度 `history`（三条数组等长、日期升序）；
- 抽样股票：`indicator`/`signal`/`position`/`equity`/`benchmark_equity` 与 `dates` 等长，
  `position` = `signal` 后移一天，`signal` 只有 0/1，买卖点都在 `dates` 里，`stats` 八个字段，
  基准 `win_rate` 为 `null`；
- 指数：`NDX` 不在 `symbols` 里；`NDX.json` 里两个算法都有 `equity`/`benchmark_equity` 与
  `stats.stat/benchmark` 的 `wealth_ratio`、`beta`。

## 四、界面核对（`python -m gui`）

| 数据集 | 股票表行数 | 可选到我的算法 | 股票页 | 指数行页数 |
|---|---|---|---|---|
| tech-us-10y | 40 | 是 | 5 | — |
| manufacturing-us-10y | 46 | 是 | 5 | — |
| healthcare-us-10y | 26 | 是 | 5 | — |
| us-agri | 28 | 是 | 5 | — |
| nasdaq100 | 101（含指数行） | 是 | 5 | 2 |
| nasdaq-xbrl | 95（含指数行） | 是 | 5 | 2 |

共 176 次绘制（每数据集取首尾 3 只股票 + 指数行），无异常、无缺字段提示。

## 五、复跑命令

```powershell
python -m algorithms.ma_cross   --data-dir data/tech-us-10y           --dataset-id tech-us-10y           --dataset-name 科技美股近十年日线
python -m algorithms.ma_cross   --data-dir data/manufacturing-us-10y  --dataset-id manufacturing-us-10y  --dataset-name 美国制造业近十年日线
python -m algorithms.ma_cross   --data-dir data/healthcare-us-10y     --dataset-id healthcare-us-10y     --dataset-name 美国医疗健康近十年日线
python -m algorithms.ma_cross   --data-dir data/us-agri               --dataset-id us-agri               --dataset-name "美国农业股（28 只，5 年）"
python -m algorithms.ma_cross   --data-dir data/nasdaq100             --dataset-id nasdaq100             --dataset-name 纳斯达克100
python -m algorithms.ma_cross   --data-dir data/nasdaq100             --dataset-id nasdaq-xbrl           --dataset-name 纳斯达克XBRL
```

`bollinger` 同理，把 `ma_cross` 换成 `bollinger`。

## 六、实现说明与已知限制

1. **指数代码不进成分股**：`contract.export_dataset` 优先沿用已有 `index.json` 的 `symbols`，
   并把 `benchmark.id`（NDX）排除；`nasdaq-xbrl` 因此只跑 94 只，而不是 `data/nasdaq100` 里的 100 只。
2. **日历对齐**：有指数的数据集以指数日历为准（NDX 缺 2019-11-08，成分股多出的一天要丢掉）；
   其他数据集以已提交 JSON 的 `dates` 为准（`us-agri` 现有 JSON 是 1240 天，行情 CSV 有 1255 天）。
   这样合并后长度才对得上，也不会和别人的数组错位。
3. **指数页**：组合是各股策略日收益的等权平均（每日再平衡），指数那条是指数买入持有；
   资金比、贝塔、近 10 年与 2021 年以来的统计复用 `algorithms/index_enhance.py` 的 `_pair_stats`，
   与指数增强口径一致。
4. 2026 年的数据不完整（截至 10 月），全局统计页按年展示时看到的是年内的数。
5. 本次新增 outdata 约 154 MB（六个数据集合计 472 MB）。按任务要求每只股票都要写两个算法，
   体积无法再压；推送到 GitHub 前请确认网络与仓库配额。
6. 提交作者仍是本地占位身份，推送前用 `git commit --amend --reset-author` 换成自己的。
