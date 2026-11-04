# 六数据集算法测试完成报告

日期：2026-11-02  
算法来源：Dove56  
算法：`macd_trend`、`momentum_12_1`、`donchian_breakout`

## 结果

| 数据集 | dataset_id | 股票数 | 原有算法 | 最终算法 |
|---|---|---:|---|---|
| 科技美股近十年日线 | `tech-us-10y` | 40 | `rsi` | `rsi` + 三个新算法 |
| 美国制造业近十年日线 | `manufacturing-us-10y` | 46 | `rsi` | `rsi` + 三个新算法 |
| 美国医疗健康近十年日线 | `healthcare-us-10y` | 26 | `macd`、`random_forest` | 原有 + 三个新算法 |
| 美国农业股（28 只，5 年） | `us-agri` | 28 | `mom_rotation` | 原有 + 三个新算法 |
| 纳斯达克100 | `nasdaq100` | 100 | 七个 `enhance-g1..g7` | 原有 + 三个新算法 |
| 纳斯达克XBRL | `nasdaq-xbrl` | 94 | `fscore`、`transformer` | 原有 + 三个新算法 |

共 334 只股票，每只写入 3 个算法，合计 1002 条算法结果。六个 `index.json` 均已更新。

## 校验

- 每个股票 JSON 都包含 `macd_trend`、`momentum_12_1`、`donchian_breakout`
- 每个数据集原有的算法条目都还在
- `signal` / `position` 均为 0/1，`position` 严格等于 `signal` 后移一天
- `indicator`、`signal`、`position`、`equity`、`benchmark_equity` 与 `dates` 等长
- `buy_dates` / `sell_dates` 都在 `dates` 中
- `nasdaq100` 的 `NDX` 没有进入 `symbols`
- GUI 逐数据集、逐算法检查：6 个数据集 × 3 个算法 × 5 页，五页都有图/表
- 没有修改 `gui/`

## 代码改动

只改了自己的算法和共享导出模块：

- `algorithms/technical_common.py`
  - 新增 `export_dataset_dir`：按 `--data-dir` / `--dataset-id` / `--dataset-name` 写入
  - 新增 `run_from_args`
  - 新增日期对齐：已有 JSON 的日期是 CSV 的子集时，先按已有日期过滤行情再计算
- `algorithms/macd_trend.py`
- `algorithms/momentum_12_1.py`
- `algorithms/donchian_breakout.py`
  - 三个 `main()` 改用 `run_from_args`，支持说明里的六条命令

信号规则、参数、名称和 `note` 没有变化。

## 把改动同步到实际 clone

```powershell
# 先把 GitHub 的 main 拉到本地 clone，并建 feature 分支
cd C:\path\to\QT-main
git checkout main
git pull
git checkout -b feature/<你的GitHub用户名>-six-datasets

# 在 qt-main-latest 目录执行同步脚本
.\sync_task_changes.ps1 -TargetRepo C:\path\to\QT-main
```

然后在目标仓库：

```powershell
git status --short
git add algorithms/technical_common.py algorithms/macd_trend.py algorithms/momentum_12_1.py algorithms/donchian_breakout.py
git add outdata/tech-us-10y outdata/manufacturing-us-10y outdata/healthcare-us-10y outdata/us-agri outdata/nasdaq100 outdata/nasdaq-xbrl
git commit -m "Run my technical algorithms on all six datasets"
git push -u origin feature/<你的GitHub用户名>-six-datasets
```

然后到 GitHub 开 PR 到 `main`。

## 说明

- 现有 `outdata` 的日期长度保持不变；遇到 CSV 多出几行时，按已有 JSON 的日期过滤，没有把 CSV 直接交给界面。
- 没有提交 `data/xbrl`、`data/factors`、`.venv`、密钥或 Notebook。
- 第 5 页对这三个算法使用各年综合盈利率回退显示；任务允许没有 `history` 时这样做。
