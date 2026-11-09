# 六数据集算法结果：上传包

这是覆盖到 QT-main 仓库的**增量包**，不是完整仓库。包含：

- 你上传的三个算法及修改后的共享导出模块：`algorithms/`
- 六个数据集的 `outdata` JSON：`outdata/`
- 任务报告：`TASK_REPORT.md`
- 同步脚本：`sync_to_repo.ps1`

不包含原始行情 `data/`、`gui/`、`engine/`、其他算法文件、`.venv` 和财报数据。

## 内容规模

| 数据集 | JSON 数 |
|---|---:|
| `tech-us-10y` | 41 |
| `manufacturing-us-10y` | 47 |
| `healthcare-us-10y` | 27 |
| `us-agri` | 29 |
| `nasdaq100` | 102 |
| `nasdaq-xbrl` | 96 |
| 合计 | 342 |

四个代码文件：

```text
algorithms/technical_common.py
algorithms/macd_trend.py
algorithms/momentum_12_1.py
algorithms/donchian_breakout.py
```

## 怎么用

### 1. 先 clone 最新 main 并建分支

```powershell
git clone https://github.com/zixifang/QT-main.git QT-main-work
cd QT-main-work
git checkout main
git pull
git checkout -b feature/<你的GitHub用户名>-six-datasets
```

### 2. 把本包同步进去

```powershell
cd "C:\Users\start\Documents\deepseek-harness\default-workspace\QT-six-datasets-upload"
.\sync_to_repo.ps1 -TargetRepo "C:\path\to\QT-main-work"
```

脚本会覆盖：

- `algorithms/technical_common.py`
- `algorithms/macd_trend.py`
- `algorithms/momentum_12_1.py`
- `algorithms/donchian_breakout.py`
- 六个 `outdata/<dataset_id>/*.json`

### 3. 检查并提交

```powershell
cd "C:\path\to\QT-main-work"
git status --short

git add algorithms/technical_common.py algorithms/macd_trend.py algorithms/momentum_12_1.py algorithms/donchian_breakout.py
git add outdata/tech-us-10y outdata/manufacturing-us-10y outdata/healthcare-us-10y outdata/us-agri outdata/nasdaq100 outdata/nasdaq-xbrl

git commit -m "Run my technical algorithms on all six datasets"
git push -u origin feature/<你的GitHub用户名>-six-datasets
```

然后到 GitHub 开 PR 到 `main`。

## 已完成校验

- 六个数据集共 334 只股票，每只都有 `macd_trend`、`momentum_12_1`、`donchian_breakout`
- 原有算法全部保留：
  - `tech-us-10y` / `manufacturing-us-10y`：`rsi`
  - `healthcare-us-10y`：`macd`、`random_forest`
  - `us-agri`：`mom_rotation`
  - `nasdaq100`：`enhance-g1` 到 `enhance-g7`
  - `nasdaq-xbrl`：`fscore`、`transformer`
- `signal` / `position` 均为 0/1，仓位严格后移一天
- GUI 逐数据集、逐算法检查：6 × 3 × 5 页都有图/表
- `nasdaq100` 的 `NDX` 没有进入 `symbols`

## 注意

- 当前目录不是 Git 仓库，不能直接 push；它是覆盖到 clone 的文件包。
- 不要把这个包 `git init` 后直接当成完整仓库推送，否则会丢失原仓库历史。
- `TASK_REPORT.md` 里有更详细的算法、日期对齐和验证说明。
