# 量化研究小组

仓库：[zixifang/QT-main](https://github.com/zixifang/QT-main)

计算用 pandas 和 numpy。算法读 `data` 里的 CSV，把 JSON 写到 `outdata/<数据集>/`。界面只读这些 JSON。

## 六周安排

| 周次 | 内容 |
|---|---|
| 第一周 | 学习基础论文，搭建基础框架。每人找 1–3 个数据集、2 个算法，完成最小可行程序 |
| 第二周 | 继续完成第一周内容。每人制作分享报告（算法简介和回测可视化）。周末开会核对进度 |
| 第三周 | 在第二周基础上扩展，例如补充算法、补齐落后进度 |
| 第四周 | 总结第三周的扩展，集中解决问题。周末开会 |
| 第五周 | 分工制作汇报，并进行汇报演练 |
| 第六周 | 处理遗留问题 |

## 共同约定

- 第一周确定统一的架构和接口，之后不再改动。
- 每个人都做数据清洗、算法实践和可视化。
- 数据来源可以来自论文，不限于 Kaggle。
- 运行环境使用 Python 3.10，或用 Docker 封装后再上传。
- 计算引擎固定为 pandas 与 numpy。行情用 DataFrame 整理，收益和指标用数组计算。

## 计算引擎

入口是 `engine`。各人的算法只通过这三个函数接入：

| 函数 | 输入 | 输出 |
|---|---|---|
| `prepare_close(frame)` | 含 `date`、`close` 的 DataFrame | 以日期为索引的收盘价 Series |
| `run_backtest(close, position, fee=0.0)` | 收盘价、目标仓位 | 含 `close`、`position`、`return`、`equity` 的 DataFrame |
| `summarize(result)` | 回测结果 | `cumulative_return`、`max_drawdown`、`sharpe` |

`position` 在当日收盘后确定，回测使用上一期仓位计算收益。仓位可以是多头、空仓或空头权重。

```python
from engine import prepare_close, run_backtest, summarize

close = prepare_close(frame)
result = run_backtest(close, position, fee=0.001)
print(summarize(result))
```

检查引擎是否装好：

```powershell
python -m engine
```

## 行情窗口

界面只读取 `outdata` 里的 JSON。RSI 单独从 `data` 计算后再写入 `outdata`。

```powershell
python -m algorithms.rsi
python -m gui
```

左侧上方选择数据集，中间是股票表（代码、全年涨跌、年末价格），下方选择算法。右侧三页为原始价格与成交量、买卖点与指标、净值与统计指标。约定见 [docs/开发说明.md](docs/开发说明.md)。给其他 agent 的边界见 [docs/AGENT开发须知.md](docs/AGENT开发须知.md)。

## 环境

本地：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python --version
```

Docker：

```powershell
docker compose build
docker compose run --rm app
```

镜像基于 `python:3.10-slim`。启动命令是 `python -m engine`，用来确认 pandas、numpy 和回测接口可用。

## 协作

- `main` 保存已经能运行的版本。
- 每人从 `main` 拉自己的分支：`feature/<GitHub用户名>-<内容>`。
- 最小可行程序放在 `members/<GitHub用户名>/`，里面包含数据说明、算法和可视化，避免互相覆盖。
- 完成后开 Pull Request 合并回 `main`。

大体积原始数据不要提交。数据说明里写清来源、字段和时间范围即可。
