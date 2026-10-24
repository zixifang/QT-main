# 近期美股大盘：数据 JSON 与生成程序

按需求只保留两样东西：

1. **数据 JSON**：`outdata/recent-us-2024-2026/`
2. **生成程序**：`algorithms/`、`engine/`、`data_fetch/recent_us.py`

原始 CSV 没有放进本包；运行 `python -m data_fetch.recent_us` 可重新下载并生成 `data/recent-us-2024-2026/`。

## 目录

```text
algorithms/
  macd_trend.py
  momentum_12_1.py
  donchian_breakout.py
  technical_common.py
  paper_account.py
engine/
  core.py
  __main__.py
data_fetch/
  recent_us.py
outdata/recent-us-2024-2026/
  index.json
  AAPL.json ... NFLX.json      20 只股票
  paper/
    macd_trend.json
    momentum_12_1.json
    donchian_breakout.json
verify_outputs.py
requirements.txt
```

## JSON 内容

- `index.json`：数据集名称、20 个代码、三个算法及分年统计
- `{代码}.json`：日期、OHLCV、`return_index`，三个算法的 `indicator`、`signal`、`position`、买卖点、`equity`、`benchmark_equity`、`stats`
- `paper/*.json`：10 万美元整股账户的净值、现金、持股和分年统计

数据集：2024-01-02 至 2026-10-23，20 只美股大盘股，每只 706 个交易日。

## 验证

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python verify_outputs.py
```

`verify_outputs.py` 会检查：

- 20 个股票 JSON 和 3 个算法是否齐全
- 数组长度、日期顺序、价格和 `return_index`
- `signal` / `position` 是否为 0/1，仓位是否后移一天
- 买卖点是否都在 `dates` 中
- `stats` 字段是否完整
- `paper/` 三个账户是否完整

期望输出结尾：

```text
验证通过：JSON 长度、0/1 仓位、仓位后移、买卖点、统计字段和 paper 账户全部符合契约。
```

## 重新生成

```powershell
# 1. 下载近期行情（需要网络）
python -m data_fetch.recent_us

# 2. 生成三个算法的 JSON
python -m algorithms.macd_trend --dataset recent-us-2024-2026
python -m algorithms.momentum_12_1 --dataset recent-us-2024-2026
python -m algorithms.donchian_breakout --dataset recent-us-2024-2026

# 3. 生成 paper 账户
python -m algorithms.paper_account

# 4. 再次验证
python verify_outputs.py
```

## 上传 GitHub

```powershell
git init
git config user.name "你的名字"
git config user.email "你的邮箱"
git add .
git commit -m "Add recent US large-cap JSON and generation code"
git branch -M main
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git push -u origin main
```

## 数据来源与口径

- 数据源：新浪美股日线接口
- 已按拆股比例回溯调整：WMT 3:1、NVDA 10:1、AVGO 10:1、NFLX 10:1
- 只做多，信号收盘确定，下一交易日按收盘价计算收益，单边手续费 0.1%
- 股票池为 2024 年时点的大盘股名单，存在幸存者偏差；未计分红、滑点和冲击成本
- 仅用于课程和研究回测，不构成投资建议
