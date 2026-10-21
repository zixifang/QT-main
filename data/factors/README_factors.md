# 纳斯达克 100 基本面因子

脚本 `data_fetch/build_factors.py` 读取 `data/xbrl/nq100_xbrl_fundamentals.parquet`，输出 `data/factors/factors_nq100.parquet`。每一行是一只股票的一个报告期，并带有 `filed_date`。下游做横截面排序时按 `period_end` 分组，但可见时点用 `filed_date`，避免用到尚未提交的数字。

```powershell
python -m data_fetch.build_factors
python -m data_fetch.build_factors --market data/market.parquet
```

## 期间怎么对齐

年报和季报分开计算，`form` 为 `10-K` 或 `10-Q`。修正稿并入对应报表。只有封面股数、没有资产负债或利润数据的日期会去掉。上期不是上一季度，而是：

- `10-K`：上一财年的 `10-K`
- `10-Q`：上一财年同一个 `fiscal_period`

利润表和现金流量表只保留区间长度合适的事实：年报约 300–400 天，季报约 70–120 天。年初至今的累计数不拿来当单季。资产负债表和股数只保留没有区间的时点值。同一报告期有多条时，留下首次披露那个财年里较新的一版。宽表上的 `filed_date` 是资产负债表或利润表的提交日。

财年取资产负债表或利润表第一次披露时的财年，不用后一年才出现的债务标签。平均分母是当期和上期的均值。股数也用首次披露值，所以股票分割当年的净股票发行会失真。上期没有时，ROE、ROA、应计利润和资产周转率退化为当期值，并在缺失报告里记 `prior_missing_used_current`。资产增长、净股票发行和 F-Score 的差分项不上这个退路，上期没有就是空值，原因记 `missing_prior`。分母为 0 时结果为空，原因记 `zero_denominator`。

## 因子

| 因子 | 公式 | 方向 |
|---|---|---|
| `ROE` | 净利润 / 平均股东权益 | 高更好 |
| `ROA` | 净利润 / 平均总资产 | 高更好 |
| `GrossProfitability` | 毛利润 / 当期总资产 | 高更好（Novy-Marx 2013） |
| `BM` | 股东权益 / 市值 | 高表示更便宜 |
| `Accruals` | (净利润 − 经营现金流) / 平均总资产 | 保留原符号，低通常更好 |
| `AssetTurnover` | 营业收入 / 平均总资产 | 高更好 |
| `F_ROA` … `F_dASSET_TURN` | Piotroski 九个 0/1 分项 | 满足为 1 |
| `FScore` | 有数据的分项之和，不是把缺失当成 0 | 高更好 |
| `FScore_coverage` | 有数据的分项个数 / 9 | 用来筛样本 |
| `AssetGrowth` | 总资产相对上期的变化率 | 低更好（Cooper 等 2008） |
| `NetStockIssues` | 流通股数相对上期的变化率 | 低更好 |

科目按这个顺序取，后面的是备选：净利润 `NetIncomeLoss`、`ProfitLoss`；权益含少数股东权益的长标签；总资产缺失时才用流动资产；收入依次为 `Revenues`、`RevenueFromContractWithCustomerExcludingAssessedTax`、`SalesRevenueNet`；经营现金流其次用持续经营现金流；股数其次用 `dei` 的流通股。用了备选时，缺失报告写 `fallback_concept:` 加实际标签。

股东权益为负时，ROE 和 B/M 仍保留原值，同时记 `negative_equity`，方便下游剔除。

## 市场数据

估值只在传入市场数据时计算。文件是 CSV 或 Parquet，至少包含：

- `ticker`、`date`、`market_cap`，或
- `ticker`、`date`、`close`、`shares_outstanding`（市值 = 收盘价 × 股数）

市值日期取 `filed_date` 当天或之后最近的交易日，并且不超过 10 天。对不上时 B/M 为空，原因是 `market_data_missing` 或 `market_date_unaligned`。不传 `--market` 时，B/M 全部为空。

## 缺失报告

`factors_missing_report.csv` 的列是 `ticker`、`form`、`factor`、`reason`、`n_periods`。这里只统计有问题的报告期，包括算出来了但用了备选标签、上期缺失后退化、以及权益为负。
