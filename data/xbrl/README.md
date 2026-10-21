# 纳斯达克 100 XBRL 基本面

脚本 `data_fetch/fetch_nq100_xbrl.py` 从 SEC EDGAR 的 companyfacts 接口拉取当前成分股的财务报表事实，写成一行一个概念、一个报告期的长表。

## User-Agent

SEC 会拒绝没有联系方式的请求。在项目根目录执行前设置：

```powershell
$env:SEC_USER_AGENT = "YourName you@example.com"
```

写成「名字 空格 邮箱」，不要加括号，也不要用 GitHub 的 noreply 地址。SEC 对这两种会直接返回 403。邮箱要能收到信。

## 运行

```powershell
python -m data_fetch.fetch_nq100_xbrl
```

只试两只：`python -m data_fetch.fetch_nq100_xbrl --limit 2`。已下载的 JSON 和 CIK 映射会缓存在 `data/xbrl/`。要重下：`--force`。请求间隔 0.15 秒。

成分股来自当前公开名单，不是历史上每个报告期的指数成分。官方 ticker 映射里没有的代码，会再查一次 EDGAR 公司页。

## 输出

| 文件 | 内容 |
|---|---|
| `nq100_xbrl_fundamentals.parquet` | 主表，同时有一份同名 CSV |
| `missing_concepts.csv` | 主标签缺失时实际用的备选；完全没有数据则写 `none`。债务合计的加总方式也在这里 |
| `skipped_tickers.csv` | 没有 CIK 或没有 companyfacts 的代码 |
| `raw/` | 原始 companyfacts，避免重复请求 |

列：`ticker`、`cik`、`concept`、`value`、`unit`、`period_end`、`fiscal_year`、`fiscal_period`、`form`、`filed_date`、`duration_days`。

`concept` 形如 `us-gaap_Revenues`。主标签和备选标签会同时保留，方便按报告期选用。经营现金流、流动资产和流动负债也会写入长表，供应计利润和 F-Score 使用。`derived_TotalDebt` 优先取长期债务，有 `ShortTermBorrowings` 或 `DebtCurrent` 时再相加。一家公司只要出现过这些债务标签，就不再用负债合计填空档。整家公司都没有债务标签时，才退回 `Liabilities`，并在 `missing_concepts.csv` 里标明这是含经营负债的粗略值。不要把总额和组成部分再加一次。

`duration_days` 是区间概念从 `start` 到 `end` 的天数。资产负债表项目没有区间，这里为空。同一期末可能既有三个月的利润，也有年初至今的利润，两行都保留。

只保留 `10-K`、`10-Q` 及其修正稿。用 `20-F` 申报的境外公司不会出现在表里。同一代码、概念、期末、报表和区间长度有多条时，以第一次披露的财年为准；同一财年里的修正稿取较新的一版。以后年度报表里重复出现的比较期，不再把财年和提交日改写到后一年。

做因子时先按 `filed_date` 对齐到当时已经提交的事实，不要用报告期末当天还没公开的数字。透视示例：

```python
wide = df.pivot_table(index=["ticker", "period_end"], columns="concept", values="value")
```
