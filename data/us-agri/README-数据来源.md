# us-agri 数据来源

## 结论先说

这份数据**只有 5 年**（2021-10-26 ~ 2026-10-26），不是要求的 10 年。

## 为什么

1. `yfinance` 对全部 30 只都返回 `YFRateLimitError: Too Many Requests`——
   Yahoo 把这个出口 IP 限流了；
2. 绕过 yfinance 直连 `query1/query2.finance.yahoo.com/v8/finance/chart/...`，
   不管换 `range`（1y/5y/10y）还是换 UA，一律 **403**；
3. 于是改用 **stockanalysis.com** 的公开 history 接口（无需 key，返回真实行情）；
4. 那个接口 `?range=10Y` 实际只返回 **1255 根**（2021-10 起），也就是 5 年。试过
   `range=MAX`（250 根）、`range=20Y`、`period=Monthly`，都拿不到更早的。

没有用任何模拟数据填充：抓不到的标的直接跳过并打印。

## 个股情况

- **CTVA 只有 17 行**（2026-10-02 起），大概率是该源对这只票的数据不全。
  17 根日线算不了 126 日动量，**已剔除**。
- 其余 29 只都是 1240~1255 根，起点 2021-10-26。
- SDOT 1254 行、AFRI 1240 行，少的是停牌日。

## 口径

- `Adj Close` 取自接口的 `a` 字段（复权价），并把 `a/c` 的比例折算进
  Open/High/Low/Close —— 与 LEAN 样例那份数据的处理方式一致；
- 列序 `Date,Open,High,Low,Close,Adj Close,Volume`，日期 `YYYY-MM-DD`；
- 一个代码一个文件。

## 想拿到真正的 10 年怎么办

1. 换一台没被限流的机器跑 `fetch_us_agri_yfinance.py`（需要 `pip install yfinance pandas`）；
2. 或用带 key 的数据源（Tiingo、Polygon、EODHD、Nasdaq Data Link）替换本脚本的 `fetch()`；
3. 拿到更长的 CSV 后直接覆盖 `data/us-agri/`，再跑一遍：

```powershell
python -m algorithms.momentum_rotation --data data/us-agri --dataset us-agri --top-n 5
python -m algorithms.paper_account
```

## 断点续爬

被限流时一次抓不完，所以有一个带断点的版本：

```powershell
python data_fetch/fetch_us_agri_resume.py                  # 默认 6 轮，每轮间隔 120 秒
python data_fetch/fetch_us_agri_resume.py --rounds 2 --sleep 20
python data_fetch/fetch_us_agri_resume.py --only CTVA      # 只补一只
python data_fetch/fetch_us_agri_resume.py --force          # 全部重抓
```

每抓完一只就写一次 `_fetch_state.json`，随时中断都不丢进度：

```json
{
  "ADM":  {"status": "ok",    "source": "stockanalysis", "rows": 1255,
           "first": "2021-10-26", "last": "2026-10-26", "attempts": 2},
  "CTVA": {"status": "short", "source": "stockanalysis", "rows": 17,
           "first": "2026-10-02", "last": "2026-10-26", "attempts": 4}
}
```

- `ok` 下次直接跳过；`short`（行数 < `--min-rows`，默认 500）和 `failed` 每轮重试；
- **`short` 的数据不写盘**——17 根日线混进 `data/us-agri/` 会把整个数据集的时间交集
  从 1239 天砍到 17 天，宁可先空着；
- 每轮之间的等待按 `--sleep × 轮次` 递增，避免刚被限流就继续撞；
- 跑完自动重建 `us_agri_stocks_10y.csv` 长表。

实测：删掉 `ADM.csv` 并把它改回 `pending` 再跑，输出是
「断点里已完成 28 只，本次要处理 2 只」——只补了 ADM 和 CTVA，其余 28 只一个请求都没发。

## 文件

- `fetch_us_agri_resume.py` —— **带断点**的版本，先试 yfinance（能给 10 年），
  失败退回 stockanalysis（5 年），推荐用这个
- `fetch_us_agri.py` —— 单轮版本（stockanalysis），一次性抓完用
- `fetch_us_agri_yfinance.py` —— 原始要求的 yfinance 版本，在没被限流的网络下可直接用
- `_fetch_state.json` —— 断点文件，删掉就是从头再来
