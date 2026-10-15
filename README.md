# 量化研究小组

仓库：[zixifang/QT-main](https://github.com/zixifang/QT-main)

本仓库已按新的项目计划清空。原先的「单一 RSI 均值回归 + 五人分模块」代码、接口和分工说明都已移除，后续协作以本说明为准。

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

镜像基于 `python:3.10-slim`。当前命令只确认 Python 版本，算法程序由各自分支接入后再改启动命令。

## 协作

- `main` 保存已经能运行的版本。
- 每人从 `main` 拉自己的分支：`feature/<GitHub用户名>-<内容>`。
- 最小可行程序放在 `members/<GitHub用户名>/`，里面包含数据说明、算法和可视化，避免互相覆盖。
- 完成后开 Pull Request 合并回 `main`。

大体积原始数据不要提交。数据说明里写清来源、字段和时间范围即可。
