"""项目根目录。服务层文件都从这里定位 data、config 和 cache。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
