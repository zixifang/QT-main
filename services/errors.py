"""指数增强服务的异常。界面只展示这些消息，不把计算写进 gui。"""

from __future__ import annotations


class ServiceError(Exception):
    """服务层可以展示给界面的错误。"""


class SourceUnavailable(ServiceError):
    """数据源没接上、没有 Key，或按规则不能使用。"""


class DataGapError(ServiceError):
    """区间、字段或股票缺失，调用方应告警而不是补造数据。"""


class AdjustError(ServiceError):
    """复权方式不被允许。前复权直接拒绝。"""


class LookaheadError(ServiceError):
    """公告日或未来收益被用到了当时还不能知道的位置。"""


class ConstituentGapError(ServiceError):
    """没有该再平衡日的历史成分，不能用现在的名单回填。"""


class Cancelled(ServiceError):
    """用户取消了长任务。"""
