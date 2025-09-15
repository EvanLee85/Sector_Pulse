# data_service/models.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, TypedDict
from datetime import datetime
from zoneinfo import ZoneInfo

try:
    import pandas as pd  # type: ignore
except Exception as e:
    raise ImportError("需要安装 pandas") from e


_SH_TZ = ZoneInfo("Asia/Shanghai")


@dataclass
class FetchResult:
    """
    统一的数据抓取返回结构（供 collector / providers / tests 使用）

    字段说明
    - data:        pandas.DataFrame，抓取到的原始数据（或经轻量列名统一后）
    - api:         实际使用的接口名（例如 'stock_zh_a_spot_em' 或降级后的 'stock_zh_a_spot'）
    - fetched_at:  抓取完成的时间戳（**必须为带时区**；默认 Asia/Shanghai）
    - params:      本次调用使用的参数快照（如 {'date': '20250912'}）
    - degraded:    是否触发了“主→备”降级（默认 False）
    - note:        附加说明（例如 'fallback: stock_zh_a_spot_em -> stock_zh_a_spot'）
    """
    data: pd.DataFrame
    api: str
    fetched_at: datetime
    params: Mapping[str, Any] = field(default_factory=dict)
    degraded: bool = False
    note: Optional[str] = None

    def __post_init__(self) -> None:
        # 保证 data 一定是 DataFrame
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError(f"FetchResult.data 需为 pandas.DataFrame，得到 {type(self.data)}")

        # 统一 fetched_at 为带时区（Asia/Shanghai）
        if not isinstance(self.fetched_at, datetime):
            raise TypeError(f"FetchResult.fetched_at 需为 datetime，得到 {type(self.fetched_at)}")
        if self.fetched_at.tzinfo is None:
            self.fetched_at = self.fetched_at.replace(tzinfo=_SH_TZ)

# ------------------------ 新增：指标计算输出的最小“冻结”契约 ------------------------

class IndicatorsMetrics(TypedDict, total=False):
    """
    指标结果的“最小核心字段”集合（允许 None，表示缺失或无法计算）
    只冻结以下 8 个核心键与单位/含义，不包含任何附加诊断字段：
      - limit_up_count: int | None
      - limit_down_count: int | None
      - limitup_down_ratio: float | None           # 比值，单位：无量纲
      - turnover_concentration_top20: float | None # 前20成交额占比，范围 [0,1]
      - hs300_vol_30d_annualized: float | None     # 年化波动（30D），单位：小数，如 0.18
      - vol_percentile: float | None               # 近一年分位，范围 [0,1]
      - sh_above_ma20: bool | None                 # 是否站上 MA20（昨收MA20 vs 盘中现价）
      - margin_net_repay_yi_prev: float | None     # 两融净买入（上一交易日），单位：亿元
    """
    limit_up_count: Optional[int]
    limit_down_count: Optional[int]
    limitup_down_ratio: Optional[float]
    turnover_concentration_top20: Optional[float]
    hs300_vol_30d_annualized: Optional[float]
    vol_percentile: Optional[float]
    sh_above_ma20: Optional[bool]
    margin_net_repay_yi_prev: Optional[float]


class IndicatorsPayload(TypedDict, total=False):
    """
    指标计算器（P0）的最小系统字段 + 指标集合
    只冻结以下系统字段：
      - schema_version: int
      - calc_at: datetime (带时区；Asia/Shanghai)
      - intraday_missing: bool  # 盘中关键项是否缺失
      - sources: Mapping[str, Any]  # 来源信息，保留灵活结构
    以及：
      - metrics: IndicatorsMetrics  # 以上 8 个核心指标
    """
    schema_version: int
    calc_at: datetime
    intraday_missing: bool
    sources: Mapping[str, Any]
    metrics: IndicatorsMetrics


__all__ = [
    "FetchResult",
    "IndicatorsMetrics",
    "IndicatorsPayload",
]