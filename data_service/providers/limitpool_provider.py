# data_service/providers/limitpool_provider.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional

try:
    import pandas as pd  # type: ignore
    import akshare as ak  # type: ignore
except Exception as e:
    raise ImportError("需要安装 pandas 与 akshare==1.17.40") from e

from ..models import FetchResult
from ..storage import now_shanghai, call_with_retry


def fetch_limit_up_pool(date: Optional[str] = None) -> FetchResult:
    """
    涨停池（Plan B：东财主源 → 历史端点兜底）
    - 主源:   stock_zt_pool_em(date=YYYYMMDD)
    - 兜底:   stock_zt_pool_previous_em(trade_date=YYYYMMDD)
    - 返回:   FetchResult（当触发兜底时，degraded=True 且 note=f"fallback: 主 → 备"；api 为实际使用）
    """
    if not date:
        raise ValueError("fetch_limit_up_pool: 需要提供 date（YYYYMMDD）")

    primary_api = "stock_zt_pool_em"
    try:
        df = call_with_retry(
            primary_api,
            ak.stock_zt_pool_em,
            date=date,
            timeout=18.0,
            retries=2,              # 共尝试 3 次（2 次重试）
            backoff_seq=(0.8, 1.6),
        )
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{primary_api} 未返回 DataFrame")
        return FetchResult(
            data=df,
            api=primary_api,
            fetched_at=now_shanghai(),
            params={"date": date},
        )
    except Exception:
        # 兜底：上一交易日历史端点
        fallback_api = "stock_zt_pool_previous_em"
        df = call_with_retry(
            fallback_api,
            ak.stock_zt_pool_previous_em,
            trade_date=date,        # 该端点使用 trade_date 参数名
            timeout=20.0,
            retries=2,
            backoff_seq=(1.0, 2.0),
        )
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{fallback_api} 未返回 DataFrame")
        return FetchResult(
            data=df,
            api=fallback_api,
            fetched_at=now_shanghai(),
            params={"date": date},
            degraded=True,
            note=f"fallback: {primary_api} -> {fallback_api}",
        )


def fetch_limit_down_pool(date: Optional[str] = None) -> FetchResult:
    """
    跌停池（Plan B：暂无稳定历史端点；放宽时序参数，仅主源）
    - 主源: stock_zt_pool_dtgc_em(date=YYYYMMDD)
    - 返回: FetchResult
    """
    if not date:
        raise ValueError("fetch_limit_down_pool: 需要提供 date（YYYYMMDD）")

    api = "stock_zt_pool_dtgc_em"
    df = call_with_retry(
        api,
        ak.stock_zt_pool_dtgc_em,
        date=date,
        timeout=20.0,               # 放宽以提高非交易时段/网络抖动下的成功率
        retries=2,
        backoff_seq=(0.8, 1.6),
    )
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{api} 未返回 DataFrame")

    return FetchResult(
        data=df,
        api=api,
        fetched_at=now_shanghai(),
        params={"date": date},
    )
