# -*- coding: utf-8 -*-
from __future__ import annotations

try:
    import pandas as pd  # type: ignore
    import akshare as ak  # type: ignore
except Exception as e:
    raise ImportError("需要安装 pandas 与 akshare==1.17.40") from e

from ..models import FetchResult
from ..storage import now_shanghai, cached, call_with_retry


@cached()
def fetch_market_spot() -> FetchResult:
    """
    全市场快照（东财优先 → 新浪兜底），用于后续计算“成交额前20占比”等市场集中度指标。
    返回 FetchResult；当触发兜底时，会设置：
      - degraded=True
      - note="fallback: stock_zh_a_spot_em -> stock_zh_a_spot"
      - api 为实际使用的接口名
    """
    primary_api = "stock_zh_a_spot_em"
    try:
        # 东财主源：显式放宽超时与重试，降低进入兜底概率
        df = call_with_retry(
            primary_api,
            ak.stock_zh_a_spot_em,
            timeout=18.0,           # 默认 2.5s 太紧，这里放宽
            retries=2,              # 共尝试 3 次（2 次重试）
            backoff_seq=(0.8, 1.6),
        )
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{primary_api} 未返回 DataFrame")

        # 统一 amount_yuan（若主源字段已是元或能推导，也一并处理）
        df = _ensure_amount_yuan(df)
        return FetchResult(data=df, api=primary_api, fetched_at=now_shanghai(), params={})

    except Exception:
        # 新浪兜底：再放宽窗口，尽量在测试阶段拿到结果
        fallback_api = "stock_zh_a_spot"
        df = call_with_retry(
            fallback_api,
            ak.stock_zh_a_spot,
            timeout=25.0,                # 放宽到 25s
            retries=3,                   # 共尝试 4 次（3 次重试）
            backoff_seq=(1.0, 3.0, 5.0),
        )
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{fallback_api} 未返回 DataFrame")

        df = _ensure_amount_yuan(df)
        return FetchResult(
            data=df,
            api=fallback_api,
            fetched_at=now_shanghai(),
            params={},
            degraded=True,
            note="fallback: stock_zh_a_spot_em -> stock_zh_a_spot",
        )


def _ensure_amount_yuan(df: pd.DataFrame) -> pd.DataFrame:
    """
    尽量统一补出列：amount_yuan（单位：元）
    兼容若干常见字段名及单位：'成交额(元)'、'成交额'、'amount'、'成交额(万元)'、'成交额(百万元)'
    """
    if "amount_yuan" in df.columns:
        return df

    amount_cols = ("成交额(元)", "成交额", "amount", "成交额(万元)", "成交额(百万元)")
    for c in amount_cols:
        if c in df.columns:
            ser = pd.to_numeric(df[c], errors="coerce")
            if "万元" in c:
                ser = ser * 1e4
            elif "百万元" in c:
                ser = ser * 1e6
            df = df.copy()
            df["amount_yuan"] = ser
            break
    return df
