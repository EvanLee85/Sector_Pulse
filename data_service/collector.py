# -*- coding: utf-8 -*-
"""
统一数据接口（含一主一备降级策略）。
本步仅实现 P0 六类“拉数”函数与一次性打包函数，不做指标计算。
- 优先使用东财（_em 系列）；失败则降级到备用源（新浪/雪球等），并在 FetchResult 上打标：
    - degraded=True
    - note=f"fallback: {primary_api} -> {fallback_api}"
    - api 字段始终填写 **实际**使用的接口名
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import pandas as pd  # type: ignore
    import akshare as ak  # type: ignore
except Exception as e:
    raise ImportError("需要安装 pandas 与 akshare==1.17.40") from e

from .models import FetchResult
from .storage import now_shanghai, call_with_retry
from .providers.limitpool_provider import (
    fetch_limit_up_pool as _fetch_limit_up_pool,
    fetch_limit_down_pool as _fetch_limit_down_pool,
)
from .providers.concentration_provider import (
    fetch_market_spot as _fetch_market_spot_em,
)
from .providers.margin_provider import (
    fetch_margin_sse as _fetch_margin_sse,
    fetch_margin_szse as _fetch_margin_szse,
)

# ---- 指数符号（与文档一致）----
SH_COMPOSITE = "sh000001"   # 上证指数
HS300 = "sh000300"          # 沪深300

# ---- 指数日线（东财 -> 新浪）----
def fetch_index_daily(symbol: str) -> FetchResult:
    """
    指数日线：优先东财 stock_zh_index_daily_em(symbol)，失败降级新浪 stock_zh_index_daily(symbol)
    """
    primary_api = "stock_zh_index_daily_em"
    fallback_api = "stock_zh_index_daily"
    try:
        df = call_with_retry(primary_api, ak.stock_zh_index_daily_em, symbol=symbol, timeout=10.0, retries=2, backoff_seq=(0.8, 1.5))
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{primary_api} 未返回 DataFrame")
        return FetchResult(data=df, api=primary_api, fetched_at=now_shanghai(), params={"symbol": symbol})
    except Exception:
        # fallback to 新浪
        df = call_with_retry(fallback_api, ak.stock_zh_index_daily, symbol=symbol, timeout=12.0, retries=2, backoff_seq=(1.0, 2.0))
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{fallback_api} 未返回 DataFrame")
        return FetchResult(
            data=df, api=fallback_api, fetched_at=now_shanghai(),
            params={"symbol": symbol}, degraded=True, note=f"fallback: {primary_api} -> {fallback_api}"
        )

# ---- 指数实时（东财 -> 新浪）----
def fetch_index_spot() -> FetchResult:
    """
    指数实时：优先东财 stock_zh_index_spot_em()，失败降级新浪 stock_zh_index_spot_sina()
    """
    primary_api = "stock_zh_index_spot_em"
    fallback_api = "stock_zh_index_spot_sina"
    try:
        df = call_with_retry(primary_api, ak.stock_zh_index_spot_em, timeout=10.0, retries=2, backoff_seq=(0.8, 1.5))
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{primary_api} 未返回 DataFrame")
        return FetchResult(data=df, api=primary_api, fetched_at=now_shanghai(), params={})
    except Exception:
        df = call_with_retry(fallback_api, ak.stock_zh_index_spot_sina, timeout=12.0, retries=2, backoff_seq=(1.0, 2.0))
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{fallback_api} 未返回 DataFrame")
        return FetchResult(
            data=df, api=fallback_api, fetched_at=now_shanghai(),
            params={}, degraded=True, note=f"fallback: {primary_api} -> {fallback_api}"
        )

# ---- 全市场快照（东财 -> 新浪）----
def fetch_market_spot_with_fallback() -> FetchResult:
    """
    全市场快照：优先调用 providers.concentration_provider.fetch_market_spot()（东财）。
    若失败则降级到新浪 stock_zh_a_spot()，并尽可能补齐统一字段（特别是成交额 -> amount_yuan）。
    """
    try:
        res = _fetch_market_spot_em()
        if not isinstance(res, FetchResult):
            raise TypeError("providers.fetch_market_spot 返回类型错误")
        return res
    except Exception:
        # fallback to 新浪
        fallback_api = "stock_zh_a_spot"
        df = call_with_retry(fallback_api, ak.stock_zh_a_spot, timeout=15.0, retries=2, backoff_seq=(1.0, 3.0))
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{fallback_api} 未返回 DataFrame")

        # 统一成交额列 amount_yuan（新浪字段名：'成交额'，单位：元）
        if isinstance(df, pd.DataFrame):
            if "amount_yuan" not in df.columns:
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

        return FetchResult(
            data=df, api=fallback_api, fetched_at=now_shanghai(),
            params={}, degraded=True, note="fallback: stock_zh_a_spot_em -> stock_zh_a_spot"
        )

# ---- 涨跌停池（东财，无备用）----
def fetch_limit_up_pool(date: Optional[str] = None) -> FetchResult:
    return _fetch_limit_up_pool(date=date)

def fetch_limit_down_pool(date: Optional[str] = None) -> FetchResult:
    return _fetch_limit_down_pool(date=date)

# ---- 两融（交易所披露，无备用源）----
def fetch_margin_sse(start_date: str, end_date: str) -> FetchResult:
    return _fetch_margin_sse(start_date=start_date, end_date=end_date)

def fetch_margin_szse(date: str) -> FetchResult:
    return _fetch_margin_szse(date=date)

# ---- P0 一次性打包 ----
@dataclass
class P0Bundle:
    """
    P0 采集打包结果（最小可用）
    """
    zt_pool: FetchResult
    dtgc_pool: FetchResult
    market_spot: FetchResult
    index_daily_sh: FetchResult
    index_daily_hs300: FetchResult
    index_spot: FetchResult
    margin_sse: FetchResult
    margin_szse: FetchResult

def fetch_p0_bundle(prev_trade_yyyymmdd: str,
                    sh_symbol: str = SH_COMPOSITE,
                    hs300_symbol: str = HS300) -> P0Bundle:
    """
    拉取最小 P0 包（用于“盘前情绪闸门 · Alt-Flow”）
    :param prev_trade_yyyymmdd: 上一交易日（YYYYMMDD），用于 EOD 性质的接口
    """
    # ① 涨停池/② 跌停池（东财）
    zt = fetch_limit_up_pool(prev_trade_yyyymmdd)
    dt = fetch_limit_down_pool(prev_trade_yyyymmdd)

    # ③ 全市场快照（东财 -> 新浪）
    spot = fetch_market_spot_with_fallback()

    # ④ 指数日线（东财 -> 新浪）
    d_sh = fetch_index_daily(sh_symbol)
    d_hs = fetch_index_daily(hs300_symbol)

    # ⑤ 指数实时（东财 -> 新浪）
    isp = fetch_index_spot()

    # ⑥ 两融汇总（上一交易日；交易所披露）
    sse = fetch_margin_sse(prev_trade_yyyymmdd, prev_trade_yyyymmdd)
    szs = fetch_margin_szse(prev_trade_yyyymmdd)

    return P0Bundle(
        zt_pool=zt, dtgc_pool=dt, market_spot=spot,
        index_daily_sh=d_sh, index_daily_hs300=d_hs, index_spot=isp,
        margin_sse=sse, margin_szse=szs
    )
