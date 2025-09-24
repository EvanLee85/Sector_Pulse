# -*- coding: utf-8 -*-
"""
统一数据接口（含一主两备降级策略 + 申万行业信息/成分）。
本步仅实现“拉数”与标准化，不做指标计算。
- 个股日线：优先东财(_em/_hist) → 备1 腾讯(hist_tx) → 备2 新浪(stock_zh_a_daily)
- 申万行业信息：提供 L1/L2/L3 信息与 L3 成分获取；支持传入 '801010' 或 '801010.SI'
- 所有返回统一到 FetchResult（见 data_service/models.py）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import time, math
import datetime as _dt

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

# ============================================================================
# 指数 / 市场快照 / 涨跌停 / 两融  —— 维持既有实现
# ============================================================================

def fetch_index_daily(symbol: str) -> FetchResult:
    """
    指数日线：优先东财 stock_zh_index_daily_em(symbol)，失败降级新浪 stock_zh_index_daily(symbol)
    """
    primary_api = "stock_zh_index_daily_em"
    fallback_api = "stock_zh_index_daily"
    try:
        df = call_with_retry(
            primary_api, ak.stock_zh_index_daily_em,
            symbol=symbol, timeout=10.0, retries=2, backoff_seq=(0.8, 1.5)
        )
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{primary_api} 未返回 DataFrame")
        return FetchResult(data=df, api=primary_api, fetched_at=now_shanghai(), params={"symbol": symbol})
    except Exception:
        df = call_with_retry(
            fallback_api, ak.stock_zh_index_daily,
            symbol=symbol, timeout=12.0, retries=2, backoff_seq=(1.0, 2.0)
        )
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{fallback_api} 未返回 DataFrame")
        return FetchResult(
            data=df, api=fallback_api, fetched_at=now_shanghai(),
            params={"symbol": symbol}, degraded=True, note=f"fallback: {primary_api} -> {fallback_api}"
        )


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


def fetch_market_spot_with_fallback() -> FetchResult:
    """
    全市场快照：优先 providers.concentration_provider.fetch_market_spot()（东财）。
    若失败则降级新浪 stock_zh_a_spot()，并尽可能补齐统一字段 amount_yuan。
    """
    try:
        res = _fetch_market_spot_em()
        if not isinstance(res, FetchResult):
            raise TypeError("providers.fetch_market_spot 返回类型错误")
        return res
    except Exception:
        fallback_api = "stock_zh_a_spot"
        df = call_with_retry(fallback_api, ak.stock_zh_a_spot, timeout=15.0, retries=2, backoff_seq=(1.0, 3.0))
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{fallback_api} 未返回 DataFrame")

        # 统一成交额列 amount_yuan
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


def fetch_limit_up_pool(date: Optional[str] = None) -> FetchResult:
    return _fetch_limit_up_pool(date=date)


def fetch_limit_down_pool(date: Optional[str] = None) -> FetchResult:
    return _fetch_limit_down_pool(date=date)


def fetch_margin_sse(start_date: str, end_date: str) -> FetchResult:
    return _fetch_margin_sse(start_date=start_date, end_date=end_date)


def fetch_margin_szse(date: str) -> FetchResult:
    return _fetch_margin_szse(date=date)


# ---- P0 一次性打包（供 Step2/3 使用）----
@dataclass
class P0Bundle:
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
    zt = fetch_limit_up_pool(prev_trade_yyyymmdd)
    dt = fetch_limit_down_pool(prev_trade_yyyymmdd)
    spot = fetch_market_spot_with_fallback()
    d_sh = fetch_index_daily(sh_symbol)
    d_hs = fetch_index_daily(hs300_symbol)
    isp = fetch_index_spot()
    sse = fetch_margin_sse(prev_trade_yyyymmdd, prev_trade_yyyymmdd)
    szs = fetch_margin_szse(prev_trade_yyyymmdd)
    return P0Bundle(
        zt_pool=zt, dtgc_pool=dt, market_spot=spot,
        index_daily_sh=d_sh, index_daily_hs300=d_hs, index_spot=isp,
        margin_sse=sse, margin_szse=szs
    )

# ============================================================================
# 申万行业：L1 “桶” + L3 回填
# ============================================================================

def _ensure_sw_code(code: str) -> str:
    c = code.strip().upper()
    if c.endswith(".SI"):
        return c
    if c.isdigit() and len(c) in (6, 7):  # e.g., 801010
        return f"{c}.SI"
    return c

def fetch_sw_l1_list() -> list[dict]:
    """
    返回形如：
    [
      {"code": "801010", "name": "农林牧渔", "count": 97},
      ...
    ]
    失败时返回空列表（上层做降级处理）。
    """
    try:
        import akshare as ak
        import pandas as pd  # 保证可用
    except Exception:
        return []
    try:
        df = ak.sw_index_first_info()
        if df is None or df.empty:
            return []
        out = []
        for _, row in df.iterrows():
            code = str(row.get("行业代码") or "").replace(".SI", "")
            name = str(row.get("行业名称") or "")
            cnt  = int(row.get("成份个数") or 0)
            if code and name:
                out.append({"code": code, "name": name, "count": cnt})
        return out
    except Exception:
        return []


def fetch_sw_index_first_info() -> FetchResult:
    """
    申万一级行业信息（名字、成分数等）
    """
    api = "sw_index_first_info"
    df = call_with_retry(api, ak.sw_index_first_info, timeout=12.0, retries=2, backoff_seq=(1.0, 2.0))
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{api} 未返回 DataFrame")
    return FetchResult(data=df, api=api, fetched_at=now_shanghai(), params={})


def fetch_sw_index_second_info() -> FetchResult:
    api = "sw_index_second_info"
    df = call_with_retry(api, ak.sw_index_second_info, timeout=12.0, retries=2, backoff_seq=(1.0, 2.0))
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{api} 未返回 DataFrame")
    return FetchResult(data=df, api=api, fetched_at=now_shanghai(), params={})


def fetch_sw_index_third_info() -> FetchResult:
    api = "sw_index_third_info"
    df = call_with_retry(api, ak.sw_index_third_info, timeout=12.0, retries=2, backoff_seq=(1.0, 2.0))
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{api} 未返回 DataFrame")
    return FetchResult(data=df, api=api, fetched_at=now_shanghai(), params={})


def fetch_sw_index_third_cons(symbol: str) -> FetchResult:
    """
    申万三级行业成份
    symbol 示例：'850111.SI'
    """
    symbol = _ensure_sw_code(symbol)
    api = "sw_index_third_cons"
    df = call_with_retry(api, ak.sw_index_third_cons, symbol=symbol, timeout=15.0, retries=2, backoff_seq=(1.0, 2.0))
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{api} 未返回 DataFrame")
    return FetchResult(data=df, api=api, fetched_at=now_shanghai(), params={"symbol": symbol})


def fetch_sw_l1_components(l1_code: str, fallback_l3_codes: Optional[List[str]] = None) -> List[str]:
    """
    申万一级行业成分（多口径尝试，失败则用 L3 合成回填；失败返回 []）
    - 首选：ak.sw_index_cons(l1_code 或 '801010.SI')
    - 兼容：ak.stock_board_industry_sw_cons(code='801010')
    - 回填：若 L1 失败且提供 L3 列表，则汇总其 L3 成分
    返回：‘股票代码’（交易所后缀格式，如 600519.SH/000001.SZ），若来源不带后缀则原样返回。
    """
    l1_code = _ensure_sw_code(l1_code)

    # 1) ak.sw_index_cons
    try:
        df = ak.sw_index_cons(l1_code)
        if isinstance(df, pd.DataFrame) and not df.empty:
            col = "股票代码" if "股票代码" in df.columns else ("symbol" if "symbol" in df.columns else None)
            if col:
                return [str(x).strip() for x in df[col].dropna().tolist()]
    except Exception:
        pass

    # 2) 兼容早期口径（无 .SI）
    try:
        df = ak.stock_board_industry_sw_cons(code=l1_code.split(".")[0])
        if isinstance(df, pd.DataFrame) and not df.empty:
            col = "股票代码" if "股票代码" in df.columns else ("symbol" if "symbol" in df.columns else None)
            if col:
                return [str(x).strip() for x in df[col].dropna().tolist()]
    except Exception:
        pass

    # 3) L3 回填
    symbols: List[str] = []
    if fallback_l3_codes:
        for c in fallback_l3_codes:
            try:
                r = fetch_sw_index_third_cons(_ensure_sw_code(c))
                df3 = r.data
                if isinstance(df3, pd.DataFrame) and not df3.empty:
                    col = "股票代码" if "股票代码" in df3.columns else ("symbol" if "symbol" in df3.columns else None)
                    if col:
                        symbols.extend([str(x).strip() for x in df3[col].dropna().tolist()])
            except Exception:
                continue
        symbols = sorted(list({x for x in symbols if x}))
    return symbols

# ============================================================================
# 个股日线：一用二备（统一列名：date, open, high, low, close, amount[元]）
# ============================================================================

def _normalize_hist_df(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    将不同来源的日线字段统一为:
    - date (datetime.date)
    - open, high, low, close (float)
    - amount (float, 单位：元)
    """
    d = df.copy()

    # 可能出现的列名映射
    col_map_variants = [
        {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交额": "amount"},
        {"date": "date", "open": "open", "high": "high", "low": "low", "close": "close", "amount": "amount"},
        {"时间": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交额": "amount"},
    ]
    for mp in col_map_variants:
        if set(mp.keys()).issubset(set(d.columns)):
            d = d.rename(columns=mp)
            break

    # 日期标准化为 date
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.date

    # 成交额单位统一为“元”
    if "amount" in d.columns:
        ser = pd.to_numeric(d["amount"], errors="coerce")
        # 有些来源以“万元”为单位（东财/腾讯可能不同字段）；根据典型取值粗判
        # 如果金额普遍 < 1e9，我们不直接缩放；若普遍 < 1e6 且又像万元，则 ×1e4
        if ser.dropna().median() < 5e6:  # 粗略启发式
            # 判是否“万元”或“百万元”列名已被合并，无法精确判断时采用保守不缩放
            # 若你已知具体来源字段，可在此分支细化
            pass
        d["amount"] = ser

    return d[["date", "open", "high", "low", "close", "amount"]].sort_values("date").reset_index(drop=True)


def fetch_stock_daily_qfq(symbol: str) -> FetchResult:
    """
    个股日线（前复权）一主两备：
    1) 东财：ak.stock_zh_a_hist(symbol=无交易所后缀, adjust="qfq")
    2) 腾讯：ak.stock_zh_a_hist_tx(symbol=带交易所后缀，如 600000.SH/000001.SZ)
    3) 新浪：ak.stock_zh_a_daily(symbol=sh600000/sz000001)
    """
    fetched_at = now_shanghai()

    # ---- 主：东财（无后缀代码，前复权）----
    primary_api = "stock_zh_a_hist[qfq]"
    try:
        tsym = symbol.upper().replace(".SH", "").replace(".SZ", "")
        df = call_with_retry(
            primary_api, ak.stock_zh_a_hist,
            symbol=tsym, adjust="qfq", timeout=15.0, retries=2, backoff_seq=(1.0, 2.0)
        )
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise RuntimeError("empty df")
        norm = _normalize_hist_df(df, "em")
        return FetchResult(data=norm, api=primary_api, fetched_at=fetched_at, params={"symbol": symbol})
    except Exception:
        pass

    # ---- 备1：腾讯（需要带交易所后缀）----
    fb1_api = "stock_zh_a_hist_tx"
    try:
        sym_tx = symbol.upper()  # 需要 600000.SH 或 000001.SZ
        df = call_with_retry(
            fb1_api, ak.stock_zh_a_hist_tx,
            symbol=sym_tx, timeout=15.0, retries=2, backoff_seq=(1.0, 2.0)
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            norm = _normalize_hist_df(df, "tx")
            return FetchResult(
                data=norm, api=fb1_api, fetched_at=fetched_at,
                params={"symbol": symbol}, degraded=True, note=f"fallback: {primary_api} -> {fb1_api}"
            )
    except Exception:
        pass

    # ---- 备2：新浪（需要 shXXXXXX/szXXXXXX）----
    fb2_api = "stock_zh_a_daily"
    try:
        if symbol.endswith(".SH"):
            sym_sina = "sh" + symbol.split(".")[0]
        elif symbol.endswith(".SZ"):
            sym_sina = "sz" + symbol.split(".")[0]
        else:
            # 无法判断交易所，尝试两次
            sym_sina = "sh" + symbol
        df = call_with_retry(
            fb2_api, ak.stock_zh_a_daily,
            symbol=sym_sina, adjust="qfq", timeout=15.0, retries=2, backoff_seq=(1.0, 2.0)
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            norm = _normalize_hist_df(df, "sina")
            return FetchResult(
                data=norm, api=fb2_api, fetched_at=fetched_at,
                params={"symbol": symbol}, degraded=True, note=f"fallback: {primary_api} -> {fb2_api}"
            )
    except Exception:
        pass

    # 全部失败：返回空 DF，标记 degraded
    return FetchResult(
        data=pd.DataFrame(columns=["date", "open", "high", "low", "close", "amount"]),
        api="unavailable", fetched_at=fetched_at, params={"symbol": symbol},
        degraded=True, note="all stock daily providers failed"
    )
