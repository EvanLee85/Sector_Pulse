# data_service/proxies/altflow_proxy.py
# -*- coding: utf-8 -*-
"""
指标计算器（P0 · Alt-Flow）
- 基于 collector.P0Bundle 计算本步所需的最小指标集（dict 形式，非强类型）
- 产出键：
  limit_up_count, limit_down_count, limitup_down_ratio,
  turnover_concentration_top20,
  hs300_vol_30d_annualized, vol_percentile,
  sh_above_ma20,
  margin_net_repay_yi_prev
- 系统字段：
  schema_version, calc_at(带时区), tz, window, intraday_missing, sources
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from math import sqrt
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from ..storage import now_shanghai
from ..models import FetchResult
from ..collector import P0Bundle, fetch_p0_bundle

_SH_TZ = ZoneInfo("Asia/Shanghai")


# ---------------------------- 工具函数 ----------------------------

def _pick_first_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _close_series_from_index_daily(df: pd.DataFrame) -> pd.Series:
    """
    从指数日线 DataFrame 提取按日期升序的收盘价序列（float），索引为 datetime64[ns, Asia/Shanghai]
    兼容常见列名：日期/date，收盘/close/收盘价
    """
    if df is None or df.empty:
        return pd.Series(dtype="float64")

    date_col = _pick_first_col(df, ("日期", "date", "交易日", "日期时间"))
    close_col = _pick_first_col(df, ("收盘", "close", "收盘价", "最新价"))

    if date_col is None or close_col is None:
        return pd.Series(dtype="float64")

    out = df[[date_col, close_col]].copy()
    # 解析日期
    try:
        out[date_col] = pd.to_datetime(out[date_col])
    except Exception:
        # 尝试按常见格式再次解析
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce", utc=False)
    out = out.dropna(subset=[date_col]).sort_values(date_col)
    out = out.set_index(date_col)[close_col].astype("float64")
    # 将索引标记为上海时区（若无 tz）
    if out.index.tz is None:
        out.index = out.index.tz_localize(_SH_TZ, nonexistent="shift_forward", ambiguous="NaT")
    return out


def _rolling_vol_annualized(close: pd.Series, window: int = 30, trading_days: int = 252) -> pd.Series:
    """
    计算滚动年化波动（对数收益的 rolling std * sqrt(252)）
    """
    if close is None or close.dropna().shape[0] < window + 1:
        return pd.Series(dtype="float64")
    logret = np.log(close / close.shift(1))
    vol = logret.rolling(window=window).std(ddof=0) * sqrt(trading_days)
    return vol


def _percentile_of_last(series: pd.Series, ref_days: int = 252) -> Optional[float]:
    """
    取最后一个值在最近 ref_days 样本中的分位（0~1）
    """
    s = series.dropna()
    if s.empty:
        return None
    last = s.iloc[-1]
    ref = s.iloc[-ref_days:] if s.shape[0] >= ref_days else s
    if ref.empty:
        return None
    # 分位：小于等于 last 的占比
    rank = (ref <= last).sum()
    return float(rank) / float(ref.shape[0])


def _extract_sh_spot_price(df: pd.DataFrame) -> Optional[float]:
    """
    从指数快照表中提取上证综指的“当前价/最新价”
    尝试按代码或名称匹配；兼容常见列名
    """
    if df is None or df.empty:
        return None

    # 代码列
    code_col = _pick_first_col(df, ("代码", "index_code", "symbol", "指数代码", "代码代码"))
    name_col = _pick_first_col(df, ("名称", "index_name", "指数名称"))

    row = None
    # 优先按代码匹配
    if code_col:
        codes = df[code_col].astype(str).str.upper()
        mask = codes.isin({"000001", "SH000001", "SH.000001", "SH-000001", "1A0001", "SZ399001"})  # 扩容常见写法
        cand = df[mask]
        if not cand.empty:
            row = cand.iloc[0]

    # 再按名称匹配
    if row is None and name_col:
        names = df[name_col].astype(str)
        cand = df[names.str.contains("上证指数|上证综指|上证综合", regex=True, na=False)]
        if not cand.empty:
            row = cand.iloc[0]

    if row is None:
        # 保底：尝试第一行
        row = df.iloc[0]

    price_col = _pick_first_col(df, ("最新价", "现价", "价格", "close", "收盘", "最新"))
    if price_col is None:
        return None
    try:
        return float(row[price_col])
    except Exception:
        return None


def _sum_amount_topN_ratio(df: pd.DataFrame, topN: int = 20) -> Optional[float]:
    """
    市场成交额集中度：按 amount_yuan 降序取前 N 占比
    """
    if df is None or df.empty or "amount_yuan" not in df.columns:
        return None
    s = pd.to_numeric(df["amount_yuan"], errors="coerce").dropna()
    if s.empty:
        return None
    total = float(s.sum())
    if total <= 0:
        return None
    top = float(s.nlargest(int(topN)).sum())
    return max(0.0, min(1.0, top / total))


def _pick_row_by_date(df: pd.DataFrame, date_col: str, target_yyyy_mm_dd: str) -> Optional[pd.Series]:
    """
    在 df 中根据日期列挑选与目标日期相等（或最近不晚于目标日）的最后一行
    """
    if df is None or df.empty or date_col not in df.columns:
        return None
    tmp = df.copy()
    try:
        tmp[date_col] = pd.to_datetime(tmp[date_col]).dt.strftime("%Y-%m-%d")
    except Exception:
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    tmp = tmp.dropna(subset=[date_col])
    # 优先精确匹配
    target = pd.to_datetime(target_yyyy_mm_dd).strftime("%Y-%m-%d")
    exact = tmp[tmp[date_col] == target]
    if not exact.empty:
        return exact.iloc[-1]
    # 次优：取不晚于目标日的最后一行
    le = tmp[tmp[date_col] <= target]
    if not le.empty:
        return le.iloc[-1]
    return None


# ---------------------------- 主计算入口 ----------------------------

def build_from_bundle(
    bundle: P0Bundle,
    prev_trade_yyyymmdd: str,
    *,
    topN: int = 20,
    vol_window: int = 30,
    vol_ref_days: int = 252,
) -> Dict[str, Any]:
    """
    基于已获取的 P0Bundle 计算指标（最小 P0）
    返回 dict：包含 metrics / system fields
    """
    # ---------- sources ----------
    def _src(fr: FetchResult) -> Dict[str, Any]:
        return {"api": fr.api, "fetched_at": fr.fetched_at.astimezone(_SH_TZ).isoformat()}

    sources = {
        "zt_pool": _src(bundle.zt_pool),
        "dtgc_pool": _src(bundle.dtgc_pool),
        "market_spot": _src(bundle.market_spot),
        "index_daily_sh": _src(bundle.index_daily_sh),
        "index_daily_hs300": _src(bundle.index_daily_hs300),
        "index_spot": _src(bundle.index_spot),
        "margin": {
            "sse": _src(bundle.margin_sse),
            "szse": _src(bundle.margin_szse),
        },
    }

    # ---------- 1) 涨跌停家数与比值 ----------
    try:
        limit_up_count = int(getattr(bundle.zt_pool.data, "shape", (0,))[0])
    except Exception:
        limit_up_count = None  # type: ignore

    try:
        limit_down_count = int(getattr(bundle.dtgc_pool.data, "shape", (0,))[0])
    except Exception:
        limit_down_count = None  # type: ignore

    if (limit_up_count is not None) and (limit_down_count is not None):
        limitup_down_ratio = float(limit_up_count) / float(max(1, limit_down_count))
    else:
        limitup_down_ratio = None  # type: ignore

    # ---------- 2) 成交额集中度（TopN） ----------
    try:
        turnover_concentration_top20 = _sum_amount_topN_ratio(bundle.market_spot.data, topN=topN)
    except Exception:
        turnover_concentration_top20 = None  # type: ignore

    # ---------- 3) HS300 波动与分位 ----------
    hs300_close = _close_series_from_index_daily(bundle.index_daily_hs300.data)
    vol30_series = _rolling_vol_annualized(hs300_close, window=vol_window, trading_days=252)
    hs300_vol_30d_annualized: Optional[float] = None
    vol_percentile: Optional[float] = None
    if not vol30_series.empty:
        try:
            hs300_vol_30d_annualized = float(vol30_series.iloc[-1])
            vol_percentile = _percentile_of_last(vol30_series, ref_days=vol_ref_days)
        except Exception:
            hs300_vol_30d_annualized = None
            vol_percentile = None

    # ---------- 4) 上证是否站上 MA20（昨收 MA20 vs 盘中现价） ----------
    sh_close = _close_series_from_index_daily(bundle.index_daily_sh.data)
    ma20 = sh_close.rolling(window=20).mean()
    # 取“上一交易日”的 MA20（避免前视）：用 prev_trade_yyyymmdd 对齐
    ma20_yday: Optional[float] = None
    if not ma20.empty:
        try:
            # 若索引有时区，按日期字符串对齐
            prev_dt = pd.to_datetime(prev_trade_yyyymmdd)
            prev_str = prev_dt.strftime("%Y-%m-%d")
            idx_str = pd.Index([pd.Timestamp(i).tz_convert(_SH_TZ).strftime("%Y-%m-%d") for i in ma20.index])
            s = pd.Series(ma20.values, index=idx_str)
            if prev_str in s.index:
                ma20_yday = float(s.loc[prev_str])
            else:
                # 取不晚于 prev 的最后值
                s2 = s[s.index <= prev_str]
                if not s2.empty:
                    ma20_yday = float(s2.iloc[-1])
        except Exception:
            ma20_yday = None

    sh_spot_price: Optional[float] = None
    try:
        sh_spot_price = _extract_sh_spot_price(bundle.index_spot.data)
    except Exception:
        sh_spot_price = None

    sh_above_ma20: Optional[bool] = None
    if (ma20_yday is not None) and (sh_spot_price is not None):
        try:
            sh_above_ma20 = bool(float(sh_spot_price) >= float(ma20_yday))
        except Exception:
            sh_above_ma20 = None

    # ---------- 5) 两融净买入（背景项，单位：亿元） ----------
    def _net_buy_yi(df: pd.DataFrame) -> Optional[float]:
        if df is None or df.empty:
            return None
        # 优先使用标准化列
        if "net_buy_亿" in df.columns:
            try:
                # 若有多行，取目标日那一行，否则取最后行
                row = None
                date_col = _pick_first_col(df, ("date", "日期", "交易日期"))
                if date_col:
                    row = _pick_row_by_date(df, date_col, prev_trade_yyyymmdd)
                if row is None:
                    row = df.iloc[-1]
                return float(pd.to_numeric(row["net_buy_亿"], errors="coerce"))
            except Exception:
                pass
        # 退化：从 buy/repay 推导
        buy_col = _pick_first_col(df, ("buy_亿", "buy"))
        repay_col = _pick_first_col(df, ("repay_亿", "repay"))
        if buy_col and repay_col:
            try:
                row = None
                date_col = _pick_first_col(df, ("date", "日期", "交易日期"))
                if date_col:
                    row = _pick_row_by_date(df, date_col, prev_trade_yyyymmdd)
                if row is None:
                    row = df.iloc[-1]
                b = float(pd.to_numeric(row[buy_col], errors="coerce"))
                r = float(pd.to_numeric(row[repay_col], errors="coerce"))
                # 若单位为“元”，则转为“亿”
                if buy_col == "buy":
                    b = b / 1e8
                if repay_col == "repay":
                    r = r / 1e8
                return b - r
            except Exception:
                return None
        return None

    sse_yi = _net_buy_yi(bundle.margin_sse.data)
    szse_yi = _net_buy_yi(bundle.margin_szse.data)

    # 兜底：余额差法（任意一侧缺失则尝试补齐）
    if sse_yi is None or szse_yi is None:
        try:
            from ..providers.margin_provider import fetch_margin_netbuy_yi_sse, fetch_margin_netbuy_yi_szse
            if sse_yi is None:
                sse_yi = fetch_margin_netbuy_yi_sse(prev_trade_yyyymmdd)
            if szse_yi is None:
                szse_yi = fetch_margin_netbuy_yi_szse(prev_trade_yyyymmdd)
        except Exception:
            pass

    margin_net_repay_yi_prev: Optional[float] = None
    if (sse_yi is not None) and (szse_yi is not None):
        try:
            margin_net_repay_yi_prev = float(sse_yi) + float(szse_yi)
        except Exception:
            margin_net_repay_yi_prev = None

    # ---------- 缺失标记（盘中关键项缺失即置 True） ----------
    intraday_missing = False
    critical_values = [
        limit_up_count,
        limit_down_count,
        limitup_down_ratio,
        turnover_concentration_top20,
        sh_above_ma20,
    ]
    for v in critical_values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            intraday_missing = True
            break

    # ---------- 输出结构 ----------
    metrics: Dict[str, Any] = {
        "limit_up_count": limit_up_count,
        "limit_down_count": limit_down_count,
        "limitup_down_ratio": None if limitup_down_ratio is None else float(limitup_down_ratio),
        "turnover_concentration_top20": None if turnover_concentration_top20 is None else float(turnover_concentration_top20),
        "hs300_vol_30d_annualized": hs300_vol_30d_annualized,
        "vol_percentile": vol_percentile,
        "sh_above_ma20": sh_above_ma20,
        "margin_net_repay_yi_prev": margin_net_repay_yi_prev,
    }

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "calc_at": now_shanghai(),
        "tz": "Asia/Shanghai",
        "window": {
            "prev_trade_date": prev_trade_yyyymmdd,
            "topN": topN,
            "vol_window": vol_window,
            "vol_ref_days": vol_ref_days,
        },
        "intraday_missing": intraday_missing,
        "metrics": metrics,
        "sources": sources,
    }
    return payload


def run_indicator_calculator(
    prev_trade_yyyymmdd: str,
    *,
    topN: int = 20,
    vol_window: int = 30,
    vol_ref_days: int = 252,
) -> Dict[str, Any]:
    """
    便捷入口：内部拉取 P0Bundle 后计算指标
    """
    bundle = fetch_p0_bundle(prev_trade_yyyymmdd=prev_trade_yyyymmdd)
    return build_from_bundle(
        bundle=bundle,
        prev_trade_yyyymmdd=prev_trade_yyyymmdd,
        topN=topN,
        vol_window=vol_window,
        vol_ref_days=vol_ref_days,
    )


__all__ = [
    "build_from_bundle",
    "run_indicator_calculator",
]
