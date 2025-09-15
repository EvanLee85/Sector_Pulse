# data_service/providers/margin_provider.py
# -*- coding: utf-8 -*-
from __future__ import annotations

try:
    import pandas as pd  # type: ignore
    import akshare as ak  # type: ignore
except Exception as e:
    raise ImportError("需要安装 pandas 与 akshare==1.17.40") from e

from ..models import FetchResult
from ..storage import now_shanghai, call_with_retry


def _to_numeric(series: pd.Series) -> pd.Series:
    """把带逗号/空格的数字字符串安全转为 float；无法解析的置为 NaN。"""
    try:
        return pd.to_numeric(series.astype(str).str.replace(",", "").str.strip(), errors="coerce")
    except Exception:
        return pd.to_numeric(series, errors="coerce")


def _to_yi(series: pd.Series, source: str) -> pd.Series:
    """
    统一派生“亿元”刻度：
      - SSE: 源单位为“元” → /1e8
      - SZSE: 源单位多为“亿元” → 原值即为“亿”
    """
    s = _to_numeric(series)
    if source.upper() == "SSE":
        return s / 1e8
    return s  # SZSE 视为“亿元”


def _normalize_margin_df(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    统一两融数据的关键字段名与单位（补充“亿元”便于对比）。
    - 标准字段名：
        date, financing_balance, seclending_balance, margin_balance,
        buy, repay, net_buy
    - 额外派生字段（亿元）：financing_balance_亿, seclending_balance_亿, margin_balance_亿,
                         buy_亿, repay_亿, net_buy_亿
    - 增加列：source（SSE / SZSE）
    """
    df = df.copy()

    # 列名映射（兼容 SSE/SZSE 常见字段）
    rename_map = {
        # 日期
        "日期": "date",
        "交易日期": "date",
        "信用交易日期": "date",          # SSE 返回

        # 融资/融券余额与合计（带/不带“(元)”都兼容）
        "融资余额(元)": "financing_balance",
        "融资余额": "financing_balance",
        "融券余额(元)": "seclending_balance",
        "融券余额": "seclending_balance",
        "融券余量金额": "seclending_balance",  # SSE 返回
        "融资融券余额(元)": "margin_balance",
        "融资融券余额": "margin_balance",

        # 买入/偿还（带/不带“(元)”都兼容）
        "融资买入额(元)": "buy",
        "融资买入额": "buy",
        "融资偿还额(元)": "repay",
        "融资偿还额": "repay",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # 关键数值列统一为 float（保留“源单位”的数值）
    for col in ["financing_balance", "seclending_balance", "margin_balance", "buy", "repay"]:
        if col in df.columns:
            df[col] = _to_numeric(df[col])

    # 如缺少合计列，尽力推导
    if "margin_balance" not in df.columns:
        if {"financing_balance", "seclending_balance"} <= set(df.columns):
            df["margin_balance"] = df["financing_balance"] + df["seclending_balance"]

    # 净买入（若可得）
    if {"buy", "repay"} <= set(df.columns):
        df["net_buy"] = df["buy"] - df["repay"]

    # 规范日期列为字符串 YYYY-MM-DD（若存在）
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        except Exception:
            # 保底：保留原值
            pass

    # 统一派生“亿元”刻度（核心对齐点）
    for col in ["financing_balance", "seclending_balance", "margin_balance", "buy", "repay", "net_buy"]:
        if col in df.columns:
            df[f"{col}_亿"] = _to_yi(df[col], source=source)

    # 标注来源
    df["source"] = source.upper()
    return df


def fetch_margin_sse(start_date: str, end_date: str) -> FetchResult:
    """
    上交所两融汇总（EOD）
    :param start_date: YYYYMMDD
    :param end_date:   YYYYMMDD
    api: stock_margin_sse
    """
    api = "stock_margin_sse"
    df = call_with_retry(
        api,
        ak.stock_margin_sse,
        start_date=start_date,
        end_date=end_date,
        timeout=15.0,
        retries=2,
        backoff_seq=(1.0, 2.5),
    )
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{api} 未返回 DataFrame")

    df = _normalize_margin_df(df, source="SSE")
    return FetchResult(
        data=df,
        api=api,
        fetched_at=now_shanghai(),
        params={"start_date": start_date, "end_date": end_date},
    )


def fetch_margin_szse(date: str) -> FetchResult:
    """
    深交所两融汇总（EOD）
    :param date: YYYYMMDD
    api: stock_margin_szse
    """
    api = "stock_margin_szse"
    df = call_with_retry(
        api,
        ak.stock_margin_szse,
        date=date,
        timeout=15.0,
        retries=2,
        backoff_seq=(1.0, 2.5),
    )
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{api} 未返回 DataFrame")

    df = _normalize_margin_df(df, source="SZSE")

    # 若源数据无日期列，则用调用参数补一个 YYYY-MM-DD（使用模块顶层的 pd）
    if "date" not in df.columns:
        try:
            df = df.copy()
            df["date"] = pd.to_datetime(date).strftime("%Y-%m-%d")
        except Exception:
            df["date"] = str(date)

    return FetchResult(
        data=df,
        api=api,
        fetched_at=now_shanghai(),
        params={"date": date},
    )


# ------------------------- 余额差法：净买入（亿元）兜底 -------------------------

def fetch_margin_netbuy_yi_sse(date: str) -> float | None:
    """
    计算上交所净买入（亿元），使用融资余额的相邻两日差：
      net_buy_亿 = financing_balance_亿[t] - financing_balance_亿[t-1]
    通过取区间 [date-4, date] 保证能拿到至少两行。
    """
    from datetime import timedelta
    import pandas as pd

    api = "stock_margin_sse"
    dt = pd.to_datetime(date)
    start = (dt - timedelta(days=4)).strftime("%Y%m%d")
    end = dt.strftime("%Y%m%d")

    df = call_with_retry(
        api, ak.stock_margin_sse,
        start_date=start, end_date=end,
        timeout=15.0, retries=2, backoff_seq=(1.0, 2.5),
    )
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None

    df = _normalize_margin_df(df, source="SSE")
    if "date" not in df.columns or df.shape[0] < 2:
        return None

    try:
        tmp = df[["date", "financing_balance_亿"]].dropna()
        tmp["date"] = pd.to_datetime(tmp["date"])
        tmp = tmp.sort_values("date")
        last2 = tmp.tail(2)["financing_balance_亿"].tolist()
        if len(last2) == 2:
            return float(last2[1] - last2[0])
    except Exception:
        return None
    return None


def fetch_margin_netbuy_yi_szse(date: str) -> float | None:
    """
    计算深交所净买入（亿元），用单日调用两次的余额差：
      net_buy_亿 = financing_balance_亿[t] - financing_balance_亿[t-1]
    """
    from datetime import timedelta
    import pandas as pd

    api = "stock_margin_szse"
    dt = pd.to_datetime(date)

    # t（目标日）
    df_t = call_with_retry(api, ak.stock_margin_szse, date=dt.strftime("%Y%m%d"), timeout=15.0, retries=2, backoff_seq=(1.0, 2.5))
    if not isinstance(df_t, pd.DataFrame) or df_t.empty:
        return None
    df_t = _normalize_margin_df(df_t, source="SZSE")
    try:
        ft = float(pd.to_numeric(df_t["financing_balance_亿"].iloc[-1], errors="coerce"))
    except Exception:
        return None

    # t-1（往前回退最多 4 天以避开周末/节假日）
    prev_val = None
    for k in range(1, 5):
        prev = (dt - pd.Timedelta(days=k)).strftime("%Y%m%d")
        df_p = call_with_retry(api, ak.stock_margin_szse, date=prev, timeout=15.0, retries=2, backoff_seq=(1.0, 2.5))
        if isinstance(df_p, pd.DataFrame) and not df_p.empty:
            df_p = _normalize_margin_df(df_p, source="SZSE")
            try:
                prev_val = float(pd.to_numeric(df_p["financing_balance_亿"].iloc[-1], errors="coerce"))
                break
            except Exception:
                continue

    if prev_val is None:
        return None
    return ft - prev_val
