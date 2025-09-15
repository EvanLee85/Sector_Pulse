# tests/06_test_akshare_gate_inputs_api.py
# 目标：验证步骤2所需的 AKShare 输入接口可用性 + 口径计算无异常
# 覆盖：交易日历、涨停家数、北向净流、HS300 30D 年化波动近一年分位、上证是否站上 MA20
# ！！！！！！！！！！！！！！
# 检查short_second_strategy的北向资金是如何处理的，因为24年5月13日起北向资金就不再公布了，我之前不知道；所以上一版本是如何处理的？是不是可靠的？能不能用到sectorpulse中？
import sys
import json
import math
import traceback
from datetime import datetime, date, timedelta
from typing import Optional

import pandas as pd
import numpy as np

try:
    import akshare as ak
except Exception as e:
    print(f"[FAIL] 无法导入 akshare：{e}")
    sys.exit(1)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)


def _today_cn() -> date:
    # 简化：按本地系统日历理解"今天"
    return datetime.now().date()


def get_trade_dates() -> pd.DataFrame:
    """交易日历（Sina 口径），列名标准化为 trade_date"""
    df = ak.tool_trade_date_hist_sina()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df[["trade_date"]]


def last_trading_day(trade_dates: pd.Series, ref: Optional[date] = None) -> date:
    ref = ref or _today_cn()
    s = pd.Series(sorted(list(set(trade_dates))))
    s = s[s <= ref]
    if s.empty:
        raise RuntimeError("未找到最近交易日")
    return s.iloc[-1]


def fetch_limit_up_count(td: date) -> int:
    """涨停家数：优先 stock_zt_pool_em(date=YYYYMMDD)，兜底 previous 或无参"""
    ds = td.strftime("%Y%m%d")
    # 优先：当日涨停池
    try:
        df = ak.stock_zt_pool_em(date=ds)
        if isinstance(df, pd.DataFrame):
            return int(len(df))
    except Exception:
        pass
    # 其次：昨日涨停池
    try:
        df = ak.stock_zt_pool_previous_em(trade_date=ds)
        if isinstance(df, pd.DataFrame):
            return int(len(df))
    except Exception:
        pass
    # 兜底：无参（有些版本支持）
    try:
        df = ak.stock_zt_pool_em()
        if isinstance(df, pd.DataFrame):
            return int(len(df))
    except Exception:
        pass
    raise RuntimeError("涨停家数接口不可用或无数据")


def fetch_northbound_net_flow(td: date) -> float:
    """
    北向资金净流入（单位：亿元）
    主源：stock_hsgt_fund_flow_summary_em()（使用汇总接口）
    备源1：stock_hsgt_north_net_flow_in_em()（按日期精确取值）
    备源2：stock_hsgt_hist_em()
    """
    
    # A) 主源：使用汇总接口（根据文档，这个接口更稳定）
    try:
        dfs = ak.stock_hsgt_fund_flow_summary_em()
        if isinstance(dfs, pd.DataFrame) and not dfs.empty:
            # 筛选北向资金且交易状态为3（收盘）的记录
            north_df = dfs[(dfs["交易状态"] == 3) & (dfs["资金方向"].str.contains("北向", na=False))].copy()
            
            if not north_df.empty:
                # 检查日期是否匹配
                if "交易日" in north_df.columns:
                    north_df["交易日"] = pd.to_datetime(north_df["交易日"]).dt.date
                    today_data = north_df[north_df["交易日"] == td]
                    
                    if not today_data.empty:
                        # 优先使用"资金净流入"列，其次使用"成交净买额"
                        for col in ["资金净流入", "成交净买额"]:
                            if col in today_data.columns:
                                # 汇总沪股通和深股通的数据
                                total = pd.to_numeric(today_data[col], errors="coerce").sum()
                                if pd.notna(total):
                                    return float(total)
                    else:
                        # 如果日期不匹配，使用最新的数据
                        for col in ["资金净流入", "成交净买额"]:
                            if col in north_df.columns:
                                total = pd.to_numeric(north_df[col], errors="coerce").sum()
                                if pd.notna(total):
                                    print(f"[WARN] 使用最新可用日期的北向资金数据（非{td}）")
                                    return float(total)
    except Exception as e:
        print(f"[DEBUG] stock_hsgt_fund_flow_summary_em 失败: {e}")
        pass

    # B) 备源1：日度时间序列
    try:
        dfn = ak.stock_hsgt_north_net_flow_in_em()
        dcol = "日期" if "日期" in dfn.columns else ("date" if "date" in dfn.columns else None)
        if dcol:
            dfn[dcol] = pd.to_datetime(dfn[dcol]).dt.date
            row = dfn.loc[dfn[dcol] == td]
            if not row.empty:
                for col in ["北向资金", "北上资金净流入", "净流入", "净买额", "north_money", "净流入额"]:
                    if col in row.columns:
                        val = pd.to_numeric(row.iloc[0][col], errors="coerce")
                        if pd.notna(val):
                            return float(val)
    except Exception as e:
        print(f"[DEBUG] stock_hsgt_north_net_flow_in_em 失败: {e}")
        pass

    # C) 备源2：历史表
    try:
        dfh = ak.stock_hsgt_hist_em()
        dcol = "日期" if "日期" in dfh.columns else ("date" if "date" in dfh.columns else None)
        if dcol:
            dfh[dcol] = pd.to_datetime(dfh[dcol]).dt.date
            row = dfh.loc[dfh[dcol] == td]
            if not row.empty:
                for col in ["北向资金", "北上资金净流入", "净流入", "净买额", "north_money", "净流入额"]:
                    if col in row.columns:
                        val = pd.to_numeric(row.iloc[0][col], errors="coerce")
                        if pd.notna(val):
                            return float(val)
    except Exception as e:
        print(f"[DEBUG] stock_hsgt_hist_em 失败: {e}")
        pass

    # D) 最后尝试：直接从汇总接口获取最新数据（不考虑日期）
    try:
        dfs = ak.stock_hsgt_fund_flow_summary_em()
        if isinstance(dfs, pd.DataFrame) and not dfs.empty:
            north_df = dfs[(dfs["交易状态"] == 3) & (dfs["资金方向"].str.contains("北向", na=False))]
            if not north_df.empty:
                for col in ["资金净流入", "成交净买额"]:
                    if col in north_df.columns:
                        total = pd.to_numeric(north_df[col], errors="coerce").sum()
                        if pd.notna(total):
                            print(f"[WARN] 使用最新可用的北向资金数据（可能非{td}）")
                            return float(total)
    except Exception:
        pass

    raise RuntimeError(f"无法获取{td}的北向净流(亿)，所有数据源均失败")


def fetch_northbound_summary_for_debug() -> float | None:
    """
    从 stock_hsgt_fund_flow_summary_em() 读取当日"北向+收盘"两行的合计（单位：亿）
    仅用于打印与主源对比，不参与最终数值
    """
    try:
        dfs = ak.stock_hsgt_fund_flow_summary_em()
        if not isinstance(dfs, pd.DataFrame) or dfs.empty:
            return None
        dfs2 = dfs[(dfs["交易状态"] == 3) & (dfs["资金方向"].astype(str).str.contains("北向"))].copy()
        if dfs2.empty:
            return None
        for col in ["资金净流入", "成交净买额"]:
            if col in dfs2.columns:
                s = pd.to_numeric(dfs2[col], errors="coerce")
                if s.notna().any():
                    return float(s.sum())
    except Exception:
        return None
    return None


def _try_fetch_index_daily(symbol_variants: list[str]) -> pd.DataFrame:
    """指数历史：多接口/多列名兼容，标准化为 [date, close]"""
    last_err = None
    for sym in symbol_variants:
        for fn in ("stock_zh_index_daily", "stock_zh_index_daily_em", "stock_zh_index_daily_tx"):
            try:
                api = getattr(ak, fn)
                df = api(symbol=sym)
                # 标准化列
                if "date" not in df.columns:
                    if "日期" in df.columns:
                        df = df.rename(columns={"日期": "date"})
                    elif "时间" in df.columns:
                        df = df.rename(columns={"时间": "date"})
                if "close" not in df.columns:
                    for k in ("收盘", "收盘价", "close_price", "收盘点数", "收盘指数"):
                        if k in df.columns:
                            df = df.rename(columns={k: "close"})
                            break
                if "date" in df.columns and "close" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.sort_values("date").reset_index(drop=True)
                    if len(df) > 0:
                        return df[["date", "close"]]
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"指数历史数据获取失败：{last_err}")


def fetch_vol_percentile_hs300(td: date) -> tuple[float, float]:
    """
    返回：(最新年化波动, 近一年分位[0~100])
    算法：对HS300收盘价，算对数收益的30D滚动标准差 * sqrt(252)
    """
    hs300_syms = ["sh000300", "000300.SH", "000300", "399300.SZ", "sz399300"]
    df = _try_fetch_index_daily(hs300_syms)

    # 取至目标交易日，保留最近 ~ 450 个样本
    df = df[df["date"] <= pd.Timestamp(td)].tail(450)

    ret = np.log(df["close"] / df["close"].shift(1))
    vol30 = ret.rolling(30).std() * math.sqrt(252)
    df["vol30"] = vol30

    recent = df.dropna(subset=["vol30"]).tail(252)  # 近一年
    if recent.empty:
        raise RuntimeError("HS300 滚动波动样本不足")

    latest = float(recent["vol30"].iloc[-1])
    pct = float((recent["vol30"] <= latest).mean() * 100.0)
    return latest, pct


def fetch_sh_above_ma20(td: date) -> tuple[float, float, bool]:
    """上证指数是否站上MA20"""
    sh_syms = ["sh000001", "000001.SH", "000001"]
    df = _try_fetch_index_daily(sh_syms)
    df = df[df["date"] <= pd.Timestamp(td)].tail(120)
    df["ma20"] = df["close"].rolling(20).mean()
    last = df.dropna().iloc[-1]
    close = float(last["close"])
    ma20 = float(last["ma20"])
    return close, ma20, bool(close >= ma20)


def debug_northbound_data():
    """调试函数：打印各个接口返回的数据结构，帮助诊断问题"""
    print("=" * 80)
    print("调试北向资金数据接口")
    print("=" * 80)
    
    # 1. 测试 stock_hsgt_fund_flow_summary_em
    print("\n1. stock_hsgt_fund_flow_summary_em():")
    print("-" * 40)
    try:
        df = ak.stock_hsgt_fund_flow_summary_em()
        print(f"返回类型: {type(df)}")
        if isinstance(df, pd.DataFrame):
            print(f"数据形状: {df.shape}")
            print(f"列名: {list(df.columns)}")
            print("\n前5行数据:")
            print(df.head())
            
            # 筛选北向资金
            north_df = df[(df["交易状态"] == 3) & (df["资金方向"].str.contains("北向", na=False))]
            print(f"\n北向资金记录数: {len(north_df)}")
            if not north_df.empty:
                print("北向资金数据:")
                print(north_df[["交易日", "类型", "板块", "资金方向", "资金净流入", "成交净买额"]])
                
                # 计算总和
                if "资金净流入" in north_df.columns:
                    total = pd.to_numeric(north_df["资金净流入"], errors="coerce").sum()
                    print(f"\n北向资金净流入总和: {total} 亿元")
    except Exception as e:
        print(f"错误: {e}")
    
    # 2. 测试 stock_hsgt_north_net_flow_in_em
    print("\n2. stock_hsgt_north_net_flow_in_em():")
    print("-" * 40)
    try:
        df = ak.stock_hsgt_north_net_flow_in_em()
        print(f"返回类型: {type(df)}")
        if isinstance(df, pd.DataFrame):
            print(f"数据形状: {df.shape}")
            print(f"列名: {list(df.columns)}")
            print("\n最后5行数据:")
            print(df.tail())
    except Exception as e:
        print(f"错误: {e}")
    
    # 3. 测试 stock_hsgt_hist_em
    print("\n3. stock_hsgt_hist_em():")
    print("-" * 40)
    try:
        df = ak.stock_hsgt_hist_em()
        print(f"返回类型: {type(df)}")
        if isinstance(df, pd.DataFrame):
            print(f"数据形状: {df.shape}")
            print(f"列名: {list(df.columns)}")
            print("\n最后5行数据:")
            print(df.tail())
    except Exception as e:
        print(f"错误: {e}")
    
    print("\n" + "=" * 80)


def main():
    try:
        print("[INFO] 开始接口与口径核验…")

        # 交易日历 & 最近交易日
        cal = get_trade_dates()
        assert not cal.empty, "交易日历为空"
        td = last_trading_day(cal["trade_date"])
        print(f"[OK] 最近交易日：{td}")

        # 涨停家数
        zt_count = fetch_limit_up_count(td)
        print(f"[OK] 涨停家数：{zt_count}")

        # 北向净流（主源：汇总接口；备源：日度时间序列/历史表）
        north_yi = fetch_northbound_net_flow(td)

        # 汇总接口（仅校验，不参与数值）
        summary_yi = None
        try:
            summary_yi = fetch_northbound_summary_for_debug()
        except Exception:
            summary_yi = None

        msg = f"[OK] 北向净流(亿)：{north_yi}"
        if summary_yi is not None:
            msg += f"  [summary={summary_yi}]"
            if abs(summary_yi - north_yi) > 10:
                msg += "  [WARN mismatch>10]"
        print(msg)

        # HS300 30D 年化波动及分位
        vol_val, vol_pct = fetch_vol_percentile_hs300(td)
        print(f"[OK] HS300 30D年化波动：{vol_val:.4f}，近一年分位：{vol_pct:.1f}%")

        # 上证是否站上 MA20
        sh_close, sh_ma20, sh_above = fetch_sh_above_ma20(td)
        print(f"[OK] 上证收盘={sh_close:.2f}，MA20={sh_ma20:.2f}，站上MA20={sh_above}")

        # 汇总输出
        payload = {
            "trade_date": td.isoformat(),
            "limit_up_count": int(zt_count),
            "northbound_netflow_yi": float(north_yi),
            "hs300_vol30_ann": float(vol_val),
            "hs300_vol30_pct_1y": float(vol_pct),
            "sse_close": float(sh_close),
            "sse_ma20": float(sh_ma20),
            "sse_above_ma20": bool(sh_above),
        }
        print("[SUMMARY]", json.dumps(payload, ensure_ascii=False))
        print("✅ ALL API INPUTS OK")

    except Exception as e:
        print(f"[FAIL] 异常：{e}")
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    import sys
    
    # 如果带 --debug 参数，运行调试函数
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        debug_northbound_data()
    else:
        main()