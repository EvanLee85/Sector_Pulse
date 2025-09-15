# tests/08_test_data_service.py
# -*- coding: utf-8 -*-
"""
Step 2 · 最小数据层 · P0
一次性测试以下 6 个文件的 I/O 契约、降级策略与最小可计算性：
- data_service/collector.py
- data_service/providers/concentration_provider.py
- data_service/providers/limitpool_provider.py
- data_service/providers/margin_provider.py
- data_service/models.py
- data_service/storage.py

用法：
  python -m tests.08_test_data_service --prev YYYYMMDD --strict
"""

from __future__ import annotations

import sys
import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Iterable, Tuple

# ---------- 彩色标记 ----------
OK = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"

# ---------- 允许的主/备 API 白名单 ----------
EXPECT_APIS: Dict[str, Tuple[str, ...]] = {
    "market_spot": ("stock_zh_a_spot_em", "stock_zh_a_spot"),
    "limit_up": ("stock_zt_pool_em", "stock_zt_pool_previous_em"),
    "limit_down": ("stock_zt_pool_dtgc_em",),
    "index_daily": ("stock_zh_index_daily_em", "stock_zh_index_daily"),
    "index_spot": ("stock_zh_index_spot_em", "stock_zh_index_spot_sina"),
    "margin_sse": ("stock_margin_sse",),
    "margin_szse": ("stock_margin_szse",),
}
PRIMARY_OF = {
    "market_spot": EXPECT_APIS["market_spot"][0],
    "limit_up": EXPECT_APIS["limit_up"][0],
    "limit_down": EXPECT_APIS["limit_down"][0],
    "index_daily": EXPECT_APIS["index_daily"][0],
    "index_spot": EXPECT_APIS["index_spot"][0],
    "margin_sse": EXPECT_APIS["margin_sse"][0],
    "margin_szse": EXPECT_APIS["margin_szse"][0],
}

SH_TZ = ZoneInfo("Asia/Shanghai")


def _print_ok(msg: str) -> None:
    print(f"{OK} {msg}")


def _print_warn(msg: str) -> None:
    print(f"{WARN} {msg}")


def _print_fail(msg: str) -> None:
    print(f"{FAIL} {msg}")


def _ensure_prev(prev_arg: str | None) -> str:
    """未提供 --prev 时，用上海时区 ‘今天-1天’ """
    if prev_arg:
        return prev_arg
    d = datetime.now(SH_TZ) - timedelta(days=1)
    s = d.strftime("%Y%m%d")
    _print_warn(f"未提供 --prev，临时使用 {s}；如遇周末/节假日可能导致部分 EOD 数据为空")
    return s


def _is_df_nonempty(df) -> bool:
    try:
        return getattr(df, "shape", (0,))[0] > 0
    except Exception:
        return False


def _has_any_column(df, cols: Iterable[str]) -> bool:
    try:
        columns = set(getattr(df, "columns", []))
        return any(c in columns for c in cols)
    except Exception:
        return False


def _assert_fetch_result(name: str, res, allow_apis: Tuple[str, ...], strict: bool, required_params: Iterable[str] | None = None) -> int:
    """
    通用断言：
      - 类型/时区/白名单 API
      - 若使用备源：必须 degraded=True 且 note 以 'fallback' 开头
      - 严格模式：data 非空
      - 必要参数快照检查
    返回：0 通过；1 失败
    """
    status = 0
    try:
        from data_service.models import FetchResult  # lazy import for robustness
    except Exception as e:
        _print_fail(f"{name}: 无法导入 FetchResult：{e}")
        return 1

    # 类型
    if not isinstance(res, FetchResult):
        _print_fail(f"{name}: 类型错误，应为 FetchResult，得到 {type(res)}")
        return 1

    # API 白名单
    api = getattr(res, "api", None)
    if api not in allow_apis:
        _print_fail(f"{name}: api 不在允许列表 {allow_apis}，得到 {api}")
        status = 1
    else:
        primary = None
        # index_daily 有两处调用，共享同一个 primary
        if name in ("index_daily_sh", "index_daily_hs300"):
            primary = PRIMARY_OF["index_daily"]
        else:
            # name 是 key 的情形
            base = name
            if base in PRIMARY_OF:
                primary = PRIMARY_OF[base]
        if primary and api != primary:
            if not getattr(res, "degraded", False):
                _print_fail(f"{name}: 使用了备用 {api} 但未标记 degraded=True")
                status = 1
            note = getattr(res, "note", "") or ""
            if not (isinstance(note, str) and note.lower().startswith("fallback")):
                _print_fail(f"{name}: 使用了备用 {api} 但 note 未包含 'fallback'")
                status = 1

    # fetched_at 时区
    fa = getattr(res, "fetched_at", None)
    try:
        _ = fa.astimezone(SH_TZ)
    except Exception as e:
        _print_fail(f"{name}: fetched_at 时区不合法：{e}")
        status = 1

    # params 快照
    if required_params:
        params = getattr(res, "params", {})
        for p in required_params:
            if p not in params:
                _print_fail(f"{name}: params 未包含必需字段 '{p}'")
                status = 1

    # data 非空（严格模式）
    df = getattr(res, "data", None)
    if strict:
        if not _is_df_nonempty(df):
            _print_fail(f"{name}: DataFrame 为空（严格模式）")
            status = 1
        else:
            _print_ok(f"{name}: data 行数={df.shape[0]}")
    else:
        if _is_df_nonempty(df):
            _print_ok(f"{name}: data 行数={df.shape[0]}")
        else:
            _print_warn(f"{name}: DataFrame 为空")

    return status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prev", type=str, default=None, help="上一交易日，格式 YYYYMMDD")
    parser.add_argument("--strict", action="store_true", help="严格模式：关键 DF 必须非空/满足最小阈值")
    args = parser.parse_args(argv)

    prev = _ensure_prev(args.prev)
    strict = bool(args.strict)

    # ---------- 依赖导入 ----------
    try:
        import pandas as pd  # noqa: F401
    except Exception as e:
        _print_fail(f"pandas 未安装：{e}")
        return 1

    try:
        from data_service.providers.concentration_provider import fetch_market_spot
        from data_service.providers.limitpool_provider import (
            fetch_limit_up_pool,
            fetch_limit_down_pool,
        )
        from data_service.providers.margin_provider import (
            fetch_margin_sse,
            fetch_margin_szse,
        )
        from data_service.collector import (
            fetch_index_daily,
            fetch_index_spot,
            fetch_p0_bundle,
            P0Bundle,
        )
    except Exception as e:
        _print_fail(f"导入 data_service 模块失败：{e}")
        return 1

    status = 0

    # ---------- 逐函数测试 ----------
    # 1) 市场快照（东财→新浪）
    try:
        res = fetch_market_spot()
        status |= _assert_fetch_result("market_spot", res, EXPECT_APIS["market_spot"], strict)
        # 专项：amount_yuan 必须存在且为数值列（可计算）
        df = res.data
        if "amount_yuan" not in getattr(df, "columns", []):
            _print_fail("market_spot: 缺少统一字段 amount_yuan")
            status |= 1
        else:
            try:
                nonnull = df["amount_yuan"].notna().sum()
                _print_ok(f"market_spot: amount_yuan 非空个数={int(nonnull)}")
            except Exception as e:
                _print_fail(f"market_spot: 读取 amount_yuan 异常：{e}")
                status |= 1
    except Exception as e:
        _print_fail(f"fetch_market_spot 调用失败：{e}")
        return 1

    # 2) 涨停池（主→备）
    try:
        res = fetch_limit_up_pool(date=prev)
        status |= _assert_fetch_result("limit_up", res, EXPECT_APIS["limit_up"], strict, required_params=("date",))
    except Exception as e:
        _print_fail(f"fetch_limit_up_pool 调用失败：{e}")
        return 1

    # 3) 跌停池（主）
    try:
        res = fetch_limit_down_pool(date=prev)
        status |= _assert_fetch_result("limit_down", res, EXPECT_APIS["limit_down"], strict, required_params=("date",))
    except Exception as e:
        _print_fail(f"fetch_limit_down_pool 调用失败：{e}")
        return 1

    # 4) 两融（上交）
    try:
        res = fetch_margin_sse(start_date=prev, end_date=prev)
        status |= _assert_fetch_result("margin_sse", res, EXPECT_APIS["margin_sse"], strict, required_params=("start_date", "end_date"))
        # 专项：规范字段+派生“亿元”刻度 & source
        df = res.data
        # 宽松：存在核心字段的子集即可
        core_any = any(c in getattr(df, "columns", []) for c in ["financing_balance", "seclending_balance", "margin_balance"])
        if not core_any:
            _print_fail("margin_sse: 缺少标准化核心字段（financing_balance/seclending_balance/margin_balance 至少一个）")
            status |= 1
        if "source" in getattr(df, "columns", []):
            try:
                src_ok = (df["source"].astype(str) == "SSE").any()
                if not src_ok:
                    _print_warn("margin_sse: source 列存在但未包含 'SSE'（宽松处理）")
            except Exception:
                _print_warn("margin_sse: source 列检查异常（宽松处理）")
    except Exception as e:
        _print_fail(f"fetch_margin_sse 调用失败：{e}")
        return 1

    # 5) 两融（深交）
    try:
        res = fetch_margin_szse(date=prev)
        status |= _assert_fetch_result("margin_szse", res, EXPECT_APIS["margin_szse"], strict, required_params=("date",))
        df = res.data
        core_any = any(c in getattr(df, "columns", []) for c in ["financing_balance", "seclending_balance", "margin_balance"])
        if not core_any:
            _print_fail("margin_szse: 缺少标准化核心字段（financing_balance/seclending_balance/margin_balance 至少一个）")
            status |= 1
        if "source" in getattr(df, "columns", []):
            try:
                src_ok = (df["source"].astype(str) == "SZSE").any()
                if not src_ok:
                    _print_warn("margin_szse: source 列存在但未包含 'SZSE'（宽松处理）")
            except Exception:
                _print_warn("margin_szse: source 列检查异常（宽松处理）")
    except Exception as e:
        _print_fail(f"fetch_margin_szse 调用失败：{e}")
        return 1

    # 6) 指数日线（上证/沪深300 各一次）
    try:
        res = fetch_index_daily("sh000001")
        status |= _assert_fetch_result("index_daily_sh", res, EXPECT_APIS["index_daily"], strict)
        df = res.data
        if strict and getattr(df, "shape", (0,))[0] < 20:
            _print_fail("index_daily_sh: 行数不足 20（严格模式）")
            status |= 1
    except Exception as e:
        _print_fail(f"fetch_index_daily(sh000001) 调用失败：{e}")
        return 1

    try:
        res = fetch_index_daily("sh000300")
        status |= _assert_fetch_result("index_daily_hs300", res, EXPECT_APIS["index_daily"], strict)
        df = res.data
        if getattr(df, "shape", (0,))[0] >= 20:
            _print_ok("index_daily_hs300: 行数满足 ≥20")
        elif strict:
            _print_fail("index_daily_hs300: 行数不足 20（严格模式）")
            status |= 1
        else:
            _print_warn("index_daily_hs300: 行数不足 20（宽松模式）")
    except Exception as e:
        _print_fail(f"fetch_index_daily(sh000300) 调用失败：{e}")
        return 1

    # 7) 指数实时
    try:
        res = fetch_index_spot()
        status |= _assert_fetch_result("index_spot", res, EXPECT_APIS["index_spot"], strict)
        df = res.data
        # 至少包含一个指数代码列
        if not _has_any_column(df, ("代码", "index_code", "symbol", "指数代码")):
            _print_warn("index_spot: 未发现常见指数代码列（宽松通过）")
    except Exception as e:
        _print_fail(f"fetch_index_spot 调用失败：{e}")
        return 1

    # ---------- P0 一次性打包 ----------
    try:
        bundle = fetch_p0_bundle(prev_trade_yyyymmdd=prev)
    except Exception as e:
        _print_fail(f"fetch_p0_bundle 调用失败：{e}")
        return 1

    if not isinstance(bundle, P0Bundle):
        _print_fail(f"fetch_p0_bundle 返回类型错误，应为 P0Bundle，得到 {type(bundle)}")
        return 1
    _print_ok("fetch_p0_bundle: 返回 P0Bundle")

    # 针对 8 个成员逐项执行契约检查
    members = [
        ("zt_pool", getattr(bundle, "zt_pool", None), EXPECT_APIS["limit_up"], ("date",)),
        ("dtgc_pool", getattr(bundle, "dtgc_pool", None), EXPECT_APIS["limit_down"], ("date",)),
        ("market_spot", getattr(bundle, "market_spot", None), EXPECT_APIS["market_spot"], ()),
        ("index_daily_sh", getattr(bundle, "index_daily_sh", None), EXPECT_APIS["index_daily"], ()),
        ("index_daily_hs300", getattr(bundle, "index_daily_hs300", None), EXPECT_APIS["index_daily"], ()),
        ("index_spot", getattr(bundle, "index_spot", None), EXPECT_APIS["index_spot"], ()),
        ("margin_sse", getattr(bundle, "margin_sse", None), EXPECT_APIS["margin_sse"], ("start_date", "end_date")),
        ("margin_szse", getattr(bundle, "margin_szse", None), EXPECT_APIS["margin_szse"], ("date",)),
    ]

    for name, res, allow_apis, req_params in members:
        if res is None:
            _print_fail(f"P0Bundle 缺少 {name}")
            status |= 1
            continue
        status |= _assert_fetch_result(name, res, allow_apis, strict, req_params)

    # 关键可计算性快检
    try:
        df = bundle.market_spot.data
        if "amount_yuan" not in getattr(df, "columns", []):
            _print_fail("P0Bundle.market_spot: 缺少 amount_yuan")
            status |= 1
    except Exception as e:
        _print_fail(f"P0Bundle.market_spot: 检查异常：{e}")
        status |= 1

    try:
        df = bundle.index_daily_hs300.data
        if getattr(df, "shape", (0,))[0] < 20:
            if strict:
                _print_fail("P0Bundle.index_daily_hs300: 行数不足 20（严格模式）")
                status |= 1
            else:
                _print_warn("P0Bundle.index_daily_hs300: 行数不足 20（宽松模式）")
        else:
            _print_ok("P0Bundle.index_daily_hs300: 行数满足 ≥20")
    except Exception as e:
        _print_fail(f"P0Bundle.index_daily_hs300: 检查异常：{e}")
        status |= 1

    # ---------- 汇总 ----------
    if status == 0:
        _print_ok("P0 · 六文件与 P0Bundle 的契约/降级/可计算性：全部通过")
    else:
        _print_fail("P0 · 契约/降级/可计算性：存在失败项（详见上方日志）")

    return status


if __name__ == "__main__":
    sys.exit(main())
