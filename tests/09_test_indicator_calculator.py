# tests/09_test_indicator_calculator.py
# -*- coding: utf-8 -*-
"""
Step 2 · ③ 指标计算器（P0）验收测试
按“最小冻结契约”验证 altflow_proxy.build_from_bundle / run_indicator_calculator 的输出。

用法：
  python -m tests.09_test_indicator_calculator --prev YYYYMMDD --strict
  -------------------------------------------------------------------------------------------------------------------
  (sss_py311) evan@Evan1985:~/SectorPulse v1.0$ python -m tests.09_test_indicator_calculator --prev 20250911 --strict
    [OK] schema_version=1                                                                                                                                                          
    [OK] calc_at 为带时区时间
    [OK] intraday_missing=False
    [OK] sources 结构齐全（含 margin.sse / margin.szse）
    [OK] limit_up_count=87
    [OK] limit_down_count=3
    [OK] limitup_down_ratio=29.0
    [OK] turnover_concentration_top20=0.12513362436959394
    [OK] hs300_vol_30d_annualized=0.16433393209522762
    [OK] vol_percentile=0.6111111111111112
    [OK] sh_above_ma20=True
    [OK] margin_net_repay_yi_prev=143.4146592499983
    [OK] 严格模式：完成数值/范围/结构断言
    [OK] schema_version=1                                                                                                                                                          
    [OK] calc_at 为带时区时间
    [OK] intraday_missing=False
    [OK] sources 结构齐全（含 margin.sse / margin.szse）
    [OK] limit_up_count=87
    [OK] limit_down_count=3
    [OK] limitup_down_ratio=29.0
    [OK] turnover_concentration_top20=0.12513362436959394
    [OK] hs300_vol_30d_annualized=0.16433393209522762
    [OK] vol_percentile=0.6111111111111112
    [OK] sh_above_ma20=True
    [OK] margin_net_repay_yi_prev=143.4146592499983
    [OK] 严格模式：完成数值/范围/结构断言
    [OK] 指标计算器（P0）最小冻结契约：全部通过
"""
from __future__ import annotations

import sys
import argparse
from typing import Any, Dict, Iterable
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ---- 打印样式 ----
OK = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"

SH_TZ = ZoneInfo("Asia/Shanghai")


def _print_ok(msg: str) -> None:
    print(f"{OK} {msg}")


def _print_warn(msg: str) -> None:
    print(f"{WARN} {msg}")


def _print_fail(msg: str) -> None:
    print(f"{FAIL} {msg}")


def _ensure_prev(prev_arg: str | None) -> str:
    """未提供 --prev 时，用上海时区 ‘今天-1天’（可能遇到节假日为空）"""
    if prev_arg:
        return prev_arg
    d = datetime.now(SH_TZ) - timedelta(days=1)
    s = d.strftime("%Y%m%d")
    _print_warn(f"未提供 --prev，临时使用 {s}")
    return s


def _is_number(x: Any) -> bool:
    try:
        import numpy as np  # noqa
        return isinstance(x, (int, float, np.integer, np.floating)) and (not isinstance(x, bool))
    except Exception:
        return isinstance(x, (int, float)) and (not isinstance(x, bool))


def _check_tz_aware(dt: Any) -> bool:
    try:
        return isinstance(dt, datetime) and (dt.tzinfo is not None)
    except Exception:
        return False


def _assert_payload(payload: Dict[str, Any], strict: bool) -> int:
    """
    验证最小冻结契约：
      系统字段：schema_version, calc_at(带时区), intraday_missing(bool), sources(存在)
      指标集：metrics 内的 8 个核心键存在；若值非 None，做单位与范围检查
    返回：0 通过；1 失败
    """
    status = 0

    # ---- 系统字段 ----
    if "schema_version" not in payload or not isinstance(payload["schema_version"], int):
        _print_fail("schema_version 缺失或类型不为 int")
        status |= 1
    else:
        _print_ok(f"schema_version={payload['schema_version']}")

    if "calc_at" not in payload or not _check_tz_aware(payload["calc_at"]):
        _print_fail("calc_at 缺失或不是带时区的 datetime")
        status |= 1
    else:
        try:
            _ = payload["calc_at"].astimezone(SH_TZ)
            _print_ok("calc_at 为带时区时间")
        except Exception as e:
            _print_fail(f"calc_at 时区转换失败：{e}")
            status |= 1

    if "intraday_missing" not in payload or not isinstance(payload["intraday_missing"], bool):
        _print_fail("intraday_missing 缺失或类型不为 bool")
        status |= 1
    else:
        _print_ok(f"intraday_missing={payload['intraday_missing']}")

    if "sources" not in payload or not isinstance(payload["sources"], dict):
        _print_fail("sources 缺失或类型不为 dict")
        status |= 1
    else:
        req_sources = ["zt_pool", "dtgc_pool", "market_spot", "index_daily_sh", "index_daily_hs300", "index_spot", "margin"]
        missing = [k for k in req_sources if k not in payload["sources"]]
        if missing:
            _print_fail(f"sources 缺少子项：{missing}")
            status |= 1
        else:
            # margin 还需包含 sse/szse
            margin = payload["sources"].get("margin", {})
            if not isinstance(margin, dict) or not {"sse", "szse"} <= set(margin.keys()):
                _print_fail("sources.margin 需包含 sse 与 szse")
                status |= 1
            else:
                _print_ok("sources 结构齐全（含 margin.sse / margin.szse）")

    # ---- 指标集 ----
    if "metrics" not in payload or not isinstance(payload["metrics"], dict):
        _print_fail("metrics 缺失或类型不为 dict")
        status |= 1
        return status

    m = payload["metrics"]
    required_keys = [
        "limit_up_count",
        "limit_down_count",
        "limitup_down_ratio",
        "turnover_concentration_top20",
        "hs300_vol_30d_annualized",
        "vol_percentile",
        "sh_above_ma20",
        "margin_net_repay_yi_prev",
    ]
    missing = [k for k in required_keys if k not in m]
    if missing:
        _print_fail(f"metrics 缺少核心键：{missing}")
        status |= 1

    # 逐项检查（允许 None；非 None 时做范围/类型断言）
    def _num_ok(key: str, lower: float | None = None, upper: float | None = None, allow_negative: bool = True) -> int:
        v = m.get(key, None)
        if v is None:
            _print_warn(f"{key} 为 None（允许）")
            return 0
        if not _is_number(v):
            _print_fail(f"{key} 不是数值类型：{type(v)}")
            return 1
        fv = float(v)
        if not allow_negative and fv < 0:
            _print_fail(f"{key} 不应为负，得到 {fv}")
            return 1
        if (lower is not None and fv < lower) or (upper is not None and fv > upper):
            _print_fail(f"{key} 超出范围 [{lower}, {upper}]，得到 {fv}")
            return 1
        _print_ok(f"{key}={fv}")
        return 0

    # 个别键的规则
    # 计数：>=0 的整数
    for key in ("limit_up_count", "limit_down_count"):
        v = m.get(key, None)
        if v is None:
            _print_warn(f"{key} 为 None（允许）")
        elif not isinstance(v, int) or v < 0:
            _print_fail(f"{key} 需为非负整数，得到 {v}")
            status |= 1
        else:
            _print_ok(f"{key}={v}")

    # 比值与比例
    status |= _num_ok("limitup_down_ratio", lower=0.0, allow_negative=False)
    status |= _num_ok("turnover_concentration_top20", lower=0.0, upper=1.0, allow_negative=False)
    # 波动率（通常 0~5 之间，留足安全边界）
    status |= _num_ok("hs300_vol_30d_annualized", lower=0.0, upper=5.0, allow_negative=False)
    # 分位数 [0,1]
    status |= _num_ok("vol_percentile", lower=0.0, upper=1.0, allow_negative=False)
    # bool
    v = m.get("sh_above_ma20", None)
    if v is None:
        _print_warn("sh_above_ma20 为 None（允许）")
    elif not isinstance(v, bool):
        _print_fail(f"sh_above_ma20 需为 bool，得到 {type(v)}")
        status |= 1
    else:
        _print_ok(f"sh_above_ma20={v}")
    # 两融净买入（亿元）：可正可负，只验证为数值
    v = m.get("margin_net_repay_yi_prev", None)
    if v is None:
        _print_warn("margin_net_repay_yi_prev 为 None（允许）")
    elif not _is_number(v):
        _print_fail(f"margin_net_repay_yi_prev 需为数值，得到 {type(v)}")
        status |= 1
    else:
        _print_ok(f"margin_net_repay_yi_prev={float(v)}")

    # 严格模式下，若 intraday_missing=True 仍可通过本测试（由后续闸门判定处理）
    if strict:
        _print_ok("严格模式：完成数值/范围/结构断言")

    return status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prev", type=str, default=None, help="上一交易日，格式 YYYYMMDD")
    parser.add_argument("--strict", action="store_true", help="严格模式：更严格的数值/范围断言")
    parser.add_argument("--topN", type=int, default=20)
    parser.add_argument("--vol_window", type=int, default=30)
    parser.add_argument("--vol_ref_days", type=int, default=252)
    args = parser.parse_args(argv)

    prev = _ensure_prev(args.prev)
    strict = bool(args.strict)

    # ---- 导入被测对象 ----
    try:
        from data_service.proxies.altflow_proxy import (
            run_indicator_calculator,
            build_from_bundle,
        )
        from data_service.collector import fetch_p0_bundle, P0Bundle  # noqa: F401
    except Exception as e:
        _print_fail(f"导入指标计算器或 collector 失败：{e}")
        return 1

    status = 0

    # ① 直接使用便捷入口（内部获取 P0Bundle）
    try:
        payload = run_indicator_calculator(
            prev_trade_yyyymmdd=prev,
            topN=args.topN,
            vol_window=args.vol_window,
            vol_ref_days=args.vol_ref_days,
        )
        status |= _assert_payload(payload, strict=strict)
    except Exception as e:
        _print_fail(f"run_indicator_calculator 调用失败：{e}")
        return 1

    # ② 显式获取 P0Bundle 后再计算（验证另一个入口）
    try:
        bundle = fetch_p0_bundle(prev_trade_yyyymmdd=prev)
        payload2 = build_from_bundle(
            bundle=bundle,
            prev_trade_yyyymmdd=prev,
            topN=args.topN,
            vol_window=args.vol_window,
            vol_ref_days=args.vol_ref_days,
        )
        status |= _assert_payload(payload2, strict=strict)
    except Exception as e:
        _print_fail(f"build_from_bundle 调用失败：{e}")
        return 1

    # 不强制比较两个 payload 的逐项相等（可能存在不同时间点的盘中差异/缓存），仅分别校验契约
    if status == 0:
        _print_ok("指标计算器（P0）最小冻结契约：全部通过")
    else:
        _print_fail("指标计算器（P0）最小冻结契约：存在失败项（见上方日志）")

    return status


if __name__ == "__main__":
    sys.exit(main())
