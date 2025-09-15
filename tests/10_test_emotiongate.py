# tests/10_test_emotiongate.py
# -*- coding: utf-8 -*-
"""
Step 2 · ④ 情绪闸门聚合（P0）验收测试

本测试覆盖三类场景：
1) SMOKE：直接调用 check_emotion_gate（使用真实指标与配置），仅做结构与类型断言；
2) FORCED PASS：注入构造的指标 payload，预期 passed=True，reasons=[];
3) FORCED REJECT / INTRADAY_MISSING：注入构造的指标 payload，分别预期明确的拒绝原因。

用法：
  python -m tests.10_test_emotiongate --prev YYYYMMDD --strict
可选：
  --emit    在 SMOKE 场景中实际写事件（默认不写，避免测试产生副作用）
  --window  指定窗口标识，默认 "10:00"
-------------------------------------------------------------------------------------------------------------
  (sss_py311) evan@Evan1985:~/SectorPulse v1.0$ python -m tests.10_test_emotiongate --prev 20250821 --strict
    [OK] SMOKE：check_emotion_gate 调用成功                                                                                                                                        
    [OK] passed=False
    [OK] reasons 个数=2
    [OK] calc_at 为带时区时间
    [OK] tz=Asia/Shanghai
    [OK] window=10:00
    [OK] thresholds 快照齐全
    [OK] indicators 结构齐全
    [OK] metrics 核心键齐全
    [OK] 严格模式：结构/类型断言通过
    [OK] passed=True
    [OK] reasons 个数=0
    [OK] calc_at 为带时区时间
    [OK] tz=Asia/Shanghai
    [OK] window=10:00
    [OK] thresholds 快照齐全
    [OK] indicators 结构齐全
    [OK] metrics 核心键齐全
    [OK] 严格模式：结构/类型断言通过
    [OK] FORCED PASS：passed=True 且 reasons 为空
    [OK] passed=False
    [OK] reasons 个数=5
    [OK] calc_at 为带时区时间
    [OK] tz=Asia/Shanghai
    [OK] window=10:00
    [OK] thresholds 快照齐全
    [OK] indicators 结构齐全
    [OK] metrics 核心键齐全
    [OK] 严格模式：结构/类型断言通过
    [OK] FORCED REJECT：passed=False 且 reasons 含关键项
    [OK] passed=False
    [OK] reasons 个数=1
    [OK] calc_at 为带时区时间
    [OK] tz=Asia/Shanghai
    [OK] window=10:00
    [OK] thresholds 快照齐全
    [OK] indicators 结构齐全
    [OK] metrics 核心键齐全
    [OK] 严格模式：结构/类型断言通过
    [OK] INTRADAY_MISSING：passed=False 且 reasons=['intraday_missing']
    [OK] 情绪闸门（P0）聚合验收：全部通过
"""
from __future__ import annotations

import sys
import argparse
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

SH_TZ = ZoneInfo("Asia/Shanghai")
OK, FAIL, WARN = "[OK]", "[FAIL]", "[WARN]"


def _print_ok(msg: str) -> None:
    print(f"{OK} {msg}")


def _print_fail(msg: str) -> None:
    print(f"{FAIL} {msg}")


def _print_warn(msg: str) -> None:
    print(f"{WARN} {msg}")


def _ensure_prev(prev: Optional[str]) -> str:
    if prev:
        return prev
    d = datetime.now(SH_TZ) - timedelta(days=1)
    s = d.strftime("%Y%m%d")
    _print_warn(f"未提供 --prev，临时使用 {s}")
    return s


def _tzaware(dt: Any) -> bool:
    return isinstance(dt, datetime) and dt.tzinfo is not None


def _assert_gate_payload(payload: Dict[str, Any], strict: bool) -> int:
    """
    断言 gate 聚合输出的最小冻结结构。
    返回：0=通过；1=失败
    """
    status = 0

    # 顶层键
    req_top = ["passed", "reasons", "indicators", "thresholds", "version", "calc_at", "tz", "window"]
    missing = [k for k in req_top if k not in payload]
    if missing:
        _print_fail(f"顶层缺少字段：{missing}")
        status |= 1

    # passed
    if not isinstance(payload.get("passed"), bool):
        _print_fail("passed 应为 bool")
        status |= 1
    else:
        _print_ok(f"passed={payload.get('passed')}")

    # reasons
    rs = payload.get("reasons")
    if not isinstance(rs, list) or any(not isinstance(x, str) for x in rs):
        _print_fail("reasons 应为 list[str]")
        status |= 1
    else:
        _print_ok(f"reasons 个数={len(rs)}")

    # calc_at tz-aware
    if not _tzaware(payload.get("calc_at")):
        _print_fail("calc_at 需为带时区时间")
        status |= 1
    else:
        _print_ok("calc_at 为带时区时间")

    # tz, window
    if payload.get("tz") != "Asia/Shanghai":
        _print_fail("tz 应为 'Asia/Shanghai'")
        status |= 1
    else:
        _print_ok("tz=Asia/Shanghai")
    if not isinstance(payload.get("window"), str):
        _print_fail("window 应为 str")
        status |= 1
    else:
        _print_ok(f"window={payload.get('window')}")

    # thresholds 快照（只检查存在与类型）
    th = payload.get("thresholds", {})
    need_th = [
        "emotion_gate.limit_up_count_min",
        "emotion_gate.limitup_down_ratio_min",
        "emotion_gate.turnover_concentration_max",
        "emotion_gate.vol_percentile_max",
        "emotion_gate.ma20_required",
    ]
    th_missing = [k for k in need_th if k not in th]
    if th_missing:
        _print_fail(f"thresholds 缺失：{th_missing}")
        status |= 1
    else:
        _print_ok("thresholds 快照齐全")

    # 指标容器
    indi = payload.get("indicators", {})
    need_indi = ["schema_version", "calc_at", "intraday_missing", "metrics"]
    indi_missing = [k for k in need_indi if k not in indi]
    if indi_missing:
        _print_fail(f"indicators 缺少字段：{indi_missing}")
        status |= 1
    else:
        _print_ok("indicators 结构齐全")

    # 指标核心键
    metrics = indi.get("metrics", {})
    need_metrics = [
        "limit_up_count",
        "limit_down_count",
        "limitup_down_ratio",
        "turnover_concentration_top20",
        "hs300_vol_30d_annualized",
        "vol_percentile",
        "sh_above_ma20",
        "margin_net_repay_yi_prev",
    ]
    m_missing = [k for k in need_metrics if k not in metrics]
    if m_missing:
        _print_fail(f"metrics 缺少核心键：{m_missing}")
        status |= 1
    else:
        _print_ok("metrics 核心键齐全")

    if strict and status == 0:
        _print_ok("严格模式：结构/类型断言通过")
    return status


def _make_min_indicators_payload(
    *,
    prev_yyyymmdd: str,
    intraday_missing: bool,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    构造一个最小契约的 indicators payload，供注入式测试。
    """
    # 确保 8 个核心键都在
    core_keys = [
        "limit_up_count",
        "limit_down_count",
        "limitup_down_ratio",
        "turnover_concentration_top20",
        "hs300_vol_30d_annualized",
        "vol_percentile",
        "sh_above_ma20",
        "margin_net_repay_yi_prev",
    ]
    m = {k: metrics.get(k, None) for k in core_keys}

    return {
        "schema_version": 1,
        "calc_at": datetime.now(SH_TZ),
        "intraday_missing": bool(intraday_missing),
        "window": {"prev_trade_date": prev_yyyymmdd},
        "sources": {},   # 注入式场景可留空
        "metrics": m,
    }


def _load_thresholds() -> Dict[str, Any]:
    # 仅用于构造 PASS/REJECT 案例的数值
    from config.loader import load_config
    cfg = load_config(config_dir="config")
    eg = cfg.get("emotion_gate", {}) or {}
    return {
        "limit_up_count_min": int(eg.get("limit_up_count_min", 50)),
        "limitup_down_ratio_min": float(eg.get("limitup_down_ratio_min", 0.5)),
        "turnover_concentration_max": float(eg.get("turnover_concentration_max", 0.40)),
        "vol_percentile_max": float(eg.get("vol_percentile_max", 0.85)),
        "ma20_required": bool(eg.get("ma20_required", True)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prev", type=str, default=None, help="上一交易日 YYYYMMDD")
    parser.add_argument("--strict", action="store_true", help="严格模式")
    parser.add_argument("--emit", action="store_true", help="SMOKE 场景写事件（默认不写）")
    parser.add_argument("--window", type=str, default="10:00", help="窗口标识")
    args = parser.parse_args(argv)

    prev = _ensure_prev(args.prev)
    strict = bool(args.strict)

    # 导入被测函数
    try:
        from strategy_engine.gates.emotion_gate import check_emotion_gate
    except Exception as e:
        _print_fail(f"导入 check_emotion_gate 失败：{e}")
        return 1

    status = 0

    # ① SMOKE：真实指标 + 配置（仅结构断言）
    try:
        payload = check_emotion_gate(
            prev_trade_yyyymmdd=prev,
            window=args.window,
            indicators_payload=None,   # 触发真实指标计算
            config_dir="config",
            emit_event=bool(args.emit),
        )
        _print_ok("SMOKE：check_emotion_gate 调用成功")
        status |= _assert_gate_payload(payload, strict=strict)
    except Exception as e:
        _print_fail(f"SMOKE：check_emotion_gate 失败：{e}")
        return 1

    # ② FORCED PASS：构造指标 → 预期通过
    try:
        th = _load_thresholds()
        m_pass = {
            "limit_up_count": max(th["limit_up_count_min"], 60),
            "limit_down_count": 0,
            "limitup_down_ratio": max(th["limitup_down_ratio_min"], 1.0),
            "turnover_concentration_top20": min(th["turnover_concentration_max"], 0.20),
            "hs300_vol_30d_annualized": 0.18,
            "vol_percentile": min(th["vol_percentile_max"], 0.20),
            "sh_above_ma20": True if th["ma20_required"] else True,
            "margin_net_repay_yi_prev": 100.0,
        }
        indi_payload = _make_min_indicators_payload(prev_yyyymmdd=prev, intraday_missing=False, metrics=m_pass)
        payload2 = check_emotion_gate(
            prev_trade_yyyymmdd=prev,
            window=args.window,
            indicators_payload=indi_payload,  # 注入
            config_dir="config",
            emit_event=False,
        )
        status |= _assert_gate_payload(payload2, strict=strict)
        if payload2.get("passed") is not True or payload2.get("reasons"):
            _print_fail(f"FORCED PASS 预期 passed=True, reasons=[]，实际: passed={payload2.get('passed')} reasons={payload2.get('reasons')}")
            status |= 1
        else:
            _print_ok("FORCED PASS：passed=True 且 reasons 为空")
    except Exception as e:
        _print_fail(f"FORCED PASS 构造/断言失败：{e}")
        return 1

    # ③ FORCED REJECT：构造指标 → 预期拒绝（至少包含 limit_up_count/sh_above_ma20 的原因）
    try:
        m_reject = {
            "limit_up_count": 0,
            "limit_down_count": 10,
            "limitup_down_ratio": 0.1,
            "turnover_concentration_top20": 0.9,
            "hs300_vol_30d_annualized": 0.5,
            "vol_percentile": 0.99,
            "sh_above_ma20": False,
            "margin_net_repay_yi_prev": None,
        }
        indi_payload = _make_min_indicators_payload(prev_yyyymmdd=prev, intraday_missing=False, metrics=m_reject)
        payload3 = check_emotion_gate(
            prev_trade_yyyymmdd=prev,
            window=args.window,
            indicators_payload=indi_payload,
            config_dir="config",
            emit_event=False,
        )
        status |= _assert_gate_payload(payload3, strict=strict)
        if payload3.get("passed") is not False:
            _print_fail(f"FORCED REJECT 预期 passed=False，实际: passed={payload3.get('passed')}")
            status |= 1
        rs = payload3.get("reasons", [])
        # 不做精确字符串匹配，只检查关键字出现，避免格式变动导致脆弱
        want_keys = ["limit_up_count", "sh_above_ma20"]
        for key in want_keys:
            if not any(key in str(x) for x in rs):
                _print_fail(f"FORCED REJECT：reasons 未包含关键字 '{key}'，实际：{rs}")
                status |= 1
        if status & 1 == 0:
            _print_ok("FORCED REJECT：passed=False 且 reasons 含关键项")
    except Exception as e:
        _print_fail(f"FORCED REJECT 构造/断言失败：{e}")
        return 1

    # ④ INTRADAY_MISSING：构造指标 → 预期直接因 intraday_missing 拒绝
    try:
        m_any = {
            "limit_up_count": None,
            "limit_down_count": None,
            "limitup_down_ratio": None,
            "turnover_concentration_top20": None,
            "hs300_vol_30d_annualized": 0.1,
            "vol_percentile": 0.1,
            "sh_above_ma20": None,
            "margin_net_repay_yi_prev": None,
        }
        indi_payload = _make_min_indicators_payload(prev_yyyymmdd=prev, intraday_missing=True, metrics=m_any)
        payload4 = check_emotion_gate(
            prev_trade_yyyymmdd=prev,
            window=args.window,
            indicators_payload=indi_payload,
            config_dir="config",
            emit_event=False,
        )
        status |= _assert_gate_payload(payload4, strict=strict)
        rs = payload4.get("reasons", [])
        if payload4.get("passed") is not False or rs != ["intraday_missing"]:
            _print_fail(f"INTRADAY_MISSING 预期 passed=False 且 reasons=['intraday_missing']，实际: passed={payload4.get('passed')} reasons={rs}")
            status |= 1
        else:
            _print_ok("INTRADAY_MISSING：passed=False 且 reasons=['intraday_missing']")
    except Exception as e:
        _print_fail(f"INTRADAY_MISSING 构造/断言失败：{e}")
        return 1

    if status == 0:
        _print_ok("情绪闸门（P0）聚合验收：全部通过")
    else:
        _print_fail("情绪闸门（P0）聚合验收：存在失败项（见上方日志）")

    return status


if __name__ == "__main__":
    sys.exit(main())
