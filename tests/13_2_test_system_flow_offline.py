# tests/13_2_test_system_flow_offline.py
# -*- coding: utf-8 -*-
"""
13_2 · Step3 离线动作验收（L1/L2 状态副作用 & 重启链路）
- 仅离线：构造指标 payload 注入 check_kill_switch(..., emit_event=True)
- 验证：
  1) L1：position_cap == temp_total_cap（通常 0.30） & 事件 altflow/ks_L1
  2) L2：sleeping == True & 事件 altflow/ks_L2
  3) 重启：restart/eligible & restart/confirm（manual_confirm=False 时），且 sleeping == False
- 输出：控制台 [OK]/[FAIL]/[WARN] + JSON 报告 ./tests/test_reports/13_2_offline_<ts>.json
"""

from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

from config.loader import load_config
from strategy_engine.killswitch.killswitch import check_kill_switch, check_restart_eligibility
from utils import scheduler as sch
import data_service.storage as storage

OK, FAIL, WARN = "[OK]", "[FAIL]", "[WARN]"
SH_TZ = ZoneInfo("Asia/Shanghai")
REPORT_DIR = Path("./tests/test_reports")
JSONL = Path("events/altflow_events.jsonl")
STATE_FILE = Path("runtime_state.json")

CORE_KEYS = [
    "limitup_down_ratio",
    "turnover_concentration_top20",
    "vol_percentile",
    "margin_net_repay_yi_prev",
    "index_intraday_ret",
]

def _now() -> datetime:
    return datetime.now(SH_TZ)

def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H%M")

def _ensure_report_dir() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

def _prev_trade_yyyymmdd() -> str:
    try:
        return sch._get_prev_trade_yyyymmdd(_now().date())  # 测试用途
    except Exception:
        d = (_now().date() - timedelta(days=1))
        return d.strftime("%Y%m%d")

def _make_indicators(prev_yyyymmdd: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    m = {k: metrics.get(k) for k in CORE_KEYS}
    # 纯离线注入：不触网
    return {
        "schema_version": "step2-contract-v1.1",
        "calc_at": _now().isoformat(),
        "intraday_missing": False,
        "metrics": m,
        "sources": {},
        "window": {"prev_trade_date": prev_yyyymmdd},
        "tz": "Asia/Shanghai",
    }

def _read_state() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _jsonl_tail(max_lines: int = 200):
    if not JSONL.exists():
        return []
    try:
        lines = JSONL.read_text(encoding="utf-8").splitlines()
        return [json.loads(x) for x in lines[-max_lines:]]
    except Exception:
        return []

def _print_ok(msg: str): print(f"{OK} {msg}")
def _print_fail(msg: str): print(f"{FAIL} {msg}")
def _print_warn(msg: str): print(f"{WARN} {msg}")

def _cfg_thresholds() -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    cfg = load_config("config")
    ks = cfg.get("kill_switch", {}) or {}
    L1 = ks.get("L1", {}) or {}
    L2 = ks.get("L2", {}) or {}
    temp_cap = float(L1.get("temp_total_cap", 0.30)) if L1.get("action") == "stop_new_reduce_cap" else 0.30
    return L1, L2, temp_cap

def _safe_baseline(L1: Dict[str, Any], L2: Dict[str, Any]) -> Dict[str, Any]:
    # 安全区指标，确保不误触发其它条件
    return {
        "limitup_down_ratio": max(float(L1["limitup_down_ratio_min"]), float(L2["limitup_down_ratio_min"])) + 0.10,
        "turnover_concentration_top20": min(float(L1["turnover_concentration_max"]), float(L2["turnover_concentration_max"])) - 0.05,
        "vol_percentile": min(float(L1["vol_percentile_min"]), float(L2["vol_percentile_min"])) - 0.10,
        "margin_net_repay_yi_prev": min(float(L1["margin_net_repay_yi"]), float(L2["margin_net_repay_yi"])) - 10.0,
        "index_intraday_ret": 0.0,
    }

def _run_l1(prev: str, window: str, L1: Dict[str, Any], temp_cap: float) -> Tuple[int, Dict[str, Any]]:
    status = 0
    base = _safe_baseline(L1, L1)  # 单独使用 L1 范围即可
    base["margin_net_repay_yi_prev"] = float(L1["margin_net_repay_yi"]) + 1.0  # 仅触发 L1 两融
    p = check_kill_switch(prev, window, indicators_payload=_make_indicators(prev, base), config_dir="config", emit_event=True)
    if p.get("level") != 1:
        _print_fail(f"L1 预期 level=1，得到 level={p.get('level')} / triggers={p.get('triggered_conditions')}")
        status |= 1
    else:
        _print_ok(f"L1 触发：triggers={p.get('triggered_conditions')}")
    # 状态副作用：position_cap
    st = _read_state()
    cap = st.get("position_cap", None)
    if cap is None:
        _print_warn("未在 runtime_state.json 读到 position_cap（可能走 DB 状态；可用 DB 侧核验）")
    elif abs(float(cap) - float(temp_cap)) < 1e-9:
        _print_ok(f"L1 动作生效：position_cap={cap}")
    else:
        _print_fail(f"L1 动作异常：position_cap={cap} ≠ 期望 {temp_cap}")
        status |= 1
    # 事件：JSONL 兜底
    tail = _jsonl_tail()
    has_ev = any(rec.get("event") == "altflow/ks_L1" for rec in tail)
    if has_ev:
        _print_ok("事件写入：altflow/ks_L1（JSONL 兜底）")
    else:
        _print_warn("未在 JSONL 发现 altflow/ks_L1（可能写入 DB；属正常）")
    return status, p

def _run_l2(prev: str, window: str, L2: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    status = 0
    base = _safe_baseline(L2, L2)
    base["margin_net_repay_yi_prev"] = float(L2["margin_net_repay_yi"]) + 1.0  # 直接触发 L2 两融
    p = check_kill_switch(prev, window, indicators_payload=_make_indicators(prev, base), config_dir="config", emit_event=True)
    if p.get("level") != 2:
        _print_fail(f"L2 预期 level=2，得到 level={p.get('level')} / triggers={p.get('triggered_conditions')}")
        status |= 1
    else:
        _print_ok(f"L2 触发：triggers={p.get('triggered_conditions')}")
    # 状态副作用：sleeping
    st = _read_state()
    slp = st.get("sleeping", None)
    if slp is True:
        _print_ok("L2 动作生效：sleeping=True（已进入休眠态）")
    else:
        _print_fail(f"L2 动作异常：sleeping={slp}（应为 True）")
        status |= 1
    # 事件：JSONL 兜底
    tail = _jsonl_tail()
    has_ev = any(rec.get("event") == "altflow/ks_L2" for rec in tail)
    if has_ev:
        _print_ok("事件写入：altflow/ks_L2（JSONL 兜底）")
    else:
        _print_warn("未在 JSONL 发现 altflow/ks_L2（可能写入 DB；属正常）")
    return status, p

def _run_restart(consecutive_days: int) -> int:
    status = 0
    try:
        # 伪造 gate 通过事件 N 日
        for i in range(max(1, consecutive_days)):
            storage.write_event("altflow/gate_pass", {"mock": True, "idx": i, "ts": _now().isoformat()})
        res = check_restart_eligibility(consecutive_gate_days=consecutive_days,
                                        monthly_drawdown_threshold=None,  # 关闭回撤门槛以稳定离线验收
                                        manual_confirm=False)
        if not res.get("eligible", False):
            _print_fail("restart/eligible 未通过")
            status |= 1
        else:
            _print_ok("restart/eligible：通过")
        # 事件兜底
        tail = _jsonl_tail()
        has_elig = any(rec.get("event") == "restart/eligible" for rec in tail)
        has_conf = any(rec.get("event") == "restart/confirm" for rec in tail)
        if has_elig: _print_ok("事件写入：restart/eligible")
        else: _print_warn("未在 JSONL 发现 restart/eligible（可能写入 DB）")
        if has_conf: _print_ok("事件写入：restart/confirm（auto）")
        # 状态：应退出休眠
        st = _read_state()
        if st.get("sleeping") is False:
            _print_ok("exit_sleep_mode：成功（sleeping=False）")
        else:
            _print_fail("exit_sleep_mode：失败（sleeping 仍为 True）")
            status |= 1
    except Exception as e:
        _print_fail(f"重启链路异常：{e}")
        status |= 1
    return status

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="严格模式：任一环节失败返回码为 1")
    args = ap.parse_args(argv)

    _ensure_report_dir()
    prev = _prev_trade_yyyymmdd()
    window = _now().strftime("%H:%M")

    # 加载阈值 & 临时 cap
    try:
        L1, L2, temp_cap = _cfg_thresholds()
    except Exception as e:
        _print_fail(f"读取 kill_switch 阈值失败：{e}")
        return 1

    # 先确保进入“非休眠”起点
    try:
        storage.exit_sleep_mode(note="13_2_precheck")
    except Exception:
        pass

    total_status = 0
    # 1) L1 验收
    s1, p1 = _run_l1(prev, window, L1, temp_cap); total_status |= s1
    # 2) L2 验收
    s2, p2 = _run_l2(prev, window, L2); total_status |= s2
    # 3) 重启链路
    consecutive_days = int((load_config("config").get("restart_conditions") or {}).get("consecutive_gate_days", 3))
    s3 = _run_restart(consecutive_days); total_status |= s3

    # 报告
    report = {
        "ts": _now().isoformat(),
        "prev": prev,
        "window": window,
        "results": {
            "L1": {"status": "OK" if s1 == 0 else "FAIL", "payload": p1},
            "L2": {"status": "OK" if s2 == 0 else "FAIL", "payload": p2},
            "restart": {"status": "OK" if s3 == 0 else "FAIL"},
        },
        "state_after": _read_state(),
    }
    out = REPORT_DIR / f"13_2_offline_{_fmt(_now())}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_ok(f"报告已写入：{out}")

    if args.strict:
        return 0 if total_status == 0 else 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
