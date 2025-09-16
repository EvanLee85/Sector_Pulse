# tests/13_system_flow_cli.py
# -*- coding: utf-8 -*-
"""
系统流程（0/1/2/3）协同 + 性能自检（CLI）
- 离线：--noapi  → 注入合成指标，稳定验证 Gate→KS→事件/状态
- 在线：默认     → 仅跑一次 Alt-Flow 真实取数，再注入 Gate/KS（避免重复取数）
- 输出：11/12 风格 [OK]/[FAIL]/[WARN] + JSON 报告 ./tests/test_reports/...

# 离线（稳定复现，无需外网）
python -m tests.13_system_flow_cli --noapi --strict

# 在线（一次 Alt-Flow 真实取数，可能有 WARN）
python -m tests.13_system_flow_cli --strict

"""

from __future__ import annotations
import argparse, json, os, sys, time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

# —— 现有入口（不改名）——
from config.loader import load_config
from data_service.proxies.altflow_proxy import run_indicator_calculator
from strategy_engine.gates.emotion_gate import check_emotion_gate
from strategy_engine.killswitch.killswitch import check_kill_switch
from utils import scheduler as sch

SH_TZ = ZoneInfo("Asia/Shanghai")
OK, FAIL, WARN = "[OK]", "[FAIL]", "[WARN]"
REPORT_DIR = Path("./tests/test_reports")

CORE_KEYS = [
    "limitup_down_ratio",
    "turnover_concentration_top20",
    "vol_percentile",
    "margin_net_repay_yi_prev",
    "index_intraday_ret",
]

def _now_sh() -> datetime:
    return datetime.now(SH_TZ)

def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H%M")

def _prev_trade_yyyymmdd() -> str:
    # 使用调度器里的推断逻辑（已实现、无需改名）
    try:
        return sch._get_prev_trade_yyyymmdd(_now_sh().date())  # noqa: SLF001（测试用途）
    except Exception:
        d = (_now_sh().date() - timedelta(days=1))
        return d.strftime("%Y%m%d")

def _make_indicators(prev_yyyymmdd: str, intraday_missing: bool, metrics: Dict[str, Any]) -> Dict[str, Any]:
    m = {k: metrics.get(k) for k in CORE_KEYS}
    return {
        "schema_version": "step2-contract-v1.1",
        "calc_at": _now_sh().isoformat(),
        "intraday_missing": bool(intraday_missing),
        "metrics": m,
        "sources": {},
        "window": {"prev_trade_date": prev_yyyymmdd},
        "tz": "Asia/Shanghai",
    }

def _print_ok(msg: str): print(f"{OK} {msg}")
def _print_fail(msg: str): print(f"{FAIL} {msg}")
def _print_warn(msg: str): print(f"{WARN} {msg}")

def _ensure_report_dir() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noapi", action="store_true", help="离线注入：不触网，仅注入合成指标")
    ap.add_argument("--window", type=str, default=_now_sh().strftime("%H:%M"), help="窗口，默认取当前 HH:MM")
    ap.add_argument("--prev", type=str, help="上一交易日（YYYYMMDD），默认自动推断")
    ap.add_argument("--strict", action="store_true", help="严格模式：超出 SLA 计 FAIL")
    args = ap.parse_args(argv)

    # 0) 基线与上下文
    prev = args.prev or _prev_trade_yyyymmdd()
    window = args.window
    _ensure_report_dir()
    cfg_t0 = time.perf_counter()
    cfg = load_config("config")
    cfg_t1 = time.perf_counter()
    _print_ok("配置加载成功")

    # 1) 指标阶段（离线/在线二选一）
    indi_t0 = time.perf_counter()
    indicators: Optional[Dict[str, Any]] = None
    warn_msg: Optional[str] = None
    if args.noapi:
        # —— 离线注入：构造“安全区”基线指标，便于 Gate 通过 / KS L0 —— #
        safe = {
            "limitup_down_ratio": 0.70,
            "turnover_concentration_top20": 0.30,
            "vol_percentile": 0.50,
            "margin_net_repay_yi_prev": 10.0,
            "index_intraday_ret": 0.0,
        }
        indicators = _make_indicators(prev, False, safe)
        _print_ok("离线模式：合成指标已注入")
    else:
        # —— 在线：仅跑一次 Alt-Flow 真实取数 —— #
        try:
            indicators = run_indicator_calculator(prev_trade_yyyymmdd=prev)
            _print_ok("在线模式：Alt-Flow 指标已获取")
        except Exception as e:
            warn_msg = f"Alt-Flow 真实取数失败（记 WARN，不阻塞后续）：{e}"
            _print_warn(warn_msg)
            # 回退：用“缺失”占位，确保后续链路可运行
            indicators = _make_indicators(prev, True, {
                "limitup_down_ratio": None,
                "turnover_concentration_top20": None,
                "vol_percentile": None,
                "margin_net_repay_yi_prev": None,
                "index_intraday_ret": None,
            })
    indi_t1 = time.perf_counter()

    # 2) Gate 判定（可透传 indicators，避免重复取数）
    gate_t0 = time.perf_counter()
    gate_payload = check_emotion_gate(
        prev_trade_yyyymmdd=prev,
        window=window,
        indicators_payload=indicators,
        config_dir="config",
        emit_event=True,
        calendar={
            "source": "system_flow_cli",
            "degraded": indicators.get("intraday_missing", False),
            "staleness_days": 0,
        },
    )
    gate_t1 = time.perf_counter()
    if gate_payload.get("passed") is True:
        _print_ok(f"情绪闸门：通过（reasons={gate_payload.get('reasons')})")
    else:
        _print_warn(f"情绪闸门：未通过（reasons={gate_payload.get('reasons')})")

    # 3) KS 判定（复用 Gate 的 indicators，不重复取数）
    ks_t0 = time.perf_counter()
    ks_payload = check_kill_switch(
        prev_trade_yyyymmdd=prev,
        window=window,
        indicators_payload=gate_payload.get("indicators"),
        config_dir="config",
        emit_event=True,
    )
    ks_t1 = time.perf_counter()
    _print_ok(f"Kill-Switch：level={ks_payload.get('level')} triggers={ks_payload.get('triggered_conditions')}")

    # 4) 性能与报告
    total_t = (cfg_t1 - cfg_t0) + (indi_t1 - indi_t0) + (gate_t1 - gate_t0) + (ks_t1 - ks_t0)
    report = {
        "ts": _now_sh().isoformat(),
        "prev": prev,
        "window": window,
        "warn": warn_msg,
        "durations_sec": {
            "config_load": round(cfg_t1 - cfg_t0, 4),
            "indicators": round(indi_t1 - indi_t0, 4),
            "gate": round(gate_t1 - gate_t0, 4),
            "killswitch": round(ks_t1 - ks_t0, 4),
            "total": round(total_t, 4),
        },
        "gate": {
            "passed": gate_payload.get("passed"),
            "reasons": gate_payload.get("reasons"),
        },
        "killswitch": {
            "level": ks_payload.get("level"),
            "triggers": ks_payload.get("triggered_conditions"),
            "actions": ks_payload.get("actions"),
        },
    }
    out = REPORT_DIR / f"system_flow_{_fmt(_now_sh())}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_ok(f"报告已写入：{out}")

    # 5) 初始 SLA（独立开发者口径；仅在 --strict 时生效）
    if args.strict:
        sla_ok = True
        if args.noapi:
            # 离线：总时长 ≤ 1.5s，Gate ≤ 0.2s，KS ≤ 0.2s
            sla_ok &= total_t <= 1.5 + 1e-6
            sla_ok &= (gate_t1 - gate_t0) <= 0.2 + 1e-6
            sla_ok &= (ks_t1 - ks_t0) <= 0.2 + 1e-6
        else:
            # 在线：指标 ≤ 6s，总时长 ≤ 12s
            sla_ok &= (indi_t1 - indi_t0) <= 6.0 + 1e-6
            sla_ok &= total_t <= 12.0 + 1e-6
        if sla_ok:
            _print_ok("严格模式：SLA 通过")
            return 0
        _print_fail("严格模式：SLA 未通过")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
