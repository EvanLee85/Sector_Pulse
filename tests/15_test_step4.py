# tests/14_test_step4.py
# -*- coding: utf-8 -*-
"""
Step4 状态机（迟滞带+完整约束）验收 · 离线 & 在线
用法：
  离线（纯逻辑）：
    python -m tests.15_test_step4 --noapi --strict
  在线（端到端，当前时间即窗口）：
    python -m tests.15_test_step4 --window 23:05 --strict
说明：
  - 离线：直接调用 decide_state(...) 验证迟滞带/30%约束/L2优先级，不触网、不调 Gate/KS。
  - 在线：写入一条 Account 快照 + 预置当前状态（OFFENSE），调用 run_trading_task()，
          观察是否落 state/change 与 StateSnapshot（Gate/KS 走真实流程）。
输出：
  - 控制台 [OK]/[WARN]/[FAIL]
  - 报告：./tests/test_reports/step4_<ts>.json
"""

from __future__ import annotations
import argparse, json, os, sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from sqlmodel import select
from zoneinfo import ZoneInfo

from config.loader import load_config
from models.database import (
    create_db_and_tables, get_session,
    EventLog, StateSnapshot, SystemState, Account
)
from utils import scheduler as sch  # decide_state / run_trading_task / refresh_trading_calendar

OK, FAIL, WARN = "[OK]", "[FAIL]", "[WARN]"
SH_TZ = ZoneInfo("Asia/Shanghai")
REPORT_DIR = Path("./tests/test_reports"); REPORT_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class Thresholds:
    enter_off: float
    exit_off: float
    observe_th: float

def _now() -> datetime: return datetime.now(SH_TZ)
def _fmt(dt: datetime) -> str: return dt.strftime("%Y%m%d_%H%M")

def _load_thresholds() -> Thresholds:
    cfg = load_config("config") or {}
    st = (cfg.get("state") or {})
    pos_band = (st.get("pos_band") or {})
    enter_off = float(pos_band.get("enter_offense", 0.55))
    exit_off  = float(pos_band.get("exit_offense", 0.65))
    observe_th = float(st.get("observe_position_threshold", 0.30))  # 你已在 YAML 新增；缺失则默认0.30
    return Thresholds(enter_off, exit_off, observe_th)

def _print_ok(m: str): print(f"{OK} {m}")
def _print_fail(m: str): print(f"{FAIL} {m}")
def _print_warn(m: str): print(f"{WARN} {m}")

def _insert_account_snapshot(position_ratio: Optional[float]=None,
                             total_mv: Optional[float]=None,
                             total_asset: Optional[float]=None,
                             available_cash: Optional[float]=None) -> None:
    """
    写入一条 Account 快照：
    - 若给了 position_ratio，则自动补：total_asset、total_market_value、available_cash
    - 若没给 position_ratio，则要求 total_asset 与 total_market_value 同时给出
    """
    # 默认资产规模（仅用于测试），可按需调整
    DEFAULT_ASSET = 1_000_000.0

    if position_ratio is not None:
        ta = float(total_asset) if total_asset is not None else float(DEFAULT_ASSET)
        tmv = float(total_mv) if total_mv is not None else float(position_ratio) * ta
        ac  = float(available_cash) if available_cash is not None else max(ta - tmv, 0.0)
    else:
        if total_asset is None or total_mv is None:
            raise ValueError("当未提供 position_ratio 时，必须同时提供 total_asset 与 total_market_value")
        ta = float(total_asset)
        tmv = float(total_mv)
        if available_cash is None:
            ac = max(ta - tmv, 0.0)
        else:
            ac = float(available_cash)
        # 若脚本未显式给出 position_ratio，则由 tmv/ta 计算一份
        position_ratio = (tmv / ta) if ta > 0 else 0.0

    with get_session() as s:
        acc = Account(
            total_asset=ta,
            total_market_value=tmv,
            available_cash=ac,
            position_ratio=float(position_ratio),
        )
        s.add(acc)
        s.commit()


def _preset_state(name: str, note: str="preset") -> None:
    """预置当前状态快照（便于在线场景触发状态变化并写 state/change）"""
    sn = name.upper()
    if sn not in {"OFFENSE","HOLD","WATCH","SLEEP"}: sn = "HOLD"
    with get_session() as s:
        s.add(StateSnapshot(state=SystemState[sn], pos_ratio=0.0, note=note))
        s.commit()

def _get_last_state_change_event(within_minutes: int = 10) -> Optional[EventLog]:
    """取最近的 state/change 事件"""
    cutoff = _now() - timedelta(minutes=within_minutes)
    with get_session() as s:
        q = select(EventLog).where(EventLog.code == "state/change").order_by(EventLog.ts.desc())
        ev = s.exec(q).first()
        if ev and ev.ts.replace(tzinfo=None) >= cutoff.replace(tzinfo=None):
            return ev
        return None

def offline_suite(strict: bool) -> int:
    """纯离线：直接测 decide_state 规则"""
    th = _load_thresholds()
    status = 0

    # A) 迟滞带：从 OFFENSE 出发，在 0.58~0.62 来回不应跳变到 HOLD；≥exit_off 才应 HOLD
    dec1 = sch.decide_state(True, th.enter_off + 0.03, 0, "OFFENSE",
                            enter_offense=th.enter_off, exit_offense=th.exit_off, observe_threshold=th.observe_th)
    if dec1["state"] == "OFFENSE":
        _print_ok("迟滞带：pos≈0.58~0.62 仍保持 OFFENSE（不抖动）")
    else:
        _print_fail(f"迟滞带异常：期望 OFFENSE 得到 {dec1}")
        status |= 1

    dec2 = sch.decide_state(True, th.exit_off + 0.01, 0, "OFFENSE",
                            enter_offense=th.enter_off, exit_offense=th.exit_off, observe_threshold=th.observe_th)
    if dec2["state"] == "HOLD":
        _print_ok("迟滞带：pos≥exit_off → HOLD")
    else:
        _print_fail(f"迟滞带异常：期望 HOLD 得到 {dec2}")
        status |= 1

    # B) 观望 30% 约束：emotion_false 且 pos>30% → 保持 HOLD；pos≤30% → WATCH
    dec3 = sch.decide_state(False, th.observe_th + 0.05, 0, "HOLD",
                            enter_offense=th.enter_off, exit_offense=th.exit_off, observe_threshold=th.observe_th)
    if dec3["state"] == "HOLD":
        _print_ok("观望约束：emotion_fail & pos>30% → 保持 HOLD")
    else:
        _print_fail(f"观望约束异常：期望 HOLD 得到 {dec3}")
        status |= 1

    dec4 = sch.decide_state(False, th.observe_th - 0.05, 0, "HOLD",
                            enter_offense=th.enter_off, exit_offense=th.exit_off, observe_threshold=th.observe_th)
    if dec4["state"] == "WATCH":
        _print_ok("观望约束：emotion_fail & pos≤30% → WATCH")
    else:
        _print_fail(f"观望约束异常：期望 WATCH 得到 {dec4}")
        status |= 1

    # C) L2 优先：任意输入 → SLEEP
    dec5 = sch.decide_state(True, 0.20, 2, "OFFENSE",
                            enter_offense=th.enter_off, exit_offense=th.exit_off, observe_threshold=th.observe_th)
    if dec5["state"] == "SLEEP":
        _print_ok("L2 优先：ks_level==2 → SLEEP")
    else:
        _print_fail(f"L2 优先异常：期望 SLEEP 得到 {dec5}")
        status |= 1

    # 报告
    report = {
        "ts": _now().isoformat(),
        "mode": "offline",
        "thresholds": asdict(th),
        "cases": {"dec1": dec1, "dec2": dec2, "dec3": dec3, "dec4": dec4, "dec5": dec5}
    }
    out = REPORT_DIR / f"step4_offline_{_fmt(_now())}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_ok(f"报告已写入：{out}")
    return (1 if status else 0) if strict else 0

def online_suite(window: Optional[str], strict: bool, sla_total_sec: float = 16.0) -> int:
    """在线端到端：预置状态与账户 → 执行 run_trading_task() → 读取 state/change"""
    import time
    create_db_and_tables()
    # 刷新日历（失败自动回退 weekday）
    sch.refresh_trading_calendar()

    # 临时设置窗口
    old_windows = list(sch.TRADING_WINDOWS)
    if window:
        sch.TRADING_WINDOWS[:] = [window]
        _print_ok(f"已临时设置 TRADING_WINDOWS={sch.TRADING_WINDOWS}")

    try:
        # 预置：当前状态 OFFENSE（便于产生状态变迁）
        _preset_state("OFFENSE", note="step4_online_preset")
        # 写入账户快照：pos=0.70（emotion 通过时将切换至 HOLD；即便不通过也保持 HOLD，从 OFFENSE→HOLD）
        _insert_account_snapshot(position_ratio=0.70)

        t0 = time.perf_counter()
        sch.run_trading_task()
        dt = time.perf_counter() - t0

        # 观测 state/change 事件（10 分钟内）
        ev = _get_last_state_change_event(within_minutes=10)
        if ev:
            _print_ok("已观测到 state/change 事件")
        else:
            _print_warn("未观测到 state/change（可能状态未发生变化）")

        # 读取最新状态快照
        with get_session() as s:
            snap = s.exec(select(StateSnapshot).order_by(StateSnapshot.ts.desc())).first()
        new_state = snap.state.name if snap else None
        if new_state and new_state != "OFFENSE":
            _print_ok(f"当前状态：{new_state}（已从 OFFENSE 发生变化）")
        else:
            _print_warn(f"当前状态：{new_state}（可能未变化，这取决于实时 Gate/KS）")

        # SLA（总耗时阈值）
        if dt <= sla_total_sec:
            _print_ok(f"SLA：总耗时 {dt:.3f}s ≤ {sla_total_sec}s")
            sla_ok = True
        else:
            _print_fail(f"SLA：总耗时 {dt:.3f}s 超出 {sla_total_sec}s")
            sla_ok = False

        # 报告
        payload = {
            "ts": _now().isoformat(),
            "mode": "online",
            "window": window,
            "duration_sec": dt,
            "state_change_event": bool(ev),
            "latest_state": new_state,
        }
        out = REPORT_DIR / f"step4_online_{_fmt(_now())}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _print_ok(f"报告已写入：{out}")

        return 0 if (not strict or (ev is not None and sla_ok)) else 1

    finally:
        # 还原窗口
        sch.TRADING_WINDOWS[:] = old_windows
        _print_ok(f"TRADING_WINDOWS 已还原：{sch.TRADING_WINDOWS}")

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noapi", action="store_true", help="离线模式（不调外部接口）")
    ap.add_argument("--window", type=str, default=None, help="将窗口临时设为该时间（如 22:59）以便立刻执行在线流程")
    ap.add_argument("--strict", action="store_true", help="严格模式：任一关键断言失败返回码为 1")
    ap.add_argument("--sla_total_sec", type=float, default=16.0, help="在线端到端总时长 SLA（秒）")
    args = ap.parse_args(argv)

    rc = 0
    if args.noapi:
        rc |= offline_suite(strict=args.strict)
    else:
        # 默认先跑离线（快速逻辑回归），再跑在线（端到端）
        rc |= offline_suite(strict=False)
        rc |= online_suite(window=args.window or _now().strftime("%H:%M"),
                           strict=args.strict, sla_total_sec=args.sla_total_sec)
    return rc

if __name__ == "__main__":
    sys.exit(main())
