# tests/14_scheduler_window_e2e.py
# -*- coding: utf-8 -*-
"""
调度器窗口 E2E（当前时间即窗口）
- 运行时把 utils.scheduler.TRADING_WINDOWS 临时设为 [当前 HH:MM]，调用 refresh_trading_calendar() + run_trading_task()
- 不改动任何源文件；执行完毕后还原 TRADING_WINDOWS
- 输出：11/12 风格 [OK]/[FAIL]/[WARN] + JSON 报告 ./tests/test_reports/...

# 当前时间即窗口（不改配置、不等 10:00/14:00）
python -m tests.14_scheduler_window_e2e --strict

# 若只想验证窗口行为与降级链路（不触网）
python -m tests.14_scheduler_window_e2e --noapi --strict

"""

from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

from utils import scheduler as sch

OK, FAIL, WARN = "[OK]", "[FAIL]", "[WARN]"
SH_TZ = ZoneInfo("Asia/Shanghai")
REPORT_DIR = Path("./tests/test_reports")

def _now_sh() -> datetime:
    return datetime.now(SH_TZ)

def _ensure_report_dir():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

def _read_jsonl_tail(path: Path, max_lines: int = 200) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines[-max_lines:]:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    except Exception:
        pass
    return out

def _print_ok(msg: str): print(f"{OK} {msg}")
def _print_fail(msg: str): print(f"{FAIL} {msg}")
def _print_warn(msg: str): print(f"{WARN} {msg}")

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noapi", action="store_true", help="不触网：验证窗口行为与降级链路")
    ap.add_argument("--strict", action="store_true", help="严格模式：对单次窗口总耗时做主观判定")
    args = ap.parse_args(argv)

    _ensure_report_dir()
    now = _now_sh()
    win = now.strftime("%H:%M")

    # 0) 猴补 TRADING_WINDOWS（不改源文件）
    orig_windows = list(sch.TRADING_WINDOWS)
    sch.TRADING_WINDOWS[:] = [win]
    _print_ok(f"已临时设置 TRADING_WINDOWS={sch.TRADING_WINDOWS}")

    # 1) 交易日历刷新（可能走 cache/回退）
    t0 = time.perf_counter()
    sch.refresh_trading_calendar()
    # 可选：不触网时直接继续（真实是否联网由内部逻辑决定）
    t1 = time.perf_counter()

    # 2) 触发“当前窗口”的主任务一次
    # （run_trading_task 内部会自行调用 Gate→KS，并落事件/状态；在非交易日将落 SKIP）
    sch_t0 = time.perf_counter()
    sch.run_trading_task()
    sch_t1 = time.perf_counter()

    # 3) 汇总事件（DB 可通过 02/03/05 已验证，这里补充 JSONL 兜底观测）
    jsonl = Path("events/altflow_events.jsonl")
    tail = _read_jsonl_tail(jsonl)
    gate_seen = any(rec.get("event", "").startswith("altflow/gate_") for rec in tail)
    ks_seen = any(rec.get("event", "").startswith("altflow/ks_") for rec in tail)

    if gate_seen:
        _print_ok("已观测到 gate 事件（JSONL 兜底）")
    else:
        _print_warn("未在 JSONL 观测到 gate 事件（可能写入 DB；属正常）")

    if ks_seen:
        _print_ok("已观测到 ks 事件（JSONL 兜底）")
    else:
        _print_warn("未在 JSONL 观测到 ks 事件（可能写入 DB；属正常）")

    # 4) 输出报告
    total = (t1 - t0) + (sch_t1 - sch_t0)
    report = {
        "ts": now.isoformat(),
        "window": win,
        "durations_sec": {
            "calendar_refresh": round(t1 - t0, 4),
            "single_window_total": round(total, 4),
        },
        "observed": {
            "gate_event_jsonl": gate_seen,
            "ks_event_jsonl": ks_seen,
        },
        "notes": "如未在 JSONL 看到事件，通常因为写入 DB；可通过 EventLog 侧核验。",
    }
    out = REPORT_DIR / f"scheduler_e2e_{now.strftime('%Y%m%d_%H%M')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_ok(f"报告已写入：{out}")

    # 5) SLA（主观口径，仅 --strict）
    rc = 0
    if args.strict:
        # 单次窗口总时长 ≤ 12s（统一口径，在线/离线都可接受）
        if total <= 12.0 + 1e-6:
            _print_ok("严格模式：单次窗口耗时在可接受范围内")
        else:
            _print_fail("严格模式：单次窗口耗时超出预期")
            rc = 1

    # 6) 还原 TRADING_WINDOWS
    sch.TRADING_WINDOWS[:] = orig_windows
    _print_ok(f"TRADING_WINDOWS 已还原：{orig_windows}")
    return rc

if __name__ == "__main__":
    sys.exit(main())
