# tests/05_test_scheduler_checks.py
"""
调度器交付验收（AkShare 日历 + 缓存/回退 + 降级标记 + 窗口行为 + 启停/恢复）
运行： python -m tests.05_test_scheduler_checks
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

# === 确保可以从项目根导入 ===
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlmodel import select

from models.database import (
    create_db_and_tables,
    get_session,
    EventLog,
    StateSnapshot,
    SystemState,
)
import utils.scheduler as S  # 被测模块


def ok(msg: str):
    print(f"[OK] {msg}")


def fail(msg: str):
    print(f"[FAIL] {msg}")
    sys.exit(1)


# ---------- 测试工具 ----------

def _seed_calendar_cache(include_today: bool = True, stale_days: int = 0):
    """写入 runtime_cache/trade_calendar.json，确保在无网络时可用。
       stale_days>0 可构造“陈旧缓存”。"""
    cache_dir = Path("./runtime_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / "trade_calendar.json"

    today = S.now_cn().date()
    # 让缓存包含若干日期
    dates = [today - timedelta(days=i) for i in range(10)]
    if not include_today:
        dates = [d for d in dates if d != today]

    fetched_at = S.now_cn() - timedelta(days=stale_days)
    payload = {
        "fetched_at": fetched_at.isoformat(),
        "source": "seeded_test_cache",
        "dates": [d.isoformat() for d in sorted(set(dates))],
    }
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return p


def _get_latest_event_by_code(code: str):
    with get_session() as s:
        ev = s.exec(select(EventLog).where(EventLog.code == code).order_by(EventLog.ts.desc())).first()
        return ev


def _assert_event_present(code: str, expect_keys: list[str] | None = None):
    ev = _get_latest_event_by_code(code)
    if not ev:
        fail(f"Event '{code}' not found")
    if expect_keys:
        try:
            payload = json.loads(ev.payload_json) if ev.payload_json else {}
        except Exception:
            fail(f"Event '{code}' payload is not valid JSON")
        missing = [k for k in expect_keys if k not in payload]
        if missing:
            fail(f"Event '{code}' payload missing keys: {missing}")
    ok(f"Event '{code}' present")


# ---------- 测试用例 ----------

def test_00_setup():
    create_db_and_tables()
    ok("DB ready")


def test_01_calendar_refresh_and_is_trading_day():
    # 预置缓存，确保即使 akshare 异常也能工作
    _seed_calendar_cache(include_today=True, stale_days=0)
    # 刷新交易日历（可能走 akshare 或 cache）
    S.refresh_trading_calendar()

    # 验证产生了 refresh 或 cache 类事件之一
    any_ok = False
    for code in ["calendar_refresh_ok", "calendar_use_cache", "calendar_fallback_weekday", "calendar_unavailable"]:
        ev = _get_latest_event_by_code(code)
        if ev:
            any_ok = True
            break
    if not any_ok:
        fail("No calendar_* event produced by refresh_trading_calendar()")

    # —— 关键修正：先看日历来源，再做断言 —— #
    today = S.now_cn().date()
    val = S.is_trading_day(today)
    if not isinstance(val, bool):
        fail("is_trading_day() did not return bool")

    src = S._CALENDAR_SOURCE
    print(f"[INFO] calendar_source={src}, is_trading_day({today})={val}")

    if src == "cache":
        # 我们刚刚写入的缓存包含今天 → 应为 True
        if not val:
            fail("is_trading_day(today) is False even though cache includes today")
    elif src == "weekday_fallback":
        # 回退：仅按工作日判断
        expect = today.weekday() < 5
        if val != expect:
            fail(f"weekday_fallback mismatch: got {val}, expect {expect}")
    else:
        # akshare/unavailable 场景：不强行设 True，只需是布尔即可（周末/节假日可能 False）
        pass

    ok(f"is_trading_day({today}) check OK under source={src}")


def test_02_tick_and_skip_events():
    # 强制让窗口判断为 True，交易日为 True → 触发 tick
    orig_window = S.is_in_trading_window
    orig_trading = S.is_trading_day
    try:
        S.is_in_trading_window = lambda now=None: True  # type: ignore[assignment]
        S.is_trading_day = lambda d=None: True          # type: ignore[assignment]
        S.run_trading_task()
        _assert_event_present("scheduler_tick", expect_keys=[
            "tz", "windows", "calendar_source", "calendar_fail_policy", "action"
        ])

        # 然后制造 skip：设为非交易日
        S.is_trading_day = lambda d=None: False         # type: ignore[assignment]
        S.run_trading_task()
        _assert_event_present("scheduler_skip", expect_keys=[
            "tz", "windows", "calendar_source", "calendar_fail_policy", "action"
        ])
    finally:
        S.is_in_trading_window = orig_window            # type: ignore[assignment]
        S.is_trading_day = orig_trading                 # type: ignore[assignment]
    ok("tick/skip events OK")


def test_03_state_restore():
    # 写入一条状态快照
    with get_session() as s:
        s.add(StateSnapshot(state=SystemState.OFFENSE, pos_ratio=0.42, note="restore_check"))
        s.commit()
    # 恢复
    st = S.restore_state_from_snapshot()
    if st != SystemState.OFFENSE:
        fail(f"restore_state_from_snapshot returned {st}, expect OFFENSE")
    ok("restore_state_from_snapshot OK")


def test_04_preheat_event():
    # 直接调用预热任务（不依赖 09:00）
    S.run_data_preheat_task()
    _assert_event_present("data_preheat", expect_keys=[
        "tz", "calendar_source", "calendar_cached_at", "staleness_days"
    ])
    ok("data_preheat event OK")


def test_05_start_and_stop_scheduler():
    # 启动调度器后立刻停止，检查 start/stop 事件
    sched = S.start_scheduler(auto_create_tables=False)
    _assert_event_present("scheduler_start", expect_keys=[
        "tz", "windows", "jobstore_url", "preheat_enabled",
        "calendar_source", "calendar_fail_policy", "calendar_degraded"
    ])
    S.shutdown_scheduler(sched)
    _assert_event_present("scheduler_stop", expect_keys=["tz"])
    ok("start/stop events OK")


def test_06_degraded_flags_in_payload():
    # 手动标记降级，验证 payload 透传
    S._CALENDAR_DEGRADED = True
    S._CALENDAR_STALENESS_DAYS = 99
    orig_window = S.is_in_trading_window
    orig_trading = S.is_trading_day
    try:
        S.is_in_trading_window = lambda now=None: True  # type: ignore[assignment]
        S.is_trading_day = lambda d=None: True          # type: ignore[assignment]
        S.run_trading_task()
        ev = _get_latest_event_by_code("scheduler_tick")
        if not ev:
            fail("scheduler_tick not found after degraded flag set")
        payload = json.loads(ev.payload_json) if ev.payload_json else {}
        if payload.get("calendar_degraded") is not True:
            fail("calendar_degraded flag not reflected in payload")
        if payload.get("calendar_staleness_days", 0) < 1:
            fail("calendar_staleness_days not reflected in payload")
    finally:
        S.is_in_trading_window = orig_window            # type: ignore[assignment]
        S.is_trading_day = orig_trading                 # type: ignore[assignment]
        S._CALENDAR_DEGRADED = False
        S._CALENDAR_STALENESS_DAYS = 0
    ok("degraded flags reflected in payload")


def main():
    try:
        test_00_setup()
        test_01_calendar_refresh_and_is_trading_day()
        test_02_tick_and_skip_events()
        test_03_state_restore()
        test_04_preheat_event()
        test_05_start_and_stop_scheduler()
        test_06_degraded_flags_in_payload()
        print("✅ ALL SCHEDULER TESTS PASSED")
    except SystemExit:
        raise
    except Exception as e:
        fail(f"Unhandled exception: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
