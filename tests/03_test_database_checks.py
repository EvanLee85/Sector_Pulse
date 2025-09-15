# tests/03_test_database_checks.py
"""
数据库交付自检（5项）：
1) StateSnapshot 恢复能力（写入→“重启”→读取最新）
2) EventLog 至少 1 条记录（level/code/payload）
3) OrderIntent.idempotency_key 唯一约束触发
4) 时区一致（Asia/Shanghai；时间差 ≤ 60s）
5) DB_URL 指向预期目标库（打印）

运行：
  python -m tests.03_test_database_checks
"""

from __future__ import annotations
import sys
import json
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from models.database import (
    create_db_and_tables, get_session, DB_URL,
    Account, StateSnapshot, EventLog, OrderIntent,
    SystemState, LogLevel
)


def ok(msg: str):
    print(f"[OK] {msg}")

def fail(msg: str):
    print(f"[FAIL] {msg}")
    sys.exit(1)


def test_1_statesnapshot_recovery():
    """写入一条 StateSnapshot，关闭会话后读取最新一条，验证不丢失"""
    create_db_and_tables()

    marker = f"recovery_check:{uuid.uuid4().hex[:8]}"
    with get_session() as s:
        s.add(StateSnapshot(state=SystemState.OFFENSE, pos_ratio=0.33, note=marker))
        s.commit()

    # 模拟“重启”：重新获取会话读取最新
    with get_session() as s:
        latest = s.exec(select(StateSnapshot).order_by(StateSnapshot.ts.desc())).first()

    if not latest:
        fail("StateSnapshot 读取为空")
    if latest.note != marker:
        fail(f"StateSnapshot 最新记录不匹配（expect note={marker}, got={latest.note}）")

    ok(f"StateSnapshot 恢复成功（state={latest.state}, pos_ratio={latest.pos_ratio}, note={latest.note}）")


def test_2_eventlog_exists():
    """写入一条 EventLog，读取最新事件，验证 level/code/payload 存在"""
    code = f"smoke_test_{uuid.uuid4().hex[:6]}"
    with get_session() as s:
        s.add(EventLog(level=LogLevel.INFO, code=code, payload_json=json.dumps({"ok": 1})))
        s.commit()
    with get_session() as s:
        ev = s.exec(select(EventLog).order_by(EventLog.ts.desc())).first()
    if not ev:
        fail("EventLog 读取为空")
    if not ev.payload_json:
        fail("EventLog.payload_json 为空")
    if ev.code != code:
        fail(f"EventLog code 不匹配（expect {code}, got {ev.code}）")

    ok(f"EventLog 写入读取成功（level={ev.level}, code={ev.code}）")


def test_3_orderintent_idempotency_unique():
    """验证 OrderIntent.idempotency_key 唯一约束"""
    dup_key = f"dup-{uuid.uuid4().hex[:8]}"
    with get_session() as s:
        s.add(OrderIntent(action="open", qty=1, status="proposed", idempotency_key=dup_key))
        s.commit()

        try:
            s.add(OrderIntent(action="open", qty=2, status="proposed", idempotency_key=dup_key))
            s.commit()
            # 如果能走到这里，表示唯一性没生效
            fail("OrderIntent.idempotency_key 未触发唯一性约束（重复插入成功）")
        except IntegrityError:
            s.rollback()
            ok("OrderIntent.idempotency_key 唯一性约束已触发（IntegrityError）")


def test_4_timezone_shanghai():
    """验证新建记录的 ts 时区与 Asia/Shanghai 一致，时间差 ≤ 60s"""
    acc = Account(total_asset=1, total_market_value=0, available_cash=1, position_ratio=0.0)
    if str(acc.ts.tzinfo) != "Asia/Shanghai":
        fail(f"时区不一致（got tz={acc.ts.tzinfo}, expect Asia/Shanghai）")

    delta = abs((acc.ts - datetime.now(ZoneInfo("Asia/Shanghai"))).total_seconds())
    if delta > 60:
        fail(f"时间差过大（{delta:.1f}s > 60s）")

    ok(f"时区一致（{acc.ts.tzinfo}），时间差 {delta:.1f}s")


def test_5_print_db_url():
    """打印当前 DB_URL，便于核对实际目标库"""
    print(f"[INFO] DB_URL {DB_URL}")
    ok("DB_URL 打印完成")


def main():
    try:
        test_1_statesnapshot_recovery()
        test_2_eventlog_exists()
        test_3_orderintent_idempotency_unique()
        test_4_timezone_shanghai()
        test_5_print_db_url()
        print("✅ ALL TESTS PASSED")
    except SystemExit as e:
        # fail() 已经打印原因并 exit(1)
        raise
    except Exception as e:
        fail(f"未捕获异常：{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
    