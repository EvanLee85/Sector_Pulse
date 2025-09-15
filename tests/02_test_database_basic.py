# tests/02_test_database_basic.py
# 运行方式：python -m tests.02_test_database_basic

from models.database import (
    create_db_and_tables, get_session,
    Account, StateSnapshot, SystemState,
    EventLog, LogLevel, OrderIntent, DB_URL
)
from sqlmodel import select
from zoneinfo import ZoneInfo
from datetime import datetime
from sqlalchemy.exc import IntegrityError
import json
import os


# 00) 冒烟：建表 + 基本写入（保留你原有逻辑）
def test_00_smoke_bootstrap():
    create_db_and_tables()
    acc = Account(total_asset=100000, total_market_value=0, available_cash=100000, position_ratio=0.0)
    snap = StateSnapshot(state=SystemState.OFFENSE, pos_ratio=0.3, note="smoke")
    print("Timezone:", acc.ts.tzinfo)  # 预期 Asia/Shanghai
    with get_session() as s:
        s.add(acc)
        s.add(snap)
        s.commit()
    print("✅ Database operations successful")


# 01) StateSnapshot 恢复能力（写入→“重启”→读取最新）
def test_01_snapshot_recovery():
    create_db_and_tables()
    # 步骤 A：写入一条快照
    with get_session() as s:
        s.add(StateSnapshot(state=SystemState.OFFENSE, pos_ratio=0.33, note="recovery_check"))
        s.commit()
    print("WROTE_SNAPSHOT")

    # 步骤 B：模拟“重启”后读取最新一条
    with get_session() as s:
        ss = s.exec(select(StateSnapshot).order_by(StateSnapshot.ts.desc())).first()
    assert ss is not None
    print("LATEST_SNAPSHOT", ss.state, ss.pos_ratio, ss.note)


# 02) EventLog 至少一条记录（level/code/payload）
def test_02_eventlog_nonempty():
    with get_session() as s:
        s.add(EventLog(level=LogLevel.INFO, code="smoke_test", payload_json=json.dumps({"ok": 1})))
        s.commit()
    with get_session() as s:
        ev = s.exec(select(EventLog).order_by(EventLog.ts.desc())).first()
    assert ev is not None
    print("LATEST_EVENT", ev.level, ev.code, bool(ev.payload_json))


# 03) OrderIntent.idempotency_key 唯一约束生效
def test_03_orderintent_unique():
    with get_session() as s:
        s.add(OrderIntent(action="open", qty=1, status="proposed", idempotency_key="dup-key-1"))
        s.commit()
        try:
            s.add(OrderIntent(action="open", qty=2, status="proposed", idempotency_key="dup-key-1"))
            s.commit()
            print("DUP_STATUS", "unexpected_success")
        except IntegrityError:
            s.rollback()
            print("DUP_STATUS", "IntegrityError")


# 04) 时区一致（Asia/Shanghai）
def test_04_timezone_consistency():
    acc = Account(total_asset=1, total_market_value=0, available_cash=1, position_ratio=0.0)
    print("TZINFO", acc.ts.tzinfo)
    delta = abs((acc.ts - datetime.now(ZoneInfo("Asia/Shanghai"))).total_seconds())
    print("DELTA_SEC", int(delta))
    print("DELTA_OK", delta <= 60)


# 05) DB 目标正确（写入到你预期的库）
def test_05_db_target():
    print("DB_URL", DB_URL)
    # 仅在 sqlite 场景下尝试确认文件存在
    if isinstance(DB_URL, str) and DB_URL.startswith("sqlite:///"):
        # 兼容相对路径
        path = DB_URL[len("sqlite:///"):]
        print("DB_PATH", path)
        print("DB_PATH_EXISTS", os.path.exists(path))


if __name__ == "__main__":
    test_00_smoke_bootstrap()
    test_01_snapshot_recovery()
    test_02_eventlog_nonempty()
    test_03_orderintent_unique()
    test_04_timezone_consistency()
    test_05_db_target()
