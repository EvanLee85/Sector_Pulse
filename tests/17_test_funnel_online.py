# tests/17_test_funnel_online.py
# -*- coding: utf-8 -*-
"""
步骤5 · 漏斗（在线最小可行）
- 目标：在未接入行业/板内完整在线链路时，流程不阻塞：
  * 允许产出 0 只候选，但必须写入 funnel/result 事件与报告文件
  * 不报错即视为通过
- 产出：tests/test_reports/funnel_online_YYYYMMDD_HHMM.json
"""

from __future__ import annotations
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlmodel import select
from models.database import (
    create_db_and_tables,
    get_session,
    EventLog,
    Signal,
    StateSnapshot,
    SystemState,
)
from utils.scheduler import select_candidates_funnel

_SH = ZoneInfo("Asia/Shanghai")


def _now():
    return datetime.now(_SH)


def main() -> int:
    os.makedirs("./tests/test_reports", exist_ok=True)
    create_db_and_tables()

    # 设定当前状态为 OFFENSE（正常产出候选的制度环境；即便最终为0也只因在线链路未接入）
    with get_session() as s:
        s.add(StateSnapshot(state=SystemState.OFFENSE, pos_ratio=0.20, note="funnel_online"))
        s.commit()

    # 在线模式：不注入 injected；允许行业/个股链路未接入而降级
    out = select_candidates_funnel(
        state="OFFENSE",
        ks_level=0,
        window=_now().strftime("%H:%M"),
        indicators_payload=None,
        injected=None,
        emit_event=True,
    )

    # 事件应写入（即便 selected 为 0）
    with get_session() as s:
        ev = s.exec(select(EventLog).where(EventLog.code == "funnel/result")).first()
        assert ev is not None, "期望已落 funnel/result 事件，但未找到"

    # 写报告
    fn = f"./tests/test_reports/funnel_online_{_now().strftime('%Y%m%d_%H%M')}.json"
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ts": _now().isoformat(),
                "selected": out.get("selected", []),
                "rejected": out.get("rejected", []),
                "executable": out.get("executable", False),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[OK] 在线漏斗：selected={len(out.get('selected', []))}（允许为0）")
    print(f"[OK] 报告已写入：{fn}")
    print("[OK] 严格模式：流程通过（允许降级告警）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
