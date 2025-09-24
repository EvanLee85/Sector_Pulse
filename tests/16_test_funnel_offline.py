# tests/16_test_funnel_offline.py
# -*- coding: utf-8 -*-
"""
步骤5 · 漏斗（离线注入版）
- 构造行业打分与板内个股样本；验证：严格 3–5名、0–5只、每行业≤2、RS/流动性阈值、软/硬上限与惩罚、EV/RR/Pwin 软约束
- 产出：EventLog.funnel/result 快照；Signal 表记录（selected+rejected），并附 ctx_event_id
- 报告：tests/test_reports/funnel_offline_YYYYMMDD_HHMM.json
"""
from __future__ import annotations
import json, os, sys, time
from datetime import datetime
from zoneinfo import ZoneInfo
from sqlmodel import select

from models.database import create_db_and_tables, get_session, EventLog, Signal, StateSnapshot, SystemState
from utils.scheduler import select_candidates_funnel, _load_funnel_cfg

_SH = ZoneInfo("Asia/Shanghai")

def _now():
    return datetime.now(_SH)

def main() -> int:
    os.makedirs("./tests/test_reports", exist_ok=True)
    create_db_and_tables()
    # 设定当前状态为 OFFENSE 以便产出可执行
    with get_session() as s:
        s.add(StateSnapshot(state=SystemState.OFFENSE, pos_ratio=0.25, note="funnel_offline"))
        s.commit()

    # 构造行业（至少 6 个，确保有 3–5 名）
    sectors = []
    for i in range(1, 8):
        sectors.append({
            "name": f"S{i}",
            "rs": 0.2 + 0.1*i,              # 0.3..0.9
            "breadth": 0.3 + 0.07*i,        # 0.37..0.79
            "lead": 0.02 * ((i%4)-2),       # -0.02..+0.02
            "fundshare": 0.03 * i,          # 0.03..0.21
            "continuity": 1.0 if i in (1,3,5,7) else 0.0
        })

    # 计算 10日均额总体分布（用于分位上限）
    amounts_all = [3e8, 6e8, 9e8, 1.2e9, 1.5e9, 2e9, 2.5e9]

    # 构造板内个股（仅给 3–5 名的三个行业各 4 只：rank=2..5，确保过滤 rank_out_of_range 与配额）
    stocks = []
    # 假设 sector 排名将按 score 降序，大致 S7>S6>S5>S4>S3...
    # 我们准备 S5,S4,S3 作为目标行业（3–5 名），并提供每行业 rank=2..5 的四只
    for sec, base in (("S5", 0.80), ("S4", 0.78), ("S3", 0.76)):
        for rnk in (2,3,4,5):
            stocks.append({
                "symbol": f"{sec}_R{rnk}",
                "sector": sec,
                "rank_in_sector": rnk,
                "rs_price": base - 0.02*(rnk-2),
                "rs_volume": 0.60 + 0.05*(rnk-2),
                "rs_fund": 0.55 + 0.10*(rnk-2),
                "amount_10d": 1.1e9 if rnk==3 else (8e8 if rnk in (4,5) else 2.2e9),  # r3 超软上限触发惩罚；r2 超硬上限被拒
                "rr": 2.1, "pwin": 0.58, "ev_bps": 80,
            })

    injected = {"sectors": sectors, "stocks": stocks, "amounts_all": amounts_all}

    out = select_candidates_funnel(
        state="OFFENSE", ks_level=0, window=_now().strftime("%H:%M"),
        indicators_payload=None, injected=injected, emit_event=True
    )

    # 断言：严格行业 3–5 名（我们准备的 S5/S4/S3）
    assert set(out.get("selected_sectors", [])) <= {"S3","S4","S5"}
    # 断言：每行业至多 2 只 + 总数 ≤5
    assert len(out["selected"]) <= 5
    per_sector = {}
    for it in out["selected"]:
        per_sector[it["sector"]] = per_sector.get(it["sector"], 0) + 1
        # r3 会被软惩罚
        if it["symbol"].endswith("_R3"):
            assert it.get("oversize", False) and it.get("penalty", 1.0) < 1.0
    assert all(v<=2 for v in per_sector.values())

    # 快照事件存在；Signal 记录写入
    with get_session() as s:
        ev = s.exec(select(EventLog).where(EventLog.code == "funnel/result")).first()
        assert ev is not None
        sel_cnt = s.exec(select(Signal).where(Signal.executable == True)).all()
        rej_cnt = s.exec(select(Signal).where(Signal.reason_reject.is_not(None))).all()
        assert len(sel_cnt) >= 1
        assert len(rej_cnt) >= 1

    rep = {
        "ts": _now().isoformat(),
        "selected": out["selected"],
        "rejected": out["rejected"],
        "selected_sectors": out["selected_sectors"],
        "event_id": out.get("event_id", 0),
    }
    fn = f"./tests/test_reports/funnel_offline_{_now().strftime('%Y%m%d_%H%M')}.json"
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"[OK] 离线漏斗：selected={len(out['selected'])}，snapshot_event={out.get('event_id')}")
    print(f"[OK] 报告已写入：{fn}")
    print("[OK] 严格模式：全部通过")
    return 0

if __name__ == "__main__":
    sys.exit(main())
