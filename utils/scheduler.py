# utils/scheduler.py
# SectorPulse · 调度器（AkShare 交易日历 · 降级/回退版）
# - 只在 10:00 / 14:00 触发主任务（交易日过滤）
# - 交易日来源：AkShare 主 → 本地缓存 → 回退（fail_open|fail_closed 可配置）
# - 降级标记：calendar_degraded + staleness_days，供策略层“不开新仓，只管持仓”
# - 09:00 预热：刷新交易日历（可开关）
# - 业务/落库时间一律 ZoneInfo；APScheduler timezone 参数用 pytz（库兼容要求）
# - SQLModel 查询语法修正 + 恢复函数基础错误处理

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple, List
import math
import pandas as pd

import pytz  # APScheduler 推荐 pytz
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from sqlmodel import select
from zoneinfo import ZoneInfo

from config.loader import load_config
from models.database import (
    DB_URL,
    EventLog,
    StateSnapshot,
    SystemState,
    create_db_and_tables,
    get_session,
    Account,
    Signal,
)

from config.loader import load_config

def _load_yaml_merged(config_dir: str, _ignored_main_file: str = "") -> dict:
    """
    兼容旧签名：忽略第二个参数，底层用 load_config 合并 base.yaml/strategy_core.yaml/dev.override.yaml
    """
    from config.loader import load_config
    return load_config(config_dir) or {}


# ======================
# 配置与常量
# ======================

_CFG = load_config("config")

_TZ_NAME = os.getenv("SECTORPULSE_TZ",
                     (_CFG.get("execution") or {}).get("tz") or "Asia/Shanghai")
TZ_ZI = ZoneInfo(_TZ_NAME)             # 业务/落库时间
SCHEDULER_TZ = pytz.timezone(_TZ_NAME) # APScheduler 专用

def now_cn() -> datetime:
    """统一的本地时间入口（Asia/Shanghai 或配置指定时区）。"""
    return datetime.now(TZ_ZI)

# —— 执行窗口（优先 execution.windows: List[str]；回退兼容旧 runtime.exec_windows）——
_DEF_WINDOWS = ["10:00", "14:00"]
_cfg_windows = _CFG.get("execution", {}).get("windows")
TRADING_WINDOWS = (
    [w for w in _cfg_windows if isinstance(w, str)]
    if isinstance(_cfg_windows, list) else _DEF_WINDOWS
)
TRADING_WINDOWS = [w for w in _DEF_WINDOWS if w in TRADING_WINDOWS] or _DEF_WINDOWS

# APScheduler 参数
_SCHED = _CFG.get("scheduler", {})
_COALESCE = bool(_SCHED.get("coalesce", True))
_MISFIRE = int(_SCHED.get("misfire_grace_time_sec", 300))
_MAX_INST = int(_SCHED.get("max_instances", 1))

# 交易日失败策略与缓存陈旧阈值（环境变量优先）
_FAIL_POLICY = os.getenv("SECTORPULSE_CAL_FAIL_POLICY", (_SCHED.get("calendar_fail_policy") or "fail_open")).lower()
_CACHE_STALE_DAYS = int(os.getenv("SECTORPULSE_CAL_CACHE_STALE_DAYS", str(_SCHED.get("calendar_cache_stale_days", 7))))

# 是否启用 09:00 预热（默认开启，可通过环境变量关闭）
_ENABLE_PREHEAT = os.getenv("SECTORPULSE_ENABLE_PREHEAT", "1") == "1"

# 执行器线程池（0 表示不显式设置）
_EXEC_WORKERS = int(os.getenv("SECTORPULSE_EXEC_WORKERS", "0"))

# JobStore：SQLite 场景默认单独文件以减少锁竞争；否则复用业务 DB_URL
_JOBSTORE_URL = os.getenv("SECTORPULSE_JOBSTORE_URL") or (
    "sqlite:///./sectorpulse_jobs.db" if DB_URL.startswith("sqlite") else DB_URL
)

# 运行时缓存（交易日历）
_RUNTIME_CACHE_DIR = Path("./runtime_cache")
_RUNTIME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CALENDAR_CACHE = _RUNTIME_CACHE_DIR / "trade_calendar.json"

# ======================
# 事件落库
# ======================

def _log_event(level: str, code: str, payload: Dict[str, Any]) -> None:
    with get_session() as s:
        s.add(EventLog(level=level, code=code, payload_json=json.dumps(payload, ensure_ascii=False)))
        s.commit()

# ======================
# 步骤5 · 漏斗（选板→选股）
# ======================

def _percentile_rank(series: List[float], x: float) -> float:
    vals = [v for v in series if v is not None and not math.isnan(v)]
    if not vals:
        return 0.0
    vals = sorted(vals)
    # 经典 PR = (#<=x)/n
    import bisect
    k = bisect.bisect_right(vals, x)
    return max(0.0, min(1.0, k / len(vals)))

def _load_funnel_cfg() -> Dict[str, Any]:
    from config.loader import load_config
    cfg = load_config("config")
    return (cfg or {}).get("funnel", {}) or {}

def _should_skip_by_state_and_ks(state: str, ks_level: int, cfg: Dict[str, Any]) -> Tuple[str, bool]:
    """返回 (mode, executable_flag)。mode: skip|observe_only|normal"""
    execp = (cfg.get("exec_policy") or {})
    if ks_level >= 2 or state == "SLEEP":
        return execp.get("on_sleep", "skip"), False
    if ks_level == 1:
        return execp.get("on_ks_L1", "observe_only"), False
    if state == "WATCH":
        return execp.get("on_watch", "observe_only"), False
    return execp.get("on_offense_hold", "normal"), True

def _calc_soft_hard_upper(amounts_all: List[float], state: str, cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    pol = cfg.get("amount_10d_upper_policy") or {}
    mode = (pol.get("mode") or "auto").lower()
    floors = pol.get("floors_caps") or {}
    soft_floor = float(floors.get("soft_floor", 1e9))
    hard_cap   = float(floors.get("hard_cap",   3e9))
    if mode == "hard":
        hard = float(cfg.get("amount_10d_max", 1e9))
        return soft_floor, hard, float(pol.get("penalty", 0.9))
    if not amounts_all:
        return soft_floor, hard_cap, float(pol.get("penalty", 0.9))

    p = pol.get("percentiles") or {}
    soft_map = {"OFFENSE": p.get("offense_soft", 0.80),
                "HOLD":    p.get("hold_soft",    0.75),
                "WATCH":   p.get("watch_soft",   0.70)}
    soft_pct = float(soft_map.get(state, 0.75))
    hard_pct = float(p.get("hard", 0.90))

    try:
        import numpy as np
        soft_up = float(np.quantile(amounts_all, soft_pct))
        hard_up = float(np.quantile(amounts_all, hard_pct))
    except Exception:
        xs = sorted(v for v in amounts_all if v is not None)
        def _q(q: float) -> float:
            if not xs: return 0.0
            i = int(max(0, min(len(xs)-1, round(q*(len(xs)-1)))))
            return float(xs[i])
        soft_up = _q(soft_pct)
        hard_up = _q(hard_pct)

    soft_up = max(soft_up, soft_floor)
    hard_up = min(hard_up, hard_cap)
    return soft_up, hard_up, float(pol.get("penalty", 0.9))

def _funnel_emit_snapshot(sector_snapshot: Dict[str, Any],
                          selected: List[Dict[str, Any]],
                          rejected: List[Dict[str, Any]],
                          window: str,
                          note: str = "") -> int:
    payload = {
        "window": window,
        "sector_rankings_snapshot": sector_snapshot,
        "selected_candidates": selected,
        "rejected": rejected,
        "note": note,
    }
    with get_session() as s:
        ev = EventLog(level="INFO", code="funnel/result",
                      payload_json=json.dumps(payload, ensure_ascii=False))
        s.add(ev)
        s.commit()
        return int(ev.id or 0)

def _select_candidates_funnel_offline_compatible(
    state: str,
    ks_level: int,
    window: str,
    indicators_payload: Optional[Dict[str, Any]],
    injected: Dict[str, Any],
    emit_event: bool
) -> Dict[str, Any]:
    """
    与 16_test_funnel_offline 兼容的离线路径：
    - 使用 injected["sectors"] / injected["stocks"] / injected["amounts_all"]
    - 权重、阈值、配额全部读取 funnel 配置
    - 事件快照：funnel/result
    - Signal 入库：selected 可执行与否取决于 exec policy（state / ks_level）
    """
    cfg = _load_funnel_cfg()
    mode, can_exec = _should_skip_by_state_and_ks(state, ks_level, cfg)

    # —— 行业强度（加权打分）——
    sector_scores: List[Dict[str, Any]] = []
    w = cfg.get("score_weights") or {}
    rsw = float(w.get("rs", 0.40))
    bw  = float(w.get("breadth", 0.25))
    lw  = float(w.get("leadership", 0.20))
    fw  = float(w.get("fundshare", 0.10))
    cw  = float(w.get("continuity", 0.05))

    caps = cfg.get("caps") or {}
    lead_cap = float(caps.get("leadership_abs", 0.10))
    fund_cap = float(caps.get("fundshare_max", 0.15))

    for x in injected.get("sectors", []):
        lead_raw = float(x.get("lead", 0.0))
        lead_raw = max(-lead_cap, min(lead_cap, lead_raw))
        lead_norm = (lead_raw + lead_cap) / (2 * lead_cap) if lead_cap > 0 else 0.5
        fund_norm = min(float(x.get("fundshare", 0.0)), fund_cap) / fund_cap if fund_cap > 0 else 0.0
        sc = (
            rsw * float(x.get("rs", 0.0))
            + bw * float(x.get("breadth", 0.0))
            + lw * lead_norm
            + fw * fund_norm
            + cw * float(x.get("continuity", 0.0))
        )
        sector_scores.append({
            "name": x.get("name"),
            "score": sc,
            "parts": {
                "rs": x.get("rs", 0.0),
                "breadth": x.get("breadth", 0.0),
                "lead_norm": lead_norm,
                "fund_norm": fund_norm,
                "continuity": x.get("continuity", 0.0),
            }
        })

    sector_scores = sorted(sector_scores, key=lambda d: d["score"], reverse=True)
    sr = cfg.get("sector_rank_range") or [3, 5]
    rank_lo, rank_hi = int(sr[0]), int(sr[1])
    selected_sectors = [d["name"] for i, d in enumerate(sector_scores, start=1) if rank_lo <= i <= rank_hi]

    # —— 板内个股 ——（RS_adjusted + 10D 均额上下限 + EV/RR/Pwin 软约束）
    wmap = (cfg.get("rs_adjusted", {}).get("regime_weights") or {}).get(state, {})
    wp = float(wmap.get("price", 0.55 if state == "OFFENSE" else 0.45))
    wv = float(wmap.get("volume", 0.30))
    wf = float(wmap.get("fund", 0.15))

    rs_min = float(cfg.get("rs_min", 0.75))
    stock_rank_range = cfg.get("stock_rank_range") or [3, 5]
    rlo, rhi = int(stock_rank_range[0]), int(stock_rank_range[1])

    per_sector_cap = int(cfg.get("per_sector_cap", 2))
    per_window_cap = int(cfg.get("max_candidates_per_window", 5))

    amounts_all = injected.get("amounts_all") or []
    soft_up, hard_up, penalty = _calc_soft_hard_upper(amounts_all, state, cfg)

    selected: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    per_sector_count: Dict[str, int] = {}

    for row in injected.get("stocks", []):
        sym = str(row["symbol"])
        sec = str(row.get("sector", ""))

        # 行业是否在 3–5 名
        if sec not in selected_sectors:
            rejected.append({"symbol": sym, "sector": sec, "reason_reject": "sector_rank_out_of_range"})
            continue

        # 计算 rs_adj
        rprice = float(row.get("rs_price", 0.0))
        rvol   = float(row.get("rs_volume", 0.0))
        rfund  = float(row.get("rs_fund", 0.0))
        rs_adj = wp * rprice + wv * rvol + wf * rfund

        a10 = float(row.get("amount_10d", 0.0))
        rr  = float(row.get("rr", 0.0))
        pwin = float(row.get("pwin", 0.0))
        ev_bps = float(row.get("ev_bps", 0.0))
        rnk = int(row.get("rank_in_sector", 99))

        # 严格 rank 3–5
        if not (rlo <= rnk <= rhi):
            rejected.append({
                "symbol": sym, "sector": sec, "reason_reject": "rank_out_of_range",
                "rank_in_sector": rnk, "rs_adj": rs_adj
            })
            continue

        # rs 下限
        if rs_adj < rs_min:
            rejected.append({"symbol": sym, "sector": sec, "reason_reject": "rs_below_min",
                             "rs_adj": rs_adj, "rs_min": rs_min})
            continue

        # 10D 均额：下限 + 上限/惩罚
        if a10 < float(cfg.get("amount_10d_min", 5e8)):
            rejected.append({"symbol": sym, "sector": sec, "reason_reject": "amount_10d_below_min",
                             "amount_10d": a10})
            continue
        oversize = False
        penalty_applied = 1.0
        if a10 > hard_up:
            rejected.append({"symbol": sym, "sector": sec, "reason_reject": "amount_10d_above_max",
                             "amount_10d": a10, "hard_upper": hard_up})
            continue
        elif a10 > soft_up:
            oversize = True
            penalty_applied = penalty
            rs_adj *= penalty

        # EV/RR/Pwin 软约束（标注，不强拒）
        ev_cfg = cfg.get("ev_rr_pwin") or {}
        t = ev_cfg.get("thresholds") or {}
        rr_min = float(t.get("rr_min", 2.0))
        pwin_min = float(t.get("pwin_min", 0.55))
        ev_min = float(t.get("ev_bps_min", 60))
        tags = []
        if rr and rr < rr_min: tags.append("rr_low")
        if pwin and pwin < pwin_min: tags.append("pwin_low")
        if ev_bps and ev_bps < ev_min: tags.append("ev_low")

        # 行业/窗口配额
        if per_sector_count.get(sec, 0) >= per_sector_cap:
            rejected.append({"symbol": sym, "sector": sec, "reason_reject": "sector_quota_exceeded"})
            continue
        if len(selected) >= per_window_cap:
            rejected.append({"symbol": sym, "sector": sec, "reason_reject": "window_quota_exceeded"})
            break

        selected.append({
            "symbol": sym, "sector": sec, "rank_in_sector": rnk,
            "rs_price": rprice, "rs_volume": rvol, "rs_fund": rfund,
            "rs_adj": rs_adj, "penalty": penalty_applied, "oversize": oversize,
            "amount_10d": a10, "rr": rr, "pwin": pwin, "ev_bps": ev_bps,
        })
        per_sector_count[sec] = per_sector_count.get(sec, 0) + 1

    # —— 快照事件 + Signal 入库 —— 
    ev_id = 0
    if emit_event:
        sector_snapshot = {"scores": sector_scores, "rank_range": [rank_lo, rank_hi]}
        ev_id = _funnel_emit_snapshot(sector_snapshot, selected, rejected, window)

    from models.database import get_session, Signal, SystemState
    with get_session() as s:
        for it in selected:
            s.add(Signal(
                symbol=it["symbol"], side="buy", price_ref=None,
                state_at_emit=SystemState(state) if state in SystemState.__members__ else None,
                emotion_gate_passed=None,
                executable=bool(can_exec and (mode == "normal")),
                reason_reject=None, window=window,
                microstructure_checks=json.dumps({
                    "rs_price": it["rs_price"], "rs_volume": it["rs_volume"], "rs_fund": it["rs_fund"],
                    "rs_adj": it["rs_adj"], "amount_10d": it["amount_10d"],
                    "oversize": it.get("oversize", False), "penalty": it.get("penalty", 1.0),
                    "ctx_event_id": ev_id,
                }, ensure_ascii=False)
            ))
        for it in rejected:
            s.add(Signal(
                symbol=it.get("symbol", "?"), side="buy", price_ref=None,
                state_at_emit=SystemState(state) if state in SystemState.__members__ else None,
                emotion_gate_passed=None,
                executable=False,
                reason_reject=it.get("reason_reject", ""),
                window=window,
                microstructure_checks=json.dumps(
                    {k: v for k, v in it.items() if k not in ("symbol", "reason_reject")},
                    ensure_ascii=False
                )
            ))
        s.commit()

    return {
        "selected": selected,
        "rejected": rejected,
        "executable": bool(can_exec and (mode == "normal")),
        "event_id": ev_id,
        "selected_sectors": selected_sectors,
    }


def select_candidates_funnel(state: str,
                             ks_level: int,
                             window: str,
                             indicators_payload: Optional[Dict[str, Any]] = None,
                             injected: Optional[Dict[str, Any]] = None,
                             emit_event: bool = True) -> Dict[str, Any]:
    """
    步骤5主函数（在线优先，离线注入可覆盖）：
    - 在线：申万 L1 桶 + L1 成分（必要时 L3 回填） + 全市场快照 + 前复权日线，按既定口径打分/筛选
    - 离线：若提供 injected，则走原离线路径（与你当前一致）
    - 结果：≤5只、每行业≤2只；严格板内3–5名约束；事件与Signal入库
    """
    cfg = _load_funnel_cfg()
    mode, can_exec = _should_skip_by_state_and_ks(state, ks_level, cfg)
    if mode == "skip":
        if emit_event:
            _log_event("WARN", "funnel/skipped", {
                "window": window, "state": state, "ks_level": ks_level, "reason": "exec_policy_skip"
            })
        return {"selected": [], "rejected": [], "executable": False, "skipped": True}

    # ------------------------------
    # A) 若提供 injected → 走你原本离线路径（保持不变）
    # ------------------------------
    if injected and ("sectors" in injected or "stocks" in injected):
        return _select_candidates_funnel_offline_compatible(
            state=state, ks_level=ks_level, window=window,
            indicators_payload=indicators_payload, injected=injected, emit_event=emit_event
        )

    # ------------------------------
    # B) 在线链路（真实数据）
    # ------------------------------
    from data_service.collector import (
        fetch_sw_index_first_info, fetch_sw_l1_components, fetch_sw_index_third_info,
        fetch_market_spot_with_fallback, fetch_limit_up_pool, fetch_limit_down_pool,
        fetch_stock_daily_qfq
    )
    from models.database import get_session, EventLog, Signal, SystemState

    # 配置参数
    sr = cfg.get("sector_rank_range") or [3, 5]
    rank_lo, rank_hi = int(sr[0]), int(sr[1])
    w = cfg.get("score_weights") or {}
    w_rs = float(w.get("rs", 0.40))
    w_bd = float(w.get("breadth", 0.25))
    w_ld = float(w.get("leadership", 0.20))
    w_fs = float(w.get("fundshare", 0.10))
    w_ct = float(w.get("continuity", 0.05))  # 目前在线先置 0（可后续接入）

    caps = cfg.get("caps") or {}
    lead_cap = float(caps.get("leadership_abs", 0.10))
    fund_cap = float(caps.get("fundshare_max", 0.15))

    rs_win = int(cfg.get("rs_window_days", 20))
    rs_min = float(cfg.get("rs_min", 0.75))
    stock_rank_range = cfg.get("stock_rank_range") or [3, 5]
    rlo, rhi = int(stock_rank_range[0]), int(stock_rank_range[1])

    per_sector_cap = int(cfg.get("per_sector_cap", 2))
    per_window_cap = int(cfg.get("max_candidates_per_window", 5))

    # 取上一个交易日（用于涨/跌停池）
    now = now_cn()
    prev_yyyymmdd = _get_prev_trade_yyyymmdd(now.date())

    # 1) 准备全市场快照（用于资金份额） + 昨日涨/跌停池（用于 leadership）
    try:
        spot_res = fetch_market_spot_with_fallback()
        df_spot = spot_res.data if hasattr(spot_res, "data") else None
        if not isinstance(df_spot, pd.DataFrame) or df_spot.empty:
            raise RuntimeError("market spot empty")
        # 尽量统一 amount 列
        if "amount_yuan" not in df_spot.columns:
            # 简单兜底：若没有 amount_yuan，尝试常见列名
            for c in ("成交额", "成交额(元)", "amount"):
                if c in df_spot.columns:
                    df_spot = df_spot.copy()
                    df_spot["amount_yuan"] = pd.to_numeric(df_spot[c], errors="coerce")
                    break
        total_amount = float(pd.to_numeric(df_spot.get("amount_yuan", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    except Exception as e:
        _log_event("ERROR", "funnel/error", {"window": window, "stage": "market_spot", "err": str(e)})
        total_amount = 0.0
        df_spot = pd.DataFrame()

    try:
        zt_res = fetch_limit_up_pool(prev_yyyymmdd); df_zt = zt_res.data if hasattr(zt_res, "data") else None
    except Exception:
        df_zt = None
    try:
        dt_res = fetch_limit_down_pool(prev_yyyymmdd); df_dt = dt_res.data if hasattr(dt_res, "data") else None
    except Exception:
        df_dt = None

    zt_set = set()
    dt_set = set()
    if isinstance(df_zt, pd.DataFrame) and not df_zt.empty:
        col = "股票代码" if "股票代码" in df_zt.columns else ("code" if "code" in df_zt.columns else None)
        if col:
            zt_set = {str(x).strip() for x in df_zt[col].dropna().tolist()}
    if isinstance(df_dt, pd.DataFrame) and not df_dt.empty:
        col = "股票代码" if "股票代码" in df_dt.columns else ("code" if "code" in df_dt.columns else None)
        if col:
            dt_set = {str(x).strip() for x in df_dt[col].dropna().tolist()}

    # 2) 取申万 L1 列表
    try:
        l1_info = fetch_sw_index_first_info()
        df_l1 = l1_info.data
        if not isinstance(df_l1, pd.DataFrame) or df_l1.empty:
            raise RuntimeError("sw_index_first_info empty")
        # 统一列
        df_l1 = df_l1.rename(columns={"行业代码": "l1_code", "行业名称": "l1_name", "成份个数": "count"})
        df_l1["l1_code"] = df_l1["l1_code"].astype(str).str.replace(".SI", "", regex=False)
    except Exception as e:
        _log_event("ERROR", "funnel/error", {"window": window, "stage": "sw_l1_list", "err": str(e)})
        df_l1 = pd.DataFrame(columns=["l1_code", "l1_name", "count"])

    # 3) 行业打分（rs / breadth / leadership / fundshare / continuity）
    sector_scores: List[Dict[str, Any]] = []

    for _, row in df_l1.iterrows():
        code = str(row.get("l1_code") or "").strip()
        name = str(row.get("l1_name") or "").strip()
        if not code or not name:
            continue

        # 3.1 取成分（L1 主口径；失败则尝试 L3 回填）
        try:
            symbols = fetch_sw_l1_components(code)  # 返回 6位+后缀 或 6位（源而定）
        except Exception:
            symbols = []
        if not symbols:
            # 回填尝试（用 L3 汇总，当前 collector 内部已做容错，这里简单再判空即可）
            symbols = fetch_sw_l1_components(code, fallback_l3_codes=None)
        if not symbols:
            continue

        # 将不带后缀的，尽量与 df_spot 合并时做 left-join 容错
        sym_set = set(str(s).upper() for s in symbols)

        # 3.2 leadership：用昨日涨/跌停池
        #     leadership_raw = (涨停数 - 跌停数) / 成分数，裁剪到 [-lead_cap, +lead_cap] 并映射到 [0,1]
        if sym_set:
            zt_n = len(sym_set & {s.upper() for s in zt_set})
            dt_n = len(sym_set & {s.upper() for s in dt_set})
            raw = 0.0
            try:
                raw = (zt_n - dt_n) / max(len(sym_set), 1)
            except Exception:
                raw = 0.0
            ld_raw = max(-lead_cap, min(lead_cap, float(raw)))
            ld = (ld_raw + lead_cap) / (2 * lead_cap) if lead_cap > 0 else 0.5
        else:
            ld = 0.5

        # 3.3 fundshare：用当日全市场成交额占比（若无成交额列，则置 0）
        fs = 0.0
        if isinstance(df_spot, pd.DataFrame) and not df_spot.empty and total_amount > 0:
            # 尝试用“代码/股票代码/证券代码”等字段名匹配
            code_cols = [c for c in ("代码", "股票代码", "code", "symbol") if c in df_spot.columns]
            if code_cols:
                ccol = code_cols[0]
                sub = df_spot[df_spot[ccol].astype(str).str.upper().isin(sym_set)]
                fs_amount = float(pd.to_numeric(sub.get("amount_yuan", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
                fs_raw = fs_amount / total_amount
                fs = min(fs_raw, fund_cap) / fund_cap if fund_cap > 0 else 0.0

        # 3.4 rs & breadth：对成分**抽样**拉 20d 日线，计算均值（控制耗时）
        #     - 为避免过重，这里最多抽样 30 只（可按需调大）
        rs_vals = []
        up_flags = []
        sample_list = list(sym_set)[:30]
        for sym in sample_list:
            try:
                r = fetch_stock_daily_qfq(sym)
                df = r.data
                if not isinstance(df, pd.DataFrame) or df.shape[0] < (rs_win + 1):
                    continue
                # 取最近 rs_win+1 根，计算 N 日涨跌幅与当日涨跌
                d = df.tail(rs_win + 1).reset_index(drop=True)
                p0, p1 = float(d.loc[0, "close"]), float(d.loc[len(d) - 1, "close"])
                if p0 > 0:
                    rs_vals.append((p1 / p0) - 1.0)
                # 当日涨跌
                if len(d) >= 2:
                    up_flags.append(float(d.loc[len(d) - 1, "close"]) >= float(d.loc[len(d) - 2, "close"]))
            except Exception:
                continue

        sec_rs = float(pd.Series(rs_vals, dtype=float).mean()) if rs_vals else 0.0
        breadth = float(pd.Series(up_flags, dtype=float).mean()) if up_flags else 0.0

        score = w_rs * sec_rs + w_bd * breadth + w_ld * ld + w_fs * fs + w_ct * 0.0
        sector_scores.append({
            "name": name, "code": code, "score": score,
            "parts": {"rs": sec_rs, "breadth": breadth, "lead_norm": ld, "fund_norm": fs, "continuity": 0.0}
        })

    # 3.5 排序取 3–5 名行业
    sector_scores = sorted(sector_scores, key=lambda d: d["score"], reverse=True)
    selected_sectors = [d["name"] for i, d in enumerate(sector_scores, start=1) if rank_lo <= i <= rank_hi]
    selected_sector_codes = [d["code"] for i, d in enumerate(sector_scores, start=1) if rank_lo <= i <= rank_hi]

    # 4) 入选行业 → 板内个股 RS_adjusted（价×量×资份额代理），按 rs_adj 排名后严格取 rank 3–5
    #    - 资份额代理：用 “当日 amount / 近10日均额” 的相对强弱（再按行业内分位归一）
    #    - 10日均额下限 + 自适应上限
    selected: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    amounts_all = []
    # 准备 regime 权重
    wmap = (cfg.get("rs_adjusted", {}).get("regime_weights") or {}).get(state, {})
    wp = float(wmap.get("price", 0.55 if state == "OFFENSE" else 0.45))
    wv = float(wmap.get("volume", 0.30))
    wf = float(wmap.get("fund", 0.15))

    soft_up, hard_up, penalty = _calc_soft_hard_upper(amounts_all, state, cfg)
    per_selected_cap = 0

    for sec_code, sec_name in zip(selected_sector_codes, selected_sectors):
        comps = fetch_sw_l1_components(sec_code)
        if not comps:
            continue

        # 拉所有成分的近 60 日（足够覆盖 20D 与 10D 计算）
        rows: List[Dict[str, Any]] = []
        for sym in comps:
            try:
                r = fetch_stock_daily_qfq(sym)
                df = r.data
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                # 价格 RS（20日）
                if df.shape[0] < (rs_win + 1):
                    continue
                dd = df.tail(max(rs_win + 10, 35)).reset_index(drop=True)  # 多留些给均量
                p0, p1 = float(dd.loc[len(dd) - (rs_win + 1), "close"]), float(dd.loc[len(dd) - 1, "close"])
                if p0 <= 0:
                    continue
                rs_price = (p1 / p0) - 1.0
                # 量（3/20）
                vol = pd.to_numeric(dd["amount"], errors="coerce").fillna(0.0)
                sma3 = float(vol.tail(3).mean()) if len(vol) >= 3 else 0.0
                sma20 = float(vol.tail(20).mean()) if len(vol) >= 20 else max(sma3, 1.0)
                rs_volume = 0.0 if sma20 <= 0 else (sma3 / sma20)
                # 资份额代理：当日 amount / 10日均额（行业内分位再归一）
                a10 = float(vol.tail(10).mean()) if len(vol) >= 10 else float(vol.mean())
                a_today = float(vol.iloc[-1]) if len(vol) else 0.0
                fund_ratio = 0.0 if a10 <= 0 else (a_today / a10)
                rows.append({
                    "symbol": sym, "sector": sec_name,
                    "rs_price": rs_price, "rs_volume": rs_volume,
                    "fund_ratio": fund_ratio, "amount_10d": a10,
                })
                amounts_all.append(a10)
            except Exception:
                continue

        if not rows:
            continue

        # 行业内对 fund_ratio 做分位归一 → rs_fund
        fr = pd.Series([x["fund_ratio"] for x in rows], dtype=float)
        q = fr.rank(pct=True, method="average")  # 0~1
        for i, x in enumerate(rows):
            x["rs_fund"] = float(q.iloc[i])

        # 计算 rs_adj
        for x in rows:
            x["rs_adj"] = wp * float(x["rs_price"]) + wv * float(x["rs_volume"]) + wf * float(x["rs_fund"])

        # 行业内按 rs_adj 排序，赋 rank（1..N），严格筛第 3–5 名
        rows_sorted = sorted(rows, key=lambda d: d["rs_adj"], reverse=True)
        for rk, x in enumerate(rows_sorted, start=1):
            x["rank_in_sector"] = rk

        # 应用筛选规则（rank、RS 下限、10D 均额上下限、自适应惩罚、EV/RR/Pwin 软约束）
        for x in rows_sorted:
            sym = x["symbol"]; sec = x["sector"]; rnk = int(x["rank_in_sector"])
            rs_adj = float(x["rs_adj"]); a10 = float(x["amount_10d"])
            # rank 越界
            if not (rlo <= rnk <= rhi):
                rejected.append({"symbol": sym, "sector": sec, "reason_reject": "rank_out_of_range",
                                 "rank_in_sector": rnk, "rs_adj": rs_adj})
                continue
            # rs 下限
            if rs_adj < rs_min:
                rejected.append({"symbol": sym, "sector": sec, "reason_reject": "rs_below_min",
                                 "rs_adj": rs_adj, "rs_min": rs_min})
                continue
            # 金额下限
            if a10 < float(cfg.get("amount_10d_min", 5e8)):
                rejected.append({"symbol": sym, "sector": sec, "reason_reject": "amount_10d_below_min",
                                 "amount_10d": a10})
                continue
            # 上限与惩罚
            oversize = False
            penalty_applied = 1.0
            if a10 > hard_up:
                rejected.append({"symbol": sym, "sector": sec, "reason_reject": "amount_10d_above_max",
                                 "amount_10d": a10, "hard_upper": hard_up})
                continue
            elif a10 > soft_up:
                oversize = True
                penalty_applied = penalty
                rs_adj *= penalty

            # 配额
            if sum(1 for it in selected if it["sector"] == sec) >= per_sector_cap:
                rejected.append({"symbol": sym, "sector": sec, "reason_reject": "sector_quota_exceeded"})
                continue
            if len(selected) >= per_window_cap:
                rejected.append({"symbol": sym, "sector": sec, "reason_reject": "window_quota_exceeded"})
                break

            selected.append({
                "symbol": sym, "sector": sec, "rank_in_sector": rnk,
                "rs_price": x["rs_price"], "rs_volume": x["rs_volume"], "rs_fund": x["rs_fund"],
                "rs_adj": rs_adj, "penalty": penalty_applied, "oversize": oversize,
                "amount_10d": a10, "rr": None, "pwin": None, "ev_bps": None,  # 软约束字段（此步不强行拉）
            })

    # 5) 快照 + Signal 入库
    ev_id = 0
    if emit_event:
        sector_snapshot = {"scores": sector_scores, "rank_range": [rank_lo, rank_hi]}
        ev_id = _funnel_emit_snapshot(sector_snapshot, selected, rejected, window)

    with get_session() as s:
        for it in selected:
            s.add(Signal(
                symbol=it["symbol"], side="buy", price_ref=None,
                state_at_emit=SystemState(state) if state in SystemState.__members__ else None,
                emotion_gate_passed=None, executable=bool(can_exec and (mode == "normal")),
                reason_reject=None, window=window,
                microstructure_checks=json.dumps({
                    "rs_price": it["rs_price"], "rs_volume": it["rs_volume"], "rs_fund": it["rs_fund"],
                    "rs_adj": it["rs_adj"], "amount_10d": it["amount_10d"],
                    "oversize": it.get("oversize", False), "penalty": it.get("penalty", 1.0),
                    "ctx_event_id": ev_id,
                }, ensure_ascii=False)
            ))
        for it in rejected:
            s.add(Signal(
                symbol=it.get("symbol", "?"), side="buy", price_ref=None,
                state_at_emit=SystemState(state) if state in SystemState.__members__ else None,
                emotion_gate_passed=None, executable=False,
                reason_reject=it.get("reason_reject", ""), window=window,
                microstructure_checks=json.dumps({k: v for k, v in it.items() if k not in ("symbol", "reason_reject")},
                                                 ensure_ascii=False)
            ))
        s.commit()

    return {
        "selected": selected,
        "rejected": rejected,
        "executable": can_exec and (mode == "normal"),
        "event_id": ev_id,
        "selected_sectors": selected_sectors,
    }


# ======================
# 交易日历：AkShare 主来源 + 本地缓存 + 回退
# ======================

_TRADING_DATES: Set[date] = set()
_CALENDAR_SOURCE: str = "unknown"           # akshare | cache | weekday_fallback | unavailable | unknown
_CALENDAR_FETCHED_AT: Optional[str] = None  # ISO 时间
_CALENDAR_DEGRADED: bool = False            # 降级模式（供策略层判断是否禁止开新仓）
_CALENDAR_STALENESS_DAYS: int = 0           # 缓存距离“今天”的陈旧天数

def _parse_trade_date(v: Any) -> Optional[date]:
    """把 'YYYY-MM-DD' 或 'YYYYMMDD' 解析为 date。解析失败返回 None。"""
    if v is None:
        return None
    s = str(v).strip()
    try:
        if "-" in s:
            return datetime.fromisoformat(s).date()
        if len(s) == 8 and s.isdigit():
            return date(int(s[0:4]), int(s[4:6]), int(s[6:8]))
    except Exception:
        return None
    return None

def _load_calendar_cache() -> Tuple[Set[date], Optional[str]]:
    if not _CALENDAR_CACHE.exists():
        return set(), None
    try:
        data = json.loads(_CALENDAR_CACHE.read_text(encoding="utf-8"))
        dates = {datetime.fromisoformat(d).date() for d in data.get("dates", [])}
        fetched_at = data.get("fetched_at")
        return dates, fetched_at
    except Exception as e:
        _log_event("ERROR", "calendar_cache_broken", {"err": str(e)})
        return set(), None

def _save_calendar_cache(dates: Set[date], source: str) -> None:
    payload = {
        "fetched_at": now_cn().isoformat(),
        "source": source,
        "dates": [d.isoformat() for d in sorted(dates)],
    }
    _CALENDAR_CACHE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

def refresh_trading_calendar() -> None:
    """
    刷新交易日历：
    1) 优先从 AkShare 获取完整交易日列表；
    2) 失败则加载本地缓存；
    3) 无缓存时按策略回退：fail_open → weekday；fail_closed → 全 False。
    """
    global _TRADING_DATES, _CALENDAR_SOURCE, _CALENDAR_FETCHED_AT, _CALENDAR_DEGRADED, _CALENDAR_STALENESS_DAYS

    try:
        import akshare as ak  # 延迟导入，避免环境未装时阻塞其它功能
        df = ak.tool_trade_date_hist_sina()
        if "trade_date" not in df.columns:
            raise RuntimeError("trade_date column missing")

        dates: Set[date] = set()
        for v in df["trade_date"].tolist():
            d = _parse_trade_date(v)
            if d:
                dates.add(d)

        if not dates:
            raise RuntimeError("empty trading dates from akshare")

        _TRADING_DATES = dates
        _CALENDAR_SOURCE = "akshare"
        _CALENDAR_FETCHED_AT = now_cn().isoformat()
        _CALENDAR_DEGRADED = False
        _CALENDAR_STALENESS_DAYS = 0

        _save_calendar_cache(_TRADING_DATES, _CALENDAR_SOURCE)
        _log_event("INFO", "calendar_refresh_ok", {
            "source": "akshare",
            "count": len(_TRADING_DATES),
            "fetched_at": _CALENDAR_FETCHED_AT,
            "degraded": _CALENDAR_DEGRADED,
        })
        return

    except Exception as e:
        # 远端失败：尝试本地缓存
        cache_dates, fetched_at = _load_calendar_cache()
        if cache_dates:
            _TRADING_DATES = cache_dates
            _CALENDAR_SOURCE = "cache"
            _CALENDAR_FETCHED_AT = fetched_at
            try:
                _CALENDAR_STALENESS_DAYS = max((now_cn().date() - max(cache_dates)).days, 0)
            except Exception:
                _CALENDAR_STALENESS_DAYS = 9999
            _CALENDAR_DEGRADED = _CALENDAR_STALENESS_DAYS > _CACHE_STALE_DAYS

            _log_event("WARN", "calendar_use_cache", {
                "reason": str(e),
                "count": len(cache_dates),
                "cached_at": fetched_at,
                "staleness_days": _CALENDAR_STALENESS_DAYS,
                "stale_threshold": _CACHE_STALE_DAYS,
                "degraded": _CALENDAR_DEGRADED,
            })
            return

        # 无缓存：按策略回退
        if _FAIL_POLICY == "fail_open":
            _TRADING_DATES = set()
            _CALENDAR_SOURCE = "weekday_fallback"
            _CALENDAR_FETCHED_AT = None
            _CALENDAR_DEGRADED = True
            _CALENDAR_STALENESS_DAYS = 9999
            _log_event("ERROR", "calendar_fallback_weekday", {
                "reason": str(e), "policy": _FAIL_POLICY, "degraded": True
            })
            return
        else:
            _TRADING_DATES = set()
            _CALENDAR_SOURCE = "unavailable"
            _CALENDAR_FETCHED_AT = None
            _CALENDAR_DEGRADED = True
            _CALENDAR_STALENESS_DAYS = 9999
            _log_event("ERROR", "calendar_unavailable", {
                "reason": str(e), "policy": _FAIL_POLICY, "degraded": True, "note": "treat as non-trading day"
            })
            return

def is_trading_day(d: Optional[date] = None) -> bool:
    """
    判定优先级：
    1) akshare / cache ：集合判定
    2) weekday_fallback：仅根据 weekday（周一至周五）
    3) unavailable：一律 False（fail_closed）
    """
    d = d or now_cn().date()

    if _CALENDAR_SOURCE in ("akshare", "cache"):
        return d in _TRADING_DATES

    if _CALENDAR_SOURCE == "weekday_fallback":
        return d.weekday() < 5

    return False  # unavailable

def is_in_trading_window(now: Optional[datetime] = None) -> bool:
    now = now or now_cn()
    return now.strftime("%H:%M") in TRADING_WINDOWS

# ===== 新增：上一交易日推断（结合交易日历 / 工作日回退） =====

def _get_prev_trade_yyyymmdd(ref: Optional[date] = None) -> str:
    """推断上一交易日（YYYYMMDD）。优先用 _TRADING_DATES；无则按工作日回退。"""
    d = ref or now_cn().date()
    # 优先：已知交易日集合
    if _CALENDAR_SOURCE in ("akshare", "cache") and _TRADING_DATES:
        prev = max((x for x in _TRADING_DATES if x < d), default=None)
        if prev:
            return prev.strftime("%Y%m%d")
    # 回退：往前找最近的工作日
    t = d - timedelta(days=1)
    for _ in range(10):
        if t.weekday() < 5:  # 周一~周五
            return t.strftime("%Y%m%d")
        t -= timedelta(days=1)
    return (d - timedelta(days=1)).strftime("%Y%m%d")

# ======================
# 状态恢复（从 StateSnapshot 读取最近状态）
# ======================

_CURRENT_STATE: Optional[SystemState] = None

def restore_state_from_snapshot() -> Optional[SystemState]:
    """使用 sqlmodel.select + 基础错误处理；读取最近快照作为进程内当前状态。"""
    global _CURRENT_STATE
    try:
        with get_session() as s:
            latest = s.exec(select(StateSnapshot).order_by(StateSnapshot.ts.desc())).first()
            if latest:
                _CURRENT_STATE = latest.state
    except Exception as e:
        print(f"[scheduler] Failed to restore state: {e}")
    return _CURRENT_STATE

# ======================
# 状态机（步骤4）：迟滞带 + 完整约束
# ======================

def decide_state(emotion_pass: bool, position_ratio: float, ks_level: int, current_state: str, *, 
                 enter_offense: float, exit_offense: float, observe_threshold: float) -> Dict[str, Any]:
    """
    - sleep：ks_level==2 强制休眠
    - 情绪通过：迟滞带 55%/65%（enter_offense/exit_offense）
    - 情绪未通过：仅当仓位<=observe_threshold 时进入观望，否则保持持仓
    返回：{"state": "OFFENSE|HOLD|WATCH|SLEEP", "reason": "..."}
    """
    # 1) Kill-Switch L2：强制休眠
    if ks_level == 2:
        return {"state": "SLEEP", "reason": "ks_L2"}

    # 标准化当前状态名
    cs = (current_state or "").upper()
    if cs not in {"OFFENSE","HOLD","WATCH","SLEEP"}:
        cs = "HOLD"  # 合理默认

    # 2) 情绪通过 → 迟滞带
    if emotion_pass:
        # 进入进攻：当前为 watch/offense，且仓位 < enter_offense
        if cs in {"WATCH","OFFENSE"} and position_ratio < enter_offense:
            return {"state": "OFFENSE", "reason": f"emotion_pass & pos<{enter_offense:.2f}"}
        # 进入持仓：当前为 offense/hold，且仓位 >= exit_offense
        if cs in {"OFFENSE","HOLD"} and position_ratio >= exit_offense:
            return {"state": "HOLD", "reason": f"emotion_pass & pos>={exit_offense:.2f}"}
        # 其余保持原态，避免 58–62% 抖动
        return {"state": cs, "reason": "emotion_pass & hysteresis_hold"}

    # 3) 情绪未通过 → 观望约束
    if position_ratio <= observe_threshold:
        return {"state": "WATCH", "reason": f"emotion_fail & pos<={observe_threshold:.2f}"}
    else:
        return {"state": "HOLD", "reason": f"emotion_fail & pos>{observe_threshold:.2f} keep_hold"}

# ======================
# 定时任务
# ======================

def run_trading_task() -> None:
    """主任务：仅在交易日 + 10:00/14:00 窗口执行，否则落 skip 事件。"""
    now = now_cn()
    trading = is_trading_day(now.date())
    in_window = is_in_trading_window(now)

    payload = {
        "now": now.isoformat(),
        "tz": _TZ_NAME,
        "trading_day": trading,
        "in_window": in_window,
        "windows": TRADING_WINDOWS,
        "calendar_source": _CALENDAR_SOURCE,
        "calendar_cached_at": _CALENDAR_FETCHED_AT,
        "calendar_degraded": _CALENDAR_DEGRADED,
        "calendar_staleness_days": _CALENDAR_STALENESS_DAYS,
        "calendar_fail_policy": _FAIL_POLICY,
    }

    if trading and in_window:
        _log_event("INFO", "scheduler_tick", {**payload, "action": "EXECUTE"})

        # ==== 新增：在窗口内触发情绪闸门 ====
        try:
            # 透传日历上下文，满足事件契约的 calendar 字段
            calendar_ctx = {
                "source": _CALENDAR_SOURCE,
                "cached_at": _CALENDAR_FETCHED_AT,
                "fail_policy": _FAIL_POLICY,
                "degraded": _CALENDAR_DEGRADED,
                "staleness_days": _CALENDAR_STALENESS_DAYS,
            }
            prev_yyyymmdd = _get_prev_trade_yyyymmdd(now.date())

            # 为避免顶部循环依赖，这里局部导入
            from strategy_engine.gates.emotion_gate import check_emotion_gate
            gate_payload = check_emotion_gate(
                prev_trade_yyyymmdd=prev_yyyymmdd,
                window=now.strftime("%H:%M"),
                config_dir="config",
                emit_event=True,          # 由 gate 内部写 altflow/gate_* 事件
                calendar=calendar_ctx,    # 顶层 calendar 透传
            )
            # 记录一次聚合结束事件（可选）
            _log_event("INFO", "altflow_gate_done", {
                "window": now.strftime("%H:%M"),
                "prev_trade": prev_yyyymmdd,
                "passed": gate_payload.get("passed"),
                "reasons": gate_payload.get("reasons"),
            })
        except Exception as e:
            _log_event("ERROR", "altflow_gate_error", {
                "window": now.strftime("%H:%M"),
                "err": str(e),
                "calendar_source": _CALENDAR_SOURCE,
                "calendar_degraded": _CALENDAR_DEGRADED,
                "calendar_staleness_days": _CALENDAR_STALENESS_DAYS,
            })
        # ==== 新增结束 ====

        # ==== 新增：Gate→Kill-Switch 串联（复用 gate 指标；不重复取数）====
        try:
            from strategy_engine.killswitch.killswitch import check_kill_switch
            ks_payload = check_kill_switch(
                prev_trade_yyyymmdd=prev_yyyymmdd,
                window=now.strftime("%H:%M"),
                indicators_payload=gate_payload.get("indicators") if isinstance(gate_payload, dict) else None,
                config_dir="config",
                emit_event=True,   # 由 killswitch 内部写 altflow/ks_L{1,2}/ks_L0 事件
            )
            _log_event("INFO", "altflow_ks_done", {
                "window": now.strftime("%H:%M"),
                "prev_trade": prev_yyyymmdd,
                "level": ks_payload.get("level"),
                "triggered": ks_payload.get("triggered_conditions"),
            })
        except Exception as e:
            _log_event("ERROR", "altflow_ks_error", {
                "window": now.strftime("%H:%M"),
                "err": str(e),
            })
        # ==== 新增结束 ====
        
        # ==== 新增：Step5 · 漏斗（选板→选股）====
        try:
            # 读取当前系统状态（Step4 产出）；若你在上文已有 decide_state_and_persist，可直接读取返回
            with get_session() as s:
                ss = s.exec(select(StateSnapshot).order_by(StateSnapshot.ts.desc())).first()  # 最近一条
                curr_state = (ss.state.value if ss and hasattr(ss, "state") else "WATCH")

            # 离线/在线：如果上游测试注入了指标，也允许测试脚本通过 env 文件落地到 JSON 注入
            injected = None  # 在线模式默认无注入；tests 会在离线模式传入

            funnel_payload = select_candidates_funnel(
                state=curr_state,
                ks_level=int(ks_payload.get("level", 0)) if isinstance(ks_payload, dict) else 0,
                window=now.strftime("%H:%M"),
                indicators_payload=gate_payload.get("indicators") if isinstance(gate_payload, dict) else None,
                injected=injected,
                emit_event=True,
            )
            _log_event("INFO", "funnel_done", {
                "window": now.strftime("%H:%M"),
                "selected_count": len(funnel_payload.get("selected", [])),
                "executable": funnel_payload.get("executable", False),
                "state": curr_state,
                "ks_level": int(ks_payload.get("level", 0)) if isinstance(ks_payload, dict) else 0,
            })
        except Exception as e:
            _log_event("ERROR", "funnel_error", {"window": now.strftime("%H:%M"), "err": str(e)})
        # ==== 新增结束 ====

        # ==== 新增：KS（Step3）→ 状态机（Step4）串联 ====
        try:
            # 1) emotion_pass / ks_level
            emotion_pass = bool(gate_payload.get("passed")) if isinstance(gate_payload, dict) else False
            ks_level = int(ks_payload.get("level")) if isinstance(ks_payload, dict) else 0

            # 2) 读取账户最新仓位比（Account 最新快照）
            from sqlmodel import select
            from models.database import get_session, Account, StateSnapshot, SystemState

            pos_ratio = 0.0
            pr_source = None
            with get_session() as s:
                acc = s.exec(select(Account).order_by(Account.ts.desc())).first()
            if acc is not None:
                if getattr(acc, "position_ratio", None) is not None:
                    pos_ratio = float(acc.position_ratio)
                    pr_source = "field:position_ratio"
                else:
                    tv = getattr(acc, "total_market_value", None)
                    ta = getattr(acc, "total_asset", None)
                    if tv is not None and ta is not None:
                        ta_val = float(ta)
                        pos_ratio = (float(tv) / ta_val) if ta_val > 0 else 0.0
                        pr_source = "calc:tmv/ta"
            if pr_source is None:
                _log_event("WARN", "state/pos_ratio_fallback", {
                    "window": now.strftime("%H:%M"),
                    "reason": "missing Account.latest; default=0.0",
                })

            # 3) 当前状态（无则默认 HOLD 更稳）
            cur_snap = restore_state_from_snapshot()
            cur_name = cur_snap.name if cur_snap else "HOLD"

            # 4) 读取迟滞与观望阈值（只读配置，不改键名）
            from config.loader import load_config
            cfg = load_config("config")
            st = (cfg.get("state") or {})
            pos_band = (st.get("pos_band") or {})
            enter_off = float(pos_band.get("enter_offense", 0.55))
            exit_off  = float(pos_band.get("exit_offense", 0.65))
            observe_th = float(st.get("observe_position_threshold", 0.30))  # ← 你已同意新增

            # 5) 决策（decide_state 已在本文件中实现）
            dec = decide_state(
                emotion_pass=emotion_pass,
                position_ratio=pos_ratio,
                ks_level=ks_level,
                current_state=cur_name,
                enter_offense=enter_off,
                exit_offense=exit_off,
                observe_threshold=observe_th,
            )
            new_name = dec["state"]

            # 6) 落库（仅在状态变化时）
            if new_name != cur_name:
                _log_event("INFO", "state/change", {
                    "from_state": cur_name,
                    "to_state": new_name,
                    "trigger_reason": dec.get("reason"),
                    "emotion_pass": emotion_pass,
                    "position_ratio": pos_ratio,
                    "ks_level": ks_level,
                    "thresholds": {
                        "enter_offense": enter_off,
                        "exit_offense": exit_off,
                        "observe_threshold": observe_th,
                    },
                })
                with get_session() as s:
                    s.add(StateSnapshot(state=SystemState[new_name], pos_ratio=pos_ratio, note=dec.get("reason")))
                    s.commit()
        except Exception as e:
            _log_event("ERROR", "state/compute_error", {
                "window": now.strftime("%H:%M"),
                "err": str(e),
            })
        # ==== 新增结束 ====

    else:
        _log_event("INFO", "scheduler_skip", {**payload, "action": "SKIP"})


def run_data_preheat_task() -> None:
    """09:00 预热：刷新交易日历（未来可扩展行情缓存等）。"""
    now = now_cn()
    refresh_trading_calendar()
    _log_event("INFO", "data_preheat", {
        "now": now.isoformat(),
        "tz": _TZ_NAME,
        "calendar_source": _CALENDAR_SOURCE,
        "calendar_cached_at": _CALENDAR_FETCHED_AT,
        "degraded": _CALENDAR_DEGRADED,
        "staleness_days": _CALENDAR_STALENESS_DAYS,
    })

# ==== 新增：09:30 休眠态重启资格检查 ====
def _job_restart_check() -> None:
    """
    每个交易日 09:30 检查是否满足“可重启”条件：
    A) 最近连续 N 日闸门通过；B) 月度回撤 ≥ 阈值（可选）
    满足时落 restart/eligible；若配置 manual_confirm=false，则自动确认并尝试 exit_sleep_mode。
    """
    try:
        rc = (_CFG.get("restart_conditions") or {})
        consecutive_gate_days = int(rc.get("consecutive_gate_days", 3))
        monthly_dd = rc.get("monthly_drawdown_threshold", -0.03)
        manual_confirm = bool(rc.get("manual_confirm", True))

        # 局部导入，避免顶层循环依赖
        from strategy_engine.killswitch.killswitch import check_restart_eligibility
        res = check_restart_eligibility(consecutive_gate_days, monthly_dd, manual_confirm)
        _log_event("INFO", "restart_check_done", {"result": res})
    except Exception as e:
        _log_event("ERROR", "restart_check_error", {"err": str(e)})
# ==== 新增结束 ====

# ======================
# 调度器创建 / 启停
# ======================

def _cron_hours_from_windows(windows: Iterable[str]) -> str:
    hours = sorted({int(w.split(":")[0]) for w in windows})
    return ",".join(str(h) for h in hours)

def create_scheduler() -> BackgroundScheduler:
    """创建并配置调度器（10:00 / 14:00；可选 09:00 预热）。"""
    # JobStore：优先 SQLAlchemy，失败回退内存
    jobstores = {}
    try:
        jobstores = {"default": SQLAlchemyJobStore(url=_JOBSTORE_URL)}
        jobstore_kind = "sqlalchemy"
    except Exception:
        jobstores = {}
        jobstore_kind = "memory"

    kwargs = dict(
        timezone=SCHEDULER_TZ,
        jobstores=jobstores,
        job_defaults={"coalesce": _COALESCE, "max_instances": _MAX_INST, "misfire_grace_time": _MISFIRE},
    )
    if _EXEC_WORKERS > 0:
        kwargs["executors"] = {"default": ThreadPoolExecutor(max_workers=_EXEC_WORKERS)}

    scheduler = BackgroundScheduler(**kwargs)

    hours_expr = _cron_hours_from_windows(TRADING_WINDOWS)
    scheduler.add_job(
        run_trading_task,
        "cron",
        id="main_trading_job",
        day_of_week="mon-fri",
        hour=hours_expr,
        minute=0,
        replace_existing=True,
    )

    if _ENABLE_PREHEAT:
        scheduler.add_job(
            run_data_preheat_task,
            "cron",
            id="data_preheat_job",
            day_of_week="mon-fri",
            hour=9,
            minute=0,
            replace_existing=True,
        )

    # ==== 新增：09:30 重启资格检查 ====
    scheduler.add_job(
        _job_restart_check,
        "cron",
        id="restart_check_0930",
        day_of_week="mon-fri",
        hour=9,
        minute=30,
        replace_existing=True,
    )
    # ==== 新增结束 ====

    return scheduler

def start_scheduler(auto_create_tables: bool = True) -> BackgroundScheduler:
    """启动调度器：建表→恢复状态→刷新日历→启动 APScheduler→落启动事件。"""
    if auto_create_tables:
        create_db_and_tables()

    restore_state_from_snapshot()
    refresh_trading_calendar()

    sched = create_scheduler()
    sched.start()

    _log_event("INFO", "scheduler_start", {
        "tz": _TZ_NAME,
        "windows": TRADING_WINDOWS,
        "coalesce": _COALESCE,
        "misfire_grace_time": _MISFIRE,
        "max_instances": _MAX_INST,
        "jobstore_url": _JOBSTORE_URL,
        "executors": {"threadpool_workers": _EXEC_WORKERS} if _EXEC_WORKERS > 0 else {"threadpool_workers": 0},
        "preheat_enabled": _ENABLE_PREHEAT,
        "calendar_source": _CALENDAR_SOURCE,
        "calendar_cached_at": _CALENDAR_FETCHED_AT,
        "calendar_fail_policy": _FAIL_POLICY,
        "calendar_degraded": _CALENDAR_DEGRADED,
        "calendar_staleness_days": _CALENDAR_STALENESS_DAYS,
        "cache_stale_threshold": _CACHE_STALE_DAYS,
    })
    return sched

def shutdown_scheduler(scheduler: Optional[BackgroundScheduler]) -> None:
    if scheduler and scheduler.running:
        scheduler.shutdown(wait=False)
        _log_event("INFO", "scheduler_stop", {"tz": _TZ_NAME})

# ============== CLI 便捷启动（可选） ==============
if __name__ == "__main__":
    s = start_scheduler(auto_create_tables=True)
    print(
        f"[Scheduler] started TZ={_TZ_NAME}, windows={TRADING_WINDOWS}, "
        f"jobstore={_JOBSTORE_URL}, calendar_source={_CALENDAR_SOURCE}, "
        f"fail_policy={_FAIL_POLICY}"
    )
