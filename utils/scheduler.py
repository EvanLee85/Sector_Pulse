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
from typing import Any, Dict, Iterable, Optional, Set, Tuple

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
)

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
