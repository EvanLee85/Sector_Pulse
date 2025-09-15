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
from datetime import date, datetime
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
)

# ======================
# 配置与常量
# ======================

_CFG = load_config("config")

# 注意：本项目**所有业务/数据库时间**使用 ZoneInfo(_TZ_NAME)（见 now_cn()）。
# APScheduler 历史原因推荐 pytz，因此仅在构造调度器 timezone= 时使用 pytz.timezone(_TZ_NAME)。
# 其它任何地方请调用 now_cn()，不要直接用 datetime.now() 或 pytz 的 now()。
# —— 时区：仅 execution.tz（或环境变量）——
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
        # TODO（后续里程碑）：情绪闸门→状态机→信号/意向→风控
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

    # 启动事件留到 start_scheduler 里统一发送
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
