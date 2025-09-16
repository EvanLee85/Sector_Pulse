# data_service/storage.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import random
import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Hashable, Mapping, Optional, Tuple, List, Sequence
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
import json

# -------- Time / TZ --------
_SH_TZ = ZoneInfo("Asia/Shanghai")


def now_shanghai() -> datetime:
    """返回上海时区当前时间（aware datetime）。"""
    return datetime.now(_SH_TZ)


# -------- Lightweight cache --------
@dataclass
class _CacheEntry:
    value: Any
    expire_at: Optional[float]  # epoch seconds


def _make_cache_key(args: tuple, kwargs: dict) -> Tuple[Hashable, ...]:
    """
    生成可哈希缓存键。约束：
    - 仅在参数可哈希时使用；否则退化为无参键。
    - 我们在本项目中仅对无参函数使用（如 fetch_market_spot），足够稳妥。
    """
    try:
        kw_items = tuple(sorted(kwargs.items()))
        return args + kw_items  # type: ignore
    except Exception:
        # 不可哈希参数，退化
        return tuple()


def cached(ttl: Optional[float] = None, maxsize: int = 64) -> Callable:
    """
    最小缓存装饰器（进程内、非分布式）：
    - ttl: 秒；为 None 则不设过期（直到进程结束或主动失效）
    - maxsize: 简单上限；超过后随机删除一个键（足够应对本项目测试场景）
    """
    store: Dict[Tuple[Hashable, ...], _CacheEntry] = {}

    def _decorator(func: Callable) -> Callable:
        @wraps(func)
        def _wrapped(*args, **kwargs):
            key = _make_cache_key(args, kwargs)
            now = time.time()

            # 命中且未过期
            entry = store.get(key)
            if entry is not None and (entry.expire_at is None or entry.expire_at > now):
                return entry.value

            # 计算并写入
            value = func(*args, **kwargs)
            if len(store) >= maxsize:
                # 简单随机淘汰（足够轻量）
                try:
                    store.pop(random.choice(list(store.keys())))
                except Exception:
                    store.clear()

            expire_at = None if ttl is None else now + float(ttl)
            store[key] = _CacheEntry(value=value, expire_at=expire_at)
            return value

        # 暴露一个简易失效接口（可在测试中手动失效）
        def _invalidate(*args, **kwargs):
            key = _make_cache_key(args, kwargs)
            store.pop(key, None)

        _wrapped.invalidate = _invalidate  # type: ignore[attr-defined]
        return _wrapped

    return _decorator


# -------- Retry wrapper --------
def _sleep_with_jitter(base: float) -> None:
    """
    带抖动的 sleep，避免雪崩：±15% 随机扰动。
    """
    jitter = random.uniform(0.85, 1.15)
    time.sleep(max(0.0, base * jitter))


def call_with_retry(
    api: str,
    func: Callable,
    *args: Any,
    timeout: Optional[float] = None,
    retries: int = 2,
    backoff_seq: Tuple[float, ...] = (0.7, 1.5),
    **kwargs: Any,
) -> Any:
    """
    统一的“调用 + 重试”工具。
    约定：
      - retries 表示“尝试次数”（不是重试次数）。例如 retries=3 → 最多尝试 3 次。
      - backoff_seq：每次失败后的退避基数（秒），长度不足时取最后一个。
      - 若被调函数支持 `timeout` 参数且此处提供了 timeout，则传递下去；不支持则忽略。
      - 失败最终抛出 RuntimeError(f"{api} timeout after {retries} attempts")，并保留原始异常为 __cause__。
    """
    if retries <= 0:
        retries = 1

    # 过滤出 func 支持的参数，避免部分 akshare 接口不接受 timeout 时抛 TypeError
    func_sig = None
    try:
        func_sig = inspect.signature(func)
    except Exception:
        pass

    def _build_kwargs() -> dict:
        call_kwargs = dict(kwargs)
        if timeout is not None and func_sig is not None:
            if "timeout" in func_sig.parameters:
                call_kwargs["timeout"] = timeout
        elif timeout is not None and func_sig is None:
            # 不知道签名，谨慎起见也传入
            call_kwargs["timeout"] = timeout
        return call_kwargs

    last_exc: Optional[BaseException] = None
    attempts = int(retries)

    for i in range(attempts):
        try:
            call_kwargs = _build_kwargs()
            return func(*args, **call_kwargs)
        except BaseException as e:  # 捕获广义异常，统一重试逻辑
            last_exc = e
            if i >= attempts - 1:
                # 终止：包装错误信息，保留原始堆栈
                raise RuntimeError(f"{api} timeout after {attempts} attempts") from e
            # 退避等待
            idx = min(i, len(backoff_seq) - 1)
            _sleep_with_jitter(float(backoff_seq[idx]))

    # 理论不会到这
    if last_exc:
        raise last_exc
    raise RuntimeError(f"{api} failed without exception")


__all__ = [
    "now_shanghai",
    "cached",
    "call_with_retry",
]


# =========================
# 下面为 Step3（Kill-Switch / Restart）所需追加的最小实现
# - 事件写入：优先 DB → 降级 JSONL（events/altflow_events.jsonl）
# - 系统状态：position_cap / sleeping 幂等落地（降级 runtime_state.json）
# - 查询能力：query_events / get_monthly_drawdown（占位）
# =========================

# --- 可选：对接 ORM（若不可用则自动降级） ---
try:
    # 你的 ORM 若未提供下列符号，将自动走 JSONL 降级
    from models.database import get_session  # type: ignore
    from models.database import EventLog    # type: ignore
except Exception:
    get_session = None  # type: ignore
    EventLog = None     # type: ignore

_EVENTS_DIR = Path("events")
_EVENTS_DIR.mkdir(parents=True, exist_ok=True)
_EVENTS_JSONL = _EVENTS_DIR / "altflow_events.jsonl"
_STATE_JSON = Path("runtime_state.json")  # 轻量状态快照（DB 不可用时的降级持久化）


# ---------- 事件写入（优先 DB → 降级 JSONL） ----------
def write_event(event_code: str, payload: Dict[str, Any]) -> None:
    """优先写入 DB 的 EventLog；失败时降级 JSONL。"""
    if get_session and EventLog:
        try:
            with get_session() as s:
                # 这里假设 EventLog 拥有 (level, code, payload_json, created_at) 等字段
                rec = EventLog(level="INFO", code=event_code, payload_json=json.dumps(payload, ensure_ascii=False))
                s.add(rec)
                s.commit()
                return
        except Exception:
            # 回退到 JSONL
            pass
    append_event_jsonl(event_code, payload)


def append_event_jsonl(event_code: str, payload: Dict[str, Any]) -> None:
    """降级 JSONL：每行一条事件，含时间戳。"""
    at = datetime.now(_SH_TZ).isoformat()
    with _EVENTS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": event_code, "payload": payload, "at": at}, ensure_ascii=False) + "\n")


# 为兼容不同调用名，提供别名
def log_event(event_code: str, payload: Dict[str, Any]) -> None:
    write_event(event_code, payload)

def append_event(event_code: str, payload: Dict[str, Any]) -> None:
    write_event(event_code, payload)


# ---------- 轻量系统状态（幂等；优先 DB，降级 JSON） ----------
def _load_state() -> Dict[str, Any]:
    if _STATE_JSON.exists():
        try:
            return json.loads(_STATE_JSON.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_state(state: Dict[str, Any]) -> None:
    _STATE_JSON.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def set_position_cap(cap: float, note: str = "") -> None:
    """设置全局最大仓位上限（0..1），降级写 runtime_state.json。"""
    cap = float(cap)
    state = _load_state()
    state["position_cap"] = cap
    if note:
        state["position_cap_note"] = note
    state["position_cap_at"] = datetime.now(_SH_TZ).isoformat()
    _save_state(state)

def enter_sleep_mode(note: str = "") -> None:
    """进入休眠态（sleeping=True；记录开始时间与备注）。"""
    state = _load_state()
    state["sleeping"] = True
    state["sleep_since"] = datetime.now(_SH_TZ).isoformat()
    if note:
        state["sleep_note"] = note
    _save_state(state)

def exit_sleep_mode(note: str = "") -> None:
    """退出休眠态（恢复活跃；不自动恢复仓位上限，交由上层策略决定）。"""
    state = _load_state()
    state["sleeping"] = False
    state["sleep_cleared_at"] = datetime.now(_SH_TZ).isoformat()
    if note:
        state["sleep_clear_note"] = note
    _save_state(state)


# ---------- 查询: restart 所需的最小能力 ----------
def query_events(code_in: Sequence[str], limit: int = 20) -> List[Dict[str, Any]]:
    """
    返回最近的事件记录（列表），元素至少包含 'event' 与 'payload'（兼容 DB/JSONL）。
    - 若 DB 可用：从 EventLog 取；否则读取 JSONL 的末尾若干行（近似）。
    """
    results: List[Dict[str, Any]] = []
    if get_session and EventLog:
        try:
            with get_session() as s:
                # 这里使用原生 SQL 或 ORM 查询皆可；为兼容性，这里给出保守实现示意
                # 假设 EventLog 有自增 id，可按 id desc 排序
                rows = (
                    s.query(EventLog)  # type: ignore[attr-defined]
                    .filter(EventLog.code.in_(list(code_in)))  # type: ignore[attr-defined]
                    .order_by(EventLog.id.desc())              # type: ignore[attr-defined]
                    .limit(limit)
                    .all()
                )
                for r in rows:
                    try:
                        payload = json.loads(getattr(r, "payload_json", "{}") or "{}")
                    except Exception:
                        payload = {}
                    results.append({
                        "event": getattr(r, "code", ""),
                        "payload": payload,
                        "at": getattr(r, "created_at", None),
                    })
                return results
        except Exception:
            # 回退 JSONL
            pass

    # JSONL 降级读取（粗略尾部 N 行）
    try:
        lines = _EVENTS_JSONL.read_text(encoding="utf-8").splitlines()[-max(50, limit * 3):]
        for line in reversed(lines):
            try:
                rec = json.loads(line)
                if rec.get("event") in code_in:
                    results.append(rec)
                    if len(results) >= limit:
                        break
            except Exception:
                continue
    except Exception:
        pass
    return results


def get_monthly_drawdown() -> Optional[float]:
    """
    占位：返回当月回撤（-0.05 表示 -5%）。
    若未接入真实净值/权益曲线，这里返回 None；上层会据此判定 B 条件是否开启。
    """
    return None


# 导出新增符号
__all__ += [
    "write_event",
    "append_event_jsonl",
    "log_event",
    "append_event",
    "set_position_cap",
    "enter_sleep_mode",
    "exit_sleep_mode",
    "query_events",
    "get_monthly_drawdown",
]
