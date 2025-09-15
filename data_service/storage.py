# data_service/storage.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import random
import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Hashable, Mapping, Optional, Tuple
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


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
