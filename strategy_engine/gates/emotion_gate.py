# strategy_engine/gates/emotion_gate.py
# -*- coding: utf-8 -*-
"""
Alt-Flow · 情绪闸门聚合（Step2 · 4）
- 仅消费第 3 步的指标结果（不触发任何外部行情调用）
- 阈值来源：config/strategy_core.yaml -> emotion_gate.*
- 结果产出（最小冻结契约）：
    {
      passed: bool,
      reasons: list[str],
      indicators: IndicatorsPayload,  # 步骤3输出的最小契约（schema_version/calc_at/intraday_missing/metrics/...）
      thresholds: { ... 快照 ... },
      window: "10:00" | "14:00",
      tz: "Asia/Shanghai",
      version: "step2-contract-v1.1",
      calc_at: datetime(tz=+08:00),
      sources: { ... },
      calendar: { ... }               # ← 新增：可选透传调度日历上下文
    }
- 事件：
    - 通过：  altflow/gate_pass
    - 拒绝：  altflow/gate_reject
  （优先使用 data_service.storage 的事件写入函数；若缺失则退化写 events/altflow_events.jsonl）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable
from pathlib import Path
from datetime import datetime
import json

from zoneinfo import ZoneInfo

# —— 项目内依赖 —— #
from config.loader import load_config                      # 与现有代码一致
from data_service.proxies.altflow_proxy import run_indicator_calculator

# 优先使用 storage 的时间函数；缺失则本地生成
try:
    from data_service.storage import now_shanghai  # type: ignore
except Exception:
    def now_shanghai() -> datetime:
        return datetime.now(ZoneInfo("Asia/Shanghai"))

# 事件写入候选函数名（按存在顺序适配）
_EVENT_WRITERS_CANDIDATES: Tuple[str, ...] = (
    "append_event_jsonl",
    "write_event",
    "log_event",
    "append_event",
)

try:
    import data_service.storage as storage  # type: ignore
except Exception:
    storage = None  # fallback to local jsonl


_SH_TZ = ZoneInfo("Asia/Shanghai")
_VERSION = "step2-contract-v1.1"


@dataclass
class GateThresholds:
    limit_up_count_min: int
    limitup_down_ratio_min: Optional[float] = None     # [0,1], 可选
    turnover_concentration_max: Optional[float] = None # [0,1], 可选
    vol_percentile_max: float = 0.85                   # [0,1]
    ma20_required: bool = True


def _load_thresholds(cfg: Dict[str, Any]) -> GateThresholds:
    eg = cfg.get("emotion_gate", {}) or {}
    return GateThresholds(
        limit_up_count_min=int(eg.get("limit_up_count_min", 50)),
        limitup_down_ratio_min=eg.get("limitup_down_ratio_min", 0.5),
        turnover_concentration_max=eg.get("turnover_concentration_max", 0.40),
        vol_percentile_max=float(eg.get("vol_percentile_max", 0.85)),
        ma20_required=bool(eg.get("ma20_required", True)),
    )


def _event_writer() -> Optional[Callable[[str, Dict[str, Any]], None]]:
    """
    返回一个事件写入函数 writer(event_code, payload)。
    优先使用 data_service.storage 中已存在的函数；否则回退到本地 JSONL。
    """
    # 优先：项目内封装
    if storage is not None:
        for name in _EVENT_WRITERS_CANDIDATES:
            if hasattr(storage, name):
                fn = getattr(storage, name)
                def _wrap(ev: str, payload: Dict[str, Any], _fn=fn):
                    try:
                        _fn(ev, payload)
                    except TypeError:
                        # 适配不同签名：某些实现可能需要 (event_code=..., payload=...)
                        _fn(event_code=ev, payload=payload)
                return _wrap

    # 退化：写本地 JSONL
    events_dir = Path("events")
    events_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = events_dir / "altflow_events.jsonl"

    def fallback_writer(ev: str, payload: Dict[str, Any]) -> None:
        rec = {"event": ev, "payload": payload, "at": now_shanghai().isoformat()}
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return fallback_writer


def _fmt_num(x: Any, nd: int = 4) -> str:
    try:
        v = float(x)
        return f"{v:.{nd}f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x)


def _build_thresholds_snapshot(t: GateThresholds) -> Dict[str, Any]:
    return {
        "emotion_gate.limit_up_count_min": t.limit_up_count_min,
        "emotion_gate.limitup_down_ratio_min": t.limitup_down_ratio_min,
        "emotion_gate.turnover_concentration_max": t.turnover_concentration_max,
        "emotion_gate.vol_percentile_max": t.vol_percentile_max,
        "emotion_gate.ma20_required": t.ma20_required,
    }


def _judge(indicators: Dict[str, Any], th: GateThresholds) -> List[str]:
    """
    返回未通过原因（空列表表示通过）
    仅检查 P0“盘中关键项 + 结构/波动”，两融项不参与是否放行。
    """
    m = indicators.get("metrics", {}) if "metrics" in indicators else indicators
    reasons: List[str] = []

    # 盘中缺失直接拒绝
    if indicators.get("intraday_missing", False):
        reasons.append("intraday_missing")
        return reasons

    lu = m.get("limit_up_count")
    if lu is None or int(lu) < th.limit_up_count_min:
        reasons.append(f"limit_up_count: {lu} vs {th.limit_up_count_min} (>=")  # 与示例保持风格

    ldr_min = th.limitup_down_ratio_min
    ldr = m.get("limitup_down_ratio")
    if ldr_min is not None and ldr is not None and float(ldr) < float(ldr_min):
        reasons.append(f"limitup_down_ratio: {_fmt_num(ldr,2)} vs {ldr_min} (>=)")

    tcm_max = th.turnover_concentration_max
    tcm = m.get("turnover_concentration_top20")
    if tcm_max is not None and tcm is not None and float(tcm) > float(tcm_max):
        reasons.append(f"turnover_concentration_top20: {_fmt_num(tcm,3)} vs {tcm_max} (<=)")

    vpm = m.get("vol_percentile")
    if vpm is not None and float(vpm) > float(th.vol_percentile_max):
        reasons.append(f"vol_percentile: {_fmt_num(vpm,2)} vs {th.vol_percentile_max} (<=)")

    if th.ma20_required:
        ma20_ok = m.get("sh_above_ma20")
        if ma20_ok is not True:
            reasons.append("sh_above_ma20: False vs True (==)")

    return reasons


def check_emotion_gate(
    *,
    prev_trade_yyyymmdd: str,
    window: str = "10:00",
    indicators_payload: Optional[Dict[str, Any]] = None,
    config_dir: str = "config",
    emit_event: bool = True,
    calendar: Optional[Dict[str, Any]] = None,  # ← 新增：允许调度透传日历上下文
) -> Dict[str, Any]:
    """
    情绪闸门主入口
    - prev_trade_yyyymmdd: 上一交易日（YYYYMMDD），用于指标计算（MA20、两融等需要）
    - window: 执行窗口标识（"10:00" | "14:00" ...）
    - indicators_payload: 可选；若已在上游算好，直接传入以避免重复计算
    - config_dir: 配置目录（默认 "config"）
    - emit_event: 是否写事件（可用于测试关闭落库）
    - calendar: 可选；把调度的交易日日历状态放进 payload 顶层 `calendar`
    """
    # 1) 准备指标
    indi = indicators_payload or run_indicator_calculator(prev_trade_yyyymmdd=prev_trade_yyyymmdd)

    # 2) 阈值快照
    cfg = load_config(config_dir=config_dir)
    th = _load_thresholds(cfg)
    thresholds_snap = _build_thresholds_snapshot(th)

    # 3) 判定
    reasons = _judge(indi, th)
    passed = len(reasons) == 0

    # 4) 聚合输出（最小冻结结构）
    payload: Dict[str, Any] = {
        "window": window,
        "tz": "Asia/Shanghai",
        "passed": passed,
        "reasons": reasons,
        "calendar": calendar or {  # 若未提供，则放一个占位上下文（不影响前端）
            "source": "unknown",
            "degraded": indi.get("intraday_missing", False),
            "staleness_days": None,
        },
        # 为对齐验收样例：sources 放在顶层；完整指标塞到 indicators 命名空间
        "sources": indi.get("sources", {}),
        "indicators": {
            "schema_version": indi.get("schema_version", 1),
            "calc_at": indi.get("calc_at", now_shanghai()),
            "intraday_missing": indi.get("intraday_missing", False),
            "metrics": indi.get("metrics", {}),
            "window": indi.get("window", {}),
            "sources": indi.get("sources", {}),
        },
        "thresholds": thresholds_snap,
        "version": _VERSION,
        "calc_at": now_shanghai(),
    }

    # 5) 事件
    event_code = "altflow/gate_pass" if passed else "altflow/gate_reject"
    if emit_event:
        writer = _event_writer()
        if writer:
            try:
                writer(event_code, payload)
            except Exception:
                # 不让异常影响主流程
                pass

    return payload


__all__ = [
    "check_emotion_gate",
]
