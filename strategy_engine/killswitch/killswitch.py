# strategy_engine/killswitch/killswitch.py
# -*- coding: utf-8 -*-
"""
Alt-Flow · Kill-Switch 分级（Step3）
（…原头注释保持不变…）
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable, List
from pathlib import Path
from datetime import datetime
import json

from zoneinfo import ZoneInfo

# —— 项目内依赖 —— #
from config.loader import load_config
from data_service.proxies.altflow_proxy import run_indicator_calculator

try:
    import data_service.storage as storage  # 复用 storage 的事件写入/时间/状态接口（若缺失则自动降级）
except Exception:
    storage = None  # type: ignore

_SH_TZ = ZoneInfo("Asia/Shanghai")
_VERSION = "step3-killswitch-v1.0"


@dataclass
class KSThresholds:
    margin_net_repay_yi_min: float
    limitup_down_ratio_max: Optional[float]
    tconcentration_min: Optional[float]
    index_down_max: Optional[float]
    vol_percentile_min: float
    actions: Dict[str, Any]           # stop_new_orders / position_cap / force_liquidation / sleep_mode


def _event_writer() -> Callable[[str, Dict[str, Any]], None]:
    # （保持不变）
    if storage is not None:
        for name in ("append_event_jsonl", "write_event", "log_event", "append_event"):
            if hasattr(storage, name):
                fn = getattr(storage, name)
                def _wrap(ev: str, payload: Dict[str, Any], _fn=fn):
                    try:
                        _fn(ev, payload)
                    except TypeError:
                        _fn(event_code=ev, payload=payload)
                return _wrap
    from pathlib import Path
    events_dir = Path("events")
    events_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = events_dir / "altflow_events.jsonl"

    def fallback_writer(ev: str, payload: Dict[str, Any]) -> None:
        at = datetime.now(_SH_TZ).isoformat()
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": ev, "payload": payload, "at": at}, ensure_ascii=False) + "\n")
    return fallback_writer


def _normalize_ks_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    维持原有别名归一逻辑，不改名，仅在缺失时填充默认值。
    """
    out: Dict[str, Any] = {}
    ks = cfg.get("kill_switch", {}) or {}
    for lvl in ("L1", "L2"):
        p = dict(ks.get(lvl, {}) or {})
        # —— 历史键名 → 语义键（仅新增，不覆盖已有）——
        if "margin_net_repay_yi_min" not in p and "margin_net_repay_yi" in p:
            p["margin_net_repay_yi_min"] = p["margin_net_repay_yi"]
        if "limitup_down_ratio_max" not in p and "limitup_down_ratio_min" in p:
            p["limitup_down_ratio_max"] = p["limitup_down_ratio_min"]
        if "tconcentration_min" not in p and "turnover_concentration_max" in p:
            p["tconcentration_min"] = p["turnover_concentration_max"]

        # —— 动作映射（兼容 action + temp_total_cap）——
        actions = p.get("actions")
        if not actions:
            act = str(p.get("action", "")).strip()
            if act in ("stop_new_reduce_cap", "stop_new"):
                actions = {"stop_new_orders": True, "position_cap": float(p.get("temp_total_cap", 0.30))}
            elif act in ("liquidate_and_sleep", "liquidate_sleep"):
                actions = {"stop_new_orders": True, "force_liquidation": True, "sleep_mode": True}
        p["actions"] = actions or {}

        # —— 默认值（仅在缺失时新增，不覆盖已有）——
        if "vol_percentile_min" not in p:
            p["vol_percentile_min"] = 0.85 if lvl == "L1" else 0.90
        if "index_down_max" not in p:
            p["index_down_max"] = -0.5 if lvl == "L1" else -1.5  # 百分数口径（-0.5%、-1.5%）

        out[lvl] = p
    out["restart_conditions"] = cfg.get("restart_conditions", {})
    return out


def _get_metric(m: Dict[str, Any], key: str) -> Optional[float]:
    v = m.get(key)
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


def _resolve_index_ret(metrics: Dict[str, Any]) -> Optional[float]:
    """
    指数当日涨跌幅优先级：
      1) index_intraday_ret | index_ret | hs300_intraday_ret | sh_intraday_ret
      2) 缺失→None（对应复合条件不触发，其它条件不受影响）
    """
    for k in ("index_intraday_ret", "index_ret", "hs300_intraday_ret", "sh_intraday_ret"):
        if k in metrics:
            try:
                return float(metrics[k])
            except Exception:
                pass
    return None


# === 新增：单位自适配（仅内部统一比较量纲；不改动任何配置或指标本身） ===
def _unify_index_threshold_scale(threshold: Optional[float], idx_ret_value: Optional[float]) -> Optional[float]:
    """
    将阈值与指标统一到同一量纲：
    - 若 idx_ret 为“百分数口径”（abs≥1），而阈值是“小数口径”（abs<0.2），则阈值×100
    - 若 idx_ret 为“小数口径”（abs<0.2），而阈值是“百分数口径”（abs≥1），则阈值÷100
    其余保持不变。
    """
    if threshold is None or idx_ret_value is None:
        return threshold
    try:
        t = float(threshold)
        r = float(idx_ret_value)
    except Exception:
        return threshold

    if abs(r) >= 1.0 and abs(t) < 0.2:
        return t * 100.0
    if abs(r) < 0.2 and abs(t) >= 1.0:
        return t / 100.0
    return t


def _apply_actions(level: int, actions: Dict[str, Any], window: str) -> List[str]:
    # （保持不变，调用 storage 的幂等状态接口）
    notes: List[str] = []
    if "position_cap" in actions and hasattr(storage, "set_position_cap"):
        try:
            storage.set_position_cap(float(actions["position_cap"]), note=f"ks_level={level}@{window}")
            notes.append(f"position_cap->{actions['position_cap']}")
        except Exception as e:
            notes.append(f"position_cap_failed:{e}")
    if level == 2 and actions.get("sleep_mode") and hasattr(storage, "enter_sleep_mode"):
        try:
            storage.enter_sleep_mode(note=f"ks_level=2@{window}")
            notes.append("sleep_mode:on")
        except Exception as e:
            notes.append(f"sleep_mode_failed:{e}")
    return notes


def check_kill_switch(
    prev_trade_yyyymmdd: str,
    window: str,
    indicators_payload: Optional[Dict[str, Any]] = None,
    config_dir: str = "config",
    emit_event: bool = True,
) -> Dict[str, Any]:
    cfg = load_config(Path(config_dir))
    ksc = _normalize_ks_config(cfg)
    writer = _event_writer()
    now = datetime.now(_SH_TZ)

    # 若未透传 Step2 输出，兜底计算一次；推荐透传 Gate 指标以避免重复取数
    if indicators_payload is None:
        indicators_payload = run_indicator_calculator(prev_trade_yyyymmdd=prev_trade_yyyymmdd)

    metrics = indicators_payload.get("metrics") if "metrics" in indicators_payload else indicators_payload
    intraday_missing = bool(indicators_payload.get("intraday_missing", False))

    if intraday_missing:
        payload = {
            "level": 0,
            "triggered_conditions": ["intraday_missing"],
            "actions": [],
            "calc_at": now.isoformat(),
            "tz": "Asia/Shanghai",
            "window": window,
            "sources": indicators_payload.get("sources", {}),
            "version": _VERSION,
        }
        if emit_event:
            writer("altflow/ks_L0", payload)
        return payload

    idx_ret = _resolve_index_ret(metrics)

    def _eval(level_key: str) -> Tuple[int, List[str], Dict[str, Any]]:
        p = ksc[level_key]
        trig: List[str] = []

        # 条件 1：两融净偿还（EOD）
        v_m = _get_metric(metrics, "margin_net_repay_yi_prev")
        if v_m is not None and v_m >= float(p.get("margin_net_repay_yi_min", 1e9)):
            trig.append(f"margin_net_repay_yi_prev≥{p['margin_net_repay_yi_min']}")

        # 条件 2：涨跌停比（盘中）—— 低于阈值触发
        v_r = _get_metric(metrics, "limitup_down_ratio")
        if v_r is not None and v_r < float(p.get("limitup_down_ratio_max", -1)):
            trig.append(f"limitup_down_ratio<{p['limitup_down_ratio_max']}")

        # 条件 3：成交额集中度 + 指数当日下跌（复合；指数缺失则不触发该条）
        v_c = _get_metric(metrics, "turnover_concentration_top20")
        if v_c is not None and idx_ret is not None:
            tmin = p.get("tconcentration_min")
            idm_raw = p.get("index_down_max")
            idm = _unify_index_threshold_scale(idm_raw, idx_ret)  # ★ 新增：单位自适配
            if tmin is not None and idm is not None and v_c > float(tmin) and idx_ret <= float(idm):
                trig.append(f"tconcentration>{tmin}+index_ret≤{idm}")

        # 条件 4：波动分位
        v_v = _get_metric(metrics, "vol_percentile")
        if v_v is not None and v_v >= float(p.get("vol_percentile_min", 0.85)):
            trig.append(f"vol_percentile≥{p['vol_percentile_min']}")

        return (2 if level_key == "L2" else 1, trig, p.get("actions", {}))

    # 先判 L2，再判 L1
    lvl, trig, acts = _eval("L2")
    if not trig:
        lvl, trig, acts = _eval("L1")
        if not trig:
            lvl, acts = 0, {}

    payload = {
        "level": lvl,
        "triggered_conditions": trig,
        "actions": [k if isinstance(v, bool) and v else f"{k}:{v}" for k, v in (acts or {}).items()],
        "calc_at": now.isoformat(),
        "tz": "Asia/Shanghai",
        "window": window,
        "sources": indicators_payload.get("sources", {}),
        "version": _VERSION,
        "metrics_snapshot": {
            "margin_net_repay_yi_prev": metrics.get("margin_net_repay_yi_prev"),
            "limitup_down_ratio": metrics.get("limitup_down_ratio"),
            "turnover_concentration_top20": metrics.get("turnover_concentration_top20"),
            "vol_percentile": metrics.get("vol_percentile"),
            "index_intraday_ret": idx_ret,
        },
    }

    if emit_event:
        if lvl == 2:
            writer("altflow/ks_L2", payload)
        elif lvl == 1:
            writer("altflow/ks_L1", payload)
        else:
            writer("altflow/ks_L0", payload)

    if lvl > 0:
        exec_notes = _apply_actions(lvl, acts, window)
        if exec_notes:
            payload["exec_notes"] = exec_notes
    return payload


def check_restart_eligibility(
    consecutive_gate_days: int,
    monthly_drawdown_threshold: Optional[float],
    manual_confirm: bool = True,
) -> Dict[str, Any]:
    # （保持你当前接口与行为不变）
    writer = _event_writer()
    okA = False
    recent_pass = 0
    try:
        if hasattr(storage, "query_events"):
            events = storage.query_events(code_in=("altflow/gate_pass",), limit=max(10, consecutive_gate_days + 5))
            recent_pass = min(consecutive_gate_days, len(events or []))
            okA = recent_pass >= consecutive_gate_days
    except Exception:
        okA = False

    okB = monthly_drawdown_threshold is None
    if monthly_drawdown_threshold is not None:
        try:
            if hasattr(storage, "get_monthly_drawdown"):
                dd = storage.get_monthly_drawdown()
                okB = dd is not None and dd >= float(monthly_drawdown_threshold)
        except Exception:
            okB = False

    eligible = okA and okB
    payload = {
        "eligible": eligible,
        "need_manual_confirm": bool(manual_confirm),
        "gate_days_required": int(consecutive_gate_days),
        "gate_days_recent": int(recent_pass),
        "mdd_threshold": monthly_drawdown_threshold,
        "mdd_ok": okB,
        "checked_at": datetime.now(_SH_TZ).isoformat(),
        "tz": "Asia/Shanghai",
        "version": _VERSION,
    }

    if eligible:
        writer("restart/eligible", payload)
        if not manual_confirm and hasattr(storage, "exit_sleep_mode"):
            try:
                storage.exit_sleep_mode(note="restart_auto_confirm")
            except Exception:
                pass
            writer("restart/confirm", payload)
    return payload
