# tests/11_step2_auto_check.py
# -*- coding: utf-8 -*-
"""
Step2 · Alt-Flow 情绪闸门 一站式自检脚本（最少外部调用：仅一次指标计算）
Checks:
  1) 配置清单（阈值键是否存在且类型正确）
  2) 产出结构：check_emotion_gate 返回的最小冻结结构
  3) 验收规则：通过/拒绝逻辑 & reasons 一致性（用阈值本地复算对比 gate 输出）
  4) 风险/回退：intraday_missing / forced reject / forced pass 注入场景（不触网）
  5) 事件落库：检测 altflow/gate_* 是否写入（JSONL 或 DB EventLog）

用法：
  python -m tests.11_test_step2_auto_check --prev 20250825 --window 10:00 --strict
  python -m tests.11_test_step2_auto_check --noapi --prev 20250911 --window 10:00 --strict
可选：
  --noapi    完全不触网（跳过真实指标计算，只跑离线注入用例）
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

OK, FAIL, WARN = "[OK]", "[FAIL]", "[WARN]"
SH_TZ = ZoneInfo("Asia/Shanghai")


def _p_ok(msg: str) -> None:
    print(f"{OK} {msg}")


def _p_fail(msg: str) -> None:
    print(f"{FAIL} {msg}")


def _p_warn(msg: str) -> None:
    print(f"{WARN} {msg}")


def _tzaware(dt: Any) -> bool:
    return isinstance(dt, datetime) and dt.tzinfo is not None


# ---------------- 配置加载与检查 ----------------

def _load_cfg() -> Dict[str, Any]:
    try:
        from config.loader import load_config
    except Exception:
        # 兼容旧路径
        from config.loader import load_config  # type: ignore
    return load_config(config_dir="config")


@dataclass
class Thresholds:
    limit_up_count_min: int
    limitup_down_ratio_min: float
    turnover_concentration_max: float
    vol_percentile_max: float
    ma20_required: bool


def _cfg_check() -> Tuple[int, Optional[Thresholds]]:
    cfg = _load_cfg()
    eg = cfg.get("emotion_gate", {}) or {}
    missing = [k for k in (
        "limit_up_count_min", "vol_percentile_max",
        "ma20_required", "limitup_down_ratio_min",
        "turnover_concentration_max"
    ) if k not in eg]
    if missing:
        _p_fail(f"配置缺少 emotion_gate.* 键：{missing}")
        return 1, None

    try:
        th = Thresholds(
            limit_up_count_min=int(eg["limit_up_count_min"]),
            limitup_down_ratio_min=float(eg["limitup_down_ratio_min"]),
            turnover_concentration_max=float(eg["turnover_concentration_max"]),
            vol_percentile_max=float(eg["vol_percentile_max"]),
            ma20_required=bool(eg["ma20_required"]),
        )
    except Exception as e:
        _p_fail(f"配置 emotion_gate.* 类型不正确：{e}")
        return 1, None

    _p_ok("配置清单：emotion_gate.* 键齐全且类型正确")
    return 0, th


# ---------------- 指标构造 & 复核 ----------------

CORE_KEYS = [
    "limit_up_count",
    "limit_down_count",
    "limitup_down_ratio",
    "turnover_concentration_top20",
    "hs300_vol_30d_annualized",
    "vol_percentile",
    "sh_above_ma20",
    "margin_net_repay_yi_prev",
]


def _make_min_indicators(prev_yyyymmdd: str, intraday_missing: bool, metrics: Dict[str, Any]) -> Dict[str, Any]:
    m = {k: metrics.get(k, None) for k in CORE_KEYS}
    return {
        "schema_version": 1,
        "calc_at": datetime.now(SH_TZ),
        "intraday_missing": bool(intraday_missing),
        "window": {"prev_trade_date": prev_yyyymmdd},
        "sources": {},
        "metrics": m,
    }


def _recompute_reasons(indi: Dict[str, Any], th: Thresholds) -> List[str]:
    """本地复算 reasons（与 gate 的判定口径保持一致），用于一致性校验。"""
    m = indi.get("metrics", {})
    reasons: List[str] = []
    if indi.get("intraday_missing", False):
        return ["intraday_missing"]

    def _gte(x, y, key, fmt="{:.2f}"):
        if x is None or y is None:
            reasons.append(f"{key}: {x} vs {y} (>=)")
            return
        if not (float(x) >= float(y)):
            try:
                xs = fmt.format(float(x))
            except Exception:
                xs = str(x)
            reasons.append(f"{key}: {xs} vs {y} (>=)")

    def _lte(x, y, key, fmt="{:.3f}"):
        if x is None or y is None:
            reasons.append(f"{key}: {x} vs {y} (<=)")
            return
        if not (float(x) <= float(y)):
            try:
                xs = fmt.format(float(x))
            except Exception:
                xs = str(x)
            reasons.append(f"{key}: {xs} vs {y} (<=)")

    def _eq_true(x, key):
        if x is not True:
            reasons.append(f"{key}: False vs True (==)")

    _gte(m.get("limit_up_count"), th.limit_up_count_min, "limit_up_count", fmt="{:.0f}")
    _gte(m.get("limitup_down_ratio"), th.limitup_down_ratio_min, "limitup_down_ratio", fmt="{:.2f}")
    _lte(m.get("turnover_concentration_top20"), th.turnover_concentration_max, "turnover_concentration_top20", fmt="{:.3f}")
    _lte(m.get("vol_percentile"), th.vol_percentile_max, "vol_percentile", fmt="{:.2f}")
    if th.ma20_required:
        _eq_true(m.get("sh_above_ma20"), "sh_above_ma20")
    return reasons


def _assert_gate_payload(payload: Dict[str, Any]) -> int:
    status = 0
    # 顶层
    for k in ("passed", "reasons", "indicators", "thresholds", "version", "calc_at", "tz", "window"):
        if k not in payload:
            _p_fail(f"gate payload 缺少字段：{k}")
            status |= 1
    if not isinstance(payload.get("passed"), bool):
        _p_fail("passed 应为 bool"); status |= 1
    else:
        _p_ok(f"passed={payload['passed']}")
    if not isinstance(payload.get("reasons"), list):
        _p_fail("reasons 应为 list[str]"); status |= 1
    else:
        _p_ok(f"reasons 个数={len(payload['reasons'])}")
    if not _tzaware(payload.get("calc_at")):
        _p_fail("calc_at 需为带时区时间"); status |= 1
    else:
        _p_ok("calc_at 为带时区时间")
    if payload.get("tz") != "Asia/Shanghai":
        _p_fail("tz 应为 Asia/Shanghai"); status |= 1
    # 指标容器
    indi = payload.get("indicators", {})
    need_indi = ["schema_version", "calc_at", "intraday_missing", "metrics"]
    miss = [k for k in need_indi if k not in indi]
    if miss:
        _p_fail(f"indicators 缺少字段：{miss}"); status |= 1
    else:
        _p_ok("indicators 结构齐全")
    metrics = indi.get("metrics", {})
    missm = [k for k in CORE_KEYS if k not in metrics]
    if missm:
        _p_fail(f"metrics 缺少核心键：{missm}"); status |= 1
    else:
        _p_ok("metrics 核心键齐全")
    return status


# ---------------- 事件检查（JSONL / DB） ----------------

def _check_event_written(expected_window: str) -> int:
    """
    首选 JSONL 降级文件（emotion_gate 内置 fallback），否则尝试 DB EventLog。
    """
    # JSONL 路径
    jsonl = Path("events/altflow_events.jsonl")
    if jsonl.exists():
        try:
            lines = jsonl.read_text(encoding="utf-8").splitlines()
            for line in reversed(lines[-50:]):  # 仅扫描末尾若干行
                rec = json.loads(line)
                ev = rec.get("event")
                if ev in ("altflow/gate_pass", "altflow/gate_reject"):
                    payload = rec.get("payload", {})
                    if payload.get("window") == expected_window:
                        _p_ok(f"事件已写入 JSONL：{ev} @ window={expected_window}")
                        return 0
            _p_warn("未在 JSONL 中找到本次 window 的 altflow/gate_*，可能采用了 DB 写入")
        except Exception as e:
            _p_warn(f"读取 JSONL 失败：{e}")

    # DB 尝试
    try:
        from models.database import get_session, EventLog  # type: ignore
        from sqlmodel import select  # type: ignore
        with get_session() as s:
            rows = s.exec(select(EventLog).order_by(EventLog.id.desc())).fetchmany(50)
            for r in rows:
                try:
                    p = json.loads(r.payload_json)
                    if r.code in ("altflow/gate_pass", "altflow/gate_reject") and p.get("window") == expected_window:
                        _p_ok(f"事件已写入 DB：{r.code} @ window={expected_window}")
                        return 0
                except Exception:
                    continue
        _p_warn("未在 DB 最近记录中找到本次 window 的 altflow/gate_*")
    except Exception as e:
        _p_warn(f"查询 DB 事件失败（可能项目未启用 DB 事件写入）：{e}")

    return 0  # 不把事件检查当成硬失败


# ---------------- 主流程 ----------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev", type=str, help="上一交易日 YYYYMMDD")
    ap.add_argument("--window", type=str, default="10:00")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--noapi", action="store_true", help="完全不触网，仅跑离线注入用例")
    args = ap.parse_args(argv)

    status = 0

    # 0) 配置清单
    st, th = _cfg_check()
    status |= st
    if th is None:
        return 1

    # 1) 一次真实指标计算（可选）
    indi_real: Optional[Dict[str, Any]] = None
    if not args.noapi:
        try:
            from data_service.proxies.altflow_proxy import run_indicator_calculator
            prev = args.prev or (datetime.now(SH_TZ) - timedelta(days=1)).strftime("%Y%m%d")
            indi_real = run_indicator_calculator(prev_trade_yyyymmdd=prev)
            _p_ok("真实指标计算完成（仅此一次触发外部获取）")
        except Exception as e:
            _p_warn(f"真实指标计算失败（将仅跑离线注入用例）：{e}")

    # 2) 使用注入完成 gate 验收（若上一步成功，则注入真实指标；否则跳过此 SMOKE）
    if indi_real is not None:
        try:
            from strategy_engine.gates.emotion_gate import check_emotion_gate
            gate_payload = check_emotion_gate(
                prev_trade_yyyymmdd=args.prev or (datetime.now(SH_TZ) - timedelta(days=1)).strftime("%Y%m%d"),
                window=args.window,
                indicators_payload=indi_real,  # 注入 → 不再触发取数
                emit_event=True,               # 写事件
            )
            status |= _assert_gate_payload(gate_payload)

            # 一致性复核：用阈值本地复算 reasons 并对比 gate 输出
            recomputed = _recompute_reasons(gate_payload["indicators"], th)
            if recomputed != gate_payload.get("reasons", []):
                _p_fail(f"reasons 一致性校验失败：gate={gate_payload.get('reasons')} vs recomputed={recomputed}")
                status |= 1
            else:
                _p_ok("验收一致性：gate reasons 与本地复算一致")

            # 事件落库检查
            status |= _check_event_written(args.window)
        except Exception as e:
            _p_fail(f"SMOKE（注入真实指标）失败：{e}")
            status |= 1

    # 3) 风险/回退 — INTRADAY_MISSING（注入，不触网）
    try:
        from strategy_engine.gates.emotion_gate import check_emotion_gate
        indi_missing = _make_min_indicators(
            prev_yyyymmdd=args.prev or "19700101",
            intraday_missing=True,
            metrics={k: None for k in CORE_KEYS},
        )
        p_missing = check_emotion_gate(
            prev_trade_yyyymmdd=args.prev or "19700101",
            window=args.window,
            indicators_payload=indi_missing,
            emit_event=False,  # 注入场景不写事件，避免噪音
        )
        if not (p_missing.get("passed") is False and p_missing.get("reasons") == ["intraday_missing"]):
            _p_fail(f"INTRADAY_MISSING 预期不通过且单一原因，实际：passed={p_missing.get('passed')} reasons={p_missing.get('reasons')}")
            status |= 1
        else:
            _p_ok("INTRADAY_MISSING 场景：passed=False 且 reasons=['intraday_missing']")
    except Exception as e:
        _p_fail(f"INTRADAY_MISSING 场景失败：{e}")
        status |= 1

    # 4) 风险/回退 — FORCED REJECT & PASS（注入，不触网）
    try:
        # REJECT
        m_reject = {
            "limit_up_count": 0,
            "limit_down_count": 10,
            "limitup_down_ratio": 0.1,
            "turnover_concentration_top20": 0.9,
            "hs300_vol_30d_annualized": 0.5,
            "vol_percentile": 0.99,
            "sh_above_ma20": False,
            "margin_net_repay_yi_prev": None,
        }
        indi_reject = _make_min_indicators(args.prev or "19700101", False, m_reject)
        p_reject = check_emotion_gate(
            prev_trade_yyyymmdd=args.prev or "19700101",
            window=args.window,
            indicators_payload=indi_reject,
            emit_event=False,
        )
        if p_reject.get("passed") is not False:
            _p_fail("FORCED REJECT 预期 passed=False"); status |= 1
        # PASS
        m_pass = {
            "limit_up_count": max(th.limit_up_count_min, 60),
            "limit_down_count": 0,
            "limitup_down_ratio": max(th.limitup_down_ratio_min, 1.0),
            "turnover_concentration_top20": min(th.turnover_concentration_max, 0.20),
            "hs300_vol_30d_annualized": 0.18,
            "vol_percentile": min(th.vol_percentile_max, 0.20),
            "sh_above_ma20": True if th.ma20_required else True,
            "margin_net_repay_yi_prev": 100.0,
        }
        indi_pass = _make_min_indicators(args.prev or "19700101", False, m_pass)
        p_pass = check_emotion_gate(
            prev_trade_yyyymmdd=args.prev or "19700101",
            window=args.window,
            indicators_payload=indi_pass,
            emit_event=False,
        )
        if not (p_pass.get("passed") is True and p_pass.get("reasons") == []):
            _p_fail(f"FORCED PASS 预期 passed=True, reasons=[]，实际：{p_pass.get('passed')}, {p_pass.get('reasons')}")
            status |= 1
        else:
            _p_ok("FORCED PASS/REJECT 场景：判定逻辑正确")
    except Exception as e:
        _p_fail(f"FORCED PASS/REJECT 场景失败：{e}")
        status |= 1

    if args.strict and status == 0:
        _p_ok("严格模式：全部检查通过")
    return status


if __name__ == "__main__":
    sys.exit(main())
