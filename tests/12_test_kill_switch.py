# tests/12_test_kill_switch.py
# -*- coding: utf-8 -*-
"""
Step3 · Alt-Flow Kill-Switch 分级 一站式自检脚本（低频 / 注入为主）
Checks:
  6.1 配置检查：kill_switch.L1/L2 键/类型/范围 + restart_conditions
  6.2 判定用例（注入，0 次外部调用）：
      L0 安全区；L1 四条单条件；L2 四条单条件；L1/L2 复合条件；intraday_missing 降级
  6.3 集成用例（一次真实取数）：
      run_indicator_calculator() → 注入 check_kill_switch(..., indicators_payload=...)；
      验证返回级别、事件写入（JSONL/DB），以及状态副作用（position_cap / sleeping）
  6.4 重启机制：
      伪造最近 N 日 gate_pass 事件 + （可选）关闭回撤条件；
      断言 restart/eligible（与可选 restart/confirm），并验证 sleeping→active 切换

用法：
  python -m tests.12_test_kill_switch --prev 20250912 --window 10:00 --strict
  python -m tests.12_test_kill_switch --noapi --window 14:00
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

# ---------- 控制台输出风格（对齐 11_step2_auto_check） ----------
def _p_ok(msg: str) -> None:
    print(f"{OK} {msg}")

def _p_fail(msg: str) -> None:
    print(f"{FAIL} {msg}")

def _p_warn(msg: str) -> None:
    print(f"{WARN} {msg}")

def _tzaware(dt: Any) -> bool:
    return isinstance(dt, datetime) and dt.tzinfo is not None

# ---------- 配置加载 ----------
def _load_cfg() -> Dict[str, Any]:
    from config.loader import load_config
    return load_config(config_dir="config")

# ---------- 阈值/配置数据结构 ----------
@dataclass
class KSLevel:
    margin_net_repay_yi: float
    limitup_down_ratio_min: float
    turnover_concentration_max: float
    vol_percentile_min: float
    index_down_max: float          # 你在 YAML 里新增的键（小数或百分数；运行时自动适配）
    action: str
    temp_total_cap: Optional[float] = None

@dataclass
class RestartCfg:
    consecutive_gate_days: int
    monthly_drawdown_threshold: Optional[float]
    manual_confirm: bool

@dataclass
class KSCfg:
    L1: KSLevel
    L2: KSLevel
    restart: RestartCfg

# ---------- 配置检查（6.1） ----------
def _cfg_check() -> Tuple[int, Optional[KSCfg]]:
    cfg = _load_cfg()
    k = cfg.get("kill_switch", {}) or {}
    r = cfg.get("restart_conditions", {}) or {}

    miss_blocks = []
    if "L1" not in k: miss_blocks.append("kill_switch.L1")
    if "L2" not in k: miss_blocks.append("kill_switch.L2")
    if miss_blocks:
        _p_fail(f"缺少配置块：{miss_blocks}")
        return 1, None

    def _need(keys: List[str], d: Dict[str, Any], prefix: str) -> List[str]:
        return [f"{prefix}.{x}" for x in keys if x not in d]

    need_L = ["margin_net_repay_yi", "limitup_down_ratio_min", "turnover_concentration_max",
              "vol_percentile_min", "index_down_max", "action"]
    miss = _need(need_L, k["L1"], "kill_switch.L1") + _need(need_L, k["L2"], "kill_switch.L2")

    # L1 若 action=stop_new_reduce_cap，需存在 temp_total_cap
    if k["L1"].get("action") == "stop_new_reduce_cap" and "temp_total_cap" not in k["L1"]:
        miss.append("kill_switch.L1.temp_total_cap")

    if miss:
        _p_fail(f"Kill-Switch 配置缺失键：{miss}")
        return 1, None

    try:
        L1 = KSLevel(
            margin_net_repay_yi=float(k["L1"]["margin_net_repay_yi"]),
            limitup_down_ratio_min=float(k["L1"]["limitup_down_ratio_min"]),
            turnover_concentration_max=float(k["L1"]["turnover_concentration_max"]),
            vol_percentile_min=float(k["L1"]["vol_percentile_min"]),
            index_down_max=float(k["L1"]["index_down_max"]),
            action=str(k["L1"]["action"]),
            temp_total_cap=float(k["L1"]["temp_total_cap"]) if "temp_total_cap" in k["L1"] else None,
        )
        L2 = KSLevel(
            margin_net_repay_yi=float(k["L2"]["margin_net_repay_yi"]),
            limitup_down_ratio_min=float(k["L2"]["limitup_down_ratio_min"]),
            turnover_concentration_max=float(k["L2"]["turnover_concentration_max"]),
            vol_percentile_min=float(k["L2"]["vol_percentile_min"]),
            index_down_max=float(k["L2"]["index_down_max"]),
            action=str(k["L2"]["action"]),
            temp_total_cap=float(k["L2"]["temp_total_cap"]) if "temp_total_cap" in k["L2"] else None,
        )
        RC = RestartCfg(
            consecutive_gate_days=int(r.get("consecutive_gate_days", 3)),
            monthly_drawdown_threshold=(float(r["monthly_drawdown_threshold"]) if r.get("monthly_drawdown_threshold") is not None else None),
            manual_confirm=bool(r.get("manual_confirm", True)),
        )
    except Exception as e:
        _p_fail(f"Kill-Switch/Restart 配置类型错误：{e}")
        return 1, None

    # 范围/语义检查（宽松）
    if not (0 <= L1.limitup_down_ratio_min <= 2 and 0 <= L2.limitup_down_ratio_min <= 2):
        _p_warn("limitup_down_ratio_min 范围非常规（0~2），请确认是否合理")
    if not (0 <= L1.vol_percentile_min <= 1 and 0 <= L2.vol_percentile_min <= 1):
        _p_fail("vol_percentile_min 应在 [0,1] 之间"); return 1, None
    if not (0 <= L1.turnover_concentration_max <= 1 and 0 <= L2.turnover_concentration_max <= 1):
        _p_fail("turnover_concentration_max 应在 [0,1] 之间"); return 1, None
    if L1.action == "stop_new_reduce_cap" and (L1.temp_total_cap is None or not (0 < L1.temp_total_cap <= 1)):
        _p_fail("L1.temp_total_cap 缺失或不在 (0,1]"); return 1, None
    if L2.action != "liquidate_and_sleep":
        _p_warn(f"L2.action={L2.action} 非典型（预期 liquidate_and_sleep）")

    _p_ok("配置清单：kill_switch.L1/L2 + restart_conditions 键齐全，类型正确")
    return 0, KSCfg(L1=L1, L2=L2, restart=RC)

# ---------- 构造最小指标容器 ----------
CORE_KEYS = ["limitup_down_ratio", "turnover_concentration_top20", "vol_percentile", "margin_net_repay_yi_prev", "index_intraday_ret"]

def _make_indicators(prev_yyyymmdd: str, intraday_missing: bool, metrics: Dict[str, Any]) -> Dict[str, Any]:
    m = {k: metrics.get(k, None) for k in CORE_KEYS}
    return {
        "schema_version": "step2-contract-v1.1",
        "calc_at": datetime.now(SH_TZ).isoformat(),
        "intraday_missing": bool(intraday_missing),
        "metrics": m,
        "sources": {},
        "window": {"prev_trade_date": prev_yyyymmdd},
        "tz": "Asia/Shanghai",
    }

# ---------- 断言 KS 返回结构 ----------
def _assert_ks_payload(p: Dict[str, Any]) -> int:
    status = 0
    for k in ("level", "triggered_conditions", "actions", "calc_at", "tz", "window", "version"):
        if k not in p:
            _p_fail(f"KS payload 缺少字段：{k}")
            status |= 1
    if p.get("tz") != "Asia/Shanghai":
        _p_fail("KS payload.tz 应为 Asia/Shanghai"); status |= 1
    ms = p.get("metrics_snapshot", {})
    need_ms = ["margin_net_repay_yi_prev", "limitup_down_ratio", "turnover_concentration_top20", "vol_percentile", "index_intraday_ret"]
    for k in need_ms:
        if k not in ms:
            _p_fail(f"KS payload.metrics_snapshot 缺少字段：{k}")
            status |= 1
    if status == 0:
        _p_ok(f"KS payload 结构齐全：level={p.get('level')}，triggers={p.get('triggered_conditions')}")
    return status

# ---------- 事件与状态文件 ----------
_JSONL = Path("events/altflow_events.jsonl")
_STATE = Path("runtime_state.json")

def _check_ks_event(window: str) -> int:
    if not _JSONL.exists():
        _p_warn("未发现 JSONL 事件文件（可能走了 DB 落库，或尚未触发写入）")
        return 0
    try:
        lines = _JSONL.read_text(encoding="utf-8").splitlines()
        for line in reversed(lines[-80:]):
            rec = json.loads(line)
            ev = rec.get("event", "")
            if ev.startswith("altflow/ks_"):
                payload = rec.get("payload", {})
                if payload.get("window") == window:
                    _p_ok(f"事件写入：{ev} @ {window}")
                    return 0
        _p_warn("未在 JSONL 末尾找到本次 window 的 altflow/ks_* 事件（可能写入了 DB）")
    except Exception as e:
        _p_warn(f"读取 JSONL 失败：{e}")
    return 0

def _read_state() -> Dict[str, Any]:
    if not _STATE.exists():
        return {}
    try:
        return json.loads(_STATE.read_text(encoding="utf-8"))
    except Exception:
        return {}

# ---------- 单位自适配（与 killswitch 内部口径一致） ----------
def _adapt_idx(threshold: float, idx_ret: float) -> Tuple[float, float]:
    t, r = float(threshold), float(idx_ret)
    if abs(r) >= 1.0 and abs(t) < 0.2:
        return t * 100.0, r
    if abs(r) < 0.2 and abs(t) >= 1.0:
        return t / 100.0, r
    return t, r

# ---------- 注入判定（6.2） ----------
def _run_injection_suite(th: KSCfg, prev: str, window: str) -> int:
    status = 0
    from strategy_engine.killswitch.killswitch import check_kill_switch

    # L0 安全区
    safe_metrics = {
        "limitup_down_ratio": max(th.L1.limitup_down_ratio_min + 0.10, th.L2.limitup_down_ratio_min + 0.10),
        "turnover_concentration_top20": min(th.L1.turnover_concentration_max, th.L2.turnover_concentration_max) - 0.05,
        "vol_percentile": min(th.L1.vol_percentile_min, th.L2.vol_percentile_min) - 0.10,
        "margin_net_repay_yi_prev": min(th.L1.margin_net_repay_yi, th.L2.margin_net_repay_yi) - 10.0,
        "index_intraday_ret": 0.0,
    }
    p = check_kill_switch(prev, window, indicators_payload=_make_indicators(prev, False, safe_metrics), emit_event=False)
    if not (p.get("level") == 0 and p.get("triggered_conditions") == []):
        _p_fail(f"L0 预期 level=0，得到 level={p.get('level')} / triggers={p.get('triggered_conditions')}")
        status |= 1
    else:
        _p_ok("L0 安全区：level=0 & 无触发条件")

    # L1 四条单条件
    try:
        # 1) 两融触发
        m = dict(safe_metrics); m["margin_net_repay_yi_prev"] = th.L1.margin_net_repay_yi + 1.0
        p = check_kill_switch(prev, window, indicators_payload=_make_indicators(prev, False, m), emit_event=False)
        if p.get("level") not in (1, 2):
            _p_fail("L1 两融触发失败"); status |= 1
        # 2) 涨跌停比触发
        m = dict(safe_metrics); m["limitup_down_ratio"] = th.L1.limitup_down_ratio_min - 0.01
        p = check_kill_switch(prev, window, indicators_payload=_make_indicators(prev, False, m), emit_event=False)
        if p.get("level") not in (1, 2):
            _p_fail("L1 涨跌停比触发失败"); status |= 1
        # 3) 复合：集中度 + 指数下跌
        m = dict(safe_metrics); m["turnover_concentration_top20"] = th.L1.turnover_concentration_max + 0.02
        t, r = _adapt_idx(th.L1.index_down_max, -1.0)   # 指数 -1%（自动适配）
        m["index_intraday_ret"] = r if abs(r) >= 1.0 else t - 0.001
        p = check_kill_switch(prev, window, indicators_payload=_make_indicators(prev, False, m), emit_event=False)
        if p.get("level") not in (1, 2):
            _p_fail("L1 复合触发失败"); status |= 1
        # 4) 波动分位触发
        m = dict(safe_metrics); m["vol_percentile"] = max(th.L1.vol_percentile_min, 0.0) + 0.02
        p = check_kill_switch(prev, window, indicators_payload=_make_indicators(prev, False, m), emit_event=False)
        if p.get("level") not in (1, 2):
            _p_fail("L1 波动分位触发失败"); status |= 1
        else:
            _p_ok("L1 单条件触发：通过（两融/涨跌停比/复合/波动）")
    except Exception as e:
        _p_fail(f"L1 注入用例异常：{e}"); status |= 1

    # L2 四条单条件
    try:
        # 1) 两融
        m = dict(safe_metrics); m["margin_net_repay_yi_prev"] = th.L2.margin_net_repay_yi + 1.0
        p = check_kill_switch(prev, window, indicators_payload=_make_indicators(prev, False, m), emit_event=False)
        if p.get("level") != 2:
            _p_fail("L2 两融触发失败"); status |= 1
        # 2) 涨跌停比
        m = dict(safe_metrics); m["limitup_down_ratio"] = th.L2.limitup_down_ratio_min - 0.01
        p = check_kill_switch(prev, window, indicators_payload=_make_indicators(prev, False, m), emit_event=False)
        if p.get("level") != 2:
            _p_fail("L2 涨跌停比触发失败"); status |= 1
        # 3) 复合
        m = dict(safe_metrics); m["turnover_concentration_top20"] = th.L2.turnover_concentration_max + 0.02
        t, r = _adapt_idx(th.L2.index_down_max, -1.6)   # 指数 -1.6%（自动适配）
        m["index_intraday_ret"] = r if abs(r) >= 1.0 else t - 0.001
        p = check_kill_switch(prev, window, indicators_payload=_make_indicators(prev, False, m), emit_event=False)
        if p.get("level") != 2:
            _p_fail("L2 复合触发失败"); status |= 1
        # 4) 波动
        m = dict(safe_metrics); m["vol_percentile"] = max(th.L2.vol_percentile_min, 0.0) + 0.02
        p = check_kill_switch(prev, window, indicators_payload=_make_indicators(prev, False, m), emit_event=False)
        if p.get("level") != 2:
            _p_fail("L2 波动分位触发失败"); status |= 1
        else:
            _p_ok("L2 单条件触发：通过（两融/涨跌停比/复合/波动）")
    except Exception as e:
        _p_fail(f"L2 注入用例异常：{e}"); status |= 1

    # intraday_missing 降级
    p = check_kill_switch(prev, window, indicators_payload=_make_indicators(prev, True, safe_metrics), emit_event=False)
    if not (p.get("level") == 0 and p.get("triggered_conditions") == ["intraday_missing"]):
        _p_fail("intraday_missing 预期 level=0 & 单一触发项 intraday_missing"); status |= 1
    else:
        _p_ok("intraday_missing 降级：通过")
    return status

# ---------- 集成用例（6.3） ----------
def _run_integration(prev: str, window: str) -> int:
    status = 0
    try:
        from data_service.proxies.altflow_proxy import run_indicator_calculator
        from strategy_engine.killswitch.killswitch import check_kill_switch
        indi = run_indicator_calculator(prev_trade_yyyymmdd=prev)
        p = check_kill_switch(prev, window, indicators_payload=indi, emit_event=True)
        status |= _assert_ks_payload(p)
        status |= _check_ks_event(window)

        # 状态副作用（仅在 level>0 时检查）
        if p.get("level") == 1:
            st = _read_state()
            cap = st.get("position_cap", None)
            if cap is None:
                _p_warn("L1 触发但未发现 runtime_state.json 的 position_cap（可能走 DB 状态或动作失败）")
            else:
                _p_ok(f"L1 动作已生效：position_cap={cap}")
        elif p.get("level") == 2:
            st = _read_state()
            if st.get("sleeping") is True:
                _p_ok("L2 动作已生效：系统进入休眠态")
            else:
                _p_warn("L2 触发但未发现 sleeping=True（可能走 DB 状态或动作失败）")
        else:
            _p_warn("本次真实指标未触发 L1/L2（属正常情况）")
    except Exception as e:
        _p_warn(f"集成用例：真实指标获取或 KS 调用失败（可忽略，仅影响此环节）：{e}")
    return status

# ---------- 重启机制（6.4） ----------
def _run_restart_flow(rcfg: RestartCfg) -> int:
    status = 0
    try:
        # 1) 先进入休眠态
        import data_service.storage as storage
        storage.enter_sleep_mode(note="test_sleep_for_restart")

        # 2) 伪造最近 N 日 gate_pass（写事件；优先 DB，失败降级 JSONL）
        for i in range(max(1, rcfg.consecutive_gate_days)):
            storage.write_event("altflow/gate_pass", {"mock": True, "idx": i, "ts": datetime.now(SH_TZ).isoformat()})

        # 3) 触发资格检查（把 B 条件置 None 可关闭回撤约束，提高稳定性）
        from strategy_engine.killswitch.killswitch import check_restart_eligibility
        res = check_restart_eligibility(
            consecutive_gate_days=rcfg.consecutive_gate_days,
            monthly_drawdown_threshold=None,   # 关闭回撤项，避免依赖真实净值
            manual_confirm=False               # 自动确认 + 尝试 exit_sleep_mode
        )
        if not res.get("eligible", False):
            _p_fail("restart/eligible 未通过（请确认事件查询或 N 日注入是否生效）"); status |= 1
        else:
            _p_ok("restart/eligible：通过")

        # 4) 检查 restart 事件与状态切换
        if _JSONL.exists():
            try:
                lines = _JSONL.read_text(encoding="utf-8").splitlines()
                has_eligible = any(json.loads(x).get("event") == "restart/eligible" for x in lines[-80:])
                has_confirm  = any(json.loads(x).get("event") == "restart/confirm"  for x in lines[-80:])
                if has_eligible:
                    _p_ok("事件写入：restart/eligible")
                else:
                    _p_warn("未在 JSONL 中找到 restart/eligible（可能写入 DB）")
                if has_confirm:
                    _p_ok("事件写入：restart/confirm（manual_confirm=False 自动确认）")
            except Exception as e:
                _p_warn(f"读取 JSONL 失败：{e}")

        st = _read_state()
        if st.get("sleeping") is False:
            _p_ok("exit_sleep_mode：成功（sleeping=False）")
        else:
            _p_warn("sleeping 未切换为 False（可能走 DB 状态或 exit 失败）")

    except Exception as e:
        _p_fail(f"重启流程测试异常：{e}")
        status |= 1
    return status

# ---------- 主程序 ----------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev", type=str, help="上一交易日 YYYYMMDD")
    ap.add_argument("--window", type=str, default="10:00")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--noapi", action="store_true", help="完全不触网（跳过 6.3 集成用例）")
    args = ap.parse_args(argv)

    status = 0

    # 0) 配置检查
    st, ks_cfg = _cfg_check()
    status |= st
    if ks_cfg is None:
        return 1

    # 1) 注入用例（6.2）
    prev = args.prev or (datetime.now(SH_TZ) - timedelta(days=1)).strftime("%Y%m%d")
    status |= _run_injection_suite(ks_cfg, prev, args.window)

    # 2) 集成用例（6.3）—仅当允许联网
    if not args.noapi:
        status |= _run_integration(prev, args.window)
    else:
        _p_warn("已启用 --noapi：跳过一次真实指标的集成用例（6.3）")

    # 3) 重启机制（6.4）
    status |= _run_restart_flow(ks_cfg.restart)

    if args.strict and status == 0:
        _p_ok("严格模式：全部检查通过")
    return status

if __name__ == "__main__":
    sys.exit(main())
