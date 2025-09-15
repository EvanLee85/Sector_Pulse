#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_test_validate_altflow_payload.py
Validate Alt-Flow gate event payloads against the contract (step2-contract-v1.1).

用法（建议放在 tests/ 目录下运行）:
# 通过样本
python -m tests.07_test_validate_altflow_payload \
  --payload docs/contracts/examples/step2_gate_payload_pass.json \
  --contract config/indicators_contract.yaml --strict

  {
  "ok": true,
  "errors": [],
  "warnings": []
}

# 拒绝-指标不达标样本（应校验通过，且 passed=false 合理）
python -m tests.07_test_validate_altflow_payload \
  --payload docs/contracts/examples/step2_gate_payload_reject_indicator.json \
  --contract config/indicators_contract.yaml --strict

  {
  "ok": true,
  "errors": [],
  "warnings": []
}

# 拒绝-盘中缺失样本（应校验通过，且 passed=false 合理）
python -m tests.07_test_validate_altflow_payload \
  --payload docs/contracts/examples/step2_gate_payload_reject_intraday_missing.json \
  --contract config/indicators_contract.yaml --strict

  {
  "ok": true,
  "errors": [],
  "warnings": []
}

"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Optional

# --- YAML loading ---
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

CN_TZ_OFFSET = timedelta(hours=8)


def iso8601_cn(dt_str: str, *, field: str) -> Tuple[Optional[datetime], Optional[str]]:
    """解析 ISO8601（必须带 +08:00 时区偏移）。返回 (datetime, error)。"""
    try:
        dt = datetime.fromisoformat(dt_str)
    except Exception:
        return None, f'{field}: invalid datetime format (expect ISO8601 with offset)'
    if dt.tzinfo is None:
        return None, f'{field}: timezone offset required (+08:00)'
    if dt.utcoffset() != CN_TZ_OFFSET:
        return None, f'{field}: timezone must be +08:00'
    return dt, None


@dataclass
class Result:
    ok: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def err(self, msg: str):
        self.ok = False
        self.errors.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)


def load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        print("ERROR: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
        sys.exit(2)
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_api_whitelist(contract: Dict[str, Any]) -> List[str]:
    """从 contract.indicators.*.sources 汇总白名单。"""
    wl = set()
    for _name, spec in (contract.get('indicators') or {}).items():
        for src in spec.get('sources') or []:
            wl.add(src)
    return sorted(wl)


def decision_expected_pass(payload: Dict[str, Any], contract: Dict[str, Any]) -> Optional[bool]:
    """
    按 YAML 中的 decision 规则推导期望 passed。
    若信息不足返回 None。
    """
    decision = contract.get('decision') or {}
    gates = decision.get('gates') or []
    inds = payload.get('indicators') or {}
    thresholds = payload.get('thresholds') or {}

    # 硬失败：intraday_missing
    for hf in decision.get('hard_fail_on', []):
        ind = hf.get('indicator')
        expected = hf.get('expected')
        if ind in inds:
            if inds[ind] != expected:
                return False
        else:
            return None

    # 按 gate 规则检验
    def get_val(key):
        if key not in inds:
            return None
        return inds[key]

    def get_thr(tk):
        return thresholds.get(tk)

    for g in gates:
        ind = g.get('indicator')
        cmp = g.get('comparator')
        tk = g.get('threshold_key')
        v = get_val(ind)
        t = get_thr(tk) if tk else None
        if v is None or (tk and t is None):
            return None

        ok = True
        if cmp == 'gte':
            ok = (v >= t)
        elif cmp == 'lte':
            ok = (v <= t)
        elif cmp == 'eq':
            ok = (v == t)
        elif cmp == 'none':
            ok = True
        else:
            return None

        if not ok:
            return False

    return True


def validate(payload: Dict[str, Any], contract: Dict[str, Any]) -> Result:
    r = Result()

    # 版本一致性
    want_ver = str(contract.get('version'))
    if str(payload.get('version')) != want_ver:
        r.err(f"version mismatch: payload={payload.get('version')} contract={want_ver}")

    # 顶层必填键
    required_top = list(contract.get('event_payload_required_keys') or [])
    for k in required_top:
        if k not in payload:
            r.err(f"missing required top-level key: {k}")

    # window & tz
    defaults = contract.get('defaults') or {}
    allowed_windows = defaults.get('windows') or ["10:00", "14:00"]
    if 'window' in payload and payload['window'] not in allowed_windows:
        r.err(f"window must be one of {allowed_windows}; got {payload['window']}")
    if payload.get('tz') != (defaults.get('tz') or "Asia/Shanghai"):
        r.err(f"tz must be {defaults.get('tz') or 'Asia/Shanghai'}; got {payload.get('tz')}")

    # calc_at 校验
    if 'calc_at' in payload:
        _, err = iso8601_cn(str(payload['calc_at']), field='calc_at')
        if err:
            r.err(err)

    # calendar
    cal = payload.get('calendar') or {}
    if isinstance(cal, dict):
        src = cal.get('source')
        if src not in {'akshare', 'cache', 'weekday_fallback', 'unavailable'}:
            r.err(f"calendar.source invalid: {src}")
        if not isinstance(cal.get('degraded'), bool):
            r.err("calendar.degraded must be boolean")
        sd = cal.get('staleness_days')
        if not (isinstance(sd, int) and sd >= 0):
            r.err("calendar.staleness_days must be int >= 0")

    # indicators 必含键
    req_inds = list(contract.get('indicators_required_keys') or [])
    inds = payload.get('indicators')
    if not isinstance(inds, dict):
        r.err("indicators must be an object")
        inds = {}
    for k in req_inds:
        if k not in inds:
            r.err(f"indicators missing required key: {k}")

    # 基本类型与范围
    def is_number(x): return isinstance(x, (int, float)) and not isinstance(x, bool)

    def chk_range(name, lo=None, hi=None):
        if name in inds:
            v = inds[name]
            if not is_number(v):
                r.err(f"{name} must be a number; got {type(v).__name__}")
                return
            if lo is not None and v < lo:
                r.err(f"{name} out of range: {v} < {lo}")
            if hi is not None and v > hi:
                r.err(f"{name} out of range: {v} > {hi}")

    # 整数家数
    for k in ('limit_up_count', 'limit_down_count'):
        if k in inds and not isinstance(inds[k], int):
            r.err(f"{k} must be int; got {type(inds[k]).__name__}")
        if k in inds and inds[k] < 0:
            r.err(f"{k} must be >= 0; got {inds[k]}")

    # 比例/分位
    chk_range('limitup_down_ratio', 0, None)
    chk_range('turnover_concentration_top20', 0, 1)
    chk_range('vol_percentile', 0, 1)

    # 年化波动（宽松上限）
    if 'hs300_vol_30d_annualized' in inds:
        v = inds['hs300_vol_30d_annualized']
        if not is_number(v):
            r.err("hs300_vol_30d_annualized must be a number")
        elif v < 0 or v > 5:
            r.warn(f"hs300_vol_30d_annualized unusual value: {v} (expected 0..1-ish)")

    # 布尔
    for k in ('sh_above_ma20', 'intraday_missing', 'calendar_degraded'):
        if k in inds and not isinstance(inds[k], bool):
            r.err(f"{k} must be boolean; got {type(inds[k]).__name__}")

    # 两融（可为负；但必须是数）
    if 'margin_net_repay_yi_prev' in inds and not is_number(inds['margin_net_repay_yi_prev']):
        r.err("margin_net_repay_yi_prev must be a number (unit: yi_yuan)")

    # 阈值存在性与类型
    thr = payload.get('thresholds') or {}
    needed_thr = set()
    for g in (contract.get('decision') or {}).get('gates', []):
        tk = g.get('threshold_key')
        if tk:
            needed_thr.add(tk)
    for _name, spec in (contract.get('indicators') or {}).items():
        tk = spec.get('threshold_key')
        if tk:
            needed_thr.add(tk)
    for k in sorted(needed_thr):
        if k not in thr:
            r.err(f"thresholds missing key: {k}")

    if 'emotion_gate.limit_up_count_min' in thr and not isinstance(thr['emotion_gate.limit_up_count_min'], int):
        r.err("emotion_gate.limit_up_count_min must be int")
    for flt_key in ('emotion_gate.limitup_down_ratio_min',
                    'emotion_gate.turnover_concentration_max',
                    'emotion_gate.vol_percentile_max'):
        if flt_key in thr and not isinstance(thr[flt_key], (int, float)):
            r.err(f"{flt_key} must be number")
    if 'emotion_gate.turnover_concentration_max' in thr:
        v = thr['emotion_gate.turnover_concentration_max']
        if not (0 <= v <= 1):
            r.err("emotion_gate.turnover_concentration_max must be in [0,1]")
    if 'emotion_gate.vol_percentile_max' in thr:
        v = thr['emotion_gate.vol_percentile_max']
        if not (0 <= v <= 1):
            r.err("emotion_gate.vol_percentile_max must be in [0,1]")
    if 'emotion_gate.ma20_required' in thr and not isinstance(thr['emotion_gate.ma20_required'], bool):
        r.err("emotion_gate.ma20_required must be bool")

    # sources 白名单与字段
    wl = set(build_api_whitelist(contract))
    srcs = payload.get('sources') or {}
    if not isinstance(srcs, dict):
        r.err("sources must be an object")
        srcs = {}

    # 对每个指标要求提供 {api, fetched_at}
    for key in inds.keys():
        s = srcs.get(key)
        if not isinstance(s, dict):
            r.err(f"sources.{key} must be an object with 'api' and 'fetched_at'")
            continue
        api = s.get('api')
        if api is None or not isinstance(api, str) or not api:
            r.err(f"sources.{key}.api must be a non-empty string")
        else:
            # 允许派生/组合标记
            if api not in wl and not api.startswith('derived(') and api not in ('spot(daily_MA20+index)', 'SSE+SZSE_margin'):
                r.warn(f"sources.{key}.api not in whitelist: {api}")
        fa = s.get('fetched_at')
        if fa is None or not isinstance(fa, str):
            r.err(f"sources.{key}.fetched_at must be ISO8601 string with +08:00")
        else:
            _, err = iso8601_cn(fa, field=f"sources.{key}.fetched_at")
            if err:
                r.err(err)

    # passed / reasons 逻辑
    if 'passed' in payload and not isinstance(payload['passed'], bool):
        r.err("passed must be boolean")
    if payload.get('passed') is True:
        if payload.get('reasons') not in ([], None):
            r.err("passed==true requires reasons to be an empty array")
    if payload.get('passed') is False:
        if not payload.get('reasons'):
            r.err("passed==false requires non-empty reasons array")

    # 与 contract.decision 的一致性校验
    exp = decision_expected_pass(payload, contract)
    if exp is not None and payload.get('passed') is not None:
        if exp != payload['passed']:
            r.err(f"passed mismatch by contract decision: expected {exp}, got {payload['passed']}")

    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", required=True, help="Path to JSON payload to validate")
    ap.add_argument("--contract", required=True, help="Path to indicators_contract.yaml")
    ap.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    args = ap.parse_args()

    contract = load_yaml(args.contract)
    payload = load_json(args.payload)

    res = validate(payload, contract)
    if args.strict and res.warnings:
        res.errors.extend(f"[warn-as-error] {w}" for w in res.warnings)
        res.warnings.clear()
        res.ok = res.ok and (len(res.errors) == 0)

    # 输出机器可读 + 人类可读
    summary = {
        "ok": res.ok,
        "errors": res.errors,
        "warnings": res.warnings,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not res.ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
