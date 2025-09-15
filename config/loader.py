# config/loader.py
# 组合并加载三个配置文件：base.yaml + strategy_core.yaml + <env>.override.yaml
# 功能：深度合并 / 北向“亿→元”统一换算 / 基础配置校验
from __future__ import annotations
from pathlib import Path
import datetime as dt
import copy
import yaml


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base_dict: dict, update_dict: dict) -> dict:
    """递归合并字典，update_dict 优先。返回新字典，不修改入参。"""
    result = copy.deepcopy(base_dict) if base_dict else {}
    for k, v in (update_dict or {}).items():
        if (
            k in result
            and isinstance(result[k], dict)
            and isinstance(v, dict)
        ):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result

def _normalize_aliases(cfg: dict) -> None:
    """将旧键名归一到文档口径，兼容历史配置"""
    eg = cfg.get("emotion_gate", {})
    # 闸门：limit_up_min → limit_up_count_min
    if "limit_up_count_min" not in eg and "limit_up_min" in eg:
        eg["limit_up_count_min"] = eg.pop("limit_up_min")
    # 闸门：limit_up_ratio_min → limitup_down_ratio_min（历史遗留）
    if "limitup_down_ratio_min" not in eg and "limit_up_ratio_min" in eg:
        eg["limitup_down_ratio_min"] = eg.pop("limit_up_ratio_min")

    # KS：limit_up_ratio_min → limitup_down_ratio_min
    ks = cfg.get("kill_switch", {})
    for level in ("L1", "L2"):
        node = ks.get(level, {})
        if "limitup_down_ratio_min" not in node and "limit_up_ratio_min" in node:
            node["limitup_down_ratio_min"] = node.pop("limit_up_ratio_min")

def _normalize_units(cfg: dict) -> None:
    """
    单位归一（可选）：
    - 可将两融‘亿’阈值转换为‘元’，新增 *_yuan 字段（若后续计算用到货币单位）。
    - Alt-Flow 指标（涨停家数、拥挤度、波动分位）无货币单位，不需转换。
    """
    yi = 1e8
    ks = cfg.get("kill_switch") or {}
    for level in ("L1", "L2"):
        node = ks.get(level) or {}
        if "margin_net_repay_yi" in node:
            node["margin_net_repay_yuan"] = float(node["margin_net_repay_yi"]) * yi

def validate_config(cfg: dict) -> None:
    """关键配置校验（Alt-Flow 版）"""
    # 顶层必需
    for field in ("state", "emotion_gate", "position_management"):
        assert field in cfg, f"Missing required config: {field}"

    # 状态与仓位上限
    band = cfg["state"]["pos_band"]
    enter_v = band["enter_offense"]
    exit_v = band["exit_offense"]
    assert 0 < enter_v < 1 and 0 < exit_v < 1 and enter_v < exit_v, "invalid state.pos_band"

    caps = cfg["state"]["caps"]
    assert 0 < caps["initial_entry_fraction"] <= caps["max_single_position"] <= 1
    assert 0 < caps["max_total_position"] <= 1

    # 情绪闸门（Alt-Flow）
    eg = cfg["emotion_gate"]
    assert eg.get("limit_up_count_min") is not None, "emotion_gate.limit_up_count_min missing"
    if "limitup_down_ratio_min" in eg:
        ldr = eg["limitup_down_ratio_min"]
        assert 0 <= ldr <= 1, "emotion_gate.limitup_down_ratio_min must be in [0,1]"
    if "turnover_concentration_max" in eg:
        tcm = eg["turnover_concentration_max"]
        assert 0 <= tcm <= 1, "emotion_gate.turnover_concentration_max must be in [0,1]"
    vpm = eg["vol_percentile_max"]
    assert 0 <= vpm <= 1, "emotion_gate.vol_percentile_max must be in [0,1]"

    # Kill-Switch（按文档为可选块）
    ks = cfg.get("kill_switch", {})
    for level in ("L1", "L2"):
        if level in ks:
            p = ks[level]
            if "margin_net_repay_yi" in p:
                assert isinstance(p["margin_net_repay_yi"], (int, float))
            if "limitup_down_ratio_min" in p:
                ldr = p["limitup_down_ratio_min"]
                assert 0 <= ldr <= 1, f"{level}.limitup_down_ratio_min must be in [0,1]"
            if "turnover_concentration_max" in p:
                tcm = p["turnover_concentration_max"]
                assert 0 <= tcm <= 1, f"{level}.turnover_concentration_max must be in [0,1]"
            assert 0 <= p["vol_percentile_min"] <= 1, f"{level}.vol_percentile_min must be in [0,1]"

    # 执行窗口：兼容两种配置
    exec_times = []
    if "execution" in cfg and "windows" in cfg["execution"]:
        exec_times = list(cfg["execution"]["windows"])
    else:
        for w in cfg.get("runtime", {}).get("exec_windows", []):
            assert "time" in w, "exec_windows item missing 'time'"
            exec_times.append(w["time"])

    assert exec_times, "No execution windows configured"
    for t in exec_times:
        dt.datetime.strptime(t, "%H:%M")

def load_config(config_dir: str = "config") -> dict:
    """
    加载顺序：
      1) base.yaml
      2) strategy_core.yaml
      3) <environment>.override.yaml
    然后：单位归一 + 基本校验
    """
    cdir = Path(config_dir)
    base = _load_yaml(cdir / "base.yaml")
    core = _load_yaml(cdir / "strategy_core.yaml")
    cfg  = deep_merge(base, core)

    env = (cfg.get("environment") or "dev").lower()
    ovr = cdir / f"{env}.override.yaml"
    if ovr.exists():
        cfg = deep_merge(cfg, _load_yaml(ovr))

    # 兼容字段：把旧 runtime.exec_windows 提取为 execution.windows（若后者缺失）
    if "execution" not in cfg:
        cfg["execution"] = {}
    if "windows" not in cfg["execution"]:
        legacy = []
        for w in cfg.get("runtime", {}).get("exec_windows", []):
            if "time" in w:
                legacy.append(w["time"])
        if legacy:
            cfg["execution"]["windows"] = legacy

    _normalize_aliases(cfg)
    _normalize_units(cfg)
    validate_config(cfg)
    return cfg

