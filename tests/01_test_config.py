# test_config.py
# 轻量自检：加载并打印关键字段；失败直接抛出 AssertionError
from config.loader import load_config
import datetime as dt

def test_config_loading():
    cfg = load_config("config")

    # 基本信息
    print("Environment:", cfg.get("environment"))
    windows = cfg.get("execution", {}).get("windows")
    print("Exec windows:", windows)
    print("Stop loss:", cfg["position_management"]["stop_loss"])

    # Alt-Flow 情绪闸门（取代北向）
    eg = cfg.get("emotion_gate", {})
    gate_info = {
        "limit_up_count_min": eg.get("limit_up_count_min"),
        "limitup_down_ratio_min": eg.get("limitup_down_ratio_min"),
        "turnover_concentration_max": eg.get("turnover_concentration_max"),
        "vol_percentile_max": eg.get("vol_percentile_max"),
        "sh_ma20_filter": eg.get("sh_ma20_filter"),
    }
    print("Alt-Flow gate:", gate_info)

    # Kill-Switch L1（示例打印，不强制要求全部存在）
    ks1 = cfg.get("kill_switch", {}).get("L1", {})
    ks1_info = {
        "margin_net_repay_yi": ks1.get("margin_net_repay_yi"),
        "limitup_down_ratio_min": ks1.get("limitup_down_ratio_min"),
        "turnover_concentration_max": ks1.get("turnover_concentration_max"),
        "vol_percentile_min": ks1.get("vol_percentile_min"),
        "action": ks1.get("action"),
        "temp_total_cap": ks1.get("temp_total_cap"),
    }
    print("KS L1:", ks1_info)

    print("✅ Config loaded & validated.")


def test_redlines():
    cfg = load_config("config")

    # 1) 状态阈值基本关系
    band = cfg["state"]["pos_band"]
    assert band["enter_offense"] < band["exit_offense"], \
        "state.pos_band: enter_offense 应小于 exit_offense"

    # 2) 仓位上限关系
    caps = cfg["state"]["caps"]
    assert 0 < caps["initial_entry_fraction"] <= caps["max_single_position"] <= caps["max_total_position"] <= 1, \
        "state.caps: 0 < initial_entry_fraction <= max_single_position <= max_total_position <= 1"

    # 3) 执行窗口（主口径：字符串列表），且必须包含 10:00 与 14:00
    windows = cfg["execution"]["windows"]
    assert isinstance(windows, list) and all(isinstance(x, str) for x in windows), \
        "execution.windows 必须为 List[str]"
    for t in windows:
        dt.datetime.strptime(t, "%H:%M")  # 校验 HH:MM 格式
    assert {"10:00", "14:00"}.issubset(set(windows)), \
        "execution.windows 必须至少包含 {'10:00','14:00'}"

    # 4) 情绪闸门（Alt-Flow）关键字段与取值范围
    eg = cfg["emotion_gate"]
    assert "limit_up_count_min" in eg, "emotion_gate.limit_up_count_min 缺失"
    assert "vol_percentile_max" in eg, "emotion_gate.vol_percentile_max 缺失"
    vpm = eg["vol_percentile_max"]
    assert 0 <= vpm <= 1, "emotion_gate.vol_percentile_max 必须在 [0,1]"
    if "limitup_down_ratio_min" in eg:
        ldr = eg["limitup_down_ratio_min"]
        assert 0 <= ldr <= 1, "emotion_gate.limitup_down_ratio_min 必须在 [0,1]"
    if "turnover_concentration_max" in eg:
        tcm = eg["turnover_concentration_max"]
        assert 0 <= tcm <= 1, "emotion_gate.turnover_concentration_max 必须在 [0,1]"

    # 5) Kill-Switch（若配置则校验范围）
    ks = cfg.get("kill_switch", {})
    for level in ("L1", "L2"):
        if level in ks:
            p = ks[level]
            if "margin_net_repay_yi" in p:
                assert isinstance(p["margin_net_repay_yi"], (int, float)), f"{level}.margin_net_repay_yi 必须为数值"
            if "limitup_down_ratio_min" in p:
                ldr = p["limitup_down_ratio_min"]
                assert 0 <= ldr <= 1, f"{level}.limitup_down_ratio_min 必须在 [0,1]"
            if "turnover_concentration_max" in p:
                tcm = p["turnover_concentration_max"]
                assert 0 <= tcm <= 1, f"{level}.turnover_concentration_max 必须在 [0,1]"
            if "vol_percentile_min" in p:
                vpmn = p["vol_percentile_min"]
                assert 0 <= vpmn <= 1, f"{level}.vol_percentile_min 必须在 [0,1]"

    print("✅ Redlines OK.")

if __name__ == "__main__":
    test_config_loading()
    test_redlines()
