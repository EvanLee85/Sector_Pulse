# strategy_engine/strategies/sector_pulse_v3.py
# -*- coding: utf-8 -*-
"""
SectorPulse v3 ｜步骤5：选板→选股漏斗（在线+离线注入两用）
- 行业强度（申万 L1）：score = 0.70 * mean(rs_adj) + 0.30 * breadth(rs_adj>=rs_min)
- 板内个股 RS：rs_adj = 0.60 * rk(ret_20d) + 0.25 * rk(amount_10d) + 0.15 * share_amt
- 严格命中 3–5 名：越界直接拒绝（rank_out_of_range）
- 约束：10日均额 ∈ [amount_10d_min, amount_10d_max]，rs_adj ≥ rs_min
- 产出：批量写入 Signal；落 event: funnel/result（含完整行业强度快照与被选明细/拒绝原因）
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from statistics import mean

# 依赖工程内已存在模块
from models.database import Signal, EventLog, get_session
from utils.scheduler import _load_funnel_cfg, _log_event, now_cn  # 复用你现有的时间/事件工具
from services.account_service import get_latest_position_ratio  # 只读 | 不强依赖
# 数据入口：使用 collector 中的两个新函数
from collector import fetch_sw_l1_components, fetch_stock_hist_basic  # 最小新增


# —— 轻量工具 —— #
def _normalize_symbol(sym: str) -> str:
    """极简归一：'600000'→'600000.SH'，'000001'→'000001.SZ'；已有后缀则原样返回。"""
    s = sym.strip().upper()
    if "." in s:
        return s
    if s and s[0] == "6":
        return f"{s}.SH"
    if s and s[0] in ("0", "3"):
        return f"{s}.SZ"
    return s


def _rank_0_1(values: List[float]) -> List[float]:
    """将序列转为 0..1 分位（越大越好）。空/常数列返回 0.5。"""
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if vmax - vmin < 1e-12:
        return [0.5] * len(values)
    return [(v - vmin) / (vmax - vmin) for v in values]


@dataclass
class StockScore:
    symbol: str
    rs_adj: float
    amount_10d: float
    rank_in_sector: int
    extras: Dict[str, Any]


# —— 在线：板内个股 RS 计算 —— #
def rank_stocks_in_sector_online(sector_code: str, components: List[str], cfg: Dict[str, Any]) -> List[StockScore]:
    rs_min: float = float(cfg["funnel"]["rs_min"])
    amount_min: float = float(cfg["funnel"]["amount_10d_min"])
    amount_max: float = float(cfg["funnel"]["amount_10d_max"])

    items: List[Tuple[str, float, float, float]] = []  # (symbol, ret20, amt10, share_amt)

    for raw in components:
        sym = _normalize_symbol(raw)
        df = fetch_stock_hist_basic(sym, cfg)  # 允许返回 None（降级）
        if df is None or df.empty:
            continue
        # 需要列：close、pct_chg（或自行计算）、amount（成交额）
        # 计算滚动：ret_20d = close / close.shift(20) - 1；amount_10d = amount.rolling(10).mean().iloc[-1]
        try:
            close = df["close"].astype("float64")
            amount = df["amount"].astype("float64")  # 单位元
        except KeyError:
            # 兼容不同字段名
            close = df.iloc[:, df.columns.str.contains("close", case=False)].iloc[:, 0].astype("float64")
            amount = df.iloc[:, df.columns.str.contains("amount", case=False)].iloc[:, 0].astype("float64")

        if len(close) < 61:  # 防御：数据太短
            continue
        ret20 = float(close.iloc[-1] / close.iloc[-21] - 1.0)
        amt10 = float(amount.rolling(10).mean().iloc[-1])

        # share_amt：日度金额/行业内最大金额的相对份额，作为“资强占比”代理
        # 需先收集全样本后再归一；此处先占位，二次遍历时归一
        items.append((sym, ret20, amt10, 0.0))

    if not items:
        return []

    # 二次处理：份额 share_amt 使用行业内 amount_10d 占比（0..1）
    amt_list = [amt for _, _, amt, _ in items]
    amt_share = _rank_0_1(amt_list)

    ret_list = [r for _, r, _, _ in items]
    rk_ret = _rank_0_1(ret_list)
    rk_amt = _rank_0_1(amt_list)

    scored: List[Tuple[str, float, float, float]] = []
    for i, (sym, _, amt10, _) in enumerate(items):
        rs_adj = 0.60 * rk_ret[i] + 0.25 * rk_amt[i] + 0.15 * amt_share[i]
        scored.append((sym, rs_adj, amt10, amt_share[i]))

    # 排序与过滤（仅用于在线环节，离线注入直接跳过）
    scored.sort(key=lambda x: x[1], reverse=True)
    out: List[StockScore] = []
    for rank, (sym, rs, amt10, share) in enumerate(scored, start=1):
        # 仅过滤“明显不合规”的，保留拒绝的原因在上层统一写 Signal
        out.append(StockScore(symbol=sym, rs_adj=rs, amount_10d=amt10, rank_in_sector=rank,
                              extras={"rs_min": rs_min, "amount_min": amount_min, "amount_max": amount_max,
                                      "share_amt": share}))
    return out


# —— 在线：行业强度快照 —— #
def compute_sector_strength_snapshot(prev_trade: str, window: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    ind_cfg = cfg.get("funnel", {}).get("industry", {}) or {}
    # 申万 L1 列表：从配置读取；为空则留空让上层自行处理（这里不强制罗列）
    l1_codes: List[str] = ind_cfg.get("l1_codes") or []
    rs_min: float = float(cfg["funnel"]["rs_min"])

    scores: List[Dict[str, Any]] = []
    for code in l1_codes:
        # 取成分
        comps = fetch_sw_l1_components(code, cfg) or []
        if not comps:
            continue
        # 板内个股排序（在线）
        stocks = rank_stocks_in_sector_online(code, comps, cfg)
        if not stocks:
            continue
        rs_vals = [s.rs_adj for s in stocks]
        br = sum(1 for v in rs_vals if v >= rs_min) / max(1, len(rs_vals))
        sc = 0.70 * mean(rs_vals) + 0.30 * br
        scores.append({"name": code, "score": sc, "parts": {"rs": mean(rs_vals), "breadth": br}, "n": len(rs_vals)})

    scores.sort(key=lambda x: x["score"], reverse=True)
    return {"scores": scores, "rank_range": [3, 5]}


# —— 入口：离线注入 / 在线实取 —— #
def select_candidates_funnel(
    state: str,
    ks_level: int,
    window: str,
    indicators_payload: Optional[Dict[str, Any]] = None,
    injected: Optional[Dict[str, Any]] = None,
    emit_event: bool = True,
) -> Dict[str, Any]:
    """
    - 离线注入：传入 injected={"sector_rankings_snapshot":..., "sector_to_stocks": {...}}
    - 在线模式：不传 injected，自动跑 申万行业→板内个股
    - 约束：严格 3–5 名；rs/amount 区间；超界拒绝
    - 落库：Signal 批量 + EventLog: funnel/result
    """
    cfg = _load_funnel_cfg()
    rs_min: float = float(cfg["funnel"]["rs_min"])
    amount_min: float = float(cfg["funnel"]["amount_10d_min"])
    amount_max: float = float(cfg["funnel"]["amount_10d_max"])

    if injected:
        sector_snapshot = injected.get("sector_rankings_snapshot") or {"scores": [], "rank_range": [3, 5]}
        sector_to_stocks: Dict[str, List[Dict[str, Any]]] = injected.get("sector_to_stocks") or {}
    else:
        # 在线：需要 prev_trade；与 Gate 同口径，近一日即可
        prev_trade = indicators_payload.get("prev_trade") if isinstance(indicators_payload, dict) else ""
        sector_snapshot = compute_sector_strength_snapshot(prev_trade, window, cfg)
        # 拉取 Top1-5 行业的板内个股打分明细，供 3–5 名筛选
        sector_to_stocks = {}
        for idx, item in enumerate(sector_snapshot["scores"][:5], start=1):
            code = item["name"]
            comps = fetch_sw_l1_components(code, cfg) or []
            if not comps:
                continue
            sector_to_stocks[code] = [
                ss.__dict__ for ss in rank_stocks_in_sector_online(code, comps, cfg)
            ]

    selected: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    # 强约束：只取行业榜的第 3–5 名
    rank_lo, rank_hi = sector_snapshot.get("rank_range", [3, 5])
    eligible_sectors = [it["name"] for i, it in enumerate(sector_snapshot.get("scores", []), start=1) if rank_lo <= i <= rank_hi]

    # 按每个行业的“板内个股排行”取第 3–5 名，逐一校验
    for sec in eligible_sectors:
        stocks = sector_to_stocks.get(sec) or []
        for st in stocks:
            # 只要板内 3–5 名
            r = int(st.get("rank_in_sector") or 0)
            if r not in (3, 4, 5):
                rejected.append({"symbol": st["symbol"], "sector": sec, "rank_in_sector": r, "reason_reject": "rank_out_of_range"})
                continue
            rs_adj = float(st["rs_adj"])
            amt10 = float(st["amount_10d"])

            if rs_adj < rs_min:
                rejected.append({"symbol": st["symbol"], "sector": sec, "reason_reject": "rs_below_min", "rs_adj": rs_adj, "rs_min": rs_min})
                continue
            if amt10 < amount_min:
                rejected.append({"symbol": st["symbol"], "sector": sec, "reason_reject": "amount_below_min", "amount_10d": amt10, "min": amount_min})
                continue
            if amount_max > 0 and amt10 > amount_max:
                rejected.append({"symbol": st["symbol"], "sector": sec, "reason_reject": "amount_above_max", "amount_10d": amt10, "max": amount_max})
                continue
            selected.append({"symbol": st["symbol"], "sector": sec, "rs_adj": rs_adj, "amount_10d": amt10})

    # —— 落库 —— #
    with get_session() as s:
        # 事件：完整快照（可溯源）
        ev = EventLog(level="INFO", code="funnel/result", payload={
            "window": window,
            "sector_rankings_snapshot": sector_snapshot,
            "selected_candidates": selected,
            "rejected": rejected,
            "note": "",
        })
        s.add(ev)
        s.commit()
        s.refresh(ev)

        # 批量写 Signal（通过/拒绝）—— 与你现有 16_test 的写法一致
        for it in selected:
            s.add(Signal(
                symbol=it["symbol"], side="buy", executable=True, reason_reject=None,
                state_at_emit=state, window=window,
                microstructure_checks={
                    "rs_adj": it["rs_adj"], "amount_10d": it["amount_10d"], "ctx_event_id": ev.id,
                }
            ))
        for it in rejected:
            s.add(Signal(
                symbol=it["symbol"], side="buy", executable=False, reason_reject=it.get("reason_reject"),
                state_at_emit=state, window=window,
                microstructure_checks={k: it[k] for k in it.keys() if k not in ("symbol", "sector", "reason_reject")}
            ))
        s.commit()

    if emit_event:
        _log_event("INFO", "funnel/emit_done", {"window": window, "selected": len(selected), "rejected": len(rejected)})

    return {"selected": selected, "rejected": rejected, "sector_rankings_snapshot": sector_snapshot}
