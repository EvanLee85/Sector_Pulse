# models/database.py
# SectorPulse：ORM 数据模型（7表）+ 引擎/会话
# 修正点：
# 1) 统一时区：使用 config.runtime.tz（默认 Asia/Shanghai）写入时间戳
# 2) 补齐字段：Position（holding_days/sector/SL/TP）、Signal（confidence_score/microstructure_checks）
# 3) 配置集成：从 config.loader 读取环境，开发环境 echo SQL；保留 env/config DB_URL 支持
# 4) 类型安全：关键字段使用枚举（SystemState/OrderStatus/LogLevel）

# from __future__ import annotations
from typing import Optional, Iterable, List
from contextlib import contextmanager
from datetime import datetime
import os
from functools import lru_cache
from enum import Enum
from zoneinfo import ZoneInfo

from sqlmodel import SQLModel, Field, Relationship, create_engine, Session

# ============== 枚举（提高类型安全） ==============
class SystemState(str, Enum):
    OFFENSE = "OFFENSE"
    HOLD = "HOLD"
    WATCH = "WATCH"
    SLEEP = "SLEEP"

class OrderStatus(str, Enum):
    PROPOSED  = "proposed"
    CONFIRMED = "confirmed"
    REJECTED  = "rejected"
    SENT      = "sent"
    FILLED    = "filled"
    CANCELED  = "canceled"

class LogLevel(str, Enum):
    INFO  = "INFO"
    WARN  = "WARN"
    ERROR = "ERROR"


# ============== 时区：与 config.runtime.tz 保持一致（默认 Asia/Shanghai） ==============
@lru_cache(maxsize=1)
def _get_tz_name() -> str:
    """
    时区解析优先级：
    1) 环境变量 SECTORPULSE_TZ
    2) 配置 execution.tz
    3) 默认 'Asia/Shanghai'
    """
    tz = os.getenv("SECTORPULSE_TZ")
    if tz:
        return tz
    try:
        from config.loader import load_config
        cfg = load_config("config")
        exc_tz = (cfg.get("execution") or {}).get("tz")
        return exc_tz or "Asia/Shanghai"
    except Exception:
        return "Asia/Shanghai"

def _now_local() -> datetime:
    return datetime.now(ZoneInfo(_get_tz_name()))


# ============== 基础三表 ==============
class Account(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    ts: datetime = Field(default_factory=_now_local, index=True)
    total_asset: float
    total_market_value: float
    available_cash: float
    position_ratio: float  # total_market_value / total_asset

class Position(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True)
    qty: int
    avg_price: float
    market_value: float
    pnl_ratio: float = 0.0
    open_ts: Optional[datetime] = None
    last_update_ts: datetime = Field(default_factory=_now_local, index=True)
    # —— 补充与策略口径对齐的字段 —— #
    holding_days: int = 0                     # 持仓天数（用于 T+3 / T+5 管理）
    sector: Optional[str] = None              # 所属板块（板块连续性检查）
    stop_loss_price: Optional[float] = None   # 止损价
    take_profit_1: Optional[float] = None     # 第一止盈价
    take_profit_2: Optional[float] = None     # 第二止盈价

class Transaction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    ts: datetime = Field(default_factory=_now_local, index=True)
    symbol: str = Field(index=True)
    side: str  # "buy" | "sell"
    qty: int
    price: float
    order_intent_id: Optional[int] = Field(default=None, foreign_key="orderintent.id")


# ============== 运行期四表 ==============
class Signal(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    ts: datetime = Field(default_factory=_now_local, index=True)
    symbol: str = Field(index=True)
    side: str                                  # "buy" | "sell"
    price_ref: Optional[float] = None
    state_at_emit: Optional[SystemState] = None  # OFFENSE | HOLD | WATCH | SLEEP
    emotion_gate_passed: Optional[bool] = None
    executable: bool = False
    reason_reject: Optional[str] = None        # intraday_missing / volratio<2 / ...
    window: Optional[str] = None               # "10:00" | "14:00" | "14:45"
    # —— 增强：记录微结构检查与置信度 —— #
    confidence_score: Optional[float] = None   # 买点确认置信度（0..1）
    microstructure_checks: Optional[str] = None  # JSON 字符串（检查项通过/失败明细）

    order_intents: List["OrderIntent"] = Relationship(back_populates="signal")

class OrderIntent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    signal_id: Optional[int] = Field(default=None, foreign_key="signal.id")
    action: str                                  # open | add | reduce | close
    qty: int
    price: Optional[float] = None
    status: OrderStatus = Field(default=OrderStatus.PROPOSED)
    idempotency_key: Optional[str] = Field(default=None, index=True, unique=True)
    risk_plan_json: Optional[str] = None

    signal: Optional["Signal"] = Relationship(back_populates="order_intents")

class StateSnapshot(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    ts: datetime = Field(default_factory=_now_local, index=True)
    state: SystemState                          # OFFENSE|HOLD|WATCH|SLEEP
    pos_ratio: float
    northbound: Optional[float] = None          # 元
    vol_pct: Optional[float] = None             # 0..1
    board_top3_json: Optional[str] = None       # JSON 字符串
    note: Optional[str] = None

class EventLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    ts: datetime = Field(default_factory=_now_local, index=True)
    level: LogLevel = Field(default=LogLevel.INFO)   # INFO | WARN | ERROR
    code: str                                        # state_change | reason_reject | risk_action | slippage
    payload_json: Optional[str] = None


# ============== 引擎与会话（集成配置） ==============
# DB_URL 优先级：环境变量 SECTORPULSE_DB_URL > config.settings.DB_URL > 默认 sqlite
def _resolve_db_url() -> str:
    env_url = os.getenv("SECTORPULSE_DB_URL")
    if env_url:
        return env_url
    try:
        from config.settings import DB_URL as CFG_URL  # 可选
        if CFG_URL:
            return CFG_URL
    except Exception:
        pass
    return "sqlite:///./sectorpulse.db"

def _engine_kwargs() -> dict:
    # 依据环境设置 echo / 连接池参数（sqlite 下忽略 pool_size）
    echo = False
    try:
        from config.loader import load_config
        cfg = load_config("config")
        env = (cfg.get("environment") or "dev").lower()
        echo = (env == "dev")
    except Exception:
        echo = False

    kwargs = {"echo": echo}
    db_url = _resolve_db_url()
    if db_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        # 非 sqlite 的简单池化参数（后续可在 settings 中细化）
        kwargs.update({"pool_pre_ping": True, "pool_recycle": 1800})
    return kwargs

DB_URL = _resolve_db_url()
engine = create_engine(DB_URL, **_engine_kwargs())

def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)

@contextmanager
def get_session() -> Iterable[Session]:
    with Session(engine) as session:
        yield session
