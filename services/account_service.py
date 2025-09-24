# services/account_service.py
from typing import Optional
from sqlalchemy.orm import Session
from models.database import get_session, Account  # 保持与现有 models 对齐
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def get_latest_position_ratio(session: Optional[Session] = None) -> float:
    """
    读取最近一条 Account.position_ratio；若库中暂无记录，则返回 0.0 并写 WARN。
    仅做“读”，不做任何写入操作。
    """
    owns_sess = False
    s = session
    if s is None:
        s = get_session()
        owns_sess = True
    try:
        row = s.query(Account).order_by(Account.ts.desc()).first()
        if not row or row.position_ratio is None:
            logger.warning("account_service.get_latest_position_ratio: no record, fallback 0.0")
            return 0.0
        return float(row.position_ratio)
    finally:
        if owns_sess:
            s.close()
