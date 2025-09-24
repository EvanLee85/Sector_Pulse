# api/routers/singals.py
# -*- coding: utf-8 -*-
from fastapi import APIRouter, Depends
from sqlmodel import select
from models.database import Signal, EventLog
from api.dependencies import get_db

router = APIRouter()

@router.get("/latest")
def latest_signals(db=Depends(get_db)):
    rows = db.exec(select(Signal).order_by(Signal.ts.desc()).limit(50)).all()
    return {"count": len(rows), "items": [r.dict() for r in rows]}

@router.get("/last-funnel-event")
def last_funnel_event(db=Depends(get_db)):
    ev = db.exec(select(EventLog).where(EventLog.code == "funnel/result").order_by(EventLog.id.desc()).limit(1)).first()
    return ev.dict() if ev else {}
