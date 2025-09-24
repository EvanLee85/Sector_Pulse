# api/main.py
# -*- coding: utf-8 -*-
from fastapi import FastAPI
from .dependencies import get_db
from .routers import singals as signals_router  # 按你的目录命名

app = FastAPI(title="SectorPulse API", version="0.1.0")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# 路由注册
app.include_router(signals_router.router, prefix="/signals", tags=["signals"])
