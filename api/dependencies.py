# api/dependencies.py
# -*- coding: utf-8 -*-
from typing import Generator
from models.database import get_session

def get_db() -> Generator:
    with get_session() as s:
        yield s
