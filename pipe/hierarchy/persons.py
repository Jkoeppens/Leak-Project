# pipe/persons.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

def ensure_persons(cfg: dict) -> pd.DataFrame:
    p = cfg.get("paths", {}).get("persons")
    if not p:
        return pd.DataFrame()
    path = Path(p)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)