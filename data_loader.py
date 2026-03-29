"""
data_loader.py — Preprocessing pipeline for the subsidy dataset.
All column renaming, type coercion, and derived features live here.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(os.getenv("DATA_PATH", "data/subsidies_2025.xlsx"))

RAW_COLS = [
    "num", "date_received", "col2", "col3", "region", "akimat",
    "app_number", "livestock_direction", "subsidy_name",
    "status", "norm", "amount", "district",
]

BACKLOG_STATUS = "Одобрена"   # approved but not yet paid — our queue


@lru_cache(maxsize=1)
def _load_raw() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Place the Excel file at data/subsidies_2025.xlsx "
            "or set the DATA_PATH environment variable."
        )
    df = pd.read_excel(DATA_PATH, header=4)
    df.columns = RAW_COLS
    df["date_parsed"]  = pd.to_datetime(df["date_received"], dayfirst=True, errors="coerce")
    df["implied_head"] = (df["amount"] / df["norm"].replace(0, np.nan)).fillna(0)
    return df


def load_backlog() -> pd.DataFrame:
    """Return all applications with status = 'Одобрена' (frozen backlog)."""
    return _load_raw()[_load_raw()["status"] == BACKLOG_STATUS].copy()


def load_regional_summary() -> pd.DataFrame:
    """Aggregate budget status by region for the crisis dashboard."""
    df = _load_raw()
    return (
        df.groupby("region")
        .agg(
            paid_amount    = ("amount", lambda x: x[df.loc[x.index, "status"] == "Исполнена"].sum()),
            backlog_amount = ("amount", lambda x: x[df.loc[x.index, "status"] == "Одобрена"].sum()),
            rejected_amount= ("amount", lambda x: x[df.loc[x.index, "status"] == "Отклонена"].sum()),
            backlog_count  = ("status", lambda x: (x == "Одобрена").sum()),
            total_count    = ("status", "count"),
            rejection_rate = ("status", lambda x: (x == "Отклонена").mean()),
        )
        .reset_index()
    )


def load_monthly_series() -> pd.DataFrame:
    """Monthly application counts and amounts per region × livestock type."""
    df = _load_raw().copy()
    df["month"] = df["date_parsed"].dt.to_period("M")
    return (
        df.groupby(["month", "region", "livestock_direction"])
        .agg(app_count=("amount", "count"), total_amount=("amount", "sum"))
        .reset_index()
    )
