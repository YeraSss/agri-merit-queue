"""
train.py — Train and save the LightGBM merit model
Run once before starting the API: python train.py

Outputs to ./artifacts/:
    lgbm_merit_model.txt   LightGBM booster file
    lookup_tables.json     Precomputed statistics for inference
"""

from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

DATA_PATH    = Path(os.getenv("DATA_PATH", "data/subsidies_2025.xlsx"))
ARTIFACT_DIR = Path("artifacts")

FEATURES = [
    "region_enc", "livestock_enc", "log_amount", "log_implied_head",
    "amount_zscore_by_livestock", "amount_zscore_by_region", "region_rej_rate",
    "livestock_rej_rate", "region_paid_rate", "amount_vs_region_median",
    "farm_size_tier", "month", "quarter", "day_of_year", "norm",
]

LGBM_PARAMS = dict(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    num_leaves=31, min_child_samples=20,
    class_weight="balanced", random_state=42, verbose=-1, n_jobs=-1,
)


def load_and_engineer(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, header=4)
    df.columns = [
        "num", "date_received", "col2", "col3", "region", "akimat",
        "app_number", "livestock_direction", "subsidy_name", "status",
        "norm", "amount", "district",
    ]
    df["date_parsed"]  = pd.to_datetime(df["date_received"], dayfirst=True, errors="coerce")
    df["implied_head"] = (df["amount"] / df["norm"].replace(0, np.nan)).fillna(0)
    df["month"]        = df["date_parsed"].dt.month
    df["quarter"]      = df["date_parsed"].dt.quarter
    df["day_of_year"]  = df["date_parsed"].dt.dayofyear
    df["log_amount"]   = np.log1p(df["amount"])
    df["log_implied_head"] = np.log1p(df["implied_head"])

    df["amount_zscore_by_livestock"] = df.groupby("livestock_direction")["amount"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )
    df["amount_zscore_by_region"] = df.groupby("region")["amount"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )

    region_rej    = df.groupby("region")["status"].apply(lambda x: (x == "Отклонена").mean())
    livestock_rej = df.groupby("livestock_direction")["status"].apply(lambda x: (x == "Отклонена").mean())
    region_paid   = df.groupby("region")["status"].apply(lambda x: (x == "Исполнена").mean())
    region_med    = df.groupby("region")["amount"].median()

    df["region_rej_rate"]       = df["region"].map(region_rej)
    df["livestock_rej_rate"]    = df["livestock_direction"].map(livestock_rej)
    df["region_paid_rate"]      = df["region"].map(region_paid)
    df["region_median_amount"]  = df["region"].map(region_med)
    df["amount_vs_region_median"] = df["amount"] / (df["region_median_amount"] + 1)
    df["farm_size_tier"] = pd.cut(
        df["implied_head"],
        bins=[0, 30, 100, 500, 2000, float("inf")],
        labels=[1, 2, 3, 4, 5],
    ).astype(float).fillna(1)

    le_region    = LabelEncoder()
    le_livestock = LabelEncoder()
    df["region_enc"]    = le_region.fit_transform(df["region"].fillna("unknown"))
    df["livestock_enc"] = le_livestock.fit_transform(df["livestock_direction"].fillna("unknown"))

    livestock_stats = df.groupby("livestock_direction")["amount"].agg(["mean", "std"])
    region_stats    = df.groupby("region")["amount"].agg(["mean", "std"])
    livestock_budget_share = (
        df[df["status"] == "Исполнена"]
        .groupby("livestock_direction")["amount"]
        .sum()
        .pipe(lambda s: s / s.sum() * 100)
    )

    df.attrs["lookup"] = {
        "region_rej_rates":           region_rej.to_dict(),
        "region_paid_rates":          region_paid.to_dict(),
        "region_medians":             region_med.to_dict(),
        "livestock_rej_rates":        livestock_rej.to_dict(),
        "livestock_budget_shares":    livestock_budget_share.to_dict(),
        "region_label_classes":       list(le_region.classes_),
        "livestock_label_classes":    list(le_livestock.classes_),
        "max_livestock_share":        float(livestock_budget_share.max()),
        "livestock_amount_means":     livestock_stats["mean"].to_dict(),
        "livestock_amount_stds":      livestock_stats["std"].fillna(1).to_dict(),
        "region_amount_means":        region_stats["mean"].to_dict(),
        "region_amount_stds":         region_stats["std"].fillna(1).to_dict(),
        "features":                   FEATURES,
    }
    return df


def train(df: pd.DataFrame) -> "lgb.LGBMClassifier":  # type: ignore
    import lightgbm as lgb

    df_model = df[df["status"].isin(["Исполнена", "Отклонена", "Одобрена", "Сформировано поручение"])].copy()
    df_model["target"] = (df_model["status"] == "Отклонена").astype(int)
    df_model = df_model.dropna(subset=FEATURES + ["target"])

    X, y = df_model[FEATURES], df_model["target"].values

    print(f"Training set: {len(df_model):,} rows  |  rejection rate: {y.mean()*100:.1f}%")

    model = lgb.LGBMClassifier(**LGBM_PARAMS)

    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs  = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"5-fold AUC: {aucs.mean():.4f} ± {aucs.std():.4f}")

    model.fit(X, y)
    return model


def save_artifacts(model, lookup: dict) -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    model.booster_.save_model(str(ARTIFACT_DIR / "lgbm_merit_model.txt"))
    with open(ARTIFACT_DIR / "lookup_tables.json", "w", encoding="utf-8") as f:
        json.dump(lookup, f, ensure_ascii=False, indent=2)
    print(f"Artifacts saved to {ARTIFACT_DIR}/")


if __name__ == "__main__":
    print(f"Loading data from {DATA_PATH} ...")
    df     = load_and_engineer(DATA_PATH)
    lookup = df.attrs["lookup"]
    model  = train(df)
    save_artifacts(model, lookup)
    print("Done. You can now start the API: uvicorn main:app --reload")
