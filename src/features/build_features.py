from __future__ import annotations

import pandas as pd


ONLINE_FEATURE_COLUMNS = [
    "distance_km",
    "eta_minutes",
    "driver_idle_minutes",
    "available_drivers",
    "open_requests",
    "hour_of_day",
    "distance_bin",
    "eta_per_idle_minute",
    "supply_demand_ratio",
]
LABEL_COLUMN = "matched"


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "distance_km",
        "eta_minutes",
        "driver_idle_minutes",
        "available_drivers",
        "open_requests",
        "hour_of_day",
        LABEL_COLUMN,
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Candidate table is missing required columns: {missing}")

    features = df.copy()
    features["distance_bin"] = pd.cut(
        features["distance_km"],
        bins=[0, 2, 4, 6, 10, float("inf")],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True,
    ).astype(float)
    features["eta_per_idle_minute"] = features["eta_minutes"] / (
        features["driver_idle_minutes"] + 1.0
    )
    features["supply_demand_ratio"] = features["available_drivers"] / (
        features["open_requests"] + 1.0
    )

    return features[ONLINE_FEATURE_COLUMNS + [LABEL_COLUMN]]
