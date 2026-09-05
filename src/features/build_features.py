from __future__ import annotations

import pandas as pd


ONLINE_FEATURE_COLUMNS = [
    "distance_km",
    "eta_minutes",
    "driver_idle_minutes",
    "available_drivers",
    "open_requests",
    "demand_supply_ratio",
    "hour_of_day",
    "is_peak",
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
    features["demand_supply_ratio"] = features["open_requests"] / features[
        "available_drivers"
    ].clip(lower=1)
    features["is_peak"] = (
        features["hour_of_day"].between(7, 9)
        | features["hour_of_day"].between(17, 19)
    ).astype(int)

    return features[ONLINE_FEATURE_COLUMNS + [LABEL_COLUMN]]
