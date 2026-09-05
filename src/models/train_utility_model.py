from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from src.features.build_features import ONLINE_FEATURE_COLUMNS, build_feature_table


TARGET_COLUMN = "candidate_utility"


def train_utility_model(input_path: str, output_path: str) -> dict[str, float]:
    data = pd.read_csv(input_path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Input CSV must contain {TARGET_COLUMN}")

    ordered = data.assign(event_time=pd.to_datetime(data["event_time"])).sort_values("event_time")
    ordered_features = build_feature_table(ordered)
    target = ordered[TARGET_COLUMN].astype(float)
    split_index = max(1, min(len(ordered) - 1, int(len(ordered) * 0.8)))
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )
    model.fit(ordered_features[ONLINE_FEATURE_COLUMNS].iloc[:split_index], target.iloc[:split_index])
    predictions = model.predict(ordered_features[ONLINE_FEATURE_COLUMNS].iloc[split_index:])
    metrics = {"validation_mae": float(mean_absolute_error(target.iloc[split_index:], predictions))}

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": ONLINE_FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "model_type": "gradient_boosting_utility_regressor",
        },
        output,
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a RideMatch utility GBDT")
    parser.add_argument("--input", default="data/processed/synthetic_matches.csv")
    parser.add_argument("--output", default="models/utility_model.pkl")
    args = parser.parse_args()
    print(train_utility_model(args.input, args.output))