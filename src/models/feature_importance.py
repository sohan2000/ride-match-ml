from __future__ import annotations

import argparse

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

from src.features.build_features import LABEL_COLUMN, ONLINE_FEATURE_COLUMNS, build_feature_table


def permutation_feature_importance(
    model_path: str,
    input_path: str,
    repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Estimate held-out feature importance by shuffling one feature at a time."""

    artifact = joblib.load(model_path)
    model = artifact["model"] if isinstance(artifact, dict) else artifact
    data = pd.read_csv(input_path)
    ordered = data.assign(event_time=pd.to_datetime(data["event_time"])).sort_values("event_time")
    features = build_feature_table(ordered)
    split_index = max(1, min(len(features) - 1, int(len(features) * 0.8)))
    X_test = features[ONLINE_FEATURE_COLUMNS].iloc[split_index:]
    target_column = artifact.get("target_column", LABEL_COLUMN) if isinstance(artifact, dict) else LABEL_COLUMN
    if target_column not in ordered.columns:
        raise ValueError(f"Input CSV must contain artifact target column: {target_column}")
    y_test = ordered[target_column].iloc[split_index:]
    scoring = "neg_mean_absolute_error" if target_column != LABEL_COLUMN else "roc_auc"
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=repeats,
        random_state=random_state,
        scoring=scoring,
    )
    return pd.DataFrame({
        "feature": ONLINE_FEATURE_COLUMNS,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute held-out RideMatch feature importance")
    parser.add_argument("--model", default="models/utility_model.pkl")
    parser.add_argument("--input", default="data/processed/synthetic_matches.csv")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    print(permutation_feature_importance(args.model, args.input, repeats=args.repeats).to_string(index=False))