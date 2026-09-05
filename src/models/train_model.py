from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from src.features.build_features import LABEL_COLUMN, build_feature_table
from src.features.build_features import ONLINE_FEATURE_COLUMNS


def train_and_evaluate(input_path: str, output_path: str) -> dict:
    df = pd.read_csv(input_path)
    ordered_df = df.assign(event_time=pd.to_datetime(df["event_time"])).sort_values("event_time")
    feature_df = build_feature_table(ordered_df)

    X = feature_df[ONLINE_FEATURE_COLUMNS]
    y = feature_df[LABEL_COLUMN].astype(int)

    split_index = max(1, min(len(X) - 1, int(len(X) * 0.8)))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    y_pred = (probabilities >= 0.5).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y_test, probabilities),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "positive_rate": float(y.mean()),
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": ONLINE_FEATURE_COLUMNS,
            "target_column": LABEL_COLUMN,
            "model_type": "gradient_boosting_classifier",
        },
        output,
    )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a ride-match classifier")
    parser.add_argument("--input", type=str, default="data/processed/synthetic_matches.csv")
    parser.add_argument("--output", type=str, default="models/model.pkl")
    args = parser.parse_args()

    metrics = train_and_evaluate(args.input, args.output)
    print(metrics)
