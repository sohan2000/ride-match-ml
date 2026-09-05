from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from src.features.build_features import LABEL_COLUMN, build_feature_table


def train_and_evaluate(input_path: str, output_path: str) -> dict:
    df = pd.read_csv(input_path)
    ordered_df = df.assign(event_time=pd.to_datetime(df["event_time"])).sort_values("event_time")
    feature_df = build_feature_table(ordered_df)

    X = feature_df.drop(columns=[LABEL_COLUMN])
    y = feature_df[LABEL_COLUMN]

    split_index = max(1, min(len(X) - 1, int(len(X) * 0.8)))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=2,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(model, f)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a ride-match classifier")
    parser.add_argument("--input", type=str, default="data/processed/synthetic_matches.csv")
    parser.add_argument("--output", type=str, default="models/model.pkl")
    args = parser.parse_args()

    metrics = train_and_evaluate(args.input, args.output)
    print(metrics)
