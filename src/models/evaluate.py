from __future__ import annotations

import argparse
import pickle

import pandas as pd

from src.features.build_features import LABEL_COLUMN, build_feature_table


def evaluate_model(model_path: str, input_path: str) -> dict:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(input_path)
    feature_df = build_feature_table(df)
    X = feature_df.drop(columns=[LABEL_COLUMN])
    y = feature_df[LABEL_COLUMN]

    preds = model.predict(X)
    scores = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else preds
    return {
        "num_rows": len(df),
        "positive_predictions": int(preds.sum()),
        "actual_positive_rate": float(y.mean()),
        "avg_predicted_score": float(scores.mean()),
    }


def compare_candidate_policies(input_path: str) -> dict:
    """Compare the generated nearest-driver baseline with a simple policy metric."""

    df = pd.read_csv(input_path)
    if "matched" not in df.columns or "realized_wait_minutes" not in df.columns:
        raise ValueError("Candidate input must contain matched and realized_wait_minutes")

    selected = df[df["matched"] == 1]
    return {
        "requests": int(df["request_id"].nunique()),
        "candidate_rows": int(len(df)),
        "matched_requests": int(len(selected)),
        "coverage": float(selected["request_id"].nunique() / df["request_id"].nunique())
        if len(df)
        else 0.0,
        "average_selected_wait_minutes": float(selected["realized_wait_minutes"].mean())
        if len(selected)
        else 0.0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained match model")
    parser.add_argument("--model", type=str, default="models/model.pkl")
    parser.add_argument("--input", type=str, default="data/processed/synthetic_matches.csv")
    args = parser.parse_args()

    print(evaluate_model(args.model, args.input))
