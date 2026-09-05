from __future__ import annotations

import argparse

import joblib
import pandas as pd

from src.features.build_features import LABEL_COLUMN, build_feature_table
from src.features.build_features import ONLINE_FEATURE_COLUMNS


def _positive_scores(model: object, feature_data: pd.DataFrame) -> pd.Series:
    values = feature_data[ONLINE_FEATURE_COLUMNS]
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(values)
        classes = list(getattr(model, "classes_", [0, 1]))
        positive_index = classes.index(1) if 1 in classes else 0
        return pd.Series(probabilities[:, positive_index], index=feature_data.index)
    return pd.Series(model.predict(values), index=feature_data.index, dtype=float)


def evaluate_ranking_policy(model_path: str, input_path: str, sla_minutes: float = 5.0) -> dict:
    """Replay model ranking against nearest-driver selection per request."""

    artifact = joblib.load(model_path)
    model = artifact["model"] if isinstance(artifact, dict) else artifact
    candidates = pd.read_csv(input_path)
    features = build_feature_table(candidates)
    candidates = candidates.assign(model_score=_positive_scores(model, features))
    selected = candidates.loc[candidates.groupby("request_id")["model_score"].idxmax()]
    baseline = candidates[candidates[LABEL_COLUMN] == 1]
    model_wait = selected["eta_minutes"]
    baseline_wait = baseline["eta_minutes"]
    horizon_minutes = max(1.0, candidates["event_time"].nunique())
    driver_capacity = max(1.0, float(candidates["available_drivers"].max()))

    def operational_metrics(rows: pd.DataFrame) -> tuple[float, float]:
        cancellation_proxy = float((rows["eta_minutes"] > sla_minutes).mean())
        utilization_proxy = float((rows["distance_km"] * 2.0).sum() / (horizon_minutes * driver_capacity))
        return cancellation_proxy, utilization_proxy

    model_cancellation, model_utilization = operational_metrics(selected)
    baseline_cancellation, baseline_utilization = operational_metrics(baseline)
    return {
        "requests_with_candidates": int(candidates["request_id"].nunique()),
        "model_ranking_accuracy": float(selected[LABEL_COLUMN].mean()),
        "model_average_eta_minutes": float(model_wait.mean()),
        "baseline_average_eta_minutes": float(baseline_wait.mean()),
        "eta_delta_minutes": float(model_wait.mean() - baseline_wait.mean()),
        "model_sla_hit_rate": float((model_wait <= sla_minutes).mean()),
        "baseline_sla_hit_rate": float((baseline_wait <= sla_minutes).mean()),
        "model_cancellation_proxy": model_cancellation,
        "baseline_cancellation_proxy": baseline_cancellation,
        "model_utilization_proxy": model_utilization,
        "baseline_utilization_proxy": baseline_utilization,
    }


def evaluate_model(model_path: str, input_path: str) -> dict:
    artifact = joblib.load(model_path)
    model = artifact["model"] if isinstance(artifact, dict) else artifact

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
    parser.add_argument("--ranking", action="store_true", help="Evaluate request-level policy ranking")
    args = parser.parse_args()

    result = evaluate_ranking_policy(args.model, args.input) if args.ranking else evaluate_model(args.model, args.input)
    print(result)
