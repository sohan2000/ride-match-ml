from __future__ import annotations

import joblib
import pandas as pd
import argparse
from pathlib import Path

from src.simulator.generate_data import generate_outcomes


def policy_kpi_score(model: pd.Series, baseline: pd.Series) -> float:
    """Score a policy relative to baseline; higher is better."""

    return float(
        2.0 * (baseline["average_wait_minutes"] - model["average_wait_minutes"])
        + 1.5 * (model["coverage"] - baseline["coverage"])
        + 1.0 * (model["sla_hit_rate"] - baseline["sla_hit_rate"])
        + 1.0 * (baseline["cancellation_rate"] - model["cancellation_rate"])
        + 0.5 * (model["utilization"] - baseline["utilization"])
    )


def replay_policy(
    policy: str,
    model_path: str | None = None,
    num_drivers: int = 40,
    num_riders: int = 400,
    steps: int = 240,
    seed: int = 42,
    sla_constraint_minutes: float | None = 5.0,
) -> dict[str, float | int | str]:
    """Replay a policy against the same deterministic request stream."""

    model = None
    if policy == "model":
        if model_path is None:
            raise ValueError("model_path is required for model policy replay")
        artifact = joblib.load(model_path)
        model = artifact["model"] if isinstance(artifact, dict) else artifact

    outcomes = generate_outcomes(
        num_drivers=num_drivers,
        num_riders=num_riders,
        steps=steps,
        seed=seed,
        policy=policy,
        model=model,
        sla_constraint_minutes=sla_constraint_minutes if policy == "model" else None,
    )
    matched = outcomes[outcomes["driver_id"].notna()]
    capacity = max(1, num_drivers * steps)
    return {
        "policy": policy,
        "requests": int(len(outcomes)),
        "matched": int(len(matched)),
        "coverage": float(len(matched) / len(outcomes)) if len(outcomes) else 0.0,
        "average_wait_minutes": float(outcomes["wait_minutes"].mean()) if len(outcomes) else 0.0,
        "p90_wait_minutes": float(outcomes["wait_minutes"].quantile(0.9)) if len(outcomes) else 0.0,
        "sla_hit_rate": float((outcomes["wait_minutes"] <= 5.0).mean()) if len(outcomes) else 0.0,
        "driver_rejection_rate": float((~outcomes["driver_accepted"]).mean()) if len(outcomes) else 0.0,
        "rider_cancellation_rate": float(outcomes["rider_cancelled"].mean()) if len(outcomes) else 0.0,
        "cancellation_rate": float(outcomes["cancelled"].mean()) if len(outcomes) else 0.0,
        "utilization": float(matched["ride_minutes"].sum() / capacity) if len(matched) else 0.0,
    }


def compare_policies(model_path: str, **simulation_kwargs: int) -> pd.DataFrame:
    results = pd.DataFrame([
        replay_policy("nearest", **simulation_kwargs),
        replay_policy("model", model_path=model_path, **simulation_kwargs),
    ])
    baseline = results[results["policy"] == "nearest"].iloc[0]
    model = results[results["policy"] == "model"].iloc[0]
    results["policy_kpi_score"] = 0.0
    results.loc[results["policy"] == "model", "policy_kpi_score"] = policy_kpi_score(model, baseline)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay RideMatch policies and report fulfillment KPIs")
    parser.add_argument("--model", required=True, help="Path to a trained joblib model artifact")
    parser.add_argument("--drivers", type=int, default=40)
    parser.add_argument("--riders", type=int, default=600)
    parser.add_argument("--steps", type=int, default=1440)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", help="Optional CSV path for the comparison table")
    args = parser.parse_args()
    results = compare_policies(
        args.model,
        num_drivers=args.drivers,
        num_riders=args.riders,
        steps=args.steps,
        seed=args.seed,
    )
    print(results.to_string(index=False))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output, index=False)


if __name__ == "__main__":
    main()