from __future__ import annotations

import joblib
import pandas as pd

from src.simulator.generate_data import generate_outcomes


def replay_policy(
    policy: str,
    model_path: str | None = None,
    num_drivers: int = 40,
    num_riders: int = 400,
    steps: int = 240,
    seed: int = 42,
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
        "cancellation_rate": float(outcomes["cancelled"].mean()) if len(outcomes) else 0.0,
        "utilization": float(matched["ride_minutes"].sum() / capacity) if len(matched) else 0.0,
    }


def compare_policies(model_path: str, **simulation_kwargs: int) -> pd.DataFrame:
    return pd.DataFrame([
        replay_policy("nearest", **simulation_kwargs),
        replay_policy("model", model_path=model_path, **simulation_kwargs),
    ])