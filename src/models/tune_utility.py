from __future__ import annotations

from itertools import product
from tempfile import TemporaryDirectory
from pathlib import Path

import pandas as pd

from src.models.policy_eval import compare_policies
from src.models.train_utility_model import train_utility_model
from src.simulator.generate_data import generate_dataset


def _policy_kpi_score(model: pd.Series, baseline: pd.Series) -> float:
    return float(
        2.0 * (baseline["average_wait_minutes"] - model["average_wait_minutes"])
        + 1.5 * (model["coverage"] - baseline["coverage"])
        + (model["sla_hit_rate"] - baseline["sla_hit_rate"])
        + (baseline["cancellation_rate"] - model["cancellation_rate"])
        + 0.5 * (model["utilization"] - baseline["utilization"])
    )


def tune_utility_weights(
    seeds: tuple[int, ...] = (7, 42, 101),
    driver_counts: tuple[int, ...] = (20, 40),
    rider_count: int = 400,
    steps: int = 240,
    idle_weights: tuple[float, ...] = (0.0, 0.05, 0.1),
    cancellation_weights: tuple[float, ...] = (4.0, 8.0, 12.0),
) -> pd.DataFrame:
    """Evaluate utility weights across seeds and supply-demand scenarios."""

    results = []
    with TemporaryDirectory() as directory:
        directory_path = Path(directory)
        for seed, drivers, idle_weight, cancellation_weight in product(
            seeds, driver_counts, idle_weights, cancellation_weights
        ):
            data_path = directory_path / "candidates.csv"
            model_path = directory_path / "utility_model.pkl"
            generate_dataset(
                num_drivers=drivers,
                num_riders=rider_count,
                steps=steps,
                seed=seed,
                idle_weight=idle_weight,
                cancellation_weight=cancellation_weight,
            ).to_csv(data_path, index=False)
            train_utility_model(str(data_path), str(model_path))
            policy_results = compare_policies(
                str(model_path),
                num_drivers=drivers,
                num_riders=rider_count,
                steps=steps,
                seed=seed,
            )
            nearest = policy_results[policy_results["policy"] == "nearest"].iloc[0]
            model = policy_results[policy_results["policy"] == "model"].iloc[0]
            results.append(
                {
                    "seed": seed,
                    "drivers": drivers,
                    "idle_weight": idle_weight,
                    "cancellation_weight": cancellation_weight,
                    "eta_delta_minutes": float(
                        model["average_wait_minutes"] - nearest["average_wait_minutes"]
                    ),
                    "coverage_delta": float(model["coverage"] - nearest["coverage"]),
                    "sla_delta": float(model["sla_hit_rate"] - nearest["sla_hit_rate"]),
                    "cancellation_delta": float(
                        model["cancellation_rate"] - nearest["cancellation_rate"]
                    ),
                    "utilization_delta": float(model["utilization"] - nearest["utilization"]),
                    "policy_kpi_score": _policy_kpi_score(model, nearest),
                    "model_average_wait_minutes": float(model["average_wait_minutes"]),
                    "baseline_average_wait_minutes": float(nearest["average_wait_minutes"]),
                }
            )
    return pd.DataFrame(results)