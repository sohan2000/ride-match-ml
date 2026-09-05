from __future__ import annotations

from itertools import product
from tempfile import TemporaryDirectory
from pathlib import Path

import pandas as pd

from src.models.evaluate import evaluate_ranking_policy
from src.models.train_utility_model import train_utility_model
from src.simulator.generate_data import generate_dataset


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
            report = evaluate_ranking_policy(str(model_path), str(data_path))
            results.append(
                {
                    "seed": seed,
                    "drivers": drivers,
                    "idle_weight": idle_weight,
                    "cancellation_weight": cancellation_weight,
                    **report,
                }
            )
    return pd.DataFrame(results)