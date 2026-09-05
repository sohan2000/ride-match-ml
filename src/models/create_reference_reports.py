from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.models.policy_eval import compare_policies


def create_reports(model_path: str, output_dir: str, seeds: tuple[int, ...] = (7, 42, 101)) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        rows.extend(
            compare_policies(
                model_path,
                num_drivers=40,
                num_riders=600,
                steps=1440,
                seed=seed,
            ).to_dict("records")
        )
    results = pd.DataFrame(rows)
    results.to_csv(output / "policy_replay_by_seed.csv", index=False)
    results.groupby("policy").mean(numeric_only=True).reset_index().to_csv(
        output / "policy_replay_summary.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create reproducible RideMatch reference reports")
    parser.add_argument("--model", default="models/utility_model.pkl")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()
    create_reports(args.model, args.output_dir)