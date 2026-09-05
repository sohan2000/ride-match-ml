from pathlib import Path

from src.models.train_model import train_and_evaluate
from src.models.evaluate import evaluate_ranking_policy
from src.models.train_utility_model import train_utility_model
from src.models.tune_utility import tune_utility_weights
from src.models.policy_eval import compare_policies
from src.models.feature_importance import permutation_feature_importance
from src.simulator.generate_data import generate_dataset


def test_training_saves_gbdt_artifact(tmp_path: Path):
    input_path = tmp_path / "candidates.csv"
    output_path = tmp_path / "model.pkl"
    generate_dataset(num_drivers=8, num_riders=80, steps=40, seed=11).to_csv(
        input_path, index=False
    )

    metrics = train_and_evaluate(str(input_path), str(output_path))

    assert output_path.exists()
    assert "roc_auc" in metrics
    assert metrics["positive_rate"] > 0

    ranking = evaluate_ranking_policy(str(output_path), str(input_path))
    assert ranking["requests_with_candidates"] > 0
    assert "eta_delta_minutes" in ranking


def test_utility_training_saves_regressor_artifact(tmp_path: Path):
    input_path = tmp_path / "candidates.csv"
    output_path = tmp_path / "utility_model.pkl"
    generate_dataset(num_drivers=8, num_riders=80, steps=40, seed=11).to_csv(
        input_path, index=False
    )

    metrics = train_utility_model(str(input_path), str(output_path))

    assert output_path.exists()
    assert metrics["validation_mae"] >= 0


def test_utility_tuning_returns_scenario_results():
    results = tune_utility_weights(
        seeds=(11,), driver_counts=(4,), rider_count=20, steps=20,
        idle_weights=(0.0,), cancellation_weights=(4.0,),
    )

    assert len(results) == 1
    assert "eta_delta_minutes" in results.columns


def test_policy_replay_returns_actual_operational_metrics(tmp_path: Path):
    input_path = tmp_path / "candidates.csv"
    output_path = tmp_path / "utility_model.pkl"
    generate_dataset(num_drivers=6, num_riders=40, steps=30, seed=11).to_csv(input_path, index=False)
    train_utility_model(str(input_path), str(output_path))

    results = compare_policies(str(output_path), num_drivers=6, num_riders=40, steps=30, seed=11)

    assert set(results["policy"]) == {"nearest", "model"}
    assert (results["utilization"] >= 0).all()
    assert {"driver_rejection_rate", "rider_cancellation_rate"}.issubset(results.columns)
    assert "policy_kpi_score" in results.columns


def test_permutation_importance_returns_all_online_features(tmp_path: Path):
    input_path = tmp_path / "candidates.csv"
    output_path = tmp_path / "utility_model.pkl"
    generate_dataset(num_drivers=8, num_riders=80, steps=40, seed=11).to_csv(input_path, index=False)
    train_utility_model(str(input_path), str(output_path))

    importance = permutation_feature_importance(str(output_path), str(input_path), repeats=2)

    assert len(importance) == 12
    assert set(importance["feature"]) == set([
        "distance_km", "eta_minutes", "driver_idle_minutes", "available_drivers",
        "open_requests", "demand_supply_ratio", "hour_of_day", "is_peak",
        "driver_acceptance_rate",
        "same_pickup_zone", "pickup_zone_supply", "pickup_zone_demand_supply_ratio",
    ])