from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.simulator.city import City, Driver, Rider
from src.simulator.schema import AssignmentOutcome, CandidateMatchRecord
from src.features.build_features import LABEL_COLUMN, ONLINE_FEATURE_COLUMNS, build_feature_table


def _simulate_market(
    num_drivers: int,
    num_riders: int,
    steps: int,
    seed: int,
    idle_weight: float = 0.05,
    cancellation_weight: float = 12.0,
    policy: str = "nearest",
    model: object | None = None,
    sla_constraint_minutes: float | None = None,
) -> tuple[list[CandidateMatchRecord], list[AssignmentOutcome]]:
    rng = np.random.default_rng(seed)
    city = City(width=40, height=40)
    drivers = city.generate_drivers(num_drivers)
    busy_until = {driver.driver_id: 0 for driver in drivers}
    idle_since = {driver.driver_id: 0 for driver in drivers}
    attempts = {driver.driver_id: 0 for driver in drivers}
    acceptances = {driver.driver_id: 0 for driver in drivers}
    start_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    step_hours = np.array([(start_time.hour + step // 60) % 24 for step in range(steps)])
    demand_weights = np.where(
        ((step_hours >= 7) & (step_hours <= 9)) | ((step_hours >= 17) & (step_hours <= 19)),
        3.0,
        1.0,
    )
    requests_by_step = rng.multinomial(num_riders, demand_weights / demand_weights.sum())
    candidates: list[CandidateMatchRecord] = []
    outcomes: list[AssignmentOutcome] = []

    for step, request_count in enumerate(requests_by_step):
        for driver in drivers:
            if busy_until[driver.driver_id] <= step:
                driver.x = float(np.clip(driver.x + rng.normal(0.0, 0.8), 0.0, city.width))
                driver.y = float(np.clip(driver.y + rng.normal(0.0, 0.8), 0.0, city.height))

        for request_number in range(int(request_count)):
            request_id = f"req-{step:04d}-{request_number:02d}"
            rider = Rider(
                rider_id=f"r-{request_id}",
                x=float(rng.uniform(0, city.width)),
                y=float(rng.uniform(0, city.height)),
            )
            available = [
                driver
                for driver in drivers
                if busy_until[driver.driver_id] <= step
            ]
            open_requests = int(requests_by_step[step])
            event_time = start_time + timedelta(minutes=step)

            if not available:
                outcomes.append(
                    AssignmentOutcome(request_id, None, "cancelled", 10.0, None, True)
                )
                continue

            distances = {driver.driver_id: city.distance(driver, rider) for driver in available}
            pickup_zone = (int(rider.x // 5), int(rider.y // 5))
            pickup_zone_supply = sum(
                (int(driver.x // 5), int(driver.y // 5)) == pickup_zone
                for driver in available
            )
            if policy == "nearest":
                selected_driver = min(available, key=lambda driver: distances[driver.driver_id])
            elif policy == "model" and model is not None:
                rows = pd.DataFrame([{
                    "distance_km": distances[driver.driver_id],
                    "eta_minutes": distances[driver.driver_id] * 2.5,
                    "driver_idle_minutes": step - idle_since[driver.driver_id],
                    "available_drivers": len(available),
                    "open_requests": open_requests,
                    "hour_of_day": event_time.hour,
                    "driver_acceptance_rate": acceptances[driver.driver_id] / max(1, attempts[driver.driver_id]),
                    "same_pickup_zone": int((int(driver.x // 5), int(driver.y // 5)) == pickup_zone),
                    "pickup_zone_supply": pickup_zone_supply,
                    "pickup_zone_demand_supply_ratio": open_requests / max(1, pickup_zone_supply),
                    LABEL_COLUMN: 0,
                } for driver in available])
                features = build_feature_table(rows)[ONLINE_FEATURE_COLUMNS]
                scores = model.predict_proba(features)[:, 1] if hasattr(model, "predict_proba") else model.predict(features)
                eligible = np.arange(len(available))
                if sla_constraint_minutes is not None:
                    eligible = np.flatnonzero(rows["eta_minutes"].to_numpy() <= sla_constraint_minutes)
                if len(eligible):
                    selected_driver = available[int(eligible[np.argmax(scores[eligible])])]
                else:
                    selected_driver = available[int(np.argmax(scores))]
            else:
                raise ValueError("policy must be 'nearest' or 'model' with a model")
            selected_distance = distances[selected_driver.driver_id]
            selected_eta = selected_distance * 2.5
            ride_minutes = selected_distance * 2.0
            demand_pressure = open_requests / max(1, len(available))
            peak_hour = event_time.hour in {7, 8, 9, 17, 18, 19}
            driver_acceptance_probability = float(
                np.clip(
                    0.9
                    - 0.012 * selected_eta
                    + 0.0015 * (step - idle_since[selected_driver.driver_id])
                    - 0.04 * demand_pressure,
                    0.15,
                    0.95,
                )
            )
            rider_cancellation_probability = float(
                np.clip(
                    0.02
                    + 0.012 * max(0.0, selected_eta - 5.0)
                    + 0.02 * demand_pressure
                    + (0.03 if peak_hour else 0.0),
                    0.01,
                    0.9,
                )
            )
            driver_accepted = bool(rng.random() < driver_acceptance_probability)
            rider_cancelled = bool(
                driver_accepted and rng.random() < rider_cancellation_probability
            )
            assignment_cancelled = not driver_accepted or rider_cancelled

            for driver in available:
                distance = distances[driver.driver_id]
                eta_minutes = distance * 2.5
                ride_minutes = distance * 2.0
                cancellation_risk = max(0.0, (eta_minutes - 5.0) / 5.0)
                demand_pressure = open_requests / max(1, len(available))
                peak_friction = 1.0 if event_time.hour in {7, 8, 9, 17, 18, 19} else 0.0
                acceptance_probability = np.clip(
                    0.9 - 0.012 * eta_minutes + 0.0015 * (step - idle_since[driver.driver_id])
                    - 0.04 * demand_pressure,
                    0.15,
                    0.95,
                )
                rider_cancellation_probability = np.clip(
                    0.02 + 0.012 * cancellation_risk * 5.0 + 0.02 * demand_pressure
                    + 0.03 * peak_friction,
                    0.01,
                    0.9,
                )
                candidates.append(
                    CandidateMatchRecord(
                        event_time=event_time,
                        request_id=request_id,
                        driver_id=driver.driver_id,
                        distance_km=round(float(distance), 3),
                        eta_minutes=round(float(eta_minutes), 3),
                        driver_idle_minutes=round(float((step - idle_since[driver.driver_id])), 3),
                        available_drivers=len(available),
                        open_requests=open_requests,
                        hour_of_day=event_time.hour,
                        matched=int(driver.driver_id == selected_driver.driver_id),
                        realized_wait_minutes=round(float(eta_minutes), 3),
                        cancelled=int(driver.driver_id != selected_driver.driver_id),
                        candidate_utility=round(
                            float(
                                -eta_minutes
                                - 0.25 * ride_minutes
                                + idle_weight * (step - idle_since[driver.driver_id])
                                - cancellation_weight * cancellation_risk
                                + 2.0 * acceptance_probability
                                + 3.0 * (acceptances[driver.driver_id] / max(1, attempts[driver.driver_id]))
                                - cancellation_weight * rider_cancellation_probability
                                - 0.5 * demand_pressure
                                - 1.0 * peak_friction
                                + 1.5 * int((int(driver.x // 5), int(driver.y // 5)) == pickup_zone)
                                - 0.25 * (open_requests / max(1, pickup_zone_supply))
                            ),
                            3,
                        ),
                        driver_acceptance_rate=round(
                            float(acceptances[driver.driver_id] / max(1, attempts[driver.driver_id])),
                            3,
                        ),
                        same_pickup_zone=int((int(driver.x // 5), int(driver.y // 5)) == pickup_zone),
                        pickup_zone_supply=pickup_zone_supply,
                        pickup_zone_demand_supply_ratio=round(
                            float(open_requests / max(1, pickup_zone_supply)), 3
                        ),
                    )
                )

            attempts[selected_driver.driver_id] += 1
            if driver_accepted:
                acceptances[selected_driver.driver_id] += 1
            if not assignment_cancelled:
                busy_until[selected_driver.driver_id] = step + max(
                    1, int(np.ceil(selected_eta + ride_minutes))
                )
                idle_since[selected_driver.driver_id] = step
                selected_driver.x = rider.x
                selected_driver.y = rider.y
            outcomes.append(
                AssignmentOutcome(
                    request_id,
                    selected_driver.driver_id if not assignment_cancelled else None,
                    "cancelled" if assignment_cancelled else "matched",
                    round(float(selected_eta), 3) if driver_accepted else 10.0,
                    round(float(ride_minutes), 3) if not assignment_cancelled else None,
                    assignment_cancelled,
                    driver_accepted,
                    rider_cancelled,
                )
            )

    return candidates, outcomes


def generate_dataset(
    num_drivers: int = 40,
    num_riders: int = 40,
    seed: int = 42,
    steps: int = 60,
    idle_weight: float = 0.05,
    cancellation_weight: float = 12.0,
    policy: str = "nearest",
    model: object | None = None,
    sla_constraint_minutes: float | None = None,
) -> pd.DataFrame:
    candidates, _ = _simulate_market(num_drivers, num_riders, steps, seed, idle_weight, cancellation_weight, policy, model, sla_constraint_minutes)
    return pd.DataFrame([candidate.to_dict() for candidate in candidates])


def generate_outcomes(
    num_drivers: int = 40,
    num_riders: int = 40,
    seed: int = 42,
    steps: int = 60,
    idle_weight: float = 0.05,
    cancellation_weight: float = 12.0,
    policy: str = "nearest",
    model: object | None = None,
    sla_constraint_minutes: float | None = None,
) -> pd.DataFrame:
    _, outcomes = _simulate_market(num_drivers, num_riders, steps, seed, idle_weight, cancellation_weight, policy, model, sla_constraint_minutes)
    return pd.DataFrame([outcome.to_dict() for outcome in outcomes])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic ride-matching data")
    parser.add_argument("--output", type=str, default="data/processed/synthetic_matches.csv")
    parser.add_argument("--drivers", type=int, default=40)
    parser.add_argument("--riders", type=int, default=40)
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(num_drivers=args.drivers, num_riders=args.riders, steps=args.steps)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} candidate match records to {output_path}")
