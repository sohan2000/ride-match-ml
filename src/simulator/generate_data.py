from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from math import hypot
from pathlib import Path

import numpy as np
import pandas as pd

from src.simulator.city import City, Driver, Rider
from src.simulator.schema import AssignmentOutcome, CandidateMatchRecord


def _simulate_market(
    num_drivers: int,
    num_riders: int,
    steps: int,
    seed: int,
) -> tuple[list[CandidateMatchRecord], list[AssignmentOutcome]]:
    rng = np.random.default_rng(seed)
    city = City(width=20, height=20)
    drivers = city.generate_drivers(num_drivers)
    busy_until = {driver.driver_id: 0 for driver in drivers}
    idle_since = {driver.driver_id: 0 for driver in drivers}
    requests_by_step = rng.multinomial(num_riders, np.full(steps, 1 / steps))
    start_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candidates: list[CandidateMatchRecord] = []
    outcomes: list[AssignmentOutcome] = []

    for step, request_count in enumerate(requests_by_step):
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
            selected_driver = min(available, key=lambda driver: distances[driver.driver_id])
            selected_distance = distances[selected_driver.driver_id]
            selected_eta = selected_distance * 2.5
            ride_minutes = hypot(0.1, 0.1) * 2.5 + 5.0

            for driver in available:
                distance = distances[driver.driver_id]
                eta_minutes = distance * 2.5
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
                    )
                )

            busy_until[selected_driver.driver_id] = step + max(1, int(np.ceil(selected_eta + ride_minutes)))
            idle_since[selected_driver.driver_id] = step
            outcomes.append(
                AssignmentOutcome(
                    request_id,
                    selected_driver.driver_id,
                    "matched",
                    round(float(selected_eta), 3),
                    round(float(ride_minutes), 3),
                    False,
                )
            )

    return candidates, outcomes


def generate_dataset(
    num_drivers: int = 40,
    num_riders: int = 40,
    seed: int = 42,
    steps: int = 60,
) -> pd.DataFrame:
    candidates, _ = _simulate_market(num_drivers, num_riders, steps, seed)
    return pd.DataFrame([candidate.to_dict() for candidate in candidates])


def generate_outcomes(
    num_drivers: int = 40,
    num_riders: int = 40,
    seed: int = 42,
    steps: int = 60,
) -> pd.DataFrame:
    _, outcomes = _simulate_market(num_drivers, num_riders, steps, seed)
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
