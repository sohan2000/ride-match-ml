# RideMatch: Real-Time Driver-Rider Matching Simulator

RideMatch is a small, portfolio-friendly marketplace ML project that simulates a city with riders and drivers and trains a model to score whether a given driver-rider pair is a good match.

## Goal

This project demonstrates a clean end-to-end ML workflow for a ride-sharing fulfillment setting:

- synthetic data generation for driver and rider events
- feature engineering for candidate matches
- model training and evaluation against a heuristic baseline
- an API for serving match decisions
- Dockerized deployment for production-style shipping

## Data strategy

RideMatch starts with a self-contained simulator rather than a public trip
dataset. Public mobility datasets generally contain completed trips, but not
the time-varying driver state, open requests, rejected candidates, or matching
counterfactuals needed to study fulfillment decisions.

The simulator is therefore the primary source of truth for the MVP. It gives
us control over driver and rider movement, demand and supply imbalance, rush
hours, driver shortages, request cancellations, and ground-truth outcomes.
Those controls make it possible to test matching policies against the KPIs the
project is designed to demonstrate: wait time, utilization, match coverage,
and SLA hit rate.

Real mobility data is deliberately deferred. After the simulator and offline
evaluation are stable, an optional transferability notebook may reuse the
feature and evaluation interfaces on a public trip dataset. That add-on would
validate broad mobility intuitions such as distance and time-of-day effects;
it would not replace the simulator or claim to reproduce Lyft's internal
fulfillment data.

## Tech stack

- Python
- pandas / numpy
- scikit-learn
- FastAPI
- Docker

## Project structure

```text
ride-match-ml/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── models/                  # generated artifacts; ignored by Git
│   └── model.pkl            # created locally or in Colab
├── src/
│   ├── __init__.py
│   ├── simulator/
│   │   ├── __init__.py
│   │   ├── city.py
│   │   ├── generate_data.py
│   │   └── schema.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py
│   │   └── evaluate.py
│   ├── service/
│   │   ├── __init__.py
│   │   └── main.py
│   └── utils/
│       ├── __init__.py
│       └── metrics.py
├── tests/
│   ├── test_metrics.py
│   ├── test_features.py
│   ├── test_schema.py
│   ├── test_service.py
│   └── test_simulator.py
└── notebooks/
    └── exploration.ipynb
```

## Quick start

```bash
cd ride-match-ml
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
python src/simulator/generate_data.py --output data/processed/synthetic_matches.csv
python src/models/train_model.py --input data/processed/synthetic_matches.csv --output models/model.pkl
python src/models/evaluate.py --model models/model.pkl --input data/processed/synthetic_matches.csv
uvicorn src.service.main:app --reload
```

Generated CSVs under `data/processed/` and model artifacts under `models/` are
ignored by Git. Commit the code, configuration, tests, and notebooks; recreate
the data and model from the commands above or from Colab.

## Colab workflow

Use [notebooks/exploration.ipynb](notebooks/exploration.ipynb) as the experiment
starting point. The notebook is intentionally a research surface: use it for
data inspection, feature experiments, model comparisons, plots, and recorded
results. Move any experiment that survives evaluation into `src/` so the API
and command-line workflow stay reproducible.

Once this repository is pushed to GitHub, open the notebook directly in Colab:

[Open RideMatch experiments in Google Colab](https://colab.research.google.com/github/sohan2000/ride-match-ml/blob/main/notebooks/exploration.ipynb)

In Colab:

```python
!git clone https://github.com/sohan2000/ride-match-ml.git
%cd ride-match-ml
!pip install -r requirements.txt
```

Then generate a larger experiment dataset:

```python
from src.simulator.generate_data import generate_dataset

candidates = generate_dataset(
    num_drivers=100,
    num_riders=10_000,
    steps=1_440,
    seed=42,
)
candidates.to_csv("data/processed/colab_candidates.csv", index=False)
```

### Modeling warning

The current `matched` label describes the simulator's nearest-driver policy.
It is useful for validating the pipeline, but a model trained on it mostly
learns to imitate that heuristic. It does not yet prove that ML improves
fulfillment. The next research task is to add a policy-independent utility or
outcome target and compare policies at the request level.

Recommended experiment order:

1. Inspect label balance, candidate counts, ETA, supply, demand, and time-based
   train/test drift.
2. Establish nearest-driver metrics as the baseline.
3. Train a class-weighted Random Forest or gradient-boosted tabular model.
4. Select one candidate per request and compare average wait, p90 wait, SLA hit
   rate, cancellation rate, coverage, and utilization.
5. Add a utility target that trades off rider wait and driver utilization, then
   promote the best reproducible implementation into `src/models/`.

Colab Pro is useful for larger simulations, repeated scenario sweeps, and
hyperparameter searches. A GPU is not required for the first tabular models;
the quality of the simulator and request-level evaluation matter more than
deep learning at this stage.

With the API running, send a request and its currently available drivers:

```bash
curl -X POST http://localhost:8000/match \
   -H 'Content-Type: application/json' \
   -d '{"request_id":"req-demo","pickup_x":4.0,"pickup_y":3.0,"drivers":[{"driver_id":"d1","x":2.0,"y":3.0,"idle_seconds":120},{"driver_id":"d2","x":8.0,"y":3.0,"idle_seconds":60}]}'
```

The response includes the selected driver, its score, and every candidate's
distance, ETA, and score. Before a model artifact exists, the service uses the
documented nearest-driver heuristic and labels the response
`heuristic_fallback`.

## MVP business KPIs

The project evaluates model quality using marketplace-inspired metrics:

- wait time reduction vs baseline
- match acceptance rate
- utilization improvement
- average ETA gap
- assignment success score

## MVP data contract

The simulator uses four records. Their executable definitions live in
`src/simulator/schema.py`; timestamps are timezone-aware ISO-8601 values and
coordinates are grid kilometers.

### `RideRequestEvent`

One rider request entering the marketplace:

| Field                    | Type     | Meaning                                           |
| ------------------------ | -------- | ------------------------------------------------- |
| `event_time`             | datetime | Request creation time                             |
| `request_id`, `rider_id` | string   | Request and rider identifiers                     |
| `pickup_x`, `pickup_y`   | float    | Pickup location                                   |
| `dropoff_x`, `dropoff_y` | float    | Dropoff location                                  |
| `status`                 | enum     | `waiting`, `matched`, `cancelled`, or `completed` |

### `DriverSnapshot`

The driver state available to the matcher at the request timestamp:

| Field          | Type     | Meaning                                     |
| -------------- | -------- | ------------------------------------------- |
| `event_time`   | datetime | Snapshot time, matching the request context |
| `driver_id`    | string   | Driver identifier                           |
| `x`, `y`       | float    | Current driver location                     |
| `status`       | enum     | `available`, `assigned`, or `offline`       |
| `idle_seconds` | integer  | Time available before this snapshot         |

### `CandidateMatchRecord`

One eligible driver-request pair. This is the training table and contains only
information known at matching time plus delayed labels:

`event_time`, `request_id`, `driver_id`, `distance_km`, `eta_minutes`,
`driver_idle_minutes`, `available_drivers`, `open_requests`, `hour_of_day`,
`matched`, `realized_wait_minutes`, and `cancelled`.

The MVP model predicts `matched` or a match probability. Candidate selection
uses the highest model score, with nearest-driver selection as the baseline.
`realized_wait_minutes` and `cancelled` are evaluation labels and must not be
used as online features.

### `AssignmentOutcome`

The delayed result of the selected assignment:

`request_id`, nullable `driver_id`, `status`, `wait_minutes`, nullable
`ride_minutes`, and `cancelled`.

## MVP feature set and milestones

The first complete experiment should stay deliberately narrow:

1. Simulate a 20x20 city in discrete one-minute steps with 40 drivers and a
   Poisson-like stream of rider requests. Drivers are `available`, `assigned`,
   or `offline`; requests can be waiting, matched, cancelled, or completed.
2. Build candidate features from distance, ETA, driver idle time, hour of day,
   available-driver count, and open-request count. Keep pickup/dropoff
   distance available for the ride-duration label, but do not leak outcomes
   into matching features.
3. Compare nearest-driver greedy matching with a random-forest ranker or
   classifier. Split train and test data by simulation time, not random rows,
   so the offline result resembles a future-serving scenario.
4. Report average wait, p90 wait, matched-within-5-minutes rate, cancellation
   rate, driver utilization, and match coverage. Every model result should be
   compared with the same requests under the greedy baseline.
5. Expose `POST /match` with a request and available driver snapshots, return
   the selected driver plus candidate scores, and log the decision record.
   Keep Docker and the existing health endpoint as the deployment surface.

Out of scope for the MVP: surge pricing optimization, pooled rides,
geospatial road-network routing, online learning, and a real external data
source. These can be follow-up experiments after the simulator produces a
stable baseline report.

## Notes

This is intentionally small enough to finish in a focused 1-2 week sprint while still presenting like a real fulfillment / marketplace MLE portfolio project.
