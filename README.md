# RideMatch: Real-Time Driver-Rider Matching Simulator

RideMatch is a small, portfolio-friendly marketplace ML project that simulates a city with riders and drivers and trains a model to score whether a given driver-rider pair is a good match.

## Goal

This project demonstrates a clean end-to-end ML workflow for a ride-sharing fulfillment setting:

- synthetic data generation for driver and rider events
- feature engineering for candidate matches
- model training and evaluation against a heuristic baseline
- an API for serving match decisions
- Dockerized deployment for production-style shipping

## Architecture

![RideMatch architecture](docs/architecture.svg)

The same feature builder is used during training, policy replay, and API
serving. The simulator supplies request-time state and delayed outcomes; the
policy evaluator replays the nearest-driver baseline and learned policy on the
same seeded request stream before computing KPIs.

For the full model card, score definitions, metric interactions, explainability
surfaces, and reproduction commands, see [Models and Metrics](models/README.md).

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
├── vercel.json
├── api/
│   └── index.py
├── configs/
│   └── config.yaml
├── data/
│   └── processed/             # generated locally; ignored by Git
├── docs/
│   └── architecture.svg
├── notebooks/
│   └── exploration.ipynb
├── reports/                   # committed reference summaries
├── src/
│   ├── features/
│   ├── models/
│   ├── service/
│   ├── simulator/
│   └── utils/
├── tests/
└── ui/
   └── index.html
```

## Quick start

```bash
cd ride-match-ml
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
python src/simulator/generate_data.py --drivers 40 --riders 600 --steps 1440 --output data/processed/synthetic_matches.csv
python src/models/train_model.py --input data/processed/synthetic_matches.csv --output models/model.pkl
python src/models/train_utility_model.py --input data/processed/synthetic_matches.csv --output models/utility_model.pkl
python src/models/evaluate.py --model models/model.pkl --input data/processed/synthetic_matches.csv
python -m src.models.policy_eval --model models/utility_model.pkl --drivers 40 --riders 600 --steps 1440 --seed 42
python -m src.models.create_reference_reports --model models/utility_model.pkl --output-dir reports
uvicorn src.service.main:app --reload
```

Open http://localhost:8000 for the interactive RideMatch dispatch console.
Enter pickup coordinates and available rider demand. Pin drops and coordinate
changes automatically send a request to `POST /match`. The screen shows the
available-driver pool, selected driver, ETA, and candidate scores.
The UI also accepts nearby rider demand and returns an estimated fare. Fare is
separate from the matching utility score and uses distance plus a capped demand
surge multiplier based on nearby riders divided by available drivers.
Nearby riders and the demo fleet are bounded from `1` to `100`; clearing the
nearby-riders field restores `1`. Adding available drivers lowers demand
pressure and therefore lowers the estimate, all else equal. The fare is an
estimate rather than a completed-ride charge; fare optimization is not part of
the model policy.
The console also surfaces the serving model, online feature count, fleet size,
city grid, and nearby search radius so the UI communicates the system behind
each decision.
The map contains a 20-cab simulated fleet. Cabs move continuously within the
40 km city grid, and clicking a new pickup location moves the pin, updates the
nearby-driver count and sorted driver list, and changes the candidate distances
used by the next match request.
Use `+` and `-` beside **Live driver data** to add or remove local demo drivers.
New drivers spawn at random coordinates across the full 40 km city rather than
being placed beside the requester; the map, driver table, fleet count, route,
and next match update together.
The demo ETA is intentionally transparent: it is straight-line grid distance
times `2.5 minutes/km` (about `24 km/h`), not a road-network route. The driver
table shows both values so the estimate can be checked directly. When a pickup pin is
set, the map draws a black straight connector to the nearest moving driver and
labels that connector with the driver, distance, and ETA.
The connector also shows the same pre-ride fare estimate and demand multiplier
used by the API. The demo starts with 20 drivers, while the simulator's
reproducible training and replay defaults use 40 drivers.
FastAPI's API documentation remains available at http://localhost:8000/docs.

## Vercel deployment

The FastAPI application is exported through `api/index.py`, which re-exports
the ASGI instance from `src/service/main.py`. Deploy from the repository root
with the Vercel CLI:

```bash
npm install -g vercel
vercel login
vercel link
vercel deploy --prod
```

The deployed endpoints are:

```text
https://your-project.vercel.app/
https://your-project.vercel.app/health
https://your-project.vercel.app/docs
```

Model binaries and generated CSVs are not part of the Git repository. A local
CLI deployment may include an existing artifact in its upload, but Git-based
Vercel deployments should assume no model is available and use the safe
heuristic fallback unless the model is provided through separate storage or an
approved artifact workflow. Local Docker Compose mounts
`models/utility_model.pkl` explicitly.

Generated CSVs under `data/processed/` and model artifacts under `models/` are
ignored by Git. Commit the code, configuration, tests, and notebooks; recreate
the data and model from the commands above or from Colab.

The small CSVs under `reports/` are committed reference summaries. Regenerate
them with `python -m src.models.create_reference_reports` after rebuilding the
ignored model artifact.

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
3. Train the GBDT classifier with the shared twelve-feature vector. XGBoost or
   LightGBM can be evaluated later without changing the serving contract.
4. Select one candidate per request and compare average wait, p90 wait, SLA hit
   rate, cancellation rate, coverage, and utilization.
5. Add or tune a utility target that trades off rider wait and driver utilization, then
   promote the best reproducible implementation into `src/models/`.

The policy CLI reports the primary model-versus-baseline table: coverage,
average wait, p90 wait, SLA hit rate, driver rejection, rider cancellation,
total cancellation, utilization, and the composite policy KPI score. These are
simulated outcomes from a dynamic driver trajectory, not claims about real
Lyft traffic.

Run a multi-scenario utility-weight sweep with:

```python
from src.models.tune_utility import tune_utility_weights

results = tune_utility_weights()
results.sort_values("policy_kpi_score", ascending=False).head()
```

For a full trajectory replay, run:

```python
from src.models.policy_eval import compare_policies

compare_policies(
   "models/utility_model.pkl",
   num_drivers=40,
   num_riders=600,
   steps=1440,
   seed=42,
)
```

This replays the identical request stream under nearest-driver and utility
policies while evolving driver busy state independently. Driver acceptance and
rider cancellation are sampled from the simulated ETA, demand pressure, rush
hour, and idle-time context. Coverage, cancellation rate, wait, p90 wait, SLA
rate, and utilization are measured from each policy's assignment outcomes.

Reference outcome-aware replay across three seeds (`600` requests, `40` drivers,
`1,440` minutes per seed):

| Metric | Nearest | Utility GBDT |
| --- | ---: | ---: |
| Coverage | 85.89% | 84.89% |
| Average wait | 6.103 min | 6.270 min |
| P90 wait | 10.000 min | 10.203 min |
| SLA hit rate | 47.28% | 44.94% |
| Cancellation rate | 14.11% | 15.11% |
| Driver rejection rate | 9.06% | 8.00% |
| Rider cancellation rate | 5.06% | 6.89% |
| Utilization | 0.1901 | 0.1860 |

These are synthetic reference results, not claims about real Lyft traffic. The
current utility policy is not better than nearest-driver under this replay:
it has higher wait and cancellation rates and lower coverage, SLA attainment,
and utilization. The result is still useful because it identifies the next
modeling problem honestly: the utility target and online features do not yet
predict acceptance and cancellation well enough to improve policy outcomes.

The utility regressor is an experiment, not a guaranteed business improvement.
The outcome-aware replay currently shows it losing to nearest-driver across
the three-seed reference average. The next Colab task is to model acceptance
and rider cancellation explicitly, then select a policy using request-level
KPIs rather than utility MAE alone.

Weight tuning now ranks candidates by the composite `policy_kpi_score`, which
rewards lower wait and cancellation, higher coverage, SLA attainment, and
utilization relative to nearest-driver. This is the selection metric; utility
validation MAE is diagnostic only. The current three-seed sweep still selects
nearest-driver as the winner, so the utility model is an active experiment,
not a claimed production improvement.

Learned-policy replay applies a five-minute candidate guard by default: when a
candidate is estimated at or below the SLA, the model ranks only those
candidates; otherwise it ranks all available drivers. In the current
three-seed run this raised learned SLA attainment to 45.61%, but coverage fell
to 84.22% and nearest-driver remained the better overall policy. The guard is
therefore an explicit experiment, not a claimed improvement.

For behavior validation, `src/models/feature_importance.py` computes permutation
importance on the held-out time slice. It uses ROC AUC for the classifier and
negative MAE for the utility regressor. The result answers how much held-out
performance worsens when one feature is shuffled.

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

The response includes the selected driver, its utility score, pre-ride fare,
demand multiplier, and every candidate's distance, ETA, and score. The service
loads `models/utility_model.pkl` by default. Before a model artifact exists, it
uses the nearest-driver heuristic and labels the response
`heuristic_fallback`. The API accepts 1–100 drivers and 1–100 open requests.

Compose serves `models/utility_model.pkl` by default. Set
`RIDEMATCH_MODEL_PATH=/app/models/model.pkl` when you want to compare the
classifier artifact instead.

## MVP business KPIs

The policy replay evaluates nearest-driver and learned-policy decisions on the
same seeded request stream using coverage, average wait, p90 wait, SLA hit rate,
driver rejection, rider cancellation, total cancellation, utilization, and the
composite `policy_kpi_score`. Utility validation MAE and classifier metrics are
diagnostic; request-level replay KPIs decide whether a policy is better.

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
`matched`, `realized_wait_minutes`, `cancelled`, `candidate_utility`,
`driver_acceptance_rate`, `same_pickup_zone`, `pickup_zone_supply`, and
`pickup_zone_demand_supply_ratio`.

The repository contains two scikit-learn GBDT artifacts. The classifier is a
`GradientBoostingClassifier` that predicts `matched` probability, while the
serving default is a `GradientBoostingRegressor` trained on `candidate_utility`.
Candidate selection uses the highest score from the loaded artifact, with
nearest-driver selection as the baseline. Both artifacts use the same
12-column online feature vector:
`distance_km`, `eta_minutes`, `driver_idle_minutes`, `available_drivers`,
`open_requests`, `demand_supply_ratio`, `hour_of_day`, `is_peak`, and the
historical `driver_acceptance_rate` available before the request. The rate is
updated only after an assignment attempt; the simulator does not leak the
current request's outcome into its feature row.
The vector also includes local spatial context: whether the driver is already
in the pickup zone, available supply in that zone, and local demand-to-supply
ratio. Zones are fixed 5 km by 5 km cells in the 40 km by 40 km city.
`realized_wait_minutes` and `cancelled` are evaluation labels and must not be
used as online features.

The API additionally returns a pre-ride fare estimate. Its base is `$3.00`,
plus `$1.45 * distance_km` and `$0.18 * eta_minutes`, multiplied by a capped
surge of `min(2.5, 1 + 0.25 * open_requests / available_drivers)`. This is a
transparent marketplace signal for the demo, separate from utility ranking.

`candidate_utility` is a delayed, policy-independent synthetic target for the
next experiment. The current simulator defines it as:

`-eta_minutes - 0.25 * ride_minutes + 0.05 * driver_idle_minutes - 12.0 * cancellation_risk + 2.0 * acceptance_probability + 3.0 * historical_acceptance - 12.0 * rider_cancellation_probability - 0.5 * demand_pressure - peak_friction + 1.5 * same_pickup_zone - 0.25 * local_demand_supply_ratio`.

The cancellation terms penalize long-ETA and rider-cancellation risk, while
idle time and historical driver acceptance add fairness and reliability
signals. Supply pressure and peak-hour friction provide market context. These
weights are experiment parameters, not facts
about real riders or drivers. Train it with
`src/models/train_utility_model.py` and rank by predicted utility; compare that
policy with the classifier and nearest-driver baseline rather than assuming it
is better.

### `AssignmentOutcome`

The delayed result of the selected assignment:

`request_id`, nullable `driver_id`, `status`, `wait_minutes`, nullable
`ride_minutes`, and `cancelled`.

## MVP feature set and milestones

The first complete experiment should stay deliberately narrow:

1. Simulate a 40x40 city in discrete one-minute steps with 40 moving drivers
   and a rush-hour-weighted stream of rider requests. Drivers are `available`,
   `assigned`, or `offline`; requests can be waiting, matched, cancelled, or
   completed.
2. Build candidate features from distance, ETA, driver idle time, hour of day,
   available-driver count, and open-request count. Keep pickup/dropoff
   distance available for the ride-duration label, but do not leak outcomes
   into matching features.
3. Compare nearest-driver greedy matching with the GBDT classifier and utility
   regressor. Split train and test data by simulation time, not random rows,
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
