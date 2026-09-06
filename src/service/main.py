from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.features.build_features import LABEL_COLUMN, build_feature_table


logger = logging.getLogger("ride_match.service")
app = FastAPI(title="RideMatch API", version="0.2.0")
MODEL_PATH = Path(
    os.getenv("RIDEMATCH_MODEL_PATH", str(Path(__file__).resolve().parents[2] / "models" / "model.pkl"))
)
UI_PATH = Path(__file__).resolve().parents[2] / "ui" / "index.html"


class DriverInput(BaseModel):
    driver_id: str
    x: float
    y: float
    idle_seconds: int = Field(default=0, ge=0)
    acceptance_rate: float = Field(default=0.8, ge=0.0, le=1.0)


class MatchRequest(BaseModel):
    request_id: str
    pickup_x: float
    pickup_y: float
    event_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    open_requests: int = Field(default=1, ge=1, le=100)
    drivers: list[DriverInput] = Field(..., min_length=1, max_length=100)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "ride-match", "model_loaded": MODEL_PATH.exists()}


@app.get("/metadata")
def metadata() -> dict:
    model_type = "heuristic_nearest_driver"
    feature_count = 0
    if MODEL_PATH.exists():
        artifact = joblib.load(MODEL_PATH)
        if isinstance(artifact, dict):
            model_type = artifact.get("model_type", model_type)
            feature_count = len(artifact.get("feature_columns", []))
    return {
        "model_loaded": MODEL_PATH.exists(),
        "model_type": model_type,
        "feature_count": feature_count,
        "fleet_size": 20,
        "city_size_km": 40,
        "nearby_radius_km": 7,
    }


@app.get("/")
def ui() -> FileResponse:
    return FileResponse(UI_PATH)


def _model_score(model: object, row: dict) -> float:
    features = build_feature_table(pd.DataFrame([{**row, LABEL_COLUMN: 0}]))
    values = features.drop(columns=[LABEL_COLUMN])
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(values)
        classes = list(getattr(model, "classes_", [0, 1]))
        return float(probabilities[0][classes.index(1)]) if 1 in classes else 0.0
    return float(model.predict(values)[0])


def _estimate_fare(distance_km: float, open_requests: int, driver_count: int) -> tuple[float, float]:
    demand_pressure = open_requests / max(1, driver_count)
    surge_multiplier = min(2.5, 1.0 + 0.25 * demand_pressure)
    base_fare = 3.0 + 1.45 * distance_km + 0.18 * (distance_km * 2.5)
    return round(base_fare * surge_multiplier, 2), round(surge_multiplier, 2)


@app.post("/match")
def match(request: MatchRequest) -> dict:
    if not request.drivers:
        raise HTTPException(status_code=400, detail="At least one driver is required")

    event_time = request.event_time
    if event_time.tzinfo is None:
        event_time = event_time.replace(tzinfo=timezone.utc)
    candidate_rows = []
    for driver in request.drivers:
        distance = ((driver.x - request.pickup_x) ** 2 + (driver.y - request.pickup_y) ** 2) ** 0.5
        pickup_zone = (int(request.pickup_x // 5), int(request.pickup_y // 5))
        pickup_zone_supply = sum(
            (int(candidate.x // 5), int(candidate.y // 5)) == pickup_zone
            for candidate in request.drivers
        )
        candidate_rows.append(
            {
                "driver_id": driver.driver_id,
                "distance_km": distance,
                "eta_minutes": distance * 2.5,
                "driver_idle_minutes": driver.idle_seconds / 60.0,
                "available_drivers": len(request.drivers),
                "open_requests": request.open_requests,
                "hour_of_day": event_time.hour,
                "driver_acceptance_rate": driver.acceptance_rate,
                "same_pickup_zone": int((int(driver.x // 5), int(driver.y // 5)) == pickup_zone),
                "pickup_zone_supply": pickup_zone_supply,
                "pickup_zone_demand_supply_ratio": request.open_requests / max(1, pickup_zone_supply),
                "estimated_fare": _estimate_fare(distance, request.open_requests, len(request.drivers))[0],
                "surge_multiplier": _estimate_fare(distance, request.open_requests, len(request.drivers))[1],
            }
        )

    model = None
    if MODEL_PATH.exists():
        artifact = joblib.load(MODEL_PATH)
        model = artifact["model"] if isinstance(artifact, dict) else artifact

    scored_candidates = []
    for row in candidate_rows:
        score = _model_score(model, row) if model is not None else 1.0 / (1.0 + row["distance_km"])
        scored_candidates.append(
            {
                "driver_id": row["driver_id"],
                "distance_km": round(row["distance_km"], 3),
                "eta_minutes": round(row["eta_minutes"], 3),
                "score": round(score, 4),
                "estimated_fare": row["estimated_fare"],
                "surge_multiplier": row["surge_multiplier"],
            }
        )

    selected = max(scored_candidates, key=lambda candidate: candidate["score"])
    decision = {
        "status": "ok" if model is not None else "heuristic_fallback",
        "request_id": request.request_id,
        "selected_driver_id": selected["driver_id"],
        "selected_score": selected["score"],
        "estimated_fare": selected["estimated_fare"],
        "surge_multiplier": selected["surge_multiplier"],
        "candidates": scored_candidates,
    }
    logger.info("match_decision=%s", decision)
    return decision
