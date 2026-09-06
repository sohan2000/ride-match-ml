import pytest
from fastapi.testclient import TestClient

from src.service import main
from src.service.main import app


client = TestClient(app)


def test_health_reports_service_without_model():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["service"] == "ride-match"


def test_metadata_reports_matching_system_contract():
    response = client.get("/metadata")

    assert response.status_code == 200
    assert response.json()["fleet_size"] == 20
    assert response.json()["city_size_km"] == 40


def test_ui_is_served():
    response = client.get("/")

    assert response.status_code == 200
    assert "RideMatch Dispatch" in response.text
    assert 'min="1" max="100" value="1"' in response.text


def test_match_fare_responds_to_demand_and_supply():
    drivers = [
        {"driver_id": "d1", "x": 2.0, "y": 3.0, "idle_seconds": 120},
        {"driver_id": "d2", "x": 8.0, "y": 3.0, "idle_seconds": 60},
    ]
    low_demand = client.post(
        "/match",
        json={"request_id": "low", "pickup_x": 4.0, "pickup_y": 3.0, "open_requests": 1, "drivers": drivers},
    ).json()
    high_demand = client.post(
        "/match",
        json={"request_id": "high", "pickup_x": 4.0, "pickup_y": 3.0, "open_requests": 20, "drivers": drivers},
    ).json()

    assert high_demand["estimated_fare"] > low_demand["estimated_fare"]
    assert high_demand["surge_multiplier"] > low_demand["surge_multiplier"]

    more_supply = drivers + [
        {"driver_id": f"d{index}", "x": 20.0, "y": 20.0, "idle_seconds": 60}
        for index in range(3, 11)
    ]
    higher_supply = client.post(
        "/match",
        json={"request_id": "supply", "pickup_x": 4.0, "pickup_y": 3.0, "open_requests": 20, "drivers": more_supply},
    ).json()

    assert higher_supply["estimated_fare"] < high_demand["estimated_fare"]
    assert higher_supply["surge_multiplier"] < high_demand["surge_multiplier"]


@pytest.mark.parametrize("open_requests", [0, 101])
def test_match_rejects_open_requests_outside_one_to_one_hundred(open_requests):
    response = client.post(
        "/match",
        json={"request_id": "req-1", "open_requests": open_requests, "drivers": [{"driver_id": "d1", "x": 0, "y": 0}]},
    )

    assert response.status_code == 422


def test_match_rejects_more_than_one_hundred_drivers():
    drivers = [{"driver_id": f"d{index}", "x": 0, "y": 0} for index in range(101)]

    response = client.post("/match", json={"request_id": "req-1", "drivers": drivers})

    assert response.status_code == 422


def test_match_selects_nearest_driver_with_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "MODEL_PATH", tmp_path / "missing-model.pkl")
    response = client.post(
        "/match",
        json={
            "request_id": "req-1",
            "pickup_x": 0.0,
            "pickup_y": 0.0,
            "drivers": [
                {"driver_id": "far", "x": 5.0, "y": 0.0, "idle_seconds": 60},
                {"driver_id": "near", "x": 1.0, "y": 0.0, "idle_seconds": 60},
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["selected_driver_id"] == "near"
    assert response.json()["status"] == "heuristic_fallback"
    assert len(response.json()["candidates"]) == 2


def test_match_rejects_empty_driver_list():
    response = client.post("/match", json={"request_id": "req-1", "drivers": []})

    assert response.status_code == 422