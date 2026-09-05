from fastapi.testclient import TestClient

from src.service import main
from src.service.main import app


client = TestClient(app)


def test_health_reports_service_without_model():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["service"] == "ride-match"


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