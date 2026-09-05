import math

from src.utils.metrics import average_wait_time, marketplace_metrics, match_quality_score


def test_average_wait_time():
    values = [3.0, 5.0, 7.0]
    assert average_wait_time(values) == 5.0


def test_match_quality_score_range():
    score = match_quality_score(0.8, 0.7, 15.0, 30.0)
    assert 0 <= score <= 1
    assert math.isfinite(score)


def test_marketplace_metrics_report_fulfillment_kpis():
    outcomes = [
        {"driver_id": "d1", "wait_minutes": 3.0, "cancelled": False},
        {"driver_id": "d2", "wait_minutes": 8.0, "cancelled": False},
        {"driver_id": None, "wait_minutes": 10.0, "cancelled": True},
    ]

    metrics = marketplace_metrics(outcomes, sla_minutes=5.0, utilization=0.4)

    assert metrics["match_coverage"] == 2 / 3
    assert metrics["sla_hit_rate"] == 1 / 3
    assert metrics["cancellation_rate"] == 1 / 3
    assert metrics["driver_utilization"] == 0.4
