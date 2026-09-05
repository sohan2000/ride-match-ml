from datetime import datetime, timezone

import pytest

from src.simulator.schema import DriverSnapshot, RideRequestEvent


def test_driver_snapshot_serializes_timestamp_as_iso8601():
    event_time = datetime(2026, 9, 5, 12, 30, tzinfo=timezone.utc)
    snapshot = DriverSnapshot(event_time, "d1", 1.0, 2.0, "available", 90)

    assert snapshot.to_dict()["event_time"] == "2026-09-05T12:30:00+00:00"


def test_request_rejects_unknown_status():
    with pytest.raises(TypeError):
        RideRequestEvent(
            datetime.now(timezone.utc),
            "req-1",
            "r1",
            1.0,
            2.0,
            3.0,
            4.0,
            "queued",
        )