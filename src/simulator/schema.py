from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal


DriverStatus = Literal["available", "assigned", "offline"]
RequestStatus = Literal["waiting", "matched", "cancelled", "completed"]

DRIVER_STATUSES = {"available", "assigned", "offline"}
REQUEST_STATUSES = {"waiting", "matched", "cancelled", "completed"}


@dataclass(frozen=True)
class DriverSnapshot:
    """Driver state observed at the time a request enters the marketplace."""

    event_time: datetime
    driver_id: str
    x: float
    y: float
    status: DriverStatus
    idle_seconds: int

    def __post_init__(self) -> None:
        if self.status not in DRIVER_STATUSES:
            raise TypeError(f"Unknown driver status: {self.status}")

    def to_dict(self) -> dict:
        return asdict(self) | {"event_time": self.event_time.isoformat()}


@dataclass(frozen=True)
class RideRequestEvent:
    """A rider request and the context available before matching."""

    event_time: datetime
    request_id: str
    rider_id: str
    pickup_x: float
    pickup_y: float
    dropoff_x: float
    dropoff_y: float
    status: RequestStatus = "waiting"

    def __post_init__(self) -> None:
        if self.status not in REQUEST_STATUSES:
            raise TypeError(f"Unknown request status: {self.status}")

    def to_dict(self) -> dict:
        return asdict(self) | {"event_time": self.event_time.isoformat()}


@dataclass(frozen=True)
class CandidateMatchRecord:
    """One request-driver pair used for ranking and offline training."""

    event_time: datetime
    request_id: str
    driver_id: str
    distance_km: float
    eta_minutes: float
    driver_idle_minutes: float
    available_drivers: int
    open_requests: int
    hour_of_day: int
    matched: int
    realized_wait_minutes: float
    cancelled: int

    def to_dict(self) -> dict:
        return asdict(self) | {"event_time": self.event_time.isoformat()}


@dataclass(frozen=True)
class AssignmentOutcome:
    """Observed result of the selected assignment for KPI evaluation."""

    request_id: str
    driver_id: str | None
    status: RequestStatus
    wait_minutes: float
    ride_minutes: float | None
    cancelled: bool

    def __post_init__(self) -> None:
        if self.status not in REQUEST_STATUSES:
            raise TypeError(f"Unknown request status: {self.status}")

    def to_dict(self) -> dict:
        return asdict(self)