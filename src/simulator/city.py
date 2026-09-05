from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from math import ceil, sqrt
from typing import List


@dataclass
class Driver:
    driver_id: str
    x: float
    y: float
    status: str = "available"


@dataclass
class Rider:
    rider_id: str
    x: float
    y: float
    status: str = "waiting"


class City:
    def __init__(self, width: float = 10.0, height: float = 10.0):
        self.width = width
        self.height = height

    def distance(self, driver: Driver, rider: Rider) -> float:
        return hypot(driver.x - rider.x, driver.y - rider.y)

    def generate_drivers(self, count: int) -> List[Driver]:
        columns = max(1, ceil(sqrt(count)))
        spacing_x = self.width / columns
        spacing_y = self.height / columns
        return [
            Driver(
                driver_id=f"d{i}",
                x=(i % columns + 0.5) * spacing_x,
                y=(i // columns + 0.5) * spacing_y,
                status="available",
            )
            for i in range(count)
        ]

    def generate_riders(self, count: int) -> List[Rider]:
        return [
            Rider(rider_id=f"r{i}", x=(i % 8) * 1.1 + 1.5, y=(i // 8) * 1.2 + 1.5, status="waiting")
            for i in range(count)
        ]
