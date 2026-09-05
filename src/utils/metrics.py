from __future__ import annotations

from typing import Iterable, Mapping


def average_wait_time(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def marketplace_metrics(
    outcomes: Iterable[Mapping[str, object]],
    sla_minutes: float = 5.0,
    utilization: float | None = None,
) -> dict[str, float | int]:
    """Summarize assignment outcomes using fulfillment-oriented KPIs."""

    rows = list(outcomes)
    waits = sorted(float(row["wait_minutes"]) for row in rows)
    matched = [row for row in rows if row.get("driver_id") is not None]
    cancellations = [row for row in rows if bool(row["cancelled"])]
    sla_hits = [wait for wait in waits if wait <= sla_minutes]
    p90_index = min(len(waits) - 1, int(len(waits) * 0.9)) if waits else 0

    return {
        "requests": len(rows),
        "matched": len(matched),
        "match_coverage": len(matched) / len(rows) if rows else 0.0,
        "average_wait_minutes": average_wait_time(waits),
        "p90_wait_minutes": waits[p90_index] if waits else 0.0,
        "sla_hit_rate": len(sla_hits) / len(rows) if rows else 0.0,
        "cancellation_rate": len(cancellations) / len(rows) if rows else 0.0,
        "driver_utilization": utilization if utilization is not None else 0.0,
    }


def match_quality_score(recall: float, precision: float, wait_time: float, baseline_wait_time: float) -> float:
    if baseline_wait_time <= 0:
        return 0.0
    wait_gain = 1.0 - (wait_time / baseline_wait_time)
    wait_gain = max(0.0, min(wait_gain, 1.0))
    return max(0.0, min(1.0, 0.5 * recall + 0.3 * precision + 0.2 * wait_gain))
