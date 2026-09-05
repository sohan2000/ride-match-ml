import pandas as pd
import pytest

from src.features.build_features import ONLINE_FEATURE_COLUMNS, build_feature_table


def test_build_feature_table_uses_only_online_features_and_match_label():
    candidates = pd.DataFrame(
        [{
            "distance_km": 2.0,
            "eta_minutes": 5.0,
            "driver_idle_minutes": 3.0,
            "available_drivers": 4,
            "open_requests": 2,
            "hour_of_day": 8,
            "driver_acceptance_rate": 0.8,
            "same_pickup_zone": 1,
            "pickup_zone_supply": 2,
            "pickup_zone_demand_supply_ratio": 1.0,
            "matched": 1,
            "realized_wait_minutes": 5.0,
            "cancelled": 0,
        }]
    )

    features = build_feature_table(candidates)

    assert list(features.columns) == ONLINE_FEATURE_COLUMNS + ["matched"]
    assert "realized_wait_minutes" not in features.columns
    assert "cancelled" not in features.columns
    assert features.loc[0, "demand_supply_ratio"] == 0.5
    assert features.loc[0, "is_peak"] == 1


def test_build_feature_table_reports_old_schema_columns():
    with pytest.raises(ValueError, match="driver_idle_minutes"):
        build_feature_table(pd.DataFrame([{"distance_km": 1.0}]))