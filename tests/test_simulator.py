from src.simulator.city import City, Driver, Rider
from src.simulator.generate_data import generate_dataset, generate_outcomes


def test_driver_and_rider_creation():
    city = City(width=10, height=10)
    driver = Driver(driver_id="d1", x=1.0, y=2.0, status="available")
    rider = Rider(rider_id="r1", x=8.0, y=9.0, status="waiting")

    assert driver.driver_id == "d1"
    assert rider.rider_id == "r1"
    assert city.width == 10
    assert city.height == 10


def test_city_distance():
    city = City(width=10, height=10)
    driver = Driver(driver_id="d1", x=0.0, y=0.0, status="available")
    rider = Rider(rider_id="r1", x=3.0, y=4.0, status="waiting")

    assert city.distance(driver, rider) == 5.0


def test_simulation_emits_time_ordered_candidate_records():
    dataset = generate_dataset(num_drivers=3, num_riders=8, steps=10, seed=7)

    assert len(dataset) > 0
    assert {"request_id", "driver_id", "hour_of_day", "matched"}.issubset(dataset.columns)
    assert dataset["matched"].sum() == dataset["request_id"].nunique()


def test_simulation_emits_one_outcome_per_request():
    outcomes = generate_outcomes(num_drivers=2, num_riders=5, steps=5, seed=7)

    assert len(outcomes) == 5
    assert outcomes["request_id"].is_unique
    assert set(outcomes["status"]).issubset({"matched", "cancelled"})
