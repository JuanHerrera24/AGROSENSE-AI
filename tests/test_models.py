import numpy as np

from agrosense_ai.services import detect_anomalies, predict_disease, predict_forecast, predict_irrigation


def irrigation(models, **data):
    return predict_irrigation(models, {"crop_type": None, **data})


# ---------------------------------------------------------------------------------------- irrigation

def test_dry_soil_is_irrigated(models):
    result = irrigation(models, soil_moisture=22, air_temperature=28, humidity_min=40, humidity_max=80)
    assert result["irrigate"] is True
    assert result["duration_minutes"] >= 20
    assert "low_moisture" in result["factors"]


def test_waterlogged_soil_is_not_irrigated(models):
    result = irrigation(models, soil_moisture=92, humidity_min=40, humidity_max=80)
    assert result["irrigate"] is False
    assert result["duration_minutes"] == 0
    assert "high_moisture" in result["factors"]


def test_soil_in_range_on_a_mild_day_is_not_irrigated(models):
    result = irrigation(models, soil_moisture=65, air_temperature=18, humidity_min=40, humidity_max=80)
    assert result["irrigate"] is False


def test_probability_decreases_with_moisture(models):
    probabilities = [irrigation(models, soil_moisture=m, air_temperature=24)["probability_irrigate"]
                     for m in (15, 35, 55, 75, 95)]
    assert probabilities == sorted(probabilities, reverse=True)


def test_expected_rain_lowers_probability(models):
    dry = irrigation(models, soil_moisture=48, air_temperature=26, rain_forecast_mm=0)
    rainy = irrigation(models, soil_moisture=48, air_temperature=26, rain_forecast_mm=25)
    assert rainy["probability_irrigate"] < dry["probability_irrigate"]


def test_heat_raises_probability(models):
    cool = irrigation(models, soil_moisture=50, air_temperature=14)
    hot = irrigation(models, soil_moisture=50, air_temperature=34)
    assert hot["probability_irrigate"] > cool["probability_irrigate"]


def test_acid_soil_and_frost_are_reported(models):
    result = irrigation(models, soil_moisture=60, ph=4.2, air_temperature=1)
    assert {"acidic_ph", "frost_risk"} <= set(result["factors"])


def test_crop_profile_supplies_limits(models):
    result = irrigation(models, soil_moisture=55, crop_type="plantain")
    assert result["limits"]["humidity_min"] == 60
    assert result["limits"]["source"] == "crop"


def test_prediction_is_explained(models):
    result = irrigation(models, soil_moisture=30, air_temperature=30, ph=6)
    assert result["explanation"][0]["feature"] == "soil_moisture"


# ------------------------------------------------------------------------------------------ forecast

def test_drying_field_crosses_minimum(models):
    history = [62 - 0.6 * i for i in range(12)]
    result = predict_forecast(models, {"soil_moisture": 54.5, "history": history, "air_temperature": 30,
                                       "humidity_min": 50, "humidity_max": 80})
    assert result["hours_until_minimum"] is not None
    assert result["forecast"][-1]["expected"] < 54.5


def test_intervals_are_ordered_and_widen(models):
    result = predict_forecast(models, {"soil_moisture": 65, "crop_type": "coffee"})
    widths = []
    for point in result["forecast"]:
        assert point["low"] <= point["expected"] <= point["high"]
        widths.append(point["high"] - point["low"])
    assert widths[-1] >= widths[0]


def test_dry_soil_needs_water_today(models):
    result = predict_forecast(models, {"soil_moisture": 30, "humidity_min": 40, "humidity_max": 80})
    assert result["hours_until_minimum"] == 0


# ----------------------------------------------------------------------------------------- anomalies

def series(n=96, seed=1):
    rng = np.random.default_rng(seed)
    return list(np.round(60 - 0.05 * np.arange(n) + rng.normal(0, 0.4, n), 2))


def detect(models, sensor_type, readings):
    result = detect_anomalies(models, {"sensor_type": sensor_type, "readings": readings})
    return {a["index"]: a["kind"] for a in result["anomalies"]}, result


def test_healthy_sensor_has_almost_no_alarms(models):
    flagged, result = detect(models, "SOIL_MOISTURE", series())
    assert len(flagged) <= 2
    assert result["health_score"] >= 0.97


def test_spike_is_detected(models):
    readings = series()
    readings[40] = 98
    assert detect(models, "SOIL_MOISTURE", readings)[0].get(40) == "spike"


def test_frozen_sensor_is_detected(models):
    readings = series()
    readings[50:70] = [readings[50]] * 20
    assert "frozen" in detect(models, "SOIL_MOISTURE", readings)[0].values()


def test_impossible_value_is_detected(models):
    readings = [6.3, 6.3, 6.3, 6.2, 6.4, 6.3, 15.2, 6.3, 6.2]
    assert detect(models, "PH", readings)[0].get(6) == "out_of_range"


def test_ph_drift_is_detected(models):
    rng = np.random.default_rng(3)
    readings = list(6.2 + rng.normal(0, 0.03, 48)) + list(6.2 + np.linspace(0, 1.6, 30) + rng.normal(0, 0.03, 30))
    assert "drift" in detect(models, "PH", readings)[0].values()


def test_light_at_night_is_not_frozen(models):
    day = [0] * 6 + [5_000, 20_000, 45_000, 70_000, 85_000, 90_000, 88_000, 75_000, 52_000, 28_000, 8_000] + [0] * 7
    assert "frozen" not in detect(models, "LIGHT", day * 3)[0].values()


# ------------------------------------------------------------------------------------------- disease

def test_warm_wet_weather_means_high_coffee_rust_risk(models):
    result = predict_disease(models, {"crop_type": "coffee", "air_temperature": 23, "relative_humidity": 95,
                                      "hours_high_humidity": 16, "leaf_wetness_hours": 12, "rain_mm": 10})
    assert result["disease"] == "coffee_rust"
    assert result["risk_level"] == "high"


def test_dry_weather_means_low_risk(models):
    result = predict_disease(models, {"crop_type": "coffee", "air_temperature": 23, "relative_humidity": 55,
                                      "hours_high_humidity": 0, "leaf_wetness_hours": 0, "rain_mm": 0})
    assert result["risk_level"] == "low"


def test_each_crop_has_its_own_disease_window(models):
    weather = {"air_temperature": 14, "relative_humidity": 96, "hours_high_humidity": 18, "leaf_wetness_hours": 14}
    potato = predict_disease(models, {"crop_type": "potato", **weather})
    plantain = predict_disease(models, {"crop_type": "plantain", **weather})
    # 14 °C suits late blight but is too cold for black sigatoka.
    assert potato["probability"] > 0.6
    assert plantain["probability"] < potato["probability"] - 0.3
