"""Synthetic training data.

There is no labelled field history yet, so the data comes from a simple soil water balance:
Hargreaves ET0 (FAO-56) times the crop coefficient, rain and irrigation inputs, and drainage above
the crop's maximum. Labels include some noise so the models learn probabilities instead of a rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .crops import CROP_CODES, CROPS

POINTS_PER_MM = 1.5  # soil moisture points per mm of water
RAIN_EFFICIENCY = 0.9
MINUTES_PER_POINT = 2.0

# (physical min, physical max, typical variation) per backend SensorType
SENSOR_SPECS: dict[str, tuple[float, float, float]] = {
    "SOIL_MOISTURE": (0, 100, 4.0),
    "AIR_TEMPERATURE": (-15, 50, 2.5),
    "SOIL_TEMPERATURE": (-5, 45, 1.2),
    "PH": (0, 14, 0.25),
    "CONDUCTIVITY": (0, 10, 0.3),
    "LIGHT": (0, 150_000, 12_000),
    "RAIN_GAUGE": (0, 200, 4.0),
    "RELATIVE_HUMIDITY": (0, 100, 6.0),
}
# Repeated values are normal here (light at night, no rain) and the level swings a lot.
FLAT_IS_NORMAL = {"LIGHT", "RAIN_GAUGE"}
# Probes that lose calibration over time.
DRIFT_PRONE = {"PH", "CONDUCTIVITY"}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def et0_daily(temp_c, rh=None):
    """Reference evapotranspiration in mm/day."""
    et0 = np.maximum(0.109 * (np.asarray(temp_c, dtype=float) + 17.8), 0.3)
    if rh is not None:
        et0 = et0 * np.clip(1.25 - 0.5 * np.asarray(rh, dtype=float) / 100.0, 0.7, 1.15)
    return et0


# ---------------------------------------------------------------------------------------- irrigation

IRRIGATION_FEATURES = [
    "soil_moisture", "air_temperature", "ph", "humidity_min", "humidity_max",
    "relative_humidity", "rain_forecast_mm", "kc",
]


def irrigation_dataset(n: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    crops = rng.choice(CROP_CODES, size=n)
    kc = np.array([CROPS[c].kc for c in crops])
    lo = np.array([CROPS[c].humidity_min for c in crops], dtype=float)
    hi = np.array([CROPS[c].humidity_max for c in crops], dtype=float)
    lo = np.clip(lo + rng.normal(0, 3, n), 15, 80)
    hi = np.clip(np.maximum(hi + rng.normal(0, 3, n), lo + 10), 30, 98)

    moisture = np.clip(rng.beta(2.2, 2.0, n) * 100, 2, 100)
    temp = rng.uniform(0, 38, n)
    rh = np.clip(95 - 1.4 * (temp - 10) + rng.normal(0, 10, n), 25, 100)
    rain = np.where(rng.random(n) < 0.35, rng.exponential(6, n), 0.0)
    ph = np.clip(rng.normal(6.2, 0.8, n), 3.5, 9.0)

    loss = kc * et0_daily(temp, rh) * POINTS_PER_MM
    projected = moisture - loss + RAIN_EFFICIENCY * rain
    threshold = lo + 0.2 * (hi - lo)
    p = sigmoid((threshold - projected) / 1.8)
    p = np.where(moisture < lo, np.maximum(p, 0.97), p)
    p = np.where(moisture > hi, 0.01, p)
    irrigate = rng.random(n) < p

    deficit = np.maximum((lo + hi) / 2 - projected, 2.0)
    minutes = deficit * MINUTES_PER_POINT * rng.lognormal(0, 0.08, n)
    minutes = np.where(irrigate, np.clip(np.round(minutes), 5, 120), 0)

    df = pd.DataFrame({
        "crop": crops, "soil_moisture": moisture, "air_temperature": temp, "ph": ph,
        "humidity_min": lo, "humidity_max": hi, "relative_humidity": rh,
        "rain_forecast_mm": rain, "kc": kc, "irrigate": irrigate.astype(int), "minutes": minutes,
    })
    # Not every request has every field.
    for column, share in [("air_temperature", 0.2), ("ph", 0.3), ("relative_humidity", 0.55),
                          ("rain_forecast_mm", 0.65), ("kc", 0.45)]:
        df.loc[rng.random(n) < share, column] = np.nan
    df.loc[rng.random(n) < 0.2, ["humidity_min", "humidity_max"]] = np.nan
    return df


# ------------------------------------------------------------------------------------ hourly series

def simulate_field(rng: np.random.Generator, crop: str, hours: int, irrigate: bool = True) -> pd.DataFrame:
    profile = CROPS[crop]
    lo, hi = profile.humidity_min, profile.humidity_max
    hour = np.arange(hours) % 24
    t_mean = rng.uniform(8, 27)
    amplitude = rng.uniform(3, 8)
    temp = t_mean + amplitude * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 0.6, hours)
    rh_mean = rng.uniform(55, 92)
    rh = np.clip(rh_mean - 2.2 * (temp - t_mean) + rng.normal(0, 3, hours), 25, 100)
    rain = np.where(rng.random(hours) < 0.025 * rh_mean / 75, rng.exponential(2.5, hours), 0.0)
    sun = np.maximum(np.sin(np.pi * (hour - 6) / 12), 0)
    et_hourly = profile.kc * et0_daily(temp, rh) * sun / 7.64

    moisture = np.empty(hours)
    irrigation = np.zeros(hours)
    m = rng.uniform(lo - 10, hi + 5)
    for i in range(hours):
        m = m - et_hourly[i] * POINTS_PER_MM + RAIN_EFFICIENCY * rain[i]
        if m > hi:
            m -= 0.08 * (m - hi)
        if irrigate and m < lo and hour[i] in (6, 17) and rng.random() < 0.6:
            irrigation[i] = rng.uniform(12, 28)
            m += irrigation[i]
        m = float(np.clip(m, 1, 100))
        moisture[i] = m
    observed = np.clip(moisture + rng.normal(0, 0.6, hours), 0, 100)
    return pd.DataFrame({
        "hour": hour, "temp": temp, "rh": rh, "rain": rain, "irrigation": irrigation,
        "moisture": moisture, "observed": observed,
    })


FORECAST_HORIZONS = (6, 12, 24)
FORECAST_FEATURES = [
    "soil_moisture", "lag_1", "lag_3", "lag_6", "lag_12", "air_temperature", "relative_humidity",
    "hour_sin", "hour_cos", "kc", "humidity_min", "humidity_max",
]


def forecast_dataset(fields: int, days: int = 20, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    max_h = max(FORECAST_HORIZONS)
    for _ in range(fields):
        crop = str(rng.choice(CROP_CODES))
        profile = CROPS[crop]
        series = simulate_field(rng, crop, days * 24)
        obs = series["observed"].to_numpy()
        irrigation = series["irrigation"].to_numpy()
        for t in rng.choice(np.arange(12, len(series) - max_h), size=min(260, len(series) - max_h - 12),
                            replace=False):
            # The forecast assumes no irrigation, so skip windows that have one.
            if irrigation[t + 1:t + max_h + 1].any():
                continue
            hour = series["hour"].iat[t]
            row = {
                "soil_moisture": obs[t], "lag_1": obs[t - 1], "lag_3": obs[t - 3], "lag_6": obs[t - 6],
                "lag_12": obs[t - 12], "air_temperature": series["temp"].iat[t],
                "relative_humidity": series["rh"].iat[t],
                "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
                "kc": profile.kc, "humidity_min": profile.humidity_min, "humidity_max": profile.humidity_max,
            }
            for h in FORECAST_HORIZONS:
                row[f"target_{h}"] = obs[t + h]
            rows.append(row)
    df = pd.DataFrame(rows)
    n = len(df)
    df.loc[rng.random(n) < 0.3, ["lag_1", "lag_3", "lag_6", "lag_12"]] = np.nan
    df.loc[rng.random(n) < 0.4, "relative_humidity"] = np.nan
    df.loc[rng.random(n) < 0.3, "kc"] = np.nan
    return df


# ----------------------------------------------------------------------------------------- anomalies

ANOMALY_FEATURES = ["dev_median", "jump", "spread", "flat_run", "drift"]
ANOMALY_KINDS = ("spike", "frozen", "drift", "out_of_range")
FROZEN_READINGS = 6


def normal_sensor_series(rng: np.random.Generator, sensor_type: str, n: int) -> np.ndarray:
    hour = np.arange(n) % 24
    if sensor_type == "SOIL_MOISTURE":
        return simulate_field(rng, str(rng.choice(CROP_CODES)), n)["observed"].to_numpy()
    if sensor_type in ("AIR_TEMPERATURE", "SOIL_TEMPERATURE"):
        amplitude = rng.uniform(3, 8) * (0.35 if sensor_type == "SOIL_TEMPERATURE" else 1)
        noise = 0.5 if sensor_type == "AIR_TEMPERATURE" else 0.15
        return rng.uniform(8, 26) + amplitude * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, noise, n)
    if sensor_type == "RELATIVE_HUMIDITY":
        return np.clip(rng.uniform(60, 90) - 15 * np.sin(2 * np.pi * (hour - 9) / 24)
                       + rng.normal(0, 2.5, n), 20, 100)
    if sensor_type == "PH":
        return np.clip(rng.uniform(5, 7.5) + np.cumsum(rng.normal(0, 0.008, n)) + rng.normal(0, 0.04, n), 3, 10)
    if sensor_type == "CONDUCTIVITY":
        return np.clip(rng.uniform(0.4, 2.5) + np.cumsum(rng.normal(0, 0.01, n)) + rng.normal(0, 0.05, n), 0, 10)
    if sensor_type == "LIGHT":
        sun = np.maximum(np.sin(np.pi * (hour - 6) / 12), 0)
        clouds = np.clip(rng.normal(0.75, 0.2, n), 0.15, 1)
        return np.round(rng.uniform(60_000, 110_000) * sun * clouds, 0)
    if sensor_type == "RAIN_GAUGE":
        return np.round(np.where(rng.random(n) < 0.05, rng.exponential(2.5, n), 0.0), 1)
    raise ValueError(sensor_type)


def anomaly_features(values: np.ndarray, sensor_type: str, window: int = 6) -> np.ndarray:
    """Per-reading features relative to its neighbours, scaled by the sensor's typical variation."""
    values = np.asarray(values, dtype=float)
    scale = SENSOR_SPECS[sensor_type][2]
    n = len(values)
    feats = np.zeros((n, len(ANOMALY_FEATURES)))
    flat = 0
    for i in range(n):
        neighbours = np.concatenate([values[max(0, i - window):i], values[i + 1:i + 1 + window]])
        median = np.median(neighbours) if len(neighbours) else values[i]
        before = values[max(0, i - window):i + 1]
        flat = flat + 1 if i > 0 and values[i] == values[i - 1] else 0
        recent = np.median(values[max(0, i - 2):i + 1])
        past = np.median(values[max(0, i - 3 * window):max(1, i - window)])
        feats[i] = [
            abs(values[i] - median) / scale,
            abs(values[i] - values[i - 1]) / scale if i > 0 else 0.0,
            np.std(before) / scale if len(before) > 1 else 0.0,
            0.0 if sensor_type in FLAT_IS_NORMAL else min(flat, 24) / FROZEN_READINGS,
            0.0 if sensor_type in FLAT_IS_NORMAL else abs(recent - past) / scale,
        ]
    return feats


def inject_anomalies(rng: np.random.Generator, values: np.ndarray, sensor_type: str):
    lo, hi, scale = SENSOR_SPECS[sensor_type]
    values = values.copy()
    labels = np.array([""] * len(values), dtype=object)
    n = len(values)
    for _ in range(rng.integers(1, 4)):
        kind = rng.choice(ANOMALY_KINDS)
        if kind in ("frozen", "drift") and sensor_type in FLAT_IS_NORMAL:
            kind = "spike"
        start = int(rng.integers(20, n - 30))
        if kind == "spike":
            values[start] += (1 if values[start] < (lo + hi) / 2 else -1) * rng.uniform(5, 12) * scale
            values[start] = np.clip(values[start], lo, hi)
            labels[start] = kind
        elif kind == "frozen":
            length = int(rng.integers(10, 24))
            values[start:start + length] = values[start]
            labels[start + FROZEN_READINGS:start + length] = kind
        elif kind == "drift":
            length = int(rng.integers(18, 30))
            sign = 1 if values[start] < (lo + hi) / 2 else -1
            ramp = np.linspace(0, rng.uniform(4, 8) * scale, length) * sign
            values[start:start + length] = np.clip(values[start:start + length] + ramp, lo, hi)
            values[start + length:] = np.clip(values[start + length:] + ramp[-1], lo, hi)
            labels[start + length // 2:start + length + 12] = kind
        else:
            values[start] = hi + rng.uniform(1, 5) * scale if rng.random() < 0.5 else lo - rng.uniform(1, 5) * scale
            labels[start] = kind
    return values, labels


def anomaly_dataset(series_per_type: int, length: int = 168, seed: int = 23, inject: bool = True):
    rng = np.random.default_rng(seed)
    features, labels, types = [], [], []
    for sensor_type in SENSOR_SPECS:
        for _ in range(series_per_type):
            values = normal_sensor_series(rng, sensor_type, length)
            label = np.array([""] * length, dtype=object)
            if inject:
                values, label = inject_anomalies(rng, values, sensor_type)
            features.append(anomaly_features(values, sensor_type))
            labels.append(label)
            types += [sensor_type] * length
    return np.vstack(features), np.concatenate(labels), np.array(types)


# ------------------------------------------------------------------------------------------- disease

DISEASE_FEATURES = ["crop", "air_temperature", "relative_humidity", "hours_high_humidity",
                    "leaf_wetness_hours", "rain_mm"]


def infection_pressure(crop_codes, temp, rh, wetness):
    out = np.empty(len(temp))
    for i, code in enumerate(crop_codes):
        d = CROPS[code].disease
        outside = max(d.temp_min - temp[i], 0) + max(temp[i] - d.temp_max, 0)
        temp_factor = np.exp(-(outside / 3.0) ** 2)
        humidity_factor = sigmoid((rh[i] - d.rh_min) / 3.0)
        wet_factor = min(wetness[i] / d.wetness_hours, 1.3)
        out[i] = temp_factor * (0.45 * humidity_factor + 0.55 * wet_factor)
    return out


def disease_dataset(n: int, seed: int = 31) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    crops = rng.choice(CROP_CODES, size=n)
    temp = rng.uniform(4, 34, n)
    rh = np.clip(rng.beta(4, 1.6, n) * 100, 30, 100)
    rain = np.where(rng.random(n) < 0.45, rng.exponential(8, n), 0.0)
    hours_high = np.clip(np.round(24 * sigmoid((rh - 88) / 4) + rng.normal(0, 2, n)), 0, 24)
    wetness = np.clip(np.round(0.8 * hours_high + 1.5 * np.sqrt(rain) + rng.normal(0, 1.5, n)), 0, 24)
    p = sigmoid(9 * (infection_pressure(crops, temp, rh, wetness) - 0.55))
    outbreak = (rng.random(n) < p).astype(int)
    df = pd.DataFrame({
        "crop": [CROP_CODES.index(c) for c in crops], "air_temperature": temp, "relative_humidity": rh,
        "hours_high_humidity": hours_high, "leaf_wetness_hours": wetness, "rain_mm": rain, "outbreak": outbreak,
    })
    for column, share in [("hours_high_humidity", 0.35), ("leaf_wetness_hours", 0.45), ("rain_mm", 0.5)]:
        df.loc[rng.random(n) < share, column] = np.nan
    return df
