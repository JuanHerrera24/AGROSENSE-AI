from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd

from ..crops import CROPS, profile

# Display labels shown to the user.
FEATURE_LABELS = {
    "soil_moisture": "humedad del suelo", "air_temperature": "temperatura del aire", "ph": "pH",
    "humidity_min": "mínimo de humedad del cultivo", "humidity_max": "máximo de humedad del cultivo",
    "relative_humidity": "humedad relativa", "rain_forecast_mm": "lluvia prevista", "kc": "tipo de cultivo",
    "lag_1": "lectura de hace 1 h", "lag_3": "lectura de hace 3 h", "lag_6": "lectura de hace 6 h",
    "lag_12": "lectura de hace 12 h", "crop": "tipo de cultivo",
    "hours_high_humidity": "horas con humedad sobre 90 %", "leaf_wetness_hours": "horas de hoja mojada",
    "rain_mm": "lluvia",
}


def fmt(value: float) -> str:
    value = round(float(value), 1)
    return str(int(value)) if value.is_integer() else str(value)


def is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def to_row(values: dict) -> dict:
    return {k: (np.nan if is_missing(v) else float(v)) for k, v in values.items()}


def resolve_limits(low, high, crop_type: str | None, defaults: tuple[float, float], attribute: str):
    """Limits from the request, then from the crop profile, then the backend defaults."""
    if not is_missing(low) and not is_missing(high):
        return float(low), float(high), "request"
    if crop_type in CROPS:
        p = profile(crop_type)
        return getattr(p, f"{attribute}_min"), getattr(p, f"{attribute}_max"), "crop"
    return defaults[0], defaults[1], "default"


def explain(predict: Callable[[pd.DataFrame], float], row: dict, reference: dict, features: list[str],
            top: int = 3) -> list[dict]:
    """Local explanation by substitution: how much the output changes when a feature takes its median."""
    base = predict(pd.DataFrame([row], columns=features))
    effects = []
    for name in features:
        if is_missing(row.get(name)) or name not in reference or name.startswith("hour_"):
            continue
        variant = {**row, name: reference[name]}
        effects.append((name, base - predict(pd.DataFrame([variant], columns=features))))
    effects.sort(key=lambda item: abs(item[1]), reverse=True)
    return [
        {"feature": name, "label": FEATURE_LABELS.get(name, name), "effect": round(float(effect), 4)}
        for name, effect in effects[:top] if abs(effect) >= 1e-4
    ]
