from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..config import DEFAULT_HUMIDITY, LOCAL_TZ
from ..crops import CROPS, profile
from ..registry import ModelRegistry
from ..simulation import FORECAST_HORIZONS
from .common import fmt, resolve_limits, to_row

QUANTILES = (0.1, 0.5, 0.9)


def predict_forecast(models: ModelRegistry, data: dict[str, Any]) -> dict:
    art = models["forecast"]
    crop_type = data.get("crop_type")
    moisture = float(data["soil_moisture"])
    history = [float(v) for v in (data.get("history") or [])]
    lo, hi, source = resolve_limits(data.get("humidity_min"), data.get("humidity_max"), crop_type,
                                    DEFAULT_HUMIDITY, "humidity")
    moment = data.get("timestamp") or datetime.now(LOCAL_TZ)
    if moment.tzinfo is not None:
        moment = moment.astimezone(LOCAL_TZ)
    hour = moment.hour + moment.minute / 60

    def lag(k: int):
        return history[-k] if len(history) >= k else None

    row = to_row({
        "soil_moisture": moisture, "lag_1": lag(1), "lag_3": lag(3), "lag_6": lag(6), "lag_12": lag(12),
        "air_temperature": data.get("air_temperature"), "relative_humidity": data.get("relative_humidity"),
        "hour_sin": math.sin(2 * math.pi * hour / 24), "hour_cos": math.cos(2 * math.pi * hour / 24),
        "kc": profile(crop_type).kc if crop_type in CROPS else None,
        "humidity_min": lo, "humidity_max": hi,
    })
    frame = pd.DataFrame([row], columns=art["features"])

    points = []
    for h in FORECAST_HORIZONS:
        low, mid, high = sorted(float(art["models"][h][q].predict(frame)[0]) for q in QUANTILES)
        margin = art["margins"][h]
        low, mid, high = (float(np.clip(v, 0, 100)) for v in (low - margin, mid, high + margin))
        points.append({"hours_ahead": h, "expected": round(mid, 1), "low": round(low, 1), "high": round(high, 1)})

    hours_until_min = _hours_until(moisture, points, lo)
    return {
        "id_crop": data.get("id_crop"),
        "current": round(moisture, 1),
        "forecast": points,
        "hours_until_minimum": hours_until_min,
        "limits": {"humidity_min": lo, "humidity_max": hi, "source": source},
        "recommendation": _recommendation(moisture, points[-1], lo, hi, hours_until_min),
        "assumes": "Sin riego en las próximas 24 horas. El rango low–high cubre el 80 % de los casos.",
        "model_used": art["version"],
    }


def _hours_until(moisture: float, points: list[dict], lo: float) -> float | None:
    """Linear interpolation of when the expected curve crosses the minimum."""
    if moisture < lo:
        return 0.0
    prev_h, prev_v = 0, moisture
    for point in points:
        if point["expected"] < lo:
            share = (prev_v - lo) / (prev_v - point["expected"])
            return round(prev_h + share * (point["hours_ahead"] - prev_h), 1)
        prev_h, prev_v = point["hours_ahead"], point["expected"]
    return None


def _recommendation(moisture, last, lo, hi, hours_until_min) -> str:
    if hours_until_min == 0:
        return f"La humedad ({fmt(moisture)} %) ya está bajo el mínimo del cultivo ({fmt(lo)} %): riega hoy."
    if hours_until_min is not None:
        return (f"Si no riegas, la humedad bajará del mínimo del cultivo ({fmt(lo)} %) en unas "
                f"{fmt(hours_until_min)} horas. Planea el riego antes de ese momento.")
    if last["expected"] > hi:
        return (f"La humedad seguirá por encima del máximo ({fmt(hi)} %) durante las próximas 24 horas: "
                "no riegues y revisa el drenaje.")
    return (f"La humedad se mantendrá en rango las próximas 24 horas (alrededor de {fmt(last['expected'])} % "
            "mañana a esta hora). No hace falta regar.")
