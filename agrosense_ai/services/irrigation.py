from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import DEFAULT_HUMIDITY, DEFAULT_PH
from ..crops import CROPS, profile
from ..registry import ModelRegistry
from .common import explain, fmt, is_missing, resolve_limits, to_row

MIN_MINUTES, MAX_MINUTES = 5, 120
HOT_C, COLD_C, FROST_C = 30, 12, 2
RAIN_MM = 5


def predict_irrigation(models: ModelRegistry, data: dict[str, Any]) -> dict:
    art = models["irrigation"]
    crop_type = data.get("crop_type")
    moisture = float(data["soil_moisture"])
    temp, ph, rain = data.get("air_temperature"), data.get("ph"), data.get("rain_forecast_mm")
    lo, hi, source = resolve_limits(data.get("humidity_min"), data.get("humidity_max"), crop_type,
                                    DEFAULT_HUMIDITY, "humidity")
    ph_lo, ph_hi, _ = resolve_limits(data.get("ph_min"), data.get("ph_max"), crop_type, DEFAULT_PH, "ph")

    # The model was trained with NaN limits meaning "backend defaults".
    row = to_row({
        "soil_moisture": moisture, "air_temperature": temp, "ph": ph,
        "humidity_min": lo if source != "default" else None,
        "humidity_max": hi if source != "default" else None,
        "relative_humidity": data.get("relative_humidity"), "rain_forecast_mm": rain,
        "kc": profile(crop_type).kc if crop_type in CROPS else None,
    })
    features = art["features"]
    frame = pd.DataFrame([row], columns=features)
    probability = float(art["classifier"].predict_proba(frame)[0, 1])
    irrigate = probability >= 0.5
    minutes = 0
    if irrigate:
        minutes = int(np.clip(round(float(art["regressor"].predict(frame)[0])), MIN_MINUTES, MAX_MINUTES))

    factors, warnings = _factors(moisture, lo, hi, irrigate, temp, ph, ph_lo, ph_hi, rain)
    text = _recommendation(irrigate, minutes, moisture, lo, hi, rain, factors)

    def p_irrigate(df: pd.DataFrame) -> float:
        return float(art["classifier"].predict_proba(df)[0, 1])

    return {
        "id_crop": data.get("id_crop"),
        "irrigate": irrigate,
        "duration_minutes": minutes,
        "recommendation": " ".join([text, *warnings]),
        "confidence": round(max(probability, 1 - probability), 4),
        "probability_irrigate": round(probability, 4),
        "model_used": art["version"],
        "factors": factors,
        "limits": {"humidity_min": lo, "humidity_max": hi, "ph_min": ph_lo, "ph_max": ph_hi, "source": source},
        "explanation": explain(p_irrigate, row, art["reference"], features),
    }


def _factors(moisture, lo, hi, irrigate, temp, ph, ph_lo, ph_hi, rain):
    factors: list[str] = []
    warnings: list[str] = []
    if moisture < lo:
        factors.append("low_moisture")
    elif moisture > hi:
        factors.append("high_moisture")
    else:
        factors.append("moisture_ok")
        if irrigate:
            factors.append("preventive_irrigation")
    if not is_missing(temp):
        if temp >= HOT_C:
            factors.append("high_temperature")
        elif temp <= COLD_C:
            factors.append("low_temperature")
        if temp <= FROST_C:
            factors.append("frost_risk")
            warnings.append(f"Hay riesgo de helada ({fmt(temp)} °C): protege el cultivo y evita regar de noche.")
    if not is_missing(ph):
        if ph < ph_lo:
            factors.append("acidic_ph")
            warnings.append(f"El pH ({fmt(ph)}) está por debajo del rango del cultivo ({fmt(ph_lo)}–{fmt(ph_hi)}): "
                            "considera encalar el suelo.")
        elif ph > ph_hi:
            factors.append("alkaline_ph")
            warnings.append(f"El pH ({fmt(ph)}) está por encima del rango del cultivo ({fmt(ph_lo)}–{fmt(ph_hi)}): "
                            "considera azufre o materia orgánica.")
    if not is_missing(rain) and rain >= RAIN_MM:
        factors.append("rain_expected")
    return factors, warnings


def _recommendation(irrigate, minutes, moisture, lo, hi, rain, factors) -> str:
    if irrigate and moisture < lo:
        return (f"Riega ahora durante unos {minutes} minutos: la humedad del suelo ({fmt(moisture)} %) está por "
                f"debajo del mínimo del cultivo ({fmt(lo)} %).")
    if irrigate:
        reason = " por el calor" if "high_temperature" in factors else ""
        return (f"Programa un riego de unos {minutes} minutos: la humedad del suelo ({fmt(moisture)} %) todavía "
                f"está en rango, pero se espera que baje del mínimo ({fmt(lo)} %) en las próximas horas{reason}.")
    if moisture > hi:
        return (f"No riegues: la humedad del suelo ({fmt(moisture)} %) supera el máximo del cultivo ({fmt(hi)} %). "
                "Revisa el drenaje del lote.")
    if "rain_expected" in factors:
        return (f"No riegues todavía: con la humedad del suelo en {fmt(moisture)} % y {fmt(rain)} mm de lluvia "
                "prevista, el cultivo se mantendrá en rango.")
    return (f"No hace falta regar: la humedad del suelo ({fmt(moisture)} %) está dentro del rango del cultivo "
            f"({fmt(lo)}–{fmt(hi)} %).")
