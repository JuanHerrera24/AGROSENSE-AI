from __future__ import annotations

from typing import Any

import pandas as pd

from ..crops import crop_index, profile
from ..registry import ModelRegistry
from .common import explain, fmt, to_row

HIGH_RISK, MEDIUM_RISK = 0.6, 0.3


def risk_level(probability: float) -> str:
    if probability >= HIGH_RISK:
        return "high"
    if probability >= MEDIUM_RISK:
        return "medium"
    return "low"


def predict_disease(models: ModelRegistry, data: dict[str, Any]) -> dict:
    art = models["disease"]
    crop = profile(data.get("crop_type"))
    row = to_row({
        "crop": crop_index(crop.code), "air_temperature": data["air_temperature"],
        "relative_humidity": data["relative_humidity"], "hours_high_humidity": data.get("hours_high_humidity"),
        "leaf_wetness_hours": data.get("leaf_wetness_hours"), "rain_mm": data.get("rain_mm"),
    })
    features = art["features"]

    def probability_of(df: pd.DataFrame) -> float:
        return float(art["model"].predict_proba(df)[0, 1])

    probability = probability_of(pd.DataFrame([row], columns=features))
    level = risk_level(probability)
    disease = crop.disease
    name = disease.name[0].lower() + disease.name[1:]
    if level == "high":
        text = f"Riesgo alto de {name}: el clima de los últimos días favorece la infección. {disease.advice}"
    elif level == "medium":
        text = f"Riesgo moderado de {name}. Vigila el cultivo en los próximos días. {disease.advice}"
    else:
        text = f"Riesgo bajo de {name}: el clima actual no favorece al hongo."

    return {
        "id_crop": data.get("id_crop"),
        "crop_type": crop.code,
        "disease": disease.code,
        "disease_name": disease.name,
        "probability": round(probability, 4),
        "risk_level": level,
        "recommendation": text,
        "favourable_conditions": {
            "temperature": f"{fmt(disease.temp_min)}–{fmt(disease.temp_max)} °C",
            "relative_humidity": f"más de {fmt(disease.rh_min)} %",
            "leaf_wetness_hours": f"{fmt(disease.wetness_hours)} h o más",
        },
        "explanation": explain(probability_of, row, art["reference"], features),
        "model_used": art["version"],
    }
