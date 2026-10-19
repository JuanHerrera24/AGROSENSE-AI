from __future__ import annotations

from typing import Any

import numpy as np

from ..registry import ModelRegistry
from ..simulation import ANOMALY_FEATURES, DRIFT_PRONE, SENSOR_SPECS, anomaly_features

SPIKE_DEV, SPIKE_JUMP = 5, 4
DRIFT_LEVEL = 2.5
OUTLIER_DEV = 2
HARD_KINDS = ("out_of_range", "frozen")

KIND_TEXT = {
    "spike": "picos bruscos",
    "frozen": "valores congelados (el sensor repite la misma lectura)",
    "drift": "deriva (las lecturas se desplazan poco a poco)",
    "out_of_range": "valores físicamente imposibles",
    "outlier": "lecturas atípicas",
}


def classify_points(art: dict, values: np.ndarray, sensor_type: str) -> list[dict]:
    """Hybrid detector: physical rules first, then the Isolation Forest score."""
    lo, hi, _ = SENSOR_SPECS[sensor_type]
    feats = anomaly_features(values, sensor_type)
    scores = -art["model"].score_samples(feats)
    col = {name: feats[:, i] for i, name in enumerate(ANOMALY_FEATURES)}
    dev, jump, flat, drift = col["dev_median"], col["jump"], col["flat_run"], col["drift"]

    found = []
    for i, value in enumerate(values):
        if value < lo or value > hi:
            kind = "out_of_range"
        elif flat[i] >= 1.0:
            kind = "frozen"
        elif dev[i] >= SPIKE_DEV and jump[i] >= SPIKE_JUMP:
            kind = "spike"
        elif sensor_type in DRIFT_PRONE and drift[i] >= DRIFT_LEVEL:
            kind = "drift"
        elif scores[i] >= art["threshold"]:
            if dev[i] >= 3 and jump[i] >= 3:
                kind = "spike"
            elif drift[i] >= DRIFT_LEVEL:
                kind = "drift"
            elif dev[i] >= OUTLIER_DEV:
                kind = "outlier"
            else:
                continue  # only unusual because of the previous reading (e.g. right after a spike)
        else:
            continue
        strength = (scores[i] - art["score_floor"]) / (art["score_ceiling"] - art["score_floor"])
        strength = float(np.clip(strength, 0, 1))
        if kind in HARD_KINDS:
            strength = max(strength, 0.5)
        found.append({"index": i, "value": float(value), "kind": kind, "score": round(strength, 3)})
    return found


def detect_anomalies(models: ModelRegistry, data: dict[str, Any]) -> dict:
    art = models["anomaly"]
    sensor_type = data["sensor_type"]
    values = np.array([float(v) for v in data["readings"]])
    found = classify_points(art, values, sensor_type)
    kinds = sorted({a["kind"] for a in found})

    if not found:
        summary = "Las lecturas se ven sanas: no se encontraron fallos."
    else:
        summary = (f"Se encontraron {len(found)} lecturas sospechosas de {len(values)}: "
                   f"{', '.join(KIND_TEXT[k] for k in kinds)}.")
        if {"frozen", "out_of_range", "drift"} & set(kinds):
            summary += " Revisa la conexión y la calibración del sensor."
    return {
        "sensor_id": data.get("sensor_id"),
        "sensor_type": sensor_type,
        "total_readings": len(values),
        "anomalies": found,
        "health_score": round(1 - len(found) / len(values), 3),
        "summary": summary,
        "model_used": art["version"],
    }
