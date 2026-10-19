"""Trains and evaluates the four models.

    python -m agrosense_ai.training           # full run
    python -m agrosense_ai.training --quick   # 10 % of the data
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from . import simulation as sim
from .config import ARTIFACTS_DIR
from .services.anomaly import classify_points

SEED = 42
TEST_SIZE = 0.2
GBM_OPTIONS = {"early_stopping": True, "n_iter_no_change": 15, "random_state": SEED}
INTERVAL_COVERAGE = 0.8


def reference_values(df: pd.DataFrame, features: list[str]) -> dict:
    return {f: float(df[f].median()) for f in features if df[f].notna().any()}


def train_irrigation(n: int) -> tuple[dict, dict]:
    df = sim.irrigation_dataset(n)
    features = sim.IRRIGATION_FEATURES
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=SEED, stratify=df["irrigate"])

    classifier = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, l2_regularization=1.0,
                                                **GBM_OPTIONS)
    classifier.fit(train[features], train["irrigate"])
    wet = train[train["irrigate"] == 1]
    regressor = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, **GBM_OPTIONS)
    regressor.fit(wet[features], wet["minutes"])

    p = classifier.predict_proba(test[features])[:, 1]
    # Baseline: the previous fixed rule, "irrigate when moisture is below the minimum".
    baseline = test["soil_moisture"] < test["humidity_min"].fillna(40)
    test_wet = test[test["irrigate"] == 1]
    minutes = regressor.predict(test_wet[features])
    metrics = {
        "train_rows": len(train), "test_rows": len(test),
        "accuracy": accuracy_score(test["irrigate"], p >= 0.5),
        "f1": f1_score(test["irrigate"], p >= 0.5),
        "roc_auc": roc_auc_score(test["irrigate"], p),
        "brier": brier_score_loss(test["irrigate"], p),
        "baseline_rule_accuracy": accuracy_score(test["irrigate"], baseline),
        "baseline_rule_f1": f1_score(test["irrigate"], baseline),
        "minutes_mae": mean_absolute_error(test_wet["minutes"], minutes),
        "minutes_r2": r2_score(test_wet["minutes"], minutes),
    }
    artifact = {"classifier": classifier, "regressor": regressor, "features": features,
                "reference": reference_values(train, features), "version": "agrosense-irrigation-gbm-2.0"}
    return artifact, metrics


def train_forecast(fields: int) -> tuple[dict, dict]:
    df = sim.forecast_dataset(fields)
    features = sim.FORECAST_FEATURES
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=SEED)
    fit, calibration = train_test_split(train, test_size=TEST_SIZE, random_state=SEED)
    models, margins = {}, {}
    metrics = {"train_rows": len(fit), "calibration_rows": len(calibration), "test_rows": len(test)}
    for h in sim.FORECAST_HORIZONS:
        target = f"target_{h}"
        models[h] = {}
        for q in (0.1, 0.5, 0.9):
            model = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=200, learning_rate=0.08,
                                                  **GBM_OPTIONS)
            models[h][q] = model.fit(fit[features], fit[target])

        # Conformalized quantile regression: widen the interval until it covers 80 % on held-out data.
        errors = np.maximum(models[h][0.1].predict(calibration[features]) - calibration[target],
                            calibration[target] - models[h][0.9].predict(calibration[features]))
        level = min(INTERVAL_COVERAGE * (1 + 1 / len(calibration)), 1.0)
        margins[h] = float(np.quantile(errors, level))

        low = models[h][0.1].predict(test[features]) - margins[h]
        high = models[h][0.9].predict(test[features]) + margins[h]
        metrics[f"h{h}_mae"] = mean_absolute_error(test[target], models[h][0.5].predict(test[features]))
        metrics[f"h{h}_baseline_persistence_mae"] = mean_absolute_error(test[target], test["soil_moisture"])
        metrics[f"h{h}_interval_coverage"] = float(np.mean((test[target] >= low) & (test[target] <= high)))
        metrics[f"h{h}_interval_width"] = float(np.mean(high - low))
    artifact = {"models": models, "margins": margins, "features": features,
                "version": "agrosense-forecast-qgbm-1.0"}
    return artifact, metrics


def train_anomaly(series_per_type: int) -> tuple[dict, dict]:
    normal, _, _ = sim.anomaly_dataset(series_per_type, inject=False)
    model = IsolationForest(n_estimators=200, max_samples=2048, random_state=SEED).fit(normal)
    scores = -model.score_samples(normal)
    artifact = {
        "model": model, "features": sim.ANOMALY_FEATURES,
        "threshold": float(np.quantile(scores, 0.995)),
        "score_floor": float(np.quantile(scores, 0.5)), "score_ceiling": float(scores.max() + 0.05),
        "version": "agrosense-anomaly-iforest-1.0",
    }
    return artifact, evaluate_anomaly(artifact, max(series_per_type // 4, 5)) | {"train_rows": len(normal)}


def evaluate_anomaly(artifact: dict, series_per_type: int) -> dict:
    """Point and event recall on new series with injected faults."""
    rng = np.random.default_rng(99)
    truth, flagged = [], []
    points = {k: [0, 0] for k in sim.ANOMALY_KINDS}
    events = {k: [0, 0] for k in sim.ANOMALY_KINDS}
    for sensor_type in sim.SENSOR_SPECS:
        for _ in range(series_per_type):
            clean = sim.normal_sensor_series(rng, sensor_type, 168)
            values, labels = sim.inject_anomalies(rng, clean, sensor_type)
            hits = np.zeros(len(values), dtype=bool)
            for a in classify_points(artifact, values, sensor_type):
                hits[a["index"]] = True
            truth.append(labels != "")
            flagged.append(hits)
            for kind in sim.ANOMALY_KINDS:
                points[kind][0] += int(hits[labels == kind].sum())
                points[kind][1] += int((labels == kind).sum())
            # An event is a run of readings with the same label; it counts as found if any reading is flagged.
            start = None
            for i in range(len(labels) + 1):
                label = labels[i] if i < len(labels) else ""
                if start is not None and label != labels[start]:
                    events[labels[start]][0] += int(hits[start:i].any())
                    events[labels[start]][1] += 1
                    start = None
                if start is None and label:
                    start = i
    truth, flagged = np.concatenate(truth), np.concatenate(flagged)
    return {
        "test_rows": len(truth),
        "precision": precision_score(truth, flagged), "recall": recall_score(truth, flagged),
        "f1": f1_score(truth, flagged), "false_alarm_rate": float(flagged[~truth].mean()),
        **{f"recall_{k}": hit / total if total else None for k, (hit, total) in points.items()},
        **{f"event_recall_{k}": hit / total if total else None for k, (hit, total) in events.items()},
    }


def train_disease(n: int) -> tuple[dict, dict]:
    df = sim.disease_dataset(n)
    features = sim.DISEASE_FEATURES
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=SEED, stratify=df["outbreak"])
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, categorical_features=[0],
                                           **GBM_OPTIONS)
    model.fit(train[features], train["outbreak"])
    p = model.predict_proba(test[features])[:, 1]
    metrics = {
        "train_rows": len(train), "test_rows": len(test),
        "accuracy": accuracy_score(test["outbreak"], p >= 0.5), "f1": f1_score(test["outbreak"], p >= 0.5),
        "roc_auc": roc_auc_score(test["outbreak"], p), "brier": brier_score_loss(test["outbreak"], p),
        "outbreak_share": float(df["outbreak"].mean()),
    }
    artifact = {"model": model, "features": features, "reference": reference_values(train, features),
                "version": "agrosense-disease-gbm-1.0"}
    return artifact, metrics


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="train on 10 %% of the data")
    parser.add_argument("--out", type=Path, default=ARTIFACTS_DIR)
    args = parser.parse_args(argv)
    size = 0.1 if args.quick else 1.0
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = {
        "irrigation": lambda: train_irrigation(int(40_000 * size)),
        "forecast": lambda: train_forecast(max(int(150 * size), 20)),
        "anomaly": lambda: train_anomaly(max(int(60 * size), 6)),
        "disease": lambda: train_disease(int(30_000 * size)),
    }
    report = {"trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
              "scikit_learn": sklearn.__version__, "quick": args.quick, "models": {}}
    for name, job in jobs.items():
        start = time.perf_counter()
        artifact, metrics = job()
        metrics = {k: round(float(v), 4) if isinstance(v, float | np.floating) else v for k, v in metrics.items()}
        metrics["seconds"] = round(time.perf_counter() - start, 1)
        artifact["metrics"] = metrics
        joblib.dump(artifact, args.out / f"{name}.joblib", compress=3)
        report["models"][name] = {"version": artifact["version"], "metrics": metrics}
        print(f"[{name}] {artifact['version']}")
        for key, value in metrics.items():
            print(f"    {key:<34} {value}")
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
