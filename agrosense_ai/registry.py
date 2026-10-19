from pathlib import Path

import joblib

from .config import ARTIFACTS_DIR

MODEL_NAMES = ("irrigation", "forecast", "anomaly", "disease")


class ModelRegistry:
    """Loads the trained artifacts once at startup."""

    def __init__(self, directory: Path = ARTIFACTS_DIR):
        missing = [name for name in MODEL_NAMES if not (directory / f"{name}.joblib").exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing models {', '.join(missing)} in {directory}. "
                "Train them with: python -m agrosense_ai.training"
            )
        self._artifacts = {name: joblib.load(directory / f"{name}.joblib") for name in MODEL_NAMES}

    def __getitem__(self, name: str) -> dict:
        return self._artifacts[name]

    def summary(self) -> dict:
        return {name: {"version": a["version"], "metrics": a["metrics"]} for name, a in self._artifacts.items()}
