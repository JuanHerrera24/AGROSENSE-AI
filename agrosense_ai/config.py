import os
from dataclasses import dataclass
from datetime import timedelta, timezone
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
STATIC_DIR = Path(__file__).parent / "static"
LOCAL_TZ = timezone(timedelta(hours=-5))  # Colombia

# Backend defaults for a crop without its own limits.
DEFAULT_HUMIDITY = (40.0, 80.0)
DEFAULT_PH = (5.5, 7.0)

MAX_BODY_BYTES = 64 * 1024


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    token: str
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            host=os.environ.get("HOST", "0.0.0.0"),
            port=int(os.environ.get("PORT", "8000")),
            token=os.environ.get("AI_SERVICE_TOKEN", "").strip(),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
        )
