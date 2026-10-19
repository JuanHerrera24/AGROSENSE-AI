from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CropType = Literal["generic", "coffee", "plantain", "potato", "corn", "cocoa"]
SensorType = Literal["SOIL_MOISTURE", "AIR_TEMPERATURE", "SOIL_TEMPERATURE", "PH", "CONDUCTIVITY", "LIGHT",
                     "RAIN_GAUGE", "RELATIVE_HUMIDITY"]

Percent = Annotated[float, Field(ge=0, le=100)]
Celsius = Annotated[float, Field(ge=-50, le=80)]
Ph = Annotated[float, Field(ge=0, le=14)]


class BaseRequest(BaseModel):
    # The backend may send fields this service does not use.
    model_config = ConfigDict(extra="ignore")


def check_range(model: BaseModel, low: str, high: str) -> None:
    a, b = getattr(model, low), getattr(model, high)
    if (a is None) != (b is None):
        raise ValueError(f"{low} and {high} must be sent together")
    if a is not None and a >= b:
        raise ValueError(f"{low} must be lower than {high}")


class IrrigationRequest(BaseRequest):
    id_crop: int | None = Field(None, gt=0)
    soil_moisture: Percent = Field(..., examples=[31.5])
    air_temperature: Celsius | None = Field(None, examples=[24])
    ph: Ph | None = Field(None, examples=[6.4])
    humidity_min: Percent | None = None
    humidity_max: Percent | None = None
    ph_min: Ph | None = None
    ph_max: Ph | None = None
    relative_humidity: Percent | None = None
    rain_forecast_mm: float | None = Field(None, ge=0, le=500, description="Expected rain in the next 24 h")
    crop_type: CropType | None = None

    @model_validator(mode="after")
    def check_ranges(self):
        check_range(self, "humidity_min", "humidity_max")
        check_range(self, "ph_min", "ph_max")
        return self


class ForecastRequest(BaseRequest):
    id_crop: int | None = Field(None, gt=0)
    soil_moisture: Percent = Field(..., examples=[58])
    history: list[Percent] = Field(default_factory=list, max_length=168,
                                   description="Previous hourly readings, oldest first")
    air_temperature: Celsius | None = Field(None, examples=[26])
    relative_humidity: Percent | None = None
    humidity_min: Percent | None = None
    humidity_max: Percent | None = None
    crop_type: CropType | None = None
    timestamp: datetime | None = None

    @model_validator(mode="after")
    def check_ranges(self):
        check_range(self, "humidity_min", "humidity_max")
        return self


class AnomalyRequest(BaseRequest):
    sensor_id: int | None = Field(None, gt=0)
    sensor_type: SensorType = Field(..., examples=["SOIL_MOISTURE"])
    readings: list[float] = Field(..., min_length=3, max_length=2000, description="Chronological, ideally hourly")


class DiseaseRequest(BaseRequest):
    id_crop: int | None = Field(None, gt=0)
    crop_type: CropType = Field("generic", examples=["coffee"])
    air_temperature: Celsius = Field(..., description="Daily mean", examples=[23])
    relative_humidity: Percent = Field(..., description="Daily mean", examples=[92])
    hours_high_humidity: float | None = Field(None, ge=0, le=24, description="Hours with RH >= 90 %")
    leaf_wetness_hours: float | None = Field(None, ge=0, le=24)
    rain_mm: float | None = Field(None, ge=0, le=500)
