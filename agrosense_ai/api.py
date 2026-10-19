from __future__ import annotations

import hmac
import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__, services
from .config import MAX_BODY_BYTES, STATIC_DIR, Settings
from .crops import CROPS
from .registry import ModelRegistry
from .schemas import AnomalyRequest, DiseaseRequest, ForecastRequest, IrrigationRequest

log = logging.getLogger("agrosense_ai")


def get_models(request: Request) -> ModelRegistry:
    return request.app.state.models


Models = Annotated[ModelRegistry, Depends(get_models)]


def create_app(models: ModelRegistry | None = None, token: str | None = None) -> FastAPI:
    secret = Settings.from_env().token if token is None else token.strip()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.models = models or ModelRegistry()
        log.info("Models loaded, token %s", "enabled" if secret else "disabled")
        yield

    app = FastAPI(title="AgroSense AI", version=__version__, lifespan=lifespan,
                  description="Irrigation, soil moisture forecast, sensor fault detection and disease risk.")

    def require_token(authorization: Annotated[str | None, Header()] = None) -> None:
        if not secret:
            return
        expected = f"Bearer {secret}".encode()
        if authorization is None or not hmac.compare_digest(authorization.strip().encode(), expected):
            raise HTTPException(401, "Missing or invalid token")

    @app.middleware("http")
    async def limit_body_and_log(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
            return JSONResponse({"message": "Request body is too large"}, status_code=413)
        response = await call_next(request)
        log.info("%s %s %s", request.method, request.url.path, response.status_code)
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        problems = []
        for error in exc.errors():
            field = ".".join(str(p) for p in error["loc"] if p != "body")
            problems.append(f"{field}: {error['msg']}" if field else error["msg"])
        return JSONResponse({"message": "; ".join(problems) or "Invalid request"}, status_code=400)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException):
        messages = {404: "Not found", 405: "Method not allowed"}
        message = messages.get(exc.status_code, exc.detail if isinstance(exc.detail, str) else "Error")
        return JSONResponse({"message": message}, status_code=exc.status_code, headers=exc.headers)

    protected = [Depends(require_token)]

    @app.api_route("/", methods=["GET", "HEAD"], tags=["service"])
    def root():
        return {"service": "agrosense-ai", "version": __version__, "docs": "/docs", "demo": "/demo"}

    @app.api_route("/health", methods=["GET", "HEAD"], tags=["service"])
    def health():
        return {"status": "UP"}

    @app.get("/models", tags=["service"])
    def model_info(models: Models):
        return models.summary()

    @app.get("/crops", tags=["service"])
    def crops():
        return [{"code": c.code, "name": c.name, "kc": c.kc, "humidity_min": c.humidity_min,
                 "humidity_max": c.humidity_max, "ph_min": c.ph_min, "ph_max": c.ph_max,
                 "disease": c.disease.name} for c in CROPS.values()]

    @app.get("/demo", include_in_schema=False)
    def demo():
        return FileResponse(STATIC_DIR / "demo.html")

    @app.post("/predict/irrigation", dependencies=protected, tags=["models"])
    def irrigation(body: IrrigationRequest, models: Models):
        return services.predict_irrigation(models, body.model_dump())

    @app.post("/predict/moisture-forecast", dependencies=protected, tags=["models"])
    def forecast(body: ForecastRequest, models: Models):
        return services.predict_forecast(models, body.model_dump())

    @app.post("/detect/anomalies", dependencies=protected, tags=["models"])
    def anomalies(body: AnomalyRequest, models: Models):
        return services.detect_anomalies(models, body.model_dump())

    @app.post("/predict/disease-risk", dependencies=protected, tags=["models"])
    def disease(body: DiseaseRequest, models: Models):
        return services.predict_disease(models, body.model_dump())

    return app
