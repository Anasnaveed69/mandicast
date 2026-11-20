"""MandiCast inference API.

    uvicorn mandicast.api.main:app --host 0.0.0.0 --port 8000

Endpoints (interactive docs at /docs):
    GET  /health                       liveness
    GET  /ready                        models + data loaded? (503 until they are)
    GET  /v1/models                    registry info of the loaded champion models
    GET  /v1/prices/{commodity}        recent cleaned prices
    GET  /v1/forecast                  latest forecast for every commodity and horizon
    GET  /v1/forecast/{commodity}      latest forecast for one commodity (?horizon=1|7)
    POST /v1/predict                   forecast from a price history you send
    POST /admin/reload                 reload models from the registry (X-Admin-Token)
    GET  /metrics                      Prometheus metrics
"""
import logging
import secrets
import threading
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response

from .. import __version__
from ..config import setup_logging
from .schemas import (
    Forecast,
    ForecastList,
    Health,
    ModelInfo,
    PredictRequest,
    PriceSeries,
    Ready,
)
from .settings import Settings
from .store import DataUnavailable, ModelNotFound, ModelStore, PriceStore

log = logging.getLogger("api")

REQUESTS = Counter("mandicast_http_requests_total", "HTTP requests", ["method", "route", "status"])
LATENCY = Histogram("mandicast_http_request_seconds", "HTTP request latency", ["route"])
PREDICTIONS = Counter("mandicast_predictions_total", "Forecasts served",
                      ["commodity", "horizon", "model", "kind"])
PREDICTED_PRICE = Gauge("mandicast_last_forecast_price", "Last served median forecast (Rs/100kg)",
                        ["commodity", "horizon"])
MODELS_LOADED = Gauge("mandicast_models_loaded", "Models currently loaded")
DATA_AGE = Gauge("mandicast_data_age_days", "Days since the last observed price")


class State:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.models = ModelStore(settings)
        self.prices = PriceStore(settings.data_path)
        self.cache: dict = {}
        self.stop = threading.Event()

    def load_models(self):
        n = self.models.load()
        self.cache.clear()
        MODELS_LOADED.set(n)
        return n

    def load_until_ready(self):
        """Background loop: the MLflow server may still be starting, keep trying."""
        while not self.stop.is_set():
            try:
                if self.load_models() > 0:
                    return
            except Exception:
                log.exception("model loading failed")
            self.stop.wait(self.settings.load_retry_seconds)


def create_app(settings: Settings | None = None) -> FastAPI:
    setup_logging()
    settings = settings or Settings.from_env()
    state = State(settings)

    @asynccontextmanager
    async def lifespan(app):
        if settings.background_load:
            threading.Thread(target=state.load_until_ready, daemon=True, name="model-loader").start()
        else:
            state.load_models()
        yield
        state.stop.set()

    app = FastAPI(
        title="MandiCast API",
        version=__version__,
        description="Daily wholesale price forecasts (Rs/100kg) for tomato, onion and potato "
                    "in the Lahore mandi. Models are the champions from the MLflow registry.",
        root_path=settings.root_path,
        lifespan=lifespan,
    )
    app.state.mc = state
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                           allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        LATENCY.labels(path).observe(time.perf_counter() - t0)
        REQUESTS.labels(request.method, path, str(response.status_code)).inc()
        return response

    # ------------------------------------------------------------ helpers
    def check_commodity(commodity: str):
        commodity = commodity.lower()
        if commodity not in settings.commodities:
            raise HTTPException(404, f"unknown commodity '{commodity}', use one of {settings.commodities}")
        return commodity

    def check_horizon(horizon: int):
        if horizon not in settings.horizons:
            raise HTTPException(422, f"horizon must be one of {settings.horizons}")
        return horizon

    def model_or_503(c, h):
        try:
            return state.models.get(c, h)
        except ModelNotFound as e:
            raise HTTPException(503, f"{e}; models are still loading or not registered yet") from e

    def stored_history():
        try:
            return state.prices.recent(settings.history_days)
        except DataUnavailable as e:
            raise HTTPException(503, f"price data not available: {e}") from e

    def latest_forecast(c, h) -> Forecast:
        m = model_or_503(c, h)
        hist = stored_history()
        last = hist["date"].max()
        key = (c, h, m.name, m.version, last)
        if key not in state.cache:
            try:
                state.cache[key] = state.models.predict(m, hist)
            except Exception as e:
                log.exception("prediction failed for %s", m.name)
                raise HTTPException(500, f"prediction failed: {type(e).__name__}: {e}") from e
            PREDICTED_PRICE.labels(c, str(h)).set(state.cache[key]["yhat"] or 0)
        PREDICTIONS.labels(c, str(h), m.model or "?", "latest").inc()
        DATA_AGE.set((pd.Timestamp.now().normalize() - last).days)
        return Forecast(**state.cache[key], model=ModelInfo(**m.info()))

    def require_admin(x_admin_token: str = Header("", alias="X-Admin-Token")):
        if not settings.admin_token or not secrets.compare_digest(x_admin_token, settings.admin_token):
            raise HTTPException(401, "missing or wrong X-Admin-Token")

    # ------------------------------------------------------------ routes
    @app.get("/", include_in_schema=False)
    def root(request: Request):
        return RedirectResponse(f"{request.scope.get('root_path', '')}/docs")

    @app.get("/health", response_model=Health, tags=["ops"])
    def health():
        return Health(status="ok", version=__version__)

    @app.get("/ready", response_model=Ready, tags=["ops"],
             responses={503: {"model": Ready, "description": "not ready yet"}})
    def ready():
        n = len(state.models.models)
        last = state.prices.last_date()
        body = Ready(ready=n > 0 and last is not None, models_loaded=n,
                     models_expected=len(settings.commodities) * len(settings.horizons),
                     model_source=settings.model_source, bundle_source=state.models.bundle_source,
                     data_last_date=last, loaded_at=state.models.loaded_at, errors=state.models.errors)
        if not body.ready:
            return Response(body.model_dump_json(), status_code=503, media_type="application/json")
        return body

    @app.get("/v1/models", response_model=list[ModelInfo], tags=["models"])
    def list_models():
        return [ModelInfo(**m.info()) for _, m in sorted(state.models.models.items())]

    @app.get("/v1/prices/{commodity}", response_model=PriceSeries, tags=["data"])
    def prices(commodity: str = Depends(check_commodity), days: int = Query(90, ge=1, le=2000)):
        try:
            s = state.prices.series(commodity, days)
        except DataUnavailable as e:
            raise HTTPException(503, f"price data not available: {e}") from e
        points = [{"date": d.date(), "price": float(p)} for d, p in zip(s["date"], s["price"])]
        return PriceSeries(commodity=commodity, points=points)

    @app.get("/v1/forecast", response_model=ForecastList, tags=["forecast"])
    def forecast_all():
        out = [latest_forecast(c, h) for c in settings.commodities for h in settings.horizons
               if (c, h) in state.models.models]
        if not out:
            raise HTTPException(503, "no models loaded yet")
        return ForecastList(data_last_date=state.prices.last_date(), forecasts=out)

    @app.get("/v1/forecast/{commodity}", response_model=ForecastList, tags=["forecast"])
    def forecast_one(commodity: str = Depends(check_commodity),
                     horizon: int | None = Query(None, description="1 or 7; both if omitted")):
        horizons = [check_horizon(horizon)] if horizon is not None else settings.horizons
        out = [latest_forecast(commodity, h) for h in horizons]
        return ForecastList(data_last_date=state.prices.last_date(), forecasts=out)

    @app.post("/v1/predict", response_model=Forecast, tags=["forecast"])
    def predict(req: PredictRequest):
        c, h = check_commodity(req.commodity), check_horizon(req.horizon)
        m = model_or_503(c, h)
        mine = pd.DataFrame([p.model_dump() for p in req.history])
        mine["date"] = pd.to_datetime(mine["date"]).astype("datetime64[ns]")
        mine["price"] = mine["price"].astype("float64")
        if (mine["price"] <= 0).any():
            raise HTTPException(422, "prices must be positive")
        mine["commodity"] = c
        # the global models also look at which commodities exist, so the other
        # series come from the stored data over the same window
        parts = [mine[["date", "commodity", "price"]]]
        try:
            others = state.prices.frame()
            parts.append(others[(others["commodity"] != c)
                                & (others["date"] >= mine["date"].min())
                                & (others["date"] <= mine["date"].max())])
        except DataUnavailable:
            pass
        hist = pd.concat([p for p in parts if not p.empty], ignore_index=True)
        try:
            res = state.models.predict(m, hist)
        except Exception as e:
            log.exception("custom prediction failed")
            raise HTTPException(422, f"could not forecast from this history: {e}") from e
        PREDICTIONS.labels(c, str(h), m.model or "?", "custom").inc()
        return Forecast(**res, model=ModelInfo(**m.info()))

    @app.post("/admin/reload", response_model=Ready, tags=["ops"], dependencies=[Depends(require_admin)])
    def reload():
        state.load_models()
        return ready()

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
