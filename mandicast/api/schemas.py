from datetime import date

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    name: str
    commodity: str
    horizon: int
    version: str | None = None
    alias: str | None = None
    model: str | None = Field(None, description="model family that won the walk-forward benchmark")
    family: str | None = None
    cv_mase: float | None = Field(None, description="walk-forward MASE from the Phase 1 benchmark")
    trained_until: str | None = None
    source: str


class Forecast(BaseModel):
    commodity: str
    horizon: int = Field(description="days ahead")
    origin_date: date = Field(description="last observed day the forecast is made from")
    target_date: date
    yhat: float | None = Field(description="median forecast, Rs per 100 kg")
    q10: float | None = Field(None, description="10% quantile (null for point models)")
    q90: float | None = Field(None, description="90% quantile (null for point models)")
    unit: str = "Rs/100kg"
    model: ModelInfo


class ForecastList(BaseModel):
    data_last_date: date | None
    forecasts: list[Forecast]


class PricePoint(BaseModel):
    date: date
    price: float


class PriceSeries(BaseModel):
    commodity: str
    unit: str = "Rs/100kg"
    points: list[PricePoint]


class PredictRequest(BaseModel):
    commodity: str = Field(examples=["tomato"])
    horizon: int = Field(7, examples=[7])
    history: list[PricePoint] = Field(
        min_length=30,
        description="daily prices for this commodity, oldest first; ~400 days gives the full "
                    "feature set. Other commodities are taken from the stored data.")


class Health(BaseModel):
    status: str
    version: str


class Ready(BaseModel):
    ready: bool
    models_loaded: int
    models_expected: int
    model_source: str
    bundle_source: str | None
    data_last_date: date | None
    loaded_at: str | None
    errors: dict[str, str]
