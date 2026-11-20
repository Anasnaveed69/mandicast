"""API configuration. Everything comes from environment variables so the same image
runs locally, in docker compose and on the server; defaults fall back to params.yaml."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from ..config import MODELS_DIR, PRICES_PATH, load_params, tracking_uri


def _list(value):
    return [v.strip() for v in value.split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    commodities: list[str]
    horizons: list[int]
    model_source: str = "mlflow"          # "mlflow" (registry) or "local" (exported bundle)
    tracking_uri: str = "sqlite:///mlflow.db"
    model_prefix: str = "mandicast"
    model_alias: str = "champion"
    model_dir: Path = MODELS_DIR / "serving"
    data_path: Path = PRICES_PATH
    history_days: int = 400                # what the features need (365-day window + lags)
    admin_token: str = ""
    root_path: str = ""
    cors_origins: list[str] = field(default_factory=list)
    background_load: bool = True           # load models in a thread and retry until they show up
    load_retry_seconds: float = 30.0

    @classmethod
    def from_env(cls, params=None):
        params = params or load_params()
        env = os.environ.get
        return cls(
            commodities=_list(env("COMMODITIES", "")) or list(params["data"]["commodities"]),
            horizons=[int(h) for h in _list(env("HORIZONS", ""))] or list(params["backtest"]["horizons"]),
            model_source=env("MODEL_SOURCE", "mlflow").lower(),
            tracking_uri=tracking_uri(params),
            model_prefix=env("MODEL_PREFIX", params["register"]["model_prefix"]),
            model_alias=env("MODEL_ALIAS", "champion"),
            model_dir=Path(env("MODEL_DIR", str(MODELS_DIR / "serving"))),
            data_path=Path(env("DATA_PATH", str(PRICES_PATH))),
            history_days=int(env("HISTORY_DAYS", "400")),
            admin_token=env("ADMIN_TOKEN", ""),
            root_path=env("API_ROOT_PATH", ""),
            cors_origins=_list(env("CORS_ORIGINS", "")),
            background_load=env("BACKGROUND_LOAD", "1") not in ("0", "false", "no"),
            load_retry_seconds=float(env("LOAD_RETRY_SECONDS", "30")),
        )

    def model_name(self, commodity, horizon):
        return f"{self.model_prefix}-{commodity}-h{horizon}"
