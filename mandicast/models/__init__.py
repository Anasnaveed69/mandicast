import json
import logging

from ..config import XGB_PARAMS_PATH
from .base import Forecaster, SeriesData
from .baselines import Naive, SeasonalNaive
from .tabular import LinearModel, XGBoostModel

log = logging.getLogger("models")


def build_model(name, params):
    """Create a forecaster from its params.yaml section."""
    cfg = {k: v for k, v in params["models"][name].items() if k not in ("enabled", "use_tuned")}
    quantiles = params["quantiles"]
    seed = params["seed"]

    if name == "naive":
        return Naive()
    if name == "seasonal_naive":
        return SeasonalNaive(**cfg)
    if name == "linear":
        return LinearModel(**cfg)
    if name == "xgboost":
        if params["models"]["xgboost"].get("use_tuned") and XGB_PARAMS_PATH.exists():
            cfg.update(json.loads(XGB_PARAMS_PATH.read_text()))
        return XGBoostModel(quantiles=quantiles, seed=seed, **cfg)
    if name == "patchtst":
        from .patchtst import PatchTSTModel  # torch import is slow, keep it lazy
        return PatchTSTModel(quantiles=quantiles, seed=seed, **cfg)
    if name == "chronos":
        from .chronos_zs import ChronosZeroShot
        return ChronosZeroShot(quantiles=quantiles, **cfg)
    raise ValueError(f"unknown model {name}")


def enabled_models(params):
    return [n for n, cfg in params["models"].items() if cfg.get("enabled", True)]


__all__ = ["Forecaster", "SeriesData", "build_model", "enabled_models"]
