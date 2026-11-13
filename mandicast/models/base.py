from dataclasses import dataclass

import numpy as np
import pandas as pd

PRED_COLUMNS = ["commodity", "date", "yhat", "q10", "q90"]


@dataclass
class SeriesData:
    """Everything a forecaster can look at.

    prices   : wide frame, daily DatetimeIndex x commodity columns
    features : long frame from features.build_features (only tabular models use it)
    """
    prices: pd.DataFrame
    features: pd.DataFrame

    @classmethod
    def from_frames(cls, prices_long, features):
        wide = prices_long.pivot(index="date", columns="commodity", values="price").sort_index()
        wide = wide.asfreq("D")
        return cls(prices=wide, features=features)

    @property
    def commodities(self):
        return list(self.prices.columns)

    def history(self, commodity, until):
        s = self.prices[commodity]
        return s[s.index <= until].dropna()


class Forecaster:
    """fit() sees prices up to `train_end` only; predict() gets the forecast origins
    and may use prices up to each origin (that is what would be known at that time)."""

    name = "base"
    family = "base"
    has_quantiles = False

    def __init__(self, **params):
        self.params = params

    def fit(self, data: SeriesData, train_end: pd.Timestamp, horizons):
        return self

    def predict(self, data: SeriesData, origins: pd.DataFrame, horizon: int) -> pd.DataFrame:
        raise NotImplementedError

    def log_params(self):
        return {k: v for k, v in self.params.items() if np.isscalar(v)}


def empty_quantiles(pred):
    pred["q10"] = np.nan
    pred["q90"] = np.nan
    return pred[PRED_COLUMNS]
