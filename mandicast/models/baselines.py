import pandas as pd

from .base import Forecaster, empty_quantiles


def _lookup(data, origins, offset_days):
    stacked = data.prices.stack()
    idx = pd.MultiIndex.from_arrays([origins["date"] + pd.Timedelta(days=offset_days),
                                     origins["commodity"]])
    return stacked.reindex(idx).to_numpy()


class Naive(Forecaster):
    """Tomorrow (or next week) costs what today costs."""
    name = "naive"
    family = "baseline"

    def predict(self, data, origins, horizon):
        pred = origins[["commodity", "date"]].copy()
        pred["yhat"] = _lookup(data, origins, 0)
        return empty_quantiles(pred)


class SeasonalNaive(Forecaster):
    """Repeat the value from the same point in the previous season."""
    name = "seasonal_naive"
    family = "baseline"

    def predict(self, data, origins, horizon):
        m = self.params.get("season", 7)
        # go back whole seasons until the reference date is <= origin
        k = -(-horizon // m)
        pred = origins[["commodity", "date"]].copy()
        pred["yhat"] = _lookup(data, origins, horizon - k * m)
        return empty_quantiles(pred)
