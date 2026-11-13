"""MLflow pyfunc wrapper so every registered model, whatever its family, is loaded
and called the same way:

    model = mlflow.pyfunc.load_model("models:/mandicast-tomato-h7@champion")
    model.predict(history_df)      # columns: date, commodity, price

The history should cover at least the last ~400 days of every commodity.
"""
import json

import cloudpickle
import mlflow.pyfunc
import numpy as np
import pandas as pd

from .features import build_features
from .models.base import SeriesData


def prepare_history(history):
    h = history[["date", "commodity", "price"]].copy()
    h["date"] = pd.to_datetime(h["date"]).dt.normalize()
    h["price"] = pd.to_numeric(h["price"], errors="coerce")
    h = h.dropna().sort_values("date").drop_duplicates(["date", "commodity"], keep="last")
    frames = []
    for c, g in h.groupby("commodity"):
        s = g.set_index("date")["price"].asfreq("D")
        s = np.exp(np.log(s).interpolate(limit_area="inside")).ffill()
        frames.append(pd.DataFrame({"date": s.index, "commodity": c, "price": s.to_numpy()}))
    return pd.concat(frames, ignore_index=True)


class MandiCastModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["forecaster"], "rb") as f:
            self.forecaster = cloudpickle.load(f)
        with open(context.artifacts["config"]) as f:
            self.cfg = json.load(f)

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        cfg = self.cfg
        prices = prepare_history(model_input)
        fp = cfg["features"]
        feats = build_features(prices, fp["lags"], fp["rolling_windows"], fp["long_window"])
        data = SeriesData.from_frames(prices, feats)

        c, h = cfg["commodity"], cfg["horizon"]
        if c not in data.commodities:
            raise ValueError(f"history has no rows for {c}")
        origin = data.history(c, data.prices.index[-1]).index[-1]
        origins = pd.DataFrame({"commodity": [c], "date": [origin]})
        out = self.forecaster.predict(data, origins, h)
        out["target_date"] = out["date"] + pd.Timedelta(days=h)
        out["horizon"] = h
        out["model"] = cfg["model"]
        out = out.rename(columns={"date": "origin_date"})
        return out[["commodity", "origin_date", "target_date", "horizon", "yhat", "q10", "q90", "model"]]
