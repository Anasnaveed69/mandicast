"""Direct multi-horizon tabular models (one model per horizon, global across commodities).

Target is the log return to the target date, log(p[t+h]) - log(p[t]), so the
three commodities share one scale and the forecast is p[t] * exp(pred).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from ..features import calendar_features, feature_columns
from .base import PRED_COLUMNS, Forecaster


def design_matrix(features, horizon):
    X = features.reset_index(drop=True)
    cols = feature_columns(X)
    tgt_cal = calendar_features(X["date"] + pd.Timedelta(days=horizon), prefix="tgt_")
    X = pd.concat([X[cols], tgt_cal], axis=1)
    return X


def make_target(features, horizon):
    f = features.sort_values(["commodity", "date"])
    future = f.groupby("commodity")["lp"].shift(-horizon)
    return (future - f["lp"]).reindex(features.index)


class _TabularForecaster(Forecaster):
    required = ["ret_28"]   # rows without enough history are dropped

    def __init__(self, **params):
        super().__init__(**params)
        self.models = {}
        self.columns = {}

    def _train_rows(self, data, train_end, horizon):
        f = data.features
        f = f[f["date"] <= train_end - pd.Timedelta(days=horizon)]
        f = f.dropna(subset=[c for c in self.required if c in f.columns])
        y = make_target(data.features, horizon).loc[f.index]
        keep = y.notna()
        return f[keep], y[keep]

    def _origin_rows(self, data, origins):
        f = data.features.set_index(["date", "commodity"])
        idx = pd.MultiIndex.from_arrays([origins["date"], origins["commodity"]])
        return f.reindex(idx).reset_index()

    def fit(self, data, train_end, horizons):
        for h in horizons:
            f, y = self._train_rows(data, train_end, h)
            X = design_matrix(f, h)
            self.columns[h] = list(X.columns)
            self.models[h] = self._fit_one(X, y.to_numpy())
        return self

    def predict(self, data, origins, horizon):
        rows = self._origin_rows(data, origins)
        # reindex: at serving time the history may not contain every commodity dummy
        X = design_matrix(rows, horizon).reindex(columns=self.columns[horizon], fill_value=0)
        q = self._predict_one(self.models[horizon], X)   # (n, 3) log-returns
        base = rows["price"].to_numpy()
        pred = origins[["commodity", "date"]].reset_index(drop=True).copy()
        pred["yhat"] = base * np.exp(q[:, 1])
        pred["q10"] = base * np.exp(q[:, 0]) if self.has_quantiles else np.nan
        pred["q90"] = base * np.exp(q[:, 2]) if self.has_quantiles else np.nan
        return pred[PRED_COLUMNS]


class LinearModel(_TabularForecaster):
    name = "linear"
    family = "classical"

    def _fit_one(self, X, y):
        model = make_pipeline(StandardScaler(), Ridge(alpha=self.params.get("alpha", 1.0)))
        return model.fit(X.fillna(0.0), y)

    def _predict_one(self, model, X):
        p = model.predict(X.fillna(0.0))
        return np.column_stack([p, p, p])


XGB_KEYS = ["n_estimators", "max_depth", "learning_rate", "subsample",
            "colsample_bytree", "min_child_weight", "reg_lambda", "gamma"]


class XGBoostModel(_TabularForecaster):
    name = "xgboost"
    family = "gradient_boosting"
    has_quantiles = True

    def __init__(self, quantiles=(0.1, 0.5, 0.9), seed=42, **params):
        super().__init__(**params)
        self.quantiles = list(quantiles)
        self.seed = seed

    def _fit_one(self, X, y):
        kw = {k: self.params[k] for k in XGB_KEYS if k in self.params}
        model = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=np.array(self.quantiles),
            tree_method="hist",
            random_state=self.seed,
            n_jobs=-1,
            **kw,
        )
        return model.fit(X, y)

    def _predict_one(self, model, X):
        q = model.predict(X)
        return np.sort(q, axis=1)   # avoid crossing quantiles
