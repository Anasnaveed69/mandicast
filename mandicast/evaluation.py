"""Walk-forward splits and forecast metrics."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    k: int
    cutoff: pd.Timestamp        # last date whose price the model may see when fitting
    origins: pd.DatetimeIndex   # forecast origins evaluated in this fold


def walk_forward_folds(dates, n_folds, test_size, max_horizon, min_train_days):
    """Expanding-window folds. The last fold ends at the last origin that still has
    a known target for the longest horizon."""
    dates = pd.DatetimeIndex(sorted(pd.unique(dates)))
    usable = dates[dates <= dates[-1] - pd.Timedelta(days=max_horizon)]
    if len(usable) < n_folds * test_size:
        raise ValueError("not enough data for the requested folds")

    folds = []
    for k in range(n_folds):
        end = len(usable) - (n_folds - k - 1) * test_size
        origins = usable[end - test_size:end]
        cutoff = origins[0]
        if (cutoff - dates[0]).days < min_train_days:
            raise ValueError(
                f"fold {k} only has {(cutoff - dates[0]).days} days of history "
                f"(min_train_days={min_train_days}); collect more data or reduce n_folds/test_size")
        folds.append(Fold(k, cutoff, origins))
    return folds


def pinball(y, q_pred, q):
    diff = y - q_pred
    return np.mean(np.maximum(q * diff, (q - 1) * diff))


def naive_scale(train_prices, horizon):
    """Mean absolute h-step naive error on the training part (MASE denominator)."""
    s = np.asarray(train_prices, dtype=float)
    if len(s) <= horizon:
        return np.nan
    return np.mean(np.abs(s[horizon:] - s[:-horizon]))


def point_metrics(y, yhat, scale):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    err = y - yhat
    mae = np.mean(np.abs(err))
    return {
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(np.abs(err) / np.abs(y)) * 100),
        "mase": float(mae / scale) if scale and scale > 0 else np.nan,
    }


def interval_metrics(y, q10, q50, q90):
    y = np.asarray(y, float)
    if np.isnan(q10).all():
        return {"pinball": np.nan, "coverage80": np.nan}
    loss = np.mean([pinball(y, q10, 0.1), pinball(y, q50, 0.5), pinball(y, q90, 0.9)])
    cover = np.mean((y >= q10) & (y <= q90))
    return {"pinball": float(loss), "coverage80": float(cover)}


def score(pred, scale):
    """pred has y, yhat, q10, q90. Returns a flat metrics dict."""
    m = point_metrics(pred["y"], pred["yhat"], scale)
    m.update(interval_metrics(pred["y"], pred["q10"].to_numpy(float),
                              pred["yhat"].to_numpy(float), pred["q90"].to_numpy(float)))
    return m


def attach_targets(pred, prices_wide, horizon):
    """Add the realised price at origin+horizon."""
    pred = pred.copy()
    target_date = pred["date"] + pd.Timedelta(days=horizon)
    stacked = prices_wide.stack()
    idx = pd.MultiIndex.from_arrays([target_date, pred["commodity"]])
    pred["target_date"] = target_date
    pred["y"] = stacked.reindex(idx).to_numpy()
    return pred
