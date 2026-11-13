import logging
import time

import numpy as np
import pandas as pd

from .config import FEATURES_PATH, PRICES_PATH
from .evaluation import attach_targets, naive_scale, score
from .models.base import SeriesData

log = logging.getLogger("backtest")


def load_data():
    prices = pd.read_parquet(PRICES_PATH)
    feats = pd.read_parquet(FEATURES_PATH)
    return SeriesData.from_frames(prices, feats)


def run_backtest(make_model, data, folds, horizons):
    """Fit once per fold, predict every horizon. Returns (predictions, timings)."""
    preds, timings = [], []
    for fold in folds:
        model = make_model()
        t0 = time.perf_counter()
        model.fit(data, fold.cutoff, horizons)
        fit_s = time.perf_counter() - t0

        origins = pd.DataFrame(
            [(c, d) for d in fold.origins for c in data.commodities],
            columns=["commodity", "date"])
        for h in horizons:
            t0 = time.perf_counter()
            p = model.predict(data, origins, h)
            pred_ms = (time.perf_counter() - t0) * 1000 / len(origins)
            p = attach_targets(p, data.prices, h)
            p["fold"], p["horizon"], p["model"] = fold.k, h, model.name
            for c in data.commodities:
                p.loc[p["commodity"] == c, "scale"] = naive_scale(
                    data.history(c, fold.cutoff).to_numpy(), h)
            preds.append(p)
            timings.append({"model": model.name, "fold": fold.k, "horizon": h,
                            "fit_seconds": fit_s, "predict_ms": pred_ms})
        log.info("%-16s fold %d  cutoff %s  fit %.1fs", model.name, fold.k, fold.cutoff.date(), fit_s)
    return pd.concat(preds, ignore_index=True), pd.DataFrame(timings)


def summarise(preds, by=("model", "commodity", "horizon")):
    """Metrics per group, computed per fold and then averaged over folds."""
    rows = []
    for key, g in preds.groupby(list(by) + ["fold"]):
        m = score(g, g["scale"].iloc[0])
        rows.append(dict(zip(list(by) + ["fold"], key)) | m)
    per_fold = pd.DataFrame(rows)
    metric_cols = [c for c in per_fold.columns if c not in list(by) + ["fold"]]
    summary = per_fold.groupby(list(by))[metric_cols].mean().reset_index()
    return summary, per_fold


def mean_mase(preds):
    summary, _ = summarise(preds)
    return float(np.nanmean(summary["mase"]))
