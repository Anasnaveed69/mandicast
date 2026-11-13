"""Lag / rolling / calendar features for the tabular models.

Every feature on row (commodity, t) only uses prices up to and including t.
Calendar features (incl. Ramadan and Eid) are deterministic, so the models are
also allowed to use them for the target date t+h.
"""
import logging
from functools import cache

import numpy as np
import pandas as pd
from hijridate import Gregorian, Hijri

from .config import FEATURES_PATH, PRICES_PATH, load_params, setup_logging

log = logging.getLogger("features")

ID_COLS = ["date", "commodity", "price", "lp", "imputed"]


@cache
def _eid_dates(year_from=2010, year_to=2035):
    fitr, adha = [], []
    for hy in range(Gregorian(year_from, 1, 1).to_hijri().year,
                    Gregorian(year_to, 12, 31).to_hijri().year + 1):
        fitr.append(pd.Timestamp(Hijri(hy, 10, 1).to_gregorian().isoformat()))
        adha.append(pd.Timestamp(Hijri(hy, 12, 10).to_gregorian().isoformat()))
    return pd.DatetimeIndex(fitr), pd.DatetimeIndex(adha)


def _days_until(dates, events, cap=60):
    pos = events.searchsorted(dates)
    pos = np.clip(pos, 0, len(events) - 1)
    days = (events[pos] - dates).days.to_numpy().astype(float)
    return np.clip(days, 0, cap)


def calendar_features(dates, prefix=""):
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    uniq = dates.unique()
    hijri_month = np.array([Gregorian(d.year, d.month, d.day).to_hijri().month for d in uniq])
    month_map = pd.Series(hijri_month, index=uniq)

    fitr, adha = _eid_dates()
    doy = dates.day_of_year.to_numpy()
    out = pd.DataFrame({
        "dow": dates.day_of_week,
        "month": dates.month,
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "is_ramadan": (month_map.reindex(dates).to_numpy() == 9).astype(int),
        "days_to_eid_fitr": _days_until(dates, fitr),
        "days_to_eid_adha": _days_until(dates, adha),
    })
    out["eid_week"] = ((out["days_to_eid_fitr"] <= 3) | (out["days_to_eid_adha"] <= 3)).astype(int)
    if prefix:
        out = out.add_prefix(prefix)
    return out


def build_features(prices, lags, rolling_windows, long_window=365):
    """prices: long frame [date, commodity, price, (imputed)] on a daily grid."""
    frames = []
    for commodity, g in prices.sort_values("date").groupby("commodity", sort=True):
        g = g.reset_index(drop=True)
        lp = np.log(g["price"])
        ret = lp.diff()
        f = pd.DataFrame({"date": g["date"], "commodity": commodity, "price": g["price"], "lp": lp})
        f["imputed"] = g["imputed"].astype(int) if "imputed" in g else 0
        for k in lags:
            f[f"ret_{k}"] = lp - lp.shift(k)
        for w in rolling_windows:
            f[f"dev_ma_{w}"] = lp - lp.rolling(w).mean()
            f[f"vol_{w}"] = ret.rolling(w).std()
        mu = lp.rolling(long_window, min_periods=90).mean()
        sd = lp.rolling(long_window, min_periods=90).std()
        f["level_z"] = (lp - mu) / sd
        f["ret_1y"] = lp - lp.shift(364)   # same weekday last year
        frames.append(f)

    feats = pd.concat(frames, ignore_index=True)
    cal = calendar_features(feats["date"])
    feats = pd.concat([feats, cal], axis=1)
    for c in sorted(prices["commodity"].unique()):
        feats[f"is_{c}"] = (feats["commodity"] == c).astype(int)
    return feats


def feature_columns(feats):
    return [c for c in feats.columns if c not in ID_COLS]


def main():
    setup_logging()
    p = load_params()["features"]
    prices = pd.read_parquet(PRICES_PATH)
    feats = build_features(prices, p["lags"], p["rolling_windows"], p["long_window"])
    feats.to_parquet(FEATURES_PATH, index=False)
    log.info("%d rows x %d features -> %s", len(feats), len(feature_columns(feats)), FEATURES_PATH)


if __name__ == "__main__":
    main()
