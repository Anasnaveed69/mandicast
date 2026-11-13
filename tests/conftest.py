import numpy as np
import pandas as pd
import pytest

from mandicast.features import build_features
from mandicast.models.base import SeriesData


def make_prices(days=700, seed=0):
    """Small fake series (yearly cycle + noise) so tests don't need the real data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    rows = []
    for i, (c, level) in enumerate([("onion", 6000), ("potato", 3500), ("tomato", 7000)]):
        t = np.arange(days)
        lp = (np.log(level) + 0.35 * np.sin(2 * np.pi * (t + 60 * i) / 365)
              + np.cumsum(rng.normal(0, 0.02, days)))
        rows.append(pd.DataFrame({"date": dates, "commodity": c,
                                  "price": np.exp(lp).round(0), "imputed": False}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="session")
def prices():
    return make_prices()


@pytest.fixture(scope="session")
def data(prices):
    feats = build_features(prices, [1, 2, 7, 14, 28], [7, 28], 365)
    return SeriesData.from_frames(prices, feats)
