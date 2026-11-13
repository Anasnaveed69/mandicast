import numpy as np
import pandas as pd

from mandicast.features import build_features, calendar_features


def test_ramadan_and_eid_dates():
    cal = calendar_features(pd.to_datetime(["2025-03-10", "2025-05-01", "2025-03-29"]))
    assert cal["is_ramadan"].tolist() == [1, 0, 1]
    # Umm al-Qura puts Eid ul Fitr 2025 on 30 March (Pakistan had it on the 31st,
    # a one day shift doesn't matter much for weekly-scale price effects)
    assert cal["days_to_eid_fitr"].iloc[2] == 1
    assert cal["eid_week"].iloc[2] == 1


def test_features_do_not_look_ahead(prices):
    lags, wins = [1, 7, 28], [7, 28]
    full = build_features(prices, lags, wins)
    cut = prices["date"].unique()[400]
    # change every price after the cut-off, features up to it must stay the same
    changed = prices.copy()
    changed.loc[changed["date"] > cut, "price"] *= 3
    alt = build_features(changed, lags, wins)
    a = full[full["date"] <= cut].drop(columns=["price", "lp"])
    b = alt[alt["date"] <= cut].drop(columns=["price", "lp"])
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_lag_definition(prices):
    f = build_features(prices, [7], [7])
    g = f[f["commodity"] == "tomato"].reset_index(drop=True)
    assert np.isclose(g.loc[30, "ret_7"], np.log(g.loc[30, "price"] / g.loc[23, "price"]))
