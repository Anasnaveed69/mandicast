import numpy as np
import pandas as pd

from mandicast.config import load_params
from mandicast.prepare import map_commodities, prepare, splice_varieties


def _raw(prices):
    raw = prices.rename(columns={"commodity": "commodity_raw", "price": "fqp"}).drop(columns="imputed")
    raw["commodity_raw"] = raw["commodity_raw"].str.title().replace({"Potato": "Potato Fresh"})
    raw["min_price"] = raw["fqp"] * 0.95
    raw["max_price"] = raw["fqp"] * 1.05
    raw["city"] = "Lahore"
    raw["source"] = "test"
    raw["scraped_at"] = "2026-01-01T00:00:00"
    return raw


def test_commodity_priority():
    names = ["Potato (Store)", "43&nbspPotato Fresh", "48&nbspTomato", "Garlic"]
    raw = pd.DataFrame({"commodity_raw": names})
    out = map_commodities(raw, load_params()["data"]["commodities"])
    assert set(out["commodity"]) == {"potato", "tomato"}
    fresh = out[out["commodity_raw"] == "Potato Fresh"]
    assert len(fresh) == 1
    assert fresh["priority"].iloc[0] == 0


def test_prepare_fills_gaps_and_spikes(prices):
    raw = _raw(prices)
    raw = raw.drop(raw[(raw["commodity_raw"] == "Tomato")].index[100:104])   # 4-day gap
    spike = raw[raw["commodity_raw"] == "Onion"].index[200]
    raw.loc[spike, "fqp"] *= 10                                             # typo-like spike
    params = load_params()
    params["prepare"]["start_date"] = "2000-01-01"
    out, quality = prepare(raw, params)

    assert quality["tomato"]["missing_days"] == 4
    assert quality["onion"]["spikes_fixed"] >= 1
    steps = out.groupby("commodity")["date"].diff().dropna()
    assert (steps == pd.Timedelta("1D")).all()
    onion = out[out["commodity"] == "onion"].set_index("date")["price"]
    true = prices[prices["commodity"] == "onion"].set_index("date")["price"]
    assert np.isclose(onion.iloc[200], true.iloc[200], rtol=0.15)


def test_potato_falls_back_only_on_long_gaps():
    days = pd.date_range("2026-01-01", periods=60)
    fresh = pd.DataFrame({"date": days, "commodity_raw": "Potato Fresh", "priority": 0, "price": 2000.0})
    fresh = fresh.drop(index=list(range(10, 13)) + list(range(30, 60)))   # 3-day gap, then gone
    sugar = pd.DataFrame({"date": days, "commodity_raw": "Potato Sugar free", "priority": 1, "price": 3000.0})
    out, used = splice_varieties(pd.concat([fresh, sugar]), max_gap=7)
    out = out.set_index("date")["price"]
    assert days[11] not in out.index   # short gap is left for interpolation, not filled
    assert (out[days[:10]] == 2000).all()
    assert (out[days[30:]] == 3000).all()
    assert used == {"Potato Fresh": 27, "Potato Sugar free": 30}
