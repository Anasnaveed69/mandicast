import numpy as np
import pandas as pd
import pytest

from mandicast.backtest import run_backtest, summarise
from mandicast.evaluation import walk_forward_folds
from mandicast.features import build_features
from mandicast.models.base import SeriesData
from mandicast.models.baselines import Naive, SeasonalNaive
from mandicast.models.patchtst import PatchTSTModel
from mandicast.models.tabular import LinearModel, XGBoostModel

SMALL_PATCH = dict(input_len=32, patch_len=8, stride=4, d_model=16, n_heads=2, n_layers=1,
                   dropout=0.1, lr=1e-3, batch_size=64, max_epochs=3, patience=2)


def _origins(data, date):
    return pd.DataFrame({"commodity": data.commodities, "date": [date] * len(data.commodities)})


def test_naive_and_seasonal(data):
    d = data.prices.index[500]
    o = _origins(data, d)
    assert np.allclose(Naive().predict(data, o, 7)["yhat"], data.prices.loc[d].to_numpy())
    sn = SeasonalNaive(season=7).predict(data, o, 1)["yhat"]
    assert np.allclose(sn, data.prices.loc[d - pd.Timedelta(days=6)].to_numpy())


@pytest.mark.parametrize("make", [
    lambda: LinearModel(alpha=1.0),
    lambda: XGBoostModel(n_estimators=30, max_depth=3),
    lambda: PatchTSTModel(**SMALL_PATCH),
])
def test_no_future_leakage(prices, make):
    """Fitting at a cut-off and predicting there must not depend on later prices."""
    lags, wins = [1, 7, 28], [7, 28]
    cut = pd.Timestamp(prices["date"].unique()[450])

    def run(p):
        data = SeriesData.from_frames(p, build_features(p, lags, wins))
        model = make().fit(data, cut, [1, 7])
        return model.predict(data, _origins(data, cut), 7)

    changed = prices.copy()
    changed.loc[changed["date"] > cut, "price"] *= 5
    a, b = run(prices), run(changed)
    assert np.allclose(a["yhat"], b["yhat"])
    assert (a["yhat"] > 0).all()


def test_quantiles_ordered(data):
    m = XGBoostModel(n_estimators=30).fit(data, data.prices.index[500], [7])
    p = m.predict(data, _origins(data, data.prices.index[500]), 7)
    assert (p["q10"] <= p["yhat"]).all() and (p["yhat"] <= p["q90"]).all()


def test_backtest_end_to_end(data):
    folds = walk_forward_folds(data.prices.index, 2, 20, 7, 365)
    preds, times = run_backtest(lambda: XGBoostModel(n_estimators=30), data, folds, [1, 7])
    assert preds["y"].notna().all()
    assert len(preds) == 2 * 20 * 3 * 2
    summary, _ = summarise(preds)
    assert set(summary["horizon"]) == {1, 7}
    assert (summary["mase"] > 0).all()
