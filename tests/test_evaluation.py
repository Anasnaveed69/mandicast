import numpy as np
import pandas as pd
import pytest

from mandicast.evaluation import interval_metrics, point_metrics, walk_forward_folds


def test_folds_are_ordered_and_leave_room_for_targets():
    dates = pd.date_range("2022-01-01", periods=800)
    folds = walk_forward_folds(dates, n_folds=4, test_size=30, max_horizon=7, min_train_days=365)
    assert len(folds) == 4
    assert folds[-1].origins[-1] == dates[-1] - pd.Timedelta(days=7)
    for a, b in zip(folds, folds[1:]):
        assert a.origins[-1] < b.origins[0]
        assert a.cutoff < b.cutoff
    for f in folds:
        assert f.cutoff == f.origins[0]


def test_folds_refuse_short_history():
    with pytest.raises(ValueError):
        walk_forward_folds(pd.date_range("2022-01-01", periods=300), 5, 30, 7, 365)


def test_metrics():
    m = point_metrics([100, 200], [110, 190], scale=20)
    assert m["mae"] == 10 and m["mase"] == 0.5 and np.isclose(m["mape"], 7.5)
    y = np.array([1.0, 5.0])
    im = interval_metrics(y, np.zeros(2), np.ones(2), np.full(2, 2.0))
    assert im["coverage80"] == 0.5
