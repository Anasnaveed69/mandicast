"""Synthetic prices + a small model bundle, so the API/UI stack can start without
the real data or an MLflow server (CI smoke tests, first `docker compose up`).

    python -m mandicast.demo --out demo
    # -> demo/data/processed/prices.parquet, demo/models/serving/{manifest.json, ...}

The models are a naive baseline (h=1) and the ridge model (h=7) trained on fake
data. Everything served from this bundle is labelled source="demo" by the API.
"""
import argparse
import json
import logging
import tempfile
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd

from .config import ROOT, load_params, setup_logging
from .export import write_manifest
from .features import build_features
from .models.base import SeriesData
from .models.baselines import Naive
from .models.tabular import LinearModel

log = logging.getLogger("demo")

LEVELS = {"onion": 6000, "potato": 3500, "tomato": 7000}


def make_prices(days=500, seed=0, end=None):
    """Yearly cycle + random walk, rounded like real mandi prices."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end or pd.Timestamp.today()).normalize()
    dates = pd.date_range(end=end, periods=days, freq="D")
    t = np.arange(days)
    rows = []
    for i, (c, level) in enumerate(LEVELS.items()):
        lp = (np.log(level) + 0.35 * np.sin(2 * np.pi * (t + 60 * i) / 365)
              + np.cumsum(rng.normal(0, 0.02, days)))
        rows.append(pd.DataFrame({"date": dates, "commodity": c,
                                  "price": np.exp(lp).round(0), "imputed": False}))
    return pd.concat(rows, ignore_index=True)


def save_pyfunc(path, forecaster, cfg, history_example):
    import mlflow.pyfunc

    from .serving import MandiCastModel

    with tempfile.TemporaryDirectory() as tmp:
        fc, cf = Path(tmp) / "forecaster.pkl", Path(tmp) / "config.json"
        with open(fc, "wb") as f:
            cloudpickle.dump(forecaster, f)
        cf.write_text(json.dumps(cfg, indent=2))
        mlflow.pyfunc.save_model(
            path=str(path),
            python_model=MandiCastModel(),
            artifacts={"forecaster": str(fc), "config": str(cf)},
            code_paths=[str(ROOT / "mandicast")],
            input_example=history_example,
            pip_requirements=[],
        )


def build_demo(out, days=500, params=None):
    params = params or load_params()
    out = Path(out)
    data_dir, serving = out / "data" / "processed", out / "models" / "serving"
    data_dir.mkdir(parents=True, exist_ok=True)
    serving.mkdir(parents=True, exist_ok=True)

    prices = make_prices(days)
    prices.to_parquet(data_dir / "prices.parquet", index=False)

    fp = params["features"]
    feats = build_features(prices, fp["lags"], fp["rolling_windows"], fp["long_window"])
    data = SeriesData.from_frames(prices, feats)
    last = data.prices.index[-1]
    horizons = params["backtest"]["horizons"]
    fitted = {"naive": Naive(), "linear": LinearModel(alpha=1.0).fit(data, last, horizons)}
    example = prices[prices["date"] > last - pd.Timedelta(days=400)][["date", "commodity", "price"]]
    example = example.astype({"date": "datetime64[ns]", "price": "float64"}).reset_index(drop=True)

    entries = []
    for c in LEVELS:
        for h in horizons:
            key = "naive" if h == 1 else "linear"
            name = f"{params['register']['model_prefix']}-{c}-h{h}"
            cfg = {"commodity": c, "horizon": int(h), "model": key, "features": fp,
                   "trained_until": str(last.date())}
            save_pyfunc(serving / name, fitted[key], cfg, example)
            entries.append({"name": name, "commodity": c, "horizon": int(h), "version": "demo",
                            "alias": "demo", "model": key, "family": fitted[key].family,
                            "cv_mase": None, "trained_until": str(last.date()), "path": name})
    write_manifest(serving, entries, source="demo")
    log.info("demo bundle with %d models and %d days of fake prices -> %s", len(entries), days, out)
    return out


def main(argv=None):
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="demo")
    ap.add_argument("--days", type=int, default=500)
    args = ap.parse_args(argv)
    build_demo(args.out, args.days)


if __name__ == "__main__":
    main()
