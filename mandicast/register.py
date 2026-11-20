"""Pick the best model per commodity and horizon, refit it on all data and put it
in the MLflow model registry.

Registered names look like `mandicast-tomato-h7`. Each version is tagged with the
data hash, git commit and its walk-forward score. The `champion` alias only moves
to a new version if it scores at least as well as the current champion
(otherwise the new version gets the `challenger` alias).
"""
import json
import logging
import shutil

import cloudpickle
import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from .backtest import load_data
from .config import (
    FEATURES_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    ROOT,
    file_md5,
    git_commit,
    load_params,
    setup_logging,
    tracking_uri,
)
from .models import build_model
from .serving import MandiCastModel

log = logging.getLogger("register")

NAME_TO_KEY = {"naive": "naive", "seasonal_naive": "seasonal_naive", "linear": "linear",
               "xgboost": "xgboost", "patchtst": "patchtst", "chronos_zeroshot": "chronos"}
OUT_DIR = MODELS_DIR / "registry"


def current_champion(client, name):
    try:
        mv = client.get_model_version_by_alias(name, "champion")
    except MlflowException:
        return None
    return mv


def main():
    setup_logging()
    params = load_params()
    rp = params["register"]
    horizons = params["backtest"]["horizons"]
    board = pd.read_csv(REPORTS_DIR / "benchmark.csv")
    data = load_data()
    last_date = data.prices.index[-1]
    data_md5, commit = file_md5(FEATURES_PATH), git_commit()

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    mlflow.set_tracking_uri(tracking_uri(params))
    mlflow.set_experiment(params["mlflow"]["registry_experiment"])
    client = MlflowClient()

    fitted = {}          # global models are fitted once and shared across commodities
    history_example = (data.prices.iloc[-400:].reset_index()
                       .melt(id_vars="date", var_name="commodity", value_name="price"))
    results = []

    for (c, h), g in board.groupby(["commodity", "horizon"]):
        g = g.sort_values(rp["metric"])
        win = g.iloc[0]
        naive = g[g["model"] == "naive"].iloc[0]
        key = NAME_TO_KEY[win["model"]]

        if key not in fitted:
            log.info("refitting %s on all data up to %s", key, last_date.date())
            fitted[key] = build_model(key, params).fit(data, last_date, horizons)
            with open(OUT_DIR / f"{key}.pkl", "wb") as f:
                cloudpickle.dump(fitted[key], f)
        model = fitted[key]

        cfg = {"commodity": c, "horizon": int(h), "model": win["model"],
               "features": params["features"], "trained_until": str(last_date.date())}
        cfg_path = OUT_DIR / f"{c}_h{h}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2))

        reg_name = f"{rp['model_prefix']}-{c}-h{h}"
        cv = {m: float(win[m]) for m in ["mae", "rmse", "mape", "mase", "pinball", "coverage80"]
              if pd.notna(win[m])}

        with mlflow.start_run(run_name=f"register-{c}-h{h}"):
            mlflow.set_tags({"commodity": c, "horizon": int(h), "model": win["model"],
                             "family": model.family, "git_commit": commit, "data_md5": data_md5})
            mlflow.log_params(model.log_params()
                              | {"model": win["model"], "trained_until": cfg["trained_until"]})
            mlflow.log_metrics({f"cv_{k}": v for k, v in cv.items()} |
                               {"cv_mase_naive": float(naive[rp["metric"]])})
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=MandiCastModel(),
                artifacts={"forecaster": str(OUT_DIR / f"{key}.pkl"), "config": str(cfg_path)},
                code_paths=[str(ROOT / "mandicast")],
                input_example=history_example,
                pip_requirements=str(ROOT / "requirements.txt"),
                registered_model_name=reg_name,
            )

        version = str(info.registered_model_version)
        client.update_registered_model(
            reg_name, description=f"MandiCast {h}-day ahead forecast for Lahore {c} (Rs/100kg).")
        for k, v in {"model": win["model"], "family": model.family, "cv_mase": f"{cv['mase']:.4f}",
                     "data_md5": data_md5, "git_commit": commit,
                     "trained_until": cfg["trained_until"]}.items():
            client.set_model_version_tag(reg_name, version, k, v)

        champ = current_champion(client, reg_name)
        champ_score = float(champ.tags.get("cv_mase", "inf")) if champ else float("inf")
        promote = (not rp["promote_only_if_better"]) or champ is None or cv["mase"] <= champ_score
        alias = "champion" if promote else "challenger"
        client.set_registered_model_alias(reg_name, alias, version)
        log.info("%s v%s (%s, cv MASE %.3f) -> @%s", reg_name, version, win["model"], cv["mase"], alias)

        results.append({"registered_model": reg_name, "version": version, "alias": alias,
                        "model": win["model"], "family": model.family,
                        "cv_mase": round(cv["mase"], 4),
                        "naive_mase": round(float(naive[rp["metric"]]), 4)})

    (REPORTS_DIR / "registry.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
