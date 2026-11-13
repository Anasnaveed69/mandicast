"""Optuna search for XGBoost.

Only data before the first benchmark fold is used (an inner walk-forward), so the
benchmark folds stay untouched and the XGBoost score there is not optimistic.
"""
import json
import logging

import mlflow
import optuna

from .backtest import load_data, mean_mase, run_backtest
from .config import XGB_PARAMS_PATH, git_commit, load_params, setup_logging
from .evaluation import walk_forward_folds
from .models.tabular import XGBoostModel

log = logging.getLogger("tune")


def main():
    setup_logging()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    params = load_params()
    bt, tp = params["backtest"], params["tune"]
    data = load_data()

    outer = walk_forward_folds(data.prices.index, bt["n_folds"], bt["test_size"],
                               max(bt["horizons"]), bt["min_train_days"])
    inner_dates = data.prices.index[data.prices.index <= outer[0].cutoff]
    inner = walk_forward_folds(inner_dates, tp["inner_folds"], bt["test_size"],
                               max(bt["horizons"]), min_train_days=180)

    base = {k: v for k, v in params["models"]["xgboost"].items() if k not in ("enabled", "use_tuned")}

    def objective(trial):
        hp = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            preds, _ = run_backtest(
                lambda: XGBoostModel(quantiles=params["quantiles"], seed=params["seed"], **hp),
                data, inner, bt["horizons"])
            score = mean_mase(preds)
            mlflow.log_params(hp)
            mlflow.log_metric("inner_mase", score)
        return score

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["tuning_experiment"])
    with mlflow.start_run(run_name="xgboost_optuna"):
        mlflow.set_tags({"git_commit": git_commit(), "model": "xgboost"})
        mlflow.log_params({"n_trials": tp["n_trials"], "inner_folds": tp["inner_folds"],
                           "inner_last_date": str(inner_dates[-1].date())})

        # trial 0 is the hand-picked config from params.yaml
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=params["seed"]))
        study.enqueue_trial({k: base[k] for k in base if k in
                             ("n_estimators", "max_depth", "learning_rate", "subsample",
                              "colsample_bytree", "min_child_weight")} | {"reg_lambda": 1.0})
        study.optimize(objective, n_trials=tp["n_trials"])

        mlflow.log_metric("best_inner_mase", study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        log.info("best inner MASE %.4f with %s", study.best_value, study.best_params)

    XGB_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    XGB_PARAMS_PATH.write_text(json.dumps(study.best_params, indent=2))


if __name__ == "__main__":
    main()
