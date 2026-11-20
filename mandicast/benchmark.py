"""Walk-forward benchmark of every enabled model family, logged to MLflow.

MLflow layout: one parent run per pipeline execution, and one nested run per
(model, commodity, horizon) with fold-level metrics logged as steps.
"""
import json
import logging
import tempfile
from pathlib import Path

import matplotlib
import mlflow
import pandas as pd

from .backtest import load_data, run_backtest, summarise
from .config import (
    FEATURES_PATH,
    REPORTS_DIR,
    file_md5,
    git_commit,
    load_params,
    setup_logging,
    tracking_uri,
)
from .evaluation import walk_forward_folds
from .models import build_model, enabled_models

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend has to be set first)

log = logging.getLogger("benchmark")

METRICS = ["mae", "rmse", "mape", "mase", "pinball", "coverage80"]


def plot_best(preds, leaderboard, out_dir):
    paths = []
    for (c, h), g in leaderboard.groupby(["commodity", "horizon"]):
        best = g.sort_values("mase").iloc[0]["model"]
        sub = preds[(preds["commodity"] == c) & (preds["horizon"] == h)]
        fig, ax = plt.subplots(figsize=(10, 3.5))
        actual = sub.drop_duplicates("target_date").sort_values("target_date")
        ax.plot(actual["target_date"], actual["y"], color="black", lw=1.2, label="actual")
        for name, style in [("naive", ":"), (best, "-")]:
            s = sub[sub["model"] == name].sort_values("target_date")
            ax.plot(s["target_date"], s["yhat"], ls=style, lw=1.2, label=name)
            if name == best and s["q10"].notna().any():
                ax.fill_between(s["target_date"], s["q10"], s["q90"], alpha=0.2, label="80% interval")
        ax.set_title(f"{c} - {h}-day ahead (walk-forward test folds)")
        ax.set_ylabel("Rs / 100 kg")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        path = Path(out_dir) / f"{c}_h{h}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths


def main():
    setup_logging()
    params = load_params()
    bt = params["backtest"]
    data = load_data()
    folds = walk_forward_folds(data.prices.index, bt["n_folds"], bt["test_size"],
                               max(bt["horizons"]), bt["min_train_days"])
    log.info("%d folds, first cutoff %s, last origin %s", len(folds),
             folds[0].cutoff.date(), folds[-1].origins[-1].date())

    mlflow.set_tracking_uri(tracking_uri(params))
    mlflow.set_experiment(params["mlflow"]["benchmark_experiment"])

    all_preds, all_times = [], []
    with mlflow.start_run(run_name="walk_forward_benchmark") as parent:
        mlflow.set_tags({"git_commit": git_commit(), "data_md5": file_md5(FEATURES_PATH),
                         "stage": "benchmark"})
        mlflow.log_params({
            "n_folds": bt["n_folds"], "test_size": bt["test_size"],
            "horizons": ",".join(map(str, bt["horizons"])),
            "data_start": str(data.prices.index[0].date()),
            "data_end": str(data.prices.index[-1].date()),
            "commodities": ",".join(data.commodities),
        })

        for key in enabled_models(params):
            try:
                build_model(key, params)
            except ImportError as e:
                log.warning("skipping %s: %s", key, e)
                continue
            try:
                preds, times = run_backtest(lambda k=key: build_model(k, params), data, folds, bt["horizons"])
            except Exception as e:
                # Chronos needs the HF hub on first run; don't lose the whole
                # benchmark because of a download problem. Anything else is a bug.
                if key != "chronos":
                    raise
                log.warning("skipping chronos, could not load %s: %s",
                            params["models"]["chronos"]["model_id"], e)
                continue
            all_preds.append(preds)
            all_times.append(times)

            summary, per_fold = summarise(preds)
            probe = build_model(key, params)
            for _, row in summary.iterrows():
                c, h = row["commodity"], int(row["horizon"])
                name = row["model"]
                with mlflow.start_run(run_name=f"{name}-{c}-h{h}", nested=True):
                    mlflow.set_tags({"model": name, "family": probe.family,
                                     "commodity": c, "horizon": h,
                                     "parent_run": parent.info.run_id})
                    mlflow.log_params(probe.log_params() | {"model": name, "commodity": c, "horizon": h})
                    mlflow.log_metrics({m: float(row[m]) for m in METRICS if pd.notna(row[m])})
                    t = times[times["horizon"] == h]
                    mlflow.log_metrics({"fit_seconds": float(t["fit_seconds"].mean()),
                                        "predict_ms": float(t["predict_ms"].mean())})
                    fold_rows = per_fold[(per_fold["commodity"] == c) & (per_fold["horizon"] == h)]
                    for _, fr in fold_rows.iterrows():
                        mlflow.log_metrics({f"fold_{m}": float(fr[m]) for m in ("mae", "mase")
                                            if pd.notna(fr[m])}, step=int(fr["fold"]))

        preds = pd.concat(all_preds, ignore_index=True)
        times = pd.concat(all_times, ignore_index=True)
        leaderboard, _ = summarise(preds)
        cost = times.groupby(["model", "horizon"])[["fit_seconds", "predict_ms"]].mean().reset_index()
        leaderboard = leaderboard.merge(cost, on=["model", "horizon"], how="left")
        leaderboard = leaderboard.sort_values(["commodity", "horizon", "mase"]).reset_index(drop=True)
        leaderboard["rank"] = leaderboard.groupby(["commodity", "horizon"]).cumcount() + 1

        REPORTS_DIR.mkdir(exist_ok=True)
        leaderboard.round(4).to_csv(REPORTS_DIR / "benchmark.csv", index=False)
        preds.to_parquet(REPORTS_DIR / "predictions.parquet", index=False)

        best = leaderboard[leaderboard["rank"] == 1]
        summary_json = {f"{r.commodity}_h{r.horizon}": {"best_model": r.model,
                                                       "mase": round(float(r.mase), 4),
                                                       "mape": round(float(r.mape), 2)}
                        for r in best.itertuples()}
        summary_json["mean_best_mase"] = round(float(best["mase"].mean()), 4)
        (REPORTS_DIR / "benchmark_summary.json").write_text(json.dumps(summary_json, indent=2))

        mlflow.log_metric("mean_best_mase", summary_json["mean_best_mase"])
        mlflow.log_artifact(str(REPORTS_DIR / "benchmark.csv"))
        with tempfile.TemporaryDirectory() as tmp:
            for p in plot_best(preds, leaderboard, tmp):
                mlflow.log_artifact(str(p), artifact_path="plots")

    cols = ["commodity", "horizon", "rank", "model", "mae", "mape", "mase", "coverage80"]
    log.info("leaderboard\n%s", leaderboard[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
