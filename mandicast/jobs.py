"""Jobs run by the `scheduler` and `pipeline` containers.

    python -m mandicast.jobs refresh     # scrape today's AMIS prices -> prices/features parquet
    python -m mandicast.jobs retrain     # dvc pull -> dvc repro -> MLflow registry -> dvc push -> API reload
    python -m mandicast.jobs schedule    # loop: refresh every day at REFRESH_AT (Asia/Karachi),
                                         #       retrain on RETRAIN_WEEKDAY if set
    python -m mandicast.jobs notify      # tell the API to reload its models

Environment:
    API_URL, ADMIN_TOKEN           where to send the reload (default http://api:8000)
    REFRESH_AT                     daily refresh time, HH:MM in Asia/Karachi (default 18:30)
    RETRAIN_WEEKDAY                e.g. "sun" to also retrain weekly from the scheduler (default off;
                                   the GitHub Actions retrain workflow normally does this)
    DVC_REMOTE_URL                 e.g. s3://bucket/mandicast, set as the local default remote
    DVC_PULL / DVC_PUSH            "1" to pull before / push after retraining (default 1 if a remote is set)
"""
import argparse
import json
import logging
import os
import subprocess
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .config import (
    FEATURES_PATH,
    PRICES_PATH,
    REPORTS_DIR,
    ROOT,
    load_params,
    setup_logging,
    tracking_uri,
)

log = logging.getLogger("jobs")
PKT = ZoneInfo("Asia/Karachi")
WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def atomic_parquet(df, path):
    """Write next to the target and rename, so the API never reads half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def rebuild_processed(params):
    """prepare + features, same code as the DVC stages, written atomically."""
    from .features import build_features
    from .prepare import prepare

    raw = pd.read_csv(ROOT / params["data"]["raw_path"])
    prices, quality = prepare(raw, params)
    fp = params["features"]
    feats = build_features(prices, fp["lags"], fp["rolling_windows"], fp["long_window"])
    atomic_parquet(prices, PRICES_PATH)
    atomic_parquet(feats, FEATURES_PATH)
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "data_quality.json").write_text(json.dumps(quality, indent=2))
    log.info("processed data rebuilt, last date %s", prices["date"].max().date())
    return prices


def refresh(params):
    from .scrape import main as scrape_main

    try:
        scrape_main(["daily"])
    except (Exception, SystemExit) as e:   # AMIS down / layout changed: keep serving old data
        log.error("daily scrape failed: %s", e)
        return False
    rebuild_processed(params)
    notify_api()
    return True


def run(cmd, check=True):
    log.info("$ %s", " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT, check=check).returncode


def _dvc_remote():
    url = os.environ.get("DVC_REMOTE_URL")
    if not url:
        return False
    run(["dvc", "remote", "add", "--local", "--force", "--default", "storage", url])
    if os.environ.get("AWS_DEFAULT_REGION"):
        run(["dvc", "remote", "modify", "--local", "storage", "region", os.environ["AWS_DEFAULT_REGION"]])
    return True


def missing_champions(params):
    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    mlflow.set_tracking_uri(tracking_uri(params))
    client = MlflowClient()
    prefix = params["register"]["model_prefix"]
    missing = []
    for c in params["data"]["commodities"]:
        for h in params["backtest"]["horizons"]:
            name = f"{prefix}-{c}-h{h}"
            try:
                client.get_model_version_by_alias(name, "champion")
            except MlflowException:
                missing.append(name)
    return missing


def retrain(params):
    has_remote = _dvc_remote()
    flag = lambda name: os.environ.get(name, "1" if has_remote else "0") == "1"  # noqa: E731
    if flag("DVC_PULL"):
        run(["dvc", "pull", "--allow-missing"], check=False)
    run(["dvc", "repro"])
    # dvc skips `register` when nothing changed, e.g. the first run against a fresh
    # MLflow server: force it so the registry always has a champion per model
    missing = missing_champions(params)
    if missing:
        log.info("no champion yet for %s, forcing the register stage", ", ".join(missing))
        run(["dvc", "repro", "--force", "--single-item", "register"])
    if flag("DVC_PUSH"):
        run(["dvc", "push"], check=False)
    notify_api()


def notify_api():
    url = os.environ.get("API_URL", "http://api:8000").rstrip("/") + "/admin/reload"
    req = urllib.request.Request(url, method="POST",
                                 headers={"X-Admin-Token": os.environ.get("ADMIN_TOKEN", "")})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            log.info("API reloaded: %s", r.read().decode()[:200])
            return True
    except Exception as e:
        log.warning("could not reach %s: %s", url, e)
        return False


def next_run(now, at):
    """Next datetime (same tz as `now`) at clock time `at` ("HH:MM")."""
    hh, mm = (int(x) for x in at.split(":"))
    t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return t if t > now else t + timedelta(days=1)


def schedule(params):
    at = os.environ.get("REFRESH_AT", "18:30")
    retrain_day = os.environ.get("RETRAIN_WEEKDAY", "").strip().lower()[:3]
    if retrain_day and retrain_day not in WEEKDAYS:
        raise SystemExit(f"RETRAIN_WEEKDAY must be one of {WEEKDAYS}")
    log.info("scheduler: refresh daily at %s PKT%s", at,
             f", retrain on {retrain_day}" if retrain_day else "")
    while True:
        wake = next_run(datetime.now(PKT), at)
        log.info("next run at %s", wake.isoformat(timespec="minutes"))
        while (left := (wake - datetime.now(PKT)).total_seconds()) > 0:
            time.sleep(min(left, 300))
        try:
            refresh(params)
            if retrain_day == WEEKDAYS[wake.weekday()]:
                retrain(params)
        except Exception:
            log.exception("scheduled job failed")


def main(argv=None):
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", choices=["refresh", "rebuild", "retrain", "schedule", "notify"])
    args = ap.parse_args(argv)
    params = load_params()
    if args.job == "refresh":
        raise SystemExit(0 if refresh(params) else 1)
    if args.job == "rebuild":
        rebuild_processed(params)
    elif args.job == "retrain":
        retrain(params)
    elif args.job == "schedule":
        schedule(params)
    else:
        notify_api()


if __name__ == "__main__":
    main()
