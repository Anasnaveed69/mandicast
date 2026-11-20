import hashlib
import logging
import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

PRICES_PATH = PROCESSED_DIR / "prices.parquet"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"
XGB_PARAMS_PATH = MODELS_DIR / "xgb_params.json"


def load_params(path=None):
    path = Path(path) if path else ROOT / "params.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def file_md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def tracking_uri(params):
    """MLflow tracking URI. MLFLOW_TRACKING_URI (set in the containers to the MLflow
    server) wins over params.yaml, which keeps the local sqlite default."""
    return os.environ.get("MLFLOW_TRACKING_URI") or params["mlflow"]["tracking_uri"]


def git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        # the API image has no .git; the image build passes the commit in instead
        return os.environ.get("GIT_COMMIT", "unknown")[:7]
