"""Export the current champion models from the MLflow registry into a self-contained
folder the API can load without an MLflow server (MODEL_SOURCE=local):

    python -m mandicast.export                       # -> models/serving/
    python -m mandicast.export --alias challenger --out /tmp/bundle

Layout:
    models/serving/manifest.json
    models/serving/mandicast-tomato-h7/MLmodel, python_model.pkl, artifacts/, code/ ...
"""
import argparse
import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

from .config import MODELS_DIR, git_commit, load_params, setup_logging, tracking_uri

log = logging.getLogger("export")


def write_manifest(out_dir, entries, source):
    manifest = {
        "source": source,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "models": entries,
    }
    (Path(out_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _model_root(path):
    path = Path(path)
    if (path / "MLmodel").exists():
        return path
    return next(path.rglob("MLmodel")).parent


def export_registry(out_dir, params, alias="champion"):
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri(params))
    client = MlflowClient()
    prefix = params["register"]["model_prefix"]
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    entries = []
    for c in params["data"]["commodities"]:
        for h in params["backtest"]["horizons"]:
            name = f"{prefix}-{c}-h{h}"
            mv = client.get_model_version_by_alias(name, alias)
            local = mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/{name}@{alias}", dst_path=str(out_dir / name))
            tags = mv.tags or {}
            entries.append({
                "name": name, "commodity": c, "horizon": int(h), "version": str(mv.version),
                "alias": alias, "model": tags.get("model"), "family": tags.get("family"),
                "cv_mase": tags.get("cv_mase"), "trained_until": tags.get("trained_until"),
                "path": str(_model_root(local).relative_to(out_dir)),
            })
            log.info("%s v%s (%s) -> %s", name, mv.version, tags.get("model"), entries[-1]["path"])
    return write_manifest(out_dir, entries, source="mlflow-registry")


def main(argv=None):
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(MODELS_DIR / "serving"))
    ap.add_argument("--alias", default="champion")
    args = ap.parse_args(argv)
    m = export_registry(args.out, load_params(), args.alias)
    log.info("exported %d models to %s", len(m["models"]), args.out)


if __name__ == "__main__":
    main()
