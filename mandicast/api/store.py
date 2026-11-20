"""Model and price stores used by the API.

ModelStore loads one pyfunc per (commodity, horizon), either from the MLflow
registry (`models:/mandicast-<c>-h<h>@champion`) or from a bundle exported with
`python -m mandicast.export` (models/serving/manifest.json + one folder per model).

PriceStore reads data/processed/prices.parquet and reloads it whenever the file
changes, so the daily refresh job shows up without restarting the API.
"""
import json
import logging
import math
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

log = logging.getLogger("api.store")


class ModelNotFound(LookupError):
    pass


class DataUnavailable(RuntimeError):
    pass


@dataclass
class LoadedModel:
    name: str
    commodity: str
    horizon: int
    version: str | None = None
    alias: str | None = None
    model: str | None = None
    family: str | None = None
    cv_mase: float | None = None
    trained_until: str | None = None
    source: str = "mlflow"
    pyfunc: Any = field(default=None, repr=False)

    def info(self):
        d = asdict(self)
        d.pop("pyfunc")
        return d


def _float_or_none(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


class ModelStore:
    def __init__(self, settings):
        self.s = settings
        self._models: dict[tuple[str, int], LoadedModel] = {}
        self._lock = threading.RLock()
        self._predict_lock = threading.Lock()
        self.errors: dict[str, str] = {}
        self.loaded_at: str | None = None
        self.bundle_source: str | None = None

    # ---------------------------------------------------------------- loading
    def load(self):
        """(Re)load every model. Keeps the old models if the new load finds nothing."""
        if self.s.model_source == "local":
            models, errors = self._load_local()
        elif self.s.model_source == "mlflow":
            models, errors = self._load_mlflow()
        else:
            raise ValueError(f"MODEL_SOURCE must be 'mlflow' or 'local', got {self.s.model_source!r}")
        for name, err in errors.items():
            log.warning("could not load %s: %s", name, err)
        with self._lock:
            if models or not self._models:
                self._models = models
            self.errors = errors
            self.loaded_at = datetime.now(UTC).isoformat(timespec="seconds")
        log.info("%d/%d models loaded from %s", len(models),
                 len(self.s.commodities) * len(self.s.horizons), self.s.model_source)
        return len(models)

    def _load_mlflow(self):
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(self.s.tracking_uri)
        client = MlflowClient()
        models, errors = {}, {}
        for c in self.s.commodities:
            for h in self.s.horizons:
                name = self.s.model_name(c, h)
                try:
                    mv = client.get_model_version_by_alias(name, self.s.model_alias)
                    pyfunc = mlflow.pyfunc.load_model(f"models:/{name}@{self.s.model_alias}")
                except Exception as e:  # registry down, model not registered yet, ...
                    errors[name] = f"{type(e).__name__}: {e}"[:300]
                    continue
                tags = mv.tags or {}
                models[(c, h)] = LoadedModel(
                    name=name, commodity=c, horizon=h, version=str(mv.version),
                    alias=self.s.model_alias, model=tags.get("model"), family=tags.get("family"),
                    cv_mase=_float_or_none(tags.get("cv_mase")),
                    trained_until=tags.get("trained_until"), source="mlflow", pyfunc=pyfunc)
        self.bundle_source = "mlflow-registry"
        return models, errors

    def _load_local(self):
        import mlflow

        manifest_path = self.s.model_dir / "manifest.json"
        if not manifest_path.exists():
            return {}, {"manifest": f"{manifest_path} not found (run python -m mandicast.export)"}
        manifest = json.loads(manifest_path.read_text())
        models, errors = {}, {}
        for e in manifest["models"]:
            c, h = e["commodity"], int(e["horizon"])
            if c not in self.s.commodities or h not in self.s.horizons:
                continue
            try:
                pyfunc = mlflow.pyfunc.load_model(str(self.s.model_dir / e["path"]))
            except Exception as err:
                errors[e["name"]] = f"{type(err).__name__}: {err}"[:300]
                continue
            models[(c, h)] = LoadedModel(
                name=e["name"], commodity=c, horizon=h, version=e.get("version"),
                alias=e.get("alias"), model=e.get("model"), family=e.get("family"),
                cv_mase=_float_or_none(e.get("cv_mase")), trained_until=e.get("trained_until"),
                source="local", pyfunc=pyfunc)
        self.bundle_source = manifest.get("source", "local")
        return models, errors

    # ---------------------------------------------------------------- access
    @property
    def models(self):
        with self._lock:
            return dict(self._models)

    def get(self, commodity, horizon) -> LoadedModel:
        with self._lock:
            m = self._models.get((commodity, horizon))
        if m is None:
            raise ModelNotFound(f"no model loaded for {commodity} h={horizon}")
        return m

    def predict(self, model: LoadedModel, history: pd.DataFrame) -> dict:
        with self._predict_lock:      # torch / xgboost models aren't guaranteed thread safe
            out = model.pyfunc.predict(history)
        row = out.iloc[0]
        return {
            "commodity": str(row["commodity"]),
            "horizon": int(row["horizon"]),
            "origin_date": pd.Timestamp(row["origin_date"]).date(),
            "target_date": pd.Timestamp(row["target_date"]).date(),
            "yhat": _float_or_none(row["yhat"]),
            "q10": _float_or_none(row.get("q10")),
            "q90": _float_or_none(row.get("q90")),
        }


class PriceStore:
    """Latest cleaned prices (long frame: date, commodity, price)."""

    def __init__(self, path):
        self.path = path
        self._mtime = None
        self._df = None
        self._lock = threading.Lock()

    def frame(self) -> pd.DataFrame:
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError as e:
            raise DataUnavailable(f"{self.path} not found") from e
        with self._lock:
            if mtime != self._mtime:
                df = pd.read_parquet(self.path, columns=["date", "commodity", "price"])
                df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]").dt.normalize()
                df["price"] = df["price"].astype("float64")
                df["commodity"] = df["commodity"].astype(str)
                self._df = df.sort_values(["commodity", "date"]).reset_index(drop=True)
                self._mtime = mtime
                log.info("prices reloaded: %d rows, last date %s", len(df), df["date"].max().date())
            return self._df

    def last_date(self):
        try:
            return self.frame()["date"].max().date()
        except DataUnavailable:
            return None

    def recent(self, days) -> pd.DataFrame:
        df = self.frame()
        start = df["date"].max() - pd.Timedelta(days=days - 1)
        return df[df["date"] >= start].reset_index(drop=True)

    def series(self, commodity, days) -> pd.DataFrame:
        df = self.recent(days)
        return df[df["commodity"] == commodity]
