"""Chronos-Bolt used zero-shot (no training on our data).

The weights come from the Hugging Face hub on first use (~35 MB for the tiny
model) and are cached by transformers after that.
"""
import logging

import numpy as np
import torch

from .base import PRED_COLUMNS, Forecaster

log = logging.getLogger("chronos")


class ChronosZeroShot(Forecaster):
    name = "chronos_zeroshot"
    family = "transformer_pretrained"
    has_quantiles = True

    def __init__(self, quantiles=(0.1, 0.5, 0.9), batch_size=64, **params):
        super().__init__(**params)
        self.quantiles = list(quantiles)
        self.batch_size = batch_size
        self.pipeline = None
        self.max_h = None
        self._cache = {}

    def _load(self):
        if self.pipeline is None:
            from chronos import BaseChronosPipeline
            self.pipeline = BaseChronosPipeline.from_pretrained(
                self.params["model_id"], device_map="cpu", torch_dtype=torch.float32)
        return self.pipeline

    def fit(self, data, train_end, horizons):
        # nothing to train; just make sure the model loads and reset the cache
        self.max_h = max(horizons)
        self._cache = {}
        self._load()
        return self

    def _forecast(self, data, keys):
        todo = [k for k in keys if k not in self._cache]
        ctx_len = self.params.get("context_length", 512)
        pipe = self._load()
        for i in range(0, len(todo), self.batch_size):
            batch = todo[i:i + self.batch_size]
            ctx = [torch.tensor(data.history(c, d).to_numpy(dtype=np.float32)[-ctx_len:])
                   for c, d in batch]
            q, _ = pipe.predict_quantiles(ctx, prediction_length=self.max_h,
                                          quantile_levels=self.quantiles)
            q = q.numpy()                       # (B, H, Q)
            for key, qq in zip(batch, q):
                self._cache[key] = qq

    def predict(self, data, origins, horizon):
        keys = list(zip(origins["commodity"], origins["date"]))
        self._forecast(data, keys)
        q = np.sort(np.stack([self._cache[k][horizon - 1] for k in keys]), axis=1)
        pred = origins[["commodity", "date"]].reset_index(drop=True).copy()
        pred["q10"], pred["yhat"], pred["q90"] = q[:, 0], q[:, 1], q[:, 2]
        return pred[PRED_COLUMNS]

    def __getstate__(self):
        # don't pickle the HF pipeline, it is reloaded from model_id when needed
        state = self.__dict__.copy()
        state["pipeline"] = None
        state["_cache"] = {}
        return state
