"""PatchTST (Nie et al., 2023) in plain PyTorch, sized for short daily series.

One global, channel-independent model: every commodity window is a sample.
Inputs are log prices with per-window normalisation (RevIN without the affine
part), the head outputs `max_h x n_quantiles` and is trained with pinball loss.
"""
import copy
import logging

import numpy as np
import torch
from torch import nn

from .base import PRED_COLUMNS, Forecaster

log = logging.getLogger("patchtst")


class PatchTSTNet(nn.Module):
    def __init__(self, input_len, horizon, n_quantiles, patch_len=8, stride=4,
                 d_model=64, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.horizon, self.n_q = horizon, n_quantiles
        n_patches = (input_len - patch_len) // stride + 1

        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2 * d_model,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(n_patches * d_model, horizon * n_quantiles)

    def forward(self, x):                      # x: (B, L) already normalised
        patches = x.unfold(1, self.patch_len, self.stride)   # (B, N, P)
        z = self.embed(patches) + self.pos
        z = self.encoder(z)
        z = self.drop(z.flatten(1))
        return self.head(z).view(-1, self.horizon, self.n_q)


def pinball_loss(pred, target, quantiles):
    # pred (B, H, Q), target (B, H)
    diff = target.unsqueeze(-1) - pred
    return torch.maximum(quantiles * diff, (quantiles - 1) * diff).mean()


class PatchTSTModel(Forecaster):
    name = "patchtst"
    family = "transformer"
    has_quantiles = True

    def __init__(self, quantiles=(0.1, 0.5, 0.9), seed=42, **params):
        super().__init__(**params)
        self.quantiles = list(quantiles)
        self.seed = seed
        self.net = None
        self.max_h = None

    def _windows(self, data, train_end, L, H):
        X, Y, ends = [], [], []
        for c in data.commodities:
            lp = np.log(data.history(c, train_end).to_numpy(dtype=np.float32))
            for end in range(L, len(lp) - H + 1):     # window lp[end-L:end], target lp[end:end+H]
                X.append(lp[end - L:end])
                Y.append(lp[end:end + H])
                ends.append(end)
        X, Y = np.stack(X), np.stack(Y)
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-3
        return (X - mu) / sd, (Y - mu) / sd, np.array(ends)

    def fit(self, data, train_end, horizons):
        p = self.params
        L, self.max_h = p["input_len"], max(horizons)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        X, Y, ends = self._windows(data, train_end, L, self.max_h)
        # time-based validation split: the latest 10% of window positions
        split = np.quantile(ends, 0.9)
        tr, va = ends < split, ends >= split
        Xtr, Ytr = torch.tensor(X[tr]), torch.tensor(Y[tr])
        Xva, Yva = torch.tensor(X[va]), torch.tensor(Y[va])

        self.net = PatchTSTNet(L, self.max_h, len(self.quantiles), p["patch_len"], p["stride"],
                               p["d_model"], p["n_heads"], p["n_layers"], p["dropout"])
        opt = torch.optim.AdamW(self.net.parameters(), lr=p["lr"], weight_decay=1e-4)
        qs = torch.tensor(self.quantiles)

        best, best_state, bad = np.inf, None, 0
        epochs_run = 0
        for _ in range(p["max_epochs"]):
            epochs_run += 1
            self.net.train()
            order = rng.permutation(len(Xtr))
            for i in range(0, len(order), p["batch_size"]):
                b = order[i:i + p["batch_size"]]
                loss = pinball_loss(self.net(Xtr[b]), Ytr[b], qs)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            self.net.eval()
            with torch.no_grad():
                val = pinball_loss(self.net(Xva), Yva, qs).item()
            if val < best - 1e-5:
                best, best_state, bad = val, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= p["patience"]:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        self.epochs_run, self.best_val_loss = epochs_run, best
        log.debug("stopped after %d epochs, val pinball %.4f", epochs_run, best)
        return self

    def predict(self, data, origins, horizon):
        L = self.params["input_len"]
        windows = []
        for c, d in zip(origins["commodity"], origins["date"]):
            hist = data.history(c, d).to_numpy(dtype=np.float32)
            windows.append(np.log(hist[-L:]))
        X = np.stack(windows)
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-3
        with torch.no_grad():
            out = self.net(torch.tensor((X - mu) / sd)).numpy()      # (B, H, Q)
        q = out[:, horizon - 1, :] * sd + mu
        q = np.exp(np.sort(q, axis=1))
        pred = origins[["commodity", "date"]].reset_index(drop=True).copy()
        pred["q10"], pred["yhat"], pred["q90"] = q[:, 0], q[:, 1], q[:, 2]
        return pred[PRED_COLUMNS]

    def log_params(self):
        out = super().log_params()
        out["quantiles"] = ",".join(map(str, self.quantiles))
        return out
