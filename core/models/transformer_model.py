"""Temporal Transformer head: learns chronological relationships between
historical bars and multi-timeframe features, over the same standardized
feature matrix the XGBoost head uses."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from torch import nn


def make_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    """Causal sliding windows: sequence ending at row i uses rows
    [i-seq_len+1, i] only — never future rows."""
    n, f = X.shape
    if n < seq_len:
        return np.empty((0, seq_len, f), dtype=np.float32)
    out = np.lib.stride_tricks.sliding_window_view(X, (seq_len, f))[:, 0, :, :]
    return out.astype(np.float32)


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class _TemporalTransformerNet(nn.Module):
    def __init__(self, n_features: int, d_model: int, nhead: int, n_layers: int):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = _PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                                             batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head_direction = nn.Linear(d_model, 3)
        self.head_return = nn.Linear(d_model, 1)
        self.head_gap = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.pos_enc(h)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        last = h[:, -1, :]
        return self.head_direction(last), self.head_return(last).squeeze(-1), self.head_gap(last).squeeze(-1)


class TemporalTransformerSuite:
    def __init__(self, seq_len: int = 32, d_model: int = 64, nhead: int = 4, n_layers: int = 2,
                 epochs: int = 15, lr: float = 1e-3, batch_size: int = 64):
        self.seq_len = seq_len
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.d_model, self.nhead, self.n_layers = d_model, nhead, n_layers
        self.net: _TemporalTransformerNet | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.feature_names: list[str] = []

    def _scale(self, X: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            self.mean_ = np.nanmean(X, axis=0)
            self.std_ = np.nanstd(X, axis=0)
            self.std_[self.std_ < 1e-8] = 1.0
        X = np.nan_to_num(X, nan=0.0)
        return (X - self.mean_) / self.std_

    def fit(self, X: pd.DataFrame, y_direction: pd.Series, y_return: pd.Series,
            y_gap_closure: pd.Series | None, progress_cb=None) -> "TemporalTransformerSuite":
        self.feature_names = list(X.columns)
        Xs = self._scale(X.to_numpy(dtype=np.float64), fit=True)
        seqs = make_sequences(Xs, self.seq_len)
        offset = self.seq_len - 1
        dir_map = {-1: 0, 0: 1, 1: 2}
        y_dir = y_direction.map(dir_map).values[offset:]
        y_ret = y_return.values[offset:]
        y_gap = y_gap_closure.values[offset:] if y_gap_closure is not None else np.full(len(y_dir), np.nan)

        valid = ~np.isnan(y_ret) & ~np.isnan(y_dir.astype(float))
        seqs, y_dir, y_ret, y_gap = seqs[valid], y_dir[valid], y_ret[valid], y_gap[valid]

        self.net = _TemporalTransformerNet(Xs.shape[1], self.d_model, self.nhead, self.n_layers)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        ce = nn.CrossEntropyLoss()
        mse = nn.MSELoss()
        bce = nn.BCEWithLogitsLoss()

        seqs_t = torch.from_numpy(seqs)
        y_dir_t = torch.from_numpy(y_dir.astype(np.int64))
        y_ret_t = torch.from_numpy(y_ret.astype(np.float32))
        y_gap_t = torch.from_numpy(np.nan_to_num(y_gap, nan=0.0).astype(np.float32))
        gap_mask_t = torch.from_numpy((~np.isnan(y_gap)).astype(np.float32))

        n = seqs_t.size(0)
        self.net.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            total_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb = seqs_t[idx]
                dir_logits, ret_pred, gap_logits = self.net(xb)
                loss = ce(dir_logits, y_dir_t[idx]) + 0.5 * mse(ret_pred, y_ret_t[idx])
                gm = gap_mask_t[idx]
                if gm.sum() > 0:
                    gap_loss = bce(gap_logits, y_gap_t[idx])
                    loss = loss + 0.3 * (gap_loss * gm).sum() / gm.sum().clamp(min=1)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += float(loss.item()) * xb.size(0)
            if progress_cb:
                progress_cb(epoch + 1, self.epochs, total_loss / max(n, 1))
        self.net.eval()
        return self

    @torch.no_grad()
    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        X = X[self.feature_names]
        Xs = self._scale(X.to_numpy(dtype=np.float64), fit=False)
        seqs = make_sequences(Xs, self.seq_len)
        offset = self.seq_len - 1
        n_total = len(X)
        out = {k: np.full(n_total, np.nan) for k in ["p_down", "p_flat", "p_up", "expected_return", "p_gap_closure"]}
        if len(seqs) == 0:
            return out
        seqs_t = torch.from_numpy(seqs)
        dir_logits, ret_pred, gap_logits = self.net(seqs_t)
        probs = torch.softmax(dir_logits, dim=-1).numpy()
        gap_prob = torch.sigmoid(gap_logits).numpy()
        out["p_down"][offset:] = probs[:, 0]
        out["p_flat"][offset:] = probs[:, 1]
        out["p_up"][offset:] = probs[:, 2]
        out["expected_return"][offset:] = ret_pred.numpy()
        out["p_gap_closure"][offset:] = gap_prob
        return out
