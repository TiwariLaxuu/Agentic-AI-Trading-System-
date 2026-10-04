"""Volatility feature block. Volatility is exposed purely as a descriptive
AI feature (ATR, realized vol, percentile rank, expansion/contraction, and a
Low/Normal/High/Extreme regime label) — never as a standalone trading signal,
per spec."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def compute_volatility(df: pd.DataFrame, atr_period: int = 14, rv_window: int = 20,
                        percentile_lookback: int = 252, low_pct: float = 25.0,
                        high_pct: float = 75.0, extreme_pct: float = 95.0) -> pd.DataFrame:
    close = df["close"]
    log_ret = np.log(close / close.shift(1))

    out = pd.DataFrame(index=df.index)
    out["atr"] = _atr(df, atr_period)
    out["atr_pct"] = out["atr"] / close * 100
    out["realized_vol"] = log_ret.rolling(rv_window).std() * np.sqrt(252 * (390 / max(rv_window, 1)))
    out["realized_vol_pct"] = log_ret.rolling(rv_window).std() * 100

    lookback = min(percentile_lookback, max(len(df), 1))
    out["vol_percentile"] = out["realized_vol_pct"].rolling(lookback, min_periods=20).apply(
        lambda s: (s.rank(pct=True).iloc[-1] * 100) if len(s.dropna()) > 5 else np.nan, raw=False
    )
    vol_ma = out["realized_vol_pct"].rolling(rv_window).mean()
    out["vol_change"] = out["realized_vol_pct"] - out["realized_vol_pct"].shift(rv_window)
    out["vol_expansion"] = (out["realized_vol_pct"] > vol_ma * 1.1).astype(int) - \
        (out["realized_vol_pct"] < vol_ma * 0.9).astype(int)

    def _regime(p):
        if pd.isna(p):
            return np.nan
        if p >= extreme_pct:
            return "Extreme"
        if p >= high_pct:
            return "High"
        if p <= low_pct:
            return "Low"
        return "Normal"

    out["vol_regime"] = out["vol_percentile"].apply(_regime)
    return out
