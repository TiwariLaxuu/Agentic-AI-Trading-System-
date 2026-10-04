"""Trend & Momentum feature block, computed per timeframe. Direction/strength
are exposed as continuous features (MA spread, slope) rather than fixed
crossover rules — the AI learns their significance."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_trend_momentum(df: pd.DataFrame, fast_ma: int = 10, slow_ma: int = 50,
                            roc_period: int = 10) -> pd.DataFrame:
    close = df["close"]
    ema_fast = close.ewm(span=fast_ma, adjust=False).mean()
    ema_slow = close.ewm(span=slow_ma, adjust=False).mean()

    out = pd.DataFrame(index=df.index)
    out["trend_direction"] = np.sign(ema_fast - ema_slow)
    out["trend_strength"] = (ema_fast - ema_slow) / ema_slow.replace(0, np.nan) * 100
    out["return_1"] = close.pct_change(1) * 100
    out["return_5"] = close.pct_change(5) * 100
    out["return_20"] = close.pct_change(20) * 100
    out["momentum"] = close - close.shift(roc_period)
    out["roc"] = (close / close.shift(roc_period) - 1) * 100
    out["momentum_accel"] = out["roc"] - out["roc"].shift(roc_period)
    out["ema_fast_slope"] = (ema_fast - ema_fast.shift(3)) / ema_fast.shift(3) * 100
    return out


def trend_alignment(trend_by_tf: dict[str, pd.DataFrame], base_index: pd.Index,
                     tf_order: list[str]) -> pd.Series:
    """Higher-timeframe alignment score: fraction of enabled timeframes whose
    trend_direction agrees with the highest timeframe, reindexed/forward-filled
    onto the base (finest) timeframe index to avoid look-ahead."""
    if not tf_order:
        return pd.Series(index=base_index, dtype=float)
    ref_tf = tf_order[-1]
    ref_dir = trend_by_tf[ref_tf]["trend_direction"].reindex(base_index, method="ffill")
    agree = pd.Series(0.0, index=base_index)
    count = 0
    for tf in tf_order:
        d = trend_by_tf[tf]["trend_direction"].reindex(base_index, method="ffill")
        agree = agree.add((d == ref_dir).astype(float), fill_value=0)
        count += 1
    return (agree / max(count, 1)).rename("htf_trend_alignment")
