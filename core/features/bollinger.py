"""Bollinger Band feature block — computed per timeframe, fully driven by
the configurable period / std-multiplier in Settings. All outputs are raw
numeric/categorical features; the AI models learn what they mean rather than
this module encoding any continuation/reversal rule."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_bollinger(df: pd.DataFrame, period: int = 20, std_mult: float = 2.0) -> pd.DataFrame:
    close = df["close"]
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    width_pct = (upper - lower) / mid.replace(0, np.nan) * 100
    position = (close - lower) / (upper - lower).replace(0, np.nan)  # 0=lower band,1=upper band

    out = pd.DataFrame(index=df.index)
    out["bb_middle"] = mid
    out["bb_upper"] = upper
    out["bb_lower"] = lower
    out["bb_position"] = position.clip(-1, 2)
    out["bb_dist_upper_pct"] = (upper - close) / close * 100
    out["bb_dist_lower_pct"] = (close - lower) / close * 100
    out["bb_width_pct"] = width_pct
    out["bb_width_change"] = width_pct.diff()
    out["bb_slope"] = (mid - mid.shift(5)) / mid.shift(5) * 100

    width_ma = width_pct.rolling(20).mean()
    out["bb_expansion"] = (width_pct > width_ma * 1.05).astype(int) - (width_pct < width_ma * 0.95).astype(int)
    out["bb_breakout_upper"] = (close > upper).astype(int)
    out["bb_breakout_lower"] = (close < lower).astype(int)
    out["bb_mean_reversion_zone"] = ((position > 0.95) | (position < 0.05)).astype(int)
    return out
