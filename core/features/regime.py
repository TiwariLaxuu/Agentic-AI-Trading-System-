"""Market Regime feature block — combines trend, volatility, BB behavior,
momentum, gap behavior and session context into descriptive regime features
(a bucket label + composite score) that the AI can condition on. This is
deliberately descriptive, not a trading rule: the model learns how setups
behave differently across regimes rather than being told what a regime means.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_market_regime(trend: pd.DataFrame, vol: pd.DataFrame, bb: pd.DataFrame,
                            gap_rolling: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=trend.index)

    trend_strength = trend["trend_strength"]
    trend_bucket = pd.cut(
        trend_strength, bins=[-np.inf, -1.0, 1.0, np.inf],
        labels=["Downtrend", "Sideways", "Uptrend"]
    ).astype(str)
    out["regime_trend_bucket"] = trend_bucket
    out["regime_vol_bucket"] = vol["vol_regime"]
    out["regime_label"] = trend_bucket.astype(str) + " / " + vol["vol_regime"].astype(str)

    bb_state = np.where(bb["bb_expansion"] > 0, "Expanding",
                 np.where(bb["bb_expansion"] < 0, "Contracting", "Stable"))
    out["regime_bb_state"] = bb_state

    z_trend = (trend_strength - trend_strength.rolling(100).mean()) / trend_strength.rolling(100).std()
    z_mom = (trend["roc"] - trend["roc"].rolling(100).mean()) / trend["roc"].rolling(100).std()
    z_vol = (vol["realized_vol_pct"] - vol["realized_vol_pct"].rolling(100).mean()) / vol["realized_vol_pct"].rolling(100).std()
    z_gap = (gap_rolling["gap_clustering"] - gap_rolling["gap_clustering"].rolling(100).mean()) / gap_rolling["gap_clustering"].rolling(100).std()
    out["regime_score"] = pd.concat([z_trend, z_mom, z_vol, z_gap], axis=1).mean(axis=1, skipna=True)
    return out
