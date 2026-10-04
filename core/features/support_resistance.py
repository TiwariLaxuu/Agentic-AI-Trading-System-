"""Support & Resistance feature block.

Per spec, S/R levels come from the existing (proprietary) trading robot —
this module never computes them. It only *consumes* an external levels feed
(timestamp, level, kind, strength, touch_count) — see
`core.data.generate_external_sr_levels` for the demo stand-in — and derives
AI-ready features: nearest level distance, strength, age, touch count, and
breakout/rejection status, using only levels already known at each bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def attach_support_resistance(df: pd.DataFrame, sr_levels: pd.DataFrame,
                                touch_tol_pct: float = 0.15) -> pd.DataFrame:
    n = len(df)
    cols = ["sr_support_price", "sr_support_dist_pct", "sr_support_strength",
            "sr_support_age", "sr_support_touch_count", "sr_breakout_support", "sr_rejection_support",
            "sr_resistance_price", "sr_resistance_dist_pct", "sr_resistance_strength",
            "sr_resistance_age", "sr_resistance_touch_count", "sr_breakout_resistance", "sr_rejection_resistance"]
    out = pd.DataFrame(np.nan, index=df.index, columns=cols)
    if sr_levels is None or sr_levels.empty:
        return out

    sr_levels = sr_levels.sort_values("timestamp")
    supports = sr_levels[sr_levels["kind"] == "support"].reset_index(drop=True)
    resistances = sr_levels[sr_levels["kind"] == "resistance"].reset_index(drop=True)

    sup_ts = supports["timestamp"].values
    res_ts = resistances["timestamp"].values
    idx = df.index.values
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    prev_close = df["close"].shift(1).values

    for i in range(n):
        t = idx[i]
        n_sup = np.searchsorted(sup_ts, t, side="right")
        n_res = np.searchsorted(res_ts, t, side="right")

        if n_sup > 0:
            known = supports.iloc[:n_sup]
            j = (known["level"] - close[i]).abs().idxmin()
            row = known.loc[j]
            out.iat[i, out.columns.get_loc("sr_support_price")] = row["level"]
            out.iat[i, out.columns.get_loc("sr_support_dist_pct")] = (close[i] - row["level"]) / close[i] * 100
            out.iat[i, out.columns.get_loc("sr_support_strength")] = row["strength"]
            out.iat[i, out.columns.get_loc("sr_support_touch_count")] = row["touch_count"]
            age = i - df.index.searchsorted(row["timestamp"])
            out.iat[i, out.columns.get_loc("sr_support_age")] = age
            broke = (not np.isnan(prev_close[i])) and prev_close[i] >= row["level"] > close[i]
            near = abs(low[i] - row["level"]) / row["level"] * 100 <= touch_tol_pct
            rejected = near and close[i] > row["level"]
            out.iat[i, out.columns.get_loc("sr_breakout_support")] = int(broke)
            out.iat[i, out.columns.get_loc("sr_rejection_support")] = int(rejected)

        if n_res > 0:
            known = resistances.iloc[:n_res]
            j = (known["level"] - close[i]).abs().idxmin()
            row = known.loc[j]
            out.iat[i, out.columns.get_loc("sr_resistance_price")] = row["level"]
            out.iat[i, out.columns.get_loc("sr_resistance_dist_pct")] = (row["level"] - close[i]) / close[i] * 100
            out.iat[i, out.columns.get_loc("sr_resistance_strength")] = row["strength"]
            out.iat[i, out.columns.get_loc("sr_resistance_touch_count")] = row["touch_count"]
            age = i - df.index.searchsorted(row["timestamp"])
            out.iat[i, out.columns.get_loc("sr_resistance_age")] = age
            broke = (not np.isnan(prev_close[i])) and prev_close[i] <= row["level"] < close[i]
            near = abs(high[i] - row["level"]) / row["level"] * 100 <= touch_tol_pct
            rejected = near and close[i] < row["level"]
            out.iat[i, out.columns.get_loc("sr_breakout_resistance")] = int(broke)
            out.iat[i, out.columns.get_loc("sr_rejection_resistance")] = int(rejected)

    return out
