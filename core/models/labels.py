"""Label construction (spec section 13).

Labels are the ONLY place future data is used. They are built once, kept in
a separate frame from the feature matrix, and joined back only after a
strict time-based train/val/test split — never used to compute a feature.

Produces, per bar:
  y_direction   : -1 / 0 / +1 (down / flat / up) over the forward horizon
  y_continuation: 1 if the forward move agrees with the current base-timeframe
                  trend direction, 0 if it opposes it, NaN if trend is flat
  y_return      : forward return over the horizon, in percent (regression target)
  y_gap_closure : 1/0, only defined on bars where a gap is currently open —
                  whether THAT gap closes within its remaining horizon
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_direction_labels(price_close: pd.Series, horizon: int, flat_threshold_pct: float = 0.05) -> pd.DataFrame:
    future_return = (price_close.shift(-horizon) / price_close - 1) * 100
    direction = pd.Series(0, index=price_close.index)
    direction[future_return > flat_threshold_pct] = 1
    direction[future_return < -flat_threshold_pct] = -1
    return pd.DataFrame({"y_return": future_return, "y_direction": direction})


def build_continuation_labels(direction: pd.Series, trend_direction: pd.Series) -> pd.Series:
    trend_sign = np.sign(trend_direction.fillna(0))
    label = pd.Series(np.nan, index=direction.index)
    has_trend = trend_sign != 0
    has_move = direction != 0
    valid = has_trend & has_move
    label[valid] = (direction[valid] == trend_sign[valid]).astype(float)
    return label.rename("y_continuation")


def build_gap_closure_labels(df_base: pd.DataFrame, gap_still_open: pd.Series, gap_age: pd.Series,
                               gap_active_direction: pd.Series, horizon: int) -> pd.Series:
    """For every bar where a gap is currently open, look forward (using
    future OHLCV — legitimate here since this is a training label, not a
    feature) up to its remaining horizon and check whether price reaches the
    pre-gap reference close. This mirrors `gaps.track_gap_state` but is
    explicitly forward-looking, so it lives in the label module only."""
    n = len(df_base)
    close = df_base["close"].values
    high = df_base["high"].values
    low = df_base["low"].values
    open_ = df_base["open"].values
    still_open = gap_still_open.fillna(0).values
    age = gap_age.fillna(-1).values
    direction = gap_active_direction.fillna(0).values

    y = np.full(n, np.nan)
    for i in range(n):
        if still_open[i] != 1 or age[i] < 0:
            continue
        remaining = int(horizon - age[i])
        if remaining <= 0:
            continue
        gap_open = open_[i - int(age[i])] if i - int(age[i]) >= 0 else np.nan
        ref_close = close[i - int(age[i]) - 1] if i - int(age[i]) - 1 >= 0 else np.nan
        if np.isnan(gap_open) or np.isnan(ref_close):
            continue
        end = min(i + remaining, n - 1)
        window_close = close[i:end + 1]
        # A gap UP closes when price retraces back DOWN to the pre-gap close;
        # a gap DOWN closes when price recovers back UP to it.
        if direction[i] > 0:
            closed = (window_close <= ref_close).any()
        else:
            closed = (window_close >= ref_close).any()
        y[i] = float(closed)
    return pd.Series(y, index=df_base.index, name="y_gap_closure")


def build_all_labels(dataset: pd.DataFrame, cfg_model) -> pd.DataFrame:
    price_close = dataset["price_close"]
    dir_labels = build_direction_labels(price_close, cfg_model.forward_horizon_bars)
    trend_col = "trendbase_trend_direction"
    trend_direction = dataset[trend_col] if trend_col in dataset.columns else pd.Series(0, index=dataset.index)
    continuation = build_continuation_labels(dir_labels["y_direction"], trend_direction)

    price_df = dataset[["price_open", "price_high", "price_low", "price_close"]].rename(
        columns=lambda c: c.replace("price_", ""))
    gap_open_col = "gap_still_open"
    gap_age_col = "gap_age"
    gap_dir_col = "gap_active_direction"
    if all(c in dataset.columns for c in (gap_open_col, gap_age_col, gap_dir_col)):
        gap_closure = build_gap_closure_labels(
            price_df, dataset[gap_open_col], dataset[gap_age_col], dataset[gap_dir_col],
            cfg_model.forward_horizon_bars,
        )
    else:
        gap_closure = pd.Series(np.nan, index=dataset.index, name="y_gap_closure")

    labels = pd.concat([dir_labels, continuation, gap_closure], axis=1)
    return labels
