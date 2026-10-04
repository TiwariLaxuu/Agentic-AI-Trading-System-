"""Gap analysis: multi-timeframe intraday gaps (section 6), out-of-hours gaps
(section 7), and gap-closure tracking. Every feature here is computed causally
— only bars up to and including "now" are used — so it is safe to feed
directly into the live/training feature pipeline. Forward-looking outcome
stats (max favorable/adverse movement, actual time-to-closure) are computed
separately as event-level analytics for the dashboard and for building model
labels; they are never joined back onto a bar as a live feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_gaps(df: pd.DataFrame, min_gap_pct: float = 0.05) -> pd.DataFrame:
    prev_close = df["close"].shift(1)
    gap_pct = (df["open"] - prev_close) / prev_close * 100
    out = pd.DataFrame(index=df.index)
    out["gap_pct"] = gap_pct
    out["gap_direction"] = np.sign(gap_pct.fillna(0))
    out["is_gap"] = gap_pct.abs() >= min_gap_pct
    return out


def compute_gap_rolling_features(gaps: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    is_gap = gaps["is_gap"].astype(int)
    signed = (gaps["gap_pct"] * gaps["is_gap"]).fillna(0)

    out = pd.DataFrame(index=gaps.index)
    out["gap_count"] = is_gap.rolling(lookback).sum()
    out["cumulative_gap"] = signed.rolling(lookback).sum()
    out["weighted_cumulative_gap"] = signed.ewm(span=lookback, adjust=False).mean() * lookback

    pos = ((gaps["gap_direction"] > 0) & gaps["is_gap"]).astype(int)
    neg = ((gaps["gap_direction"] < 0) & gaps["is_gap"]).astype(int)
    pos_roll = pos.rolling(lookback).sum()
    neg_roll = neg.rolling(lookback).sum()
    out["gap_directional_consistency"] = (pos_roll - neg_roll).abs() / out["gap_count"].replace(0, np.nan)
    out["gap_clustering"] = out["gap_count"] / lookback
    return out


def classify_gap_category(df: pd.DataFrame, is_market_open_bar: pd.Series,
                            days_since_prev_session: pd.Series) -> pd.Series:
    dow = df.index.to_series().dt.dayofweek
    cat = pd.Series("none", index=df.index)
    open_mask = is_market_open_bar.astype(bool)
    days = days_since_prev_session.fillna(0)

    cat[open_mask & (days <= 1)] = "overnight"
    cat[open_mask & (days == 3) & (dow == 0)] = "weekend"
    cat[open_mask & (days >= 4)] = "holiday"
    cat[open_mask & (days.between(1.5, 2.5))] = "holiday"
    cat[~open_mask] = "intraday"
    return cat.rename("gap_category")


def track_gap_state(df: pd.DataFrame, gaps: pd.DataFrame, closure_horizon_bars: int = 20) -> pd.DataFrame:
    """Causal per-bar state machine: tracks the most recent gap until it
    closes (price returns to the pre-gap reference close) or the horizon
    expires. Every value at bar i uses only data through bar i."""
    n = len(df)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    open_ = df["open"].values
    is_gap = gaps["is_gap"].values
    gap_dir = gaps["gap_direction"].values

    age = np.full(n, np.nan)
    still_open = np.zeros(n)
    closure_pct = np.full(n, np.nan)
    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    ref_dir = np.full(n, np.nan)

    ref_close = None
    gap_open = None
    direction = 0
    start_i = None

    for i in range(n):
        prev_close = close[i - 1] if i > 0 else np.nan
        if is_gap[i]:
            ref_close = prev_close
            gap_open = open_[i]
            direction = gap_dir[i]
            start_i = i

        if start_i is not None and direction != 0 and ref_close not in (None, 0) and not np.isnan(ref_close):
            bars_open = i - start_i
            denom = (ref_close - gap_open)
            if denom == 0:
                closed_pct = 100.0
            else:
                closed_pct = float(np.clip((close[i] - gap_open) / denom * 100, -50, 150))
            if direction > 0:
                favorable = (high[i] - gap_open) / gap_open * 100
                adverse = (gap_open - low[i]) / gap_open * 100
            else:
                favorable = (gap_open - low[i]) / gap_open * 100
                adverse = (high[i] - gap_open) / gap_open * 100

            age[i] = bars_open
            closure_pct[i] = closed_pct
            ref_dir[i] = direction
            mfe[i] = max(favorable, 0)
            mae[i] = max(adverse, 0)
            is_closed = closed_pct >= 100
            still_open[i] = 0 if (is_closed or bars_open > closure_horizon_bars) else 1
            if is_closed or bars_open > closure_horizon_bars:
                # freeze reference so subsequent bars (until next gap) show the resolved state
                start_i = start_i  # keep age/closure growing off last event but flagged closed
        # else leave NaN (no gap seen yet)

    out = pd.DataFrame({
        "gap_age": age,
        "gap_still_open": still_open,
        "gap_closure_pct_so_far": closure_pct,
        "gap_mfe_pct": mfe,
        "gap_mae_pct": mae,
        "gap_active_direction": ref_dir,
    }, index=df.index)
    return out


def build_gap_events(df: pd.DataFrame, gaps: pd.DataFrame, categories: pd.Series,
                      closure_horizon_bars: int = 20) -> pd.DataFrame:
    """Event-level table (one row per qualifying gap) with forward outcome
    stats, for the Gap Analysis dashboard and for building model labels.
    Includes a causal historical_closure_probability computed only from
    events strictly before the current one (grouped by category)."""
    idx = df.index
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    open_ = df["open"].values
    n = len(df)

    event_rows = []
    gap_idx = np.where(gaps["is_gap"].values)[0]
    for i in gap_idx:
        if i == 0:
            continue
        ref_close = close[i - 1]
        gap_open = open_[i]
        direction = gaps["gap_direction"].values[i]
        horizon_end = min(i + closure_horizon_bars, n - 1)
        window_close = close[i:horizon_end + 1]
        window_high = high[i:horizon_end + 1]
        window_low = low[i:horizon_end + 1]

        denom = (ref_close - gap_open)
        if denom == 0:
            closure_series = np.full(len(window_close), 100.0)
        else:
            closure_series = np.clip((window_close - gap_open) / denom * 100, -100, 200)
        closed_mask = closure_series >= 100
        time_to_closure = int(np.argmax(closed_mask)) if closed_mask.any() else np.nan
        gap_closure_pct = float(closure_series[-1])
        same_session = bool(idx[i].date() == idx[horizon_end].date()) if time_to_closure is not np.nan else False

        if direction > 0:
            mfe = float(((window_high - gap_open) / gap_open * 100).max())
            mae = float(((gap_open - window_low) / gap_open * 100).max())
        else:
            mfe = float(((gap_open - window_low) / gap_open * 100).max())
            mae = float(((window_high - gap_open) / gap_open * 100).max())

        event_rows.append({
            "timestamp": idx[i],
            "gap_pct": gaps["gap_pct"].values[i],
            "direction": direction,
            "category": categories.iloc[i],
            "prev_close": ref_close,
            "open_price": gap_open,
            "gap_closure_pct": gap_closure_pct,
            "closed_within_horizon": bool(closed_mask.any()),
            "time_to_closure_bars": time_to_closure,
            "same_session_closure": same_session,
            "sessions_remaining_open": closure_horizon_bars - (time_to_closure if not np.isnan(time_to_closure) else closure_horizon_bars),
            "max_favorable_pct": mfe,
            "max_adverse_pct": mae,
        })

    events = pd.DataFrame(event_rows)
    if events.empty:
        events["historical_closure_probability"] = pd.Series(dtype=float)
        return events

    events = events.sort_values("timestamp").reset_index(drop=True)
    hist_prob = np.full(len(events), np.nan)
    for cat in events["category"].unique():
        cat_mask = events["category"] == cat
        cat_idx = events.index[cat_mask]
        closed = events.loc[cat_idx, "closed_within_horizon"].astype(int).values
        expanding_mean = pd.Series(closed).expanding().mean().shift(1).values
        hist_prob[cat_idx] = expanding_mean
    events["historical_closure_probability"] = hist_prob
    return events


def attach_gap_event_features(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Forward-fills only the causal, already-known-at-the-time fields from
    the event table onto the bar index (category + historical closure prob
    as of the most recent prior gap). Forward-looking outcome columns are
    intentionally excluded here to prevent leakage."""
    cols = ["category", "historical_closure_probability"]
    out = pd.DataFrame(index=df.index, columns=["gap_category", "gap_historical_closure_prob"])
    if events.empty:
        return out
    ev = events.set_index("timestamp")[cols].rename(
        columns={"category": "gap_category", "historical_closure_probability": "gap_historical_closure_prob"})
    merged = ev.reindex(df.index.union(ev.index)).sort_index().ffill().reindex(df.index)
    return merged
