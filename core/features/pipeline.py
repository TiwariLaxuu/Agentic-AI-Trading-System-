"""Unified multi-timeframe feature engine (spec section 11).

Builds one standardized feature dataset from Bollinger Bands, Volatility,
Trend/Momentum, external Support & Resistance, Intraday + Out-of-Hours Gaps,
Gap Closure, Market Regime and Time/Session blocks — the same pipeline used
for training, validation, backtesting and live prediction.

Look-ahead safety: every higher-timeframe block is computed on its own
timeframe, then shifted forward by that timeframe's bar duration before being
forward-filled onto the base (finest) timeframe index. A 1H bar's features
only become visible to the base series once that 1H bar has actually closed —
never while it is still forming.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import AppConfig, TIMEFRAME_MINUTES
from core.features.bollinger import compute_bollinger
from core.features.volatility import compute_volatility
from core.features.trend_momentum import compute_trend_momentum, trend_alignment
from core.features.support_resistance import attach_support_resistance
from core.features.gaps import (compute_gaps, compute_gap_rolling_features, classify_gap_category,
                                  track_gap_state, build_gap_events, attach_gap_event_features)
from core.features.regime import compute_market_regime
from core.features.time_session import compute_time_session


def _make_available_from(block: pd.DataFrame, minutes: int, base_index: pd.Index) -> pd.DataFrame:
    """Shift a higher-timeframe feature block so it only becomes visible on
    the base index once that bar has fully closed, then forward-fill."""
    shifted = block.copy()
    shifted.index = shifted.index + pd.Timedelta(minutes=minutes)
    merged = shifted.reindex(shifted.index.union(base_index)).sort_index().ffill()
    return merged.reindex(base_index)


def build_timeframe_block(df_tf: pd.DataFrame, cfg: AppConfig, tf: str,
                            sr_levels: pd.DataFrame | None,
                            include_trend: bool, include_gaps: bool) -> dict[str, pd.DataFrame]:
    blocks: dict[str, pd.DataFrame] = {}
    blocks["bb"] = compute_bollinger(df_tf, cfg.bollinger.period, cfg.bollinger.std_multiplier)
    blocks["vol"] = compute_volatility(
        df_tf, cfg.volatility.atr_period, cfg.volatility.realized_vol_window,
        cfg.volatility.percentile_lookback, cfg.volatility.low_pct,
        cfg.volatility.high_pct, cfg.volatility.extreme_pct,
    )
    if include_trend:
        blocks["trend"] = compute_trend_momentum(df_tf, cfg.trend.fast_ma, cfg.trend.slow_ma, cfg.trend.roc_period)
    if include_gaps:
        gaps = compute_gaps(df_tf, cfg.gaps.min_gap_pct)
        gap_roll = compute_gap_rolling_features(gaps, cfg.gaps.cluster_lookback)
        time_tf = compute_time_session(df_tf, cfg.data.session_open_hour, cfg.data.session_open_minute,
                                         cfg.data.session_close_hour, cfg.data.session_close_minute)
        cats = classify_gap_category(df_tf, time_tf["is_market_open_bar"], time_tf["days_since_prev_session"])
        gap_state = track_gap_state(df_tf, gaps, cfg.gaps.closure_horizon_bars)
        events = build_gap_events(df_tf, gaps, cats, cfg.gaps.closure_horizon_bars)
        gap_event_feats = attach_gap_event_features(df_tf, events)
        blocks["gaps"] = pd.concat([gaps, gap_roll, gap_state, gap_event_feats], axis=1)
        blocks["_gap_events"] = events
    if sr_levels is not None:
        blocks["sr"] = attach_support_resistance(df_tf, sr_levels)
    return blocks


def build_feature_dataset(cfg: AppConfig, ohlcv_by_tf: dict[str, pd.DataFrame],
                            sr_by_tf: dict[str, pd.DataFrame] | None = None,
                            base_tf: str | None = None) -> tuple[pd.DataFrame, dict]:
    base_tf = base_tf or cfg.data.base_timeframe
    if base_tf not in ohlcv_by_tf:
        base_tf = next(iter(ohlcv_by_tf))
    base_df = ohlcv_by_tf[base_tf]
    base_index = base_df.index
    sr_by_tf = sr_by_tf or {}

    all_events = {}
    frames = [base_df.add_prefix("price_")]
    trend_frames_for_alignment: dict[str, pd.DataFrame] = {}

    trend_tfs = [tf for tf in cfg.trend.timeframes if tf in ohlcv_by_tf]
    gap_tfs = [tf for tf in cfg.gaps.timeframes if tf in ohlcv_by_tf]

    for tf in cfg.enabled_timeframes:
        if tf not in ohlcv_by_tf:
            continue
        df_tf = ohlcv_by_tf[tf]
        blocks = build_timeframe_block(
            df_tf, cfg, tf, sr_by_tf.get(tf),
            include_trend=(tf in trend_tfs), include_gaps=(tf in gap_tfs),
        )
        if "trend" in blocks:
            trend_frames_for_alignment[tf] = blocks["trend"]
        if "_gap_events" in blocks:
            all_events[tf] = blocks.pop("_gap_events")

        tf_combined = pd.concat([b for k, b in blocks.items()], axis=1)
        tf_combined = tf_combined.add_prefix(f"tf{tf}_")
        available = _make_available_from(tf_combined, TIMEFRAME_MINUTES[tf], base_index)
        frames.append(available)

    if trend_frames_for_alignment:
        ordered = [tf for tf in trend_tfs if tf in trend_frames_for_alignment]
        alignment = trend_alignment(trend_frames_for_alignment, base_index, ordered)
        frames.append(alignment.to_frame())

    time_feats = compute_time_session(base_df, cfg.data.session_open_hour, cfg.data.session_open_minute,
                                        cfg.data.session_close_hour, cfg.data.session_close_minute)
    frames.append(time_feats.add_prefix("time_"))

    base_bb = compute_bollinger(base_df, cfg.bollinger.period, cfg.bollinger.std_multiplier)
    base_vol = compute_volatility(base_df, cfg.volatility.atr_period, cfg.volatility.realized_vol_window,
                                    cfg.volatility.percentile_lookback, cfg.volatility.low_pct,
                                    cfg.volatility.high_pct, cfg.volatility.extreme_pct)
    base_gaps = compute_gaps(base_df, cfg.gaps.min_gap_pct)
    base_gap_roll = compute_gap_rolling_features(base_gaps, cfg.gaps.cluster_lookback)
    base_trend = trend_frames_for_alignment.get(base_tf)
    if base_trend is None:
        base_trend = compute_trend_momentum(base_df, cfg.trend.fast_ma, cfg.trend.slow_ma, cfg.trend.roc_period)
    base_regime = compute_market_regime(base_trend, base_vol, base_bb, base_gap_roll)
    frames.append(base_regime.add_prefix("regime_".rstrip("_") + "_"))
    frames.append(base_trend.add_prefix("trendbase_"))

    # Base-timeframe gap state, unprefixed: this is the "live" gap-tracking state used both
    # for the gap+BB interaction features below and for building gap-closure labels. Unlike the
    # tf{tf}_ blocks (which are deliberately shifted so a bar's features only appear once that
    # bar has closed), the base timeframe is the atomic unit of "now" — gap direction/size are
    # already fully known at the base bar's open, so no availability shift is needed here.
    base_gap_state = track_gap_state(base_df, base_gaps, cfg.gaps.closure_horizon_bars)
    base_events_df = all_events.get(base_tf, pd.DataFrame())
    base_gap_event_feats = attach_gap_event_features(base_df, base_events_df)
    frames.append(base_gaps)
    frames.append(base_gap_state)
    frames.append(base_gap_event_feats)

    base_sr_levels = sr_by_tf.get(base_tf)
    if base_sr_levels is not None:
        frames.append(attach_support_resistance(base_df, base_sr_levels))

    # Gap + Bollinger Band interaction (section 8), computed at the base timeframe.
    htf_trend_col = None
    if trend_tfs:
        highest_tf = trend_tfs[-1]
        htf_trend_col = f"tf{highest_tf}_trend_direction"

    interaction = pd.DataFrame(index=base_index)
    is_gap = base_gaps["is_gap"]
    bb_pos = base_bb["bb_position"]
    interaction["gap_bb_near_lower"] = (is_gap & (bb_pos <= 0.2)).astype(int)
    interaction["gap_bb_near_upper"] = (is_gap & (bb_pos >= 0.8)).astype(int)
    interaction["gap_bb_outside_bands"] = (is_gap & ((base_df["close"] > base_bb["bb_upper"]) | (base_df["close"] < base_bb["bb_lower"]))).astype(int)
    interaction["gap_bb_expanding"] = (is_gap & (base_bb["bb_expansion"] > 0)).astype(int)
    interaction["gap_bb_contracting"] = (is_gap & (base_bb["bb_expansion"] < 0)).astype(int)
    interaction["gap_after_bb_breakout"] = (is_gap & ((base_bb["bb_breakout_upper"].shift(1) == 1) | (base_bb["bb_breakout_lower"].shift(1) == 1)).fillna(False)).astype(int)
    frames.append(interaction)

    dataset = pd.concat(frames, axis=1)

    if htf_trend_col and htf_trend_col in dataset.columns:
        with_trend = dataset[["price_close"]].copy()
        gap_with_trend = (is_gap & (base_gaps["gap_direction"] == dataset[htf_trend_col])).astype(int)
        gap_against_trend = (is_gap & (base_gaps["gap_direction"] == -dataset[htf_trend_col])).astype(int)
        dataset["gap_with_htf_trend"] = gap_with_trend
        dataset["gap_against_htf_trend"] = gap_against_trend

    dataset = dataset.loc[:, ~dataset.columns.duplicated()]
    return dataset, all_events
