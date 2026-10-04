"""Market data engine.

Ships with a synthetic multi-session OHLCV generator (regime-switching
volatility + realistic overnight/weekend gaps) so the whole app runs with
zero external dependencies out of the box. `load_csv` lets a real feed be
swapped in without touching any downstream code — every feature module and
model consumes the same OHLCV schema regardless of source.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import DataConfig, TIMEFRAME_MINUTES

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _session_bar_count(cfg: DataConfig) -> int:
    open_min = cfg.session_open_hour * 60 + cfg.session_open_minute
    close_min = cfg.session_close_hour * 60 + cfg.session_close_minute
    return (close_min - open_min) // 5


def generate_base_data(cfg: DataConfig) -> pd.DataFrame:
    """Generate 5-minute OHLCV bars over regular trading sessions.

    Uses a regime-switching volatility process (calm / normal / turbulent)
    so downstream ATR/BB-width/percentile features have real structure to
    learn from, plus session-open gaps drawn from a fatter-tailed
    distribution on Mondays and after simulated holidays.
    """
    rng = np.random.default_rng(cfg.random_seed)
    bars_per_session = _session_bar_count(cfg)
    n_sessions = int(np.ceil(cfg.lookback_bars / bars_per_session)) + 5

    # Business days, with ~1 simulated holiday every ~21 sessions.
    all_days = pd.bdate_range(
        end=pd.Timestamp.now().normalize(), periods=n_sessions
    )
    holiday_mask = (np.arange(len(all_days)) % 21 == 20)
    session_days = all_days[~holiday_mask]

    price = 100.0
    # Volatility regime: 0=calm, 1=normal, 2=turbulent, Markov-switching.
    regime = 1
    trans = np.array([
        [0.96, 0.04, 0.00],
        [0.03, 0.94, 0.03],
        [0.00, 0.08, 0.92],
    ])
    regime_sigma = {0: 0.0006, 1: 0.0015, 2: 0.0038}
    regime_drift = {0: 0.00002, 1: 0.0, 2: -0.00003}

    rows = []
    prev_close = price
    for day_idx, day in enumerate(session_days):
        is_monday = day.dayofweek == 0
        was_holiday_before = holiday_mask[max(0, np.searchsorted(all_days, day) - 1)]
        gap_scale = 0.006 if (is_monday or was_holiday_before) else 0.0022
        gap = rng.standard_t(df=4) * gap_scale
        open_price = prev_close * (1 + gap)

        session_open = day + pd.Timedelta(hours=9, minutes=30)
        bar_price = open_price
        for b in range(bars_per_session):
            regime = rng.choice(3, p=trans[regime])
            sigma = regime_sigma[regime]
            drift = regime_drift[regime]
            ret = rng.normal(drift, sigma)
            o = bar_price
            c = o * (1 + ret)
            wick = abs(rng.normal(0, sigma * 1.4))
            h = max(o, c) * (1 + wick)
            l = min(o, c) * (1 - wick)
            intraday_u = 1.0 + 0.6 * np.sin(np.pi * b / bars_per_session)
            vol = rng.lognormal(mean=8.5, sigma=0.5) * intraday_u
            ts = session_open + pd.Timedelta(minutes=5 * b)
            rows.append((ts, o, h, l, c, vol))
            bar_price = c
        prev_close = bar_price

    df = pd.DataFrame(rows, columns=["timestamp", *OHLCV_COLUMNS]).set_index("timestamp")
    return df.tail(cfg.lookback_bars)


def load_csv(path: str) -> pd.DataFrame:
    """Load an external OHLCV feed. Expects a `timestamp` column plus
    open/high/low/close/volume."""
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df[OHLCV_COLUMNS].sort_index()


def resample_ohlcv(df_5m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample base 5-minute bars up to a coarser timeframe. Empty bins
    (overnight/weekend) are dropped rather than forward-filled, so gap
    detection later sees genuine session boundaries."""
    if timeframe == "5m":
        return df_5m.copy()
    minutes = TIMEFRAME_MINUTES[timeframe]
    rule = f"{minutes}min"
    agg = df_5m.resample(rule, label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    })
    return agg.dropna(subset=["open"])


def get_multi_timeframe_data(cfg: DataConfig, timeframes: list[str],
                              override_5m: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    base = override_5m if override_5m is not None else generate_base_data(cfg)
    out = {}
    for tf in timeframes:
        out[tf] = resample_ohlcv(base, tf)
    if "5m" not in out:
        out["_base_5m"] = base
    return out


def generate_external_sr_levels(df: pd.DataFrame, timeframe: str, window: int = 12,
                                 seed: int = 7) -> pd.DataFrame:
    """DEMO EXTERNAL-FEED SIMULATOR.

    Per the spec, Support/Resistance is computed by an external proprietary
    trading robot and consumed by the AI engine as an input feature — it is
    never recomputed here. This function only stands in for that external
    feed (simple swing-pivot detection) so the app is runnable standalone.
    In production, replace this with the real S/R API/webhook integration;
    everything downstream already treats S/R as an external input.
    """
    if len(df) < window * 2 + 1:
        return pd.DataFrame(columns=["timestamp", "level", "kind", "strength", "touch_count"])

    highs, lows = df["high"].values, df["low"].values
    idx = df.index
    levels = []
    rng = np.random.default_rng(seed)
    for i in range(window, len(df) - window):
        seg_h = highs[i - window:i + window + 1]
        seg_l = lows[i - window:i + window + 1]
        if highs[i] == seg_h.max():
            levels.append((idx[i], float(highs[i]), "resistance"))
        if lows[i] == seg_l.min():
            levels.append((idx[i], float(lows[i]), "support"))

    rows = []
    for ts, level, kind in levels:
        touches = int(np.sum(np.isclose(df["high" if kind == "resistance" else "low"].values,
                                         level, rtol=0.0015)))
        strength = float(np.clip(touches / 3.0 + rng.uniform(0, 0.15), 0.1, 1.0))
        rows.append({"timestamp": ts, "level": level, "kind": kind,
                      "strength": strength, "touch_count": max(touches, 1),
                      "timeframe": timeframe})
    return pd.DataFrame(rows)
