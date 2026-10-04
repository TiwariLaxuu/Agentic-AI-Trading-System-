"""Time & Session feature block — pure calendar features, computed directly
from each bar's timestamp against the configured session hours."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_time_session(df: pd.DataFrame, session_open_hour: int, session_open_minute: int,
                          session_close_hour: int, session_close_minute: int,
                          opening_period_min: int = 30, closing_period_min: int = 30) -> pd.DataFrame:
    idx = df.index
    minutes_of_day = pd.Series(idx.hour * 60 + idx.minute, index=idx)
    open_min = session_open_hour * 60 + session_open_minute
    close_min = session_close_hour * 60 + session_close_minute

    out = pd.DataFrame(index=idx)
    out["tod_minutes"] = minutes_of_day
    out["tod_sin"] = np.sin(2 * np.pi * minutes_of_day / 1440)
    out["tod_cos"] = np.cos(2 * np.pi * minutes_of_day / 1440)
    out["day_of_week"] = idx.dayofweek
    out["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)

    day_key = idx.date
    day_series = pd.Series(day_key, index=idx)
    is_first_of_day = day_series != day_series.shift(1)
    is_last_of_day = day_series != day_series.shift(-1)

    out["is_market_open_bar"] = is_first_of_day.astype(int)
    out["is_market_close_bar"] = is_last_of_day.astype(int)
    out["time_since_open_min"] = (minutes_of_day - open_min).clip(lower=0)
    out["time_until_close_min"] = (close_min - minutes_of_day).clip(lower=0)
    out["is_opening_period"] = (out["time_since_open_min"] <= opening_period_min).astype(int)
    out["is_closing_period"] = (out["time_until_close_min"] <= closing_period_min).astype(int)
    out["is_weekend_adjacent"] = (idx.dayofweek == 4).astype(int)  # Friday: next gap likely spans weekend

    prev_day = day_series.shift(1)
    gap_days = (pd.to_datetime(day_series) - pd.to_datetime(prev_day)).dt.days
    out["days_since_prev_session"] = gap_days.fillna(0)
    out["is_extended_break"] = (gap_days.fillna(1) > 3).astype(int)  # long weekend / holiday proxy
    return out
