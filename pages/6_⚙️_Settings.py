import pandas as pd
import streamlit as st

from core.config import ALL_TIMEFRAMES
from core.data import load_csv
from ui.theme import apply_theme
from ui.components import section_title
from ui.state import get_config, persist_config
from core.logging_utils import get_logger

logger = get_logger("settings")

st.set_page_config(page_title="Settings — AI Trading Engine", page_icon="⚙️", layout="wide")
apply_theme("Settings", "Every timeframe, threshold, and parameter referenced by the feature engine lives here")

cfg = get_config()

section_title("Data source")
with st.form("data_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        symbol = st.text_input("Symbol label", cfg.data.symbol)
        base_tf = st.selectbox("Base timeframe", ALL_TIMEFRAMES, index=ALL_TIMEFRAMES.index(cfg.data.base_timeframe))
    with c2:
        lookback = st.number_input("Lookback bars (base timeframe)", 500, 20000, cfg.data.lookback_bars, step=250)
        seed = st.number_input("Random seed", 0, 999999, cfg.data.random_seed)
    with c3:
        open_h = st.number_input("Session open hour", 0, 23, cfg.data.session_open_hour)
        open_m = st.number_input("Session open minute", 0, 59, cfg.data.session_open_minute)
        close_h = st.number_input("Session close hour", 0, 23, cfg.data.session_close_hour)
        close_m = st.number_input("Session close minute", 0, 59, cfg.data.session_close_minute)

    if st.form_submit_button("Save data settings"):
        cfg.data.symbol = symbol
        cfg.data.base_timeframe = base_tf
        cfg.data.lookback_bars = int(lookback)
        cfg.data.random_seed = int(seed)
        cfg.data.session_open_hour = int(open_h)
        cfg.data.session_open_minute = int(open_m)
        cfg.data.session_close_hour = int(close_h)
        cfg.data.session_close_minute = int(close_m)
        persist_config(cfg)
        st.success("Data settings saved.")
        st.rerun()

with st.expander("Use a real OHLCV feed instead of the synthetic generator"):
    st.caption("CSV must have columns: timestamp, open, high, low, close, volume (5-minute bars recommended).")
    up = st.file_uploader("Upload CSV", type=["csv"])
    cols = st.columns(2)
    if up is not None and cols[0].button("Load into app"):
        try:
            df = pd.read_csv(up, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
            missing = [c for c in ["open", "high", "low", "close", "volume"] if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                st.session_state["csv_override_5m"] = df[["open", "high", "low", "close", "volume"]]
                logger.info(f"Loaded external CSV feed with {len(df)} rows")
                st.success(f"Loaded {len(df)} rows. All pages now use this feed.")
                st.rerun()
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
    if st.session_state.get("csv_override_5m") is not None and cols[1].button("Revert to synthetic data"):
        del st.session_state["csv_override_5m"]
        st.rerun()

st.write("")
section_title("Timeframes")
with st.form("tf_form"):
    enabled = st.multiselect("Enabled timeframes (multi-timeframe market analysis)", ALL_TIMEFRAMES, default=cfg.enabled_timeframes)
    if st.form_submit_button("Save timeframes"):
        cfg.enabled_timeframes = enabled or list(ALL_TIMEFRAMES)
        persist_config(cfg)
        st.success("Timeframes saved.")
        st.rerun()

st.write("")
section_title("Bollinger Bands")
with st.form("bb_form"):
    c1, c2, c3 = st.columns(3)
    period = c1.number_input("Period", 5, 100, cfg.bollinger.period)
    std_mult = c2.number_input("Std multiplier", 0.5, 4.0, cfg.bollinger.std_multiplier, step=0.1)
    bb_tfs = c3.multiselect("Timeframes", ALL_TIMEFRAMES, default=cfg.bollinger.timeframes)
    if st.form_submit_button("Save Bollinger settings"):
        cfg.bollinger.period = int(period)
        cfg.bollinger.std_multiplier = float(std_mult)
        cfg.bollinger.timeframes = bb_tfs or list(ALL_TIMEFRAMES)
        persist_config(cfg)
        st.success("Bollinger settings saved.")
        st.rerun()

st.write("")
section_title("Volatility")
with st.form("vol_form"):
    c1, c2, c3 = st.columns(3)
    atr_period = c1.number_input("ATR period", 5, 100, cfg.volatility.atr_period)
    rv_window = c1.number_input("Realized-vol window", 5, 100, cfg.volatility.realized_vol_window)
    pct_lookback = c2.number_input("Percentile lookback (bars)", 50, 2000, cfg.volatility.percentile_lookback)
    low_pct = c2.number_input("Low regime percentile", 0.0, 50.0, cfg.volatility.low_pct)
    high_pct = c3.number_input("High regime percentile", 50.0, 99.0, cfg.volatility.high_pct)
    extreme_pct = c3.number_input("Extreme regime percentile", 90.0, 100.0, cfg.volatility.extreme_pct)
    if st.form_submit_button("Save volatility settings"):
        cfg.volatility.atr_period = int(atr_period)
        cfg.volatility.realized_vol_window = int(rv_window)
        cfg.volatility.percentile_lookback = int(pct_lookback)
        cfg.volatility.low_pct = float(low_pct)
        cfg.volatility.high_pct = float(high_pct)
        cfg.volatility.extreme_pct = float(extreme_pct)
        persist_config(cfg)
        st.success("Volatility settings saved.")
        st.rerun()

st.write("")
section_title("Trend & Momentum")
with st.form("trend_form"):
    c1, c2, c3 = st.columns(3)
    fast_ma = c1.number_input("Fast EMA", 2, 50, cfg.trend.fast_ma)
    slow_ma = c2.number_input("Slow EMA", 5, 200, cfg.trend.slow_ma)
    roc_period = c3.number_input("ROC period", 2, 50, cfg.trend.roc_period)
    trend_tfs = st.multiselect("Timeframes", ALL_TIMEFRAMES, default=cfg.trend.timeframes)
    if st.form_submit_button("Save trend settings"):
        cfg.trend.fast_ma = int(fast_ma)
        cfg.trend.slow_ma = int(slow_ma)
        cfg.trend.roc_period = int(roc_period)
        cfg.trend.timeframes = trend_tfs or list(ALL_TIMEFRAMES)
        persist_config(cfg)
        st.success("Trend settings saved.")
        st.rerun()

st.write("")
section_title("Gaps")
with st.form("gap_form"):
    c1, c2, c3 = st.columns(3)
    min_gap = c1.number_input("Minimum gap threshold (%)", 0.01, 2.0, cfg.gaps.min_gap_pct, step=0.01, format="%.2f")
    cluster_lb = c2.number_input("Clustering lookback (bars)", 2, 50, cfg.gaps.cluster_lookback)
    closure_horizon = c3.number_input("Closure horizon (bars)", 2, 200, cfg.gaps.closure_horizon_bars)
    gap_tfs = st.multiselect("Timeframes", ALL_TIMEFRAMES, default=cfg.gaps.timeframes)
    if st.form_submit_button("Save gap settings"):
        cfg.gaps.min_gap_pct = float(min_gap)
        cfg.gaps.cluster_lookback = int(cluster_lb)
        cfg.gaps.closure_horizon_bars = int(closure_horizon)
        cfg.gaps.timeframes = gap_tfs or list(ALL_TIMEFRAMES)
        persist_config(cfg)
        st.success("Gap settings saved.")
        st.rerun()

st.write("")
st.caption(
    "Model / training hyperparameters (XGBoost, Temporal Transformer, split ratios, forward "
    "horizon) live on the **Training** page next to the Train button, since they're normally "
    "tuned together with a training run rather than in isolation."
)
st.caption("Changing any setting here invalidates the cached dataset — every page recomputes on next load.")
