import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.config import TIMEFRAME_MINUTES
from core.features.bollinger import compute_bollinger
from ui.theme import apply_theme, SERIES, STATUS, INK_MUTED, GRIDLINE
from ui.components import stat_row, section_title, badge_html, fmt_pct, fmt_num, regime_status
from ui.state import get_config, get_dataset, get_training_result, get_predictions

st.set_page_config(page_title="AI Trading Engine — Dashboard", page_icon="📈", layout="wide")
apply_theme("AI Trading Engine", "Market Understanding — Multi-Timeframe AI Feature & Prediction Engine")

cfg = get_config()
dataset, events = get_dataset()

close = dataset["price_close"]
last_price = close.iloc[-1]
prev_price = close.iloc[-2] if len(close) > 1 else last_price
session_change = (last_price / prev_price - 1) * 100

vol_col = "regime_vol_regime" if "regime_vol_regime" in dataset.columns else None
vol_regime = dataset[vol_col].iloc[-1] if vol_col else "—"
trend_bucket = dataset["regime_trend_bucket"].iloc[-1] if "regime_trend_bucket" in dataset.columns else "—"
regime_label = dataset["regime_label"].iloc[-1] if "regime_label" in dataset.columns else "—"
gap_open = bool(dataset["gap_still_open"].iloc[-1]) if "gap_still_open" in dataset.columns else False

stat_row([
    {"label": "Symbol", "value": cfg.data.symbol, "sub": f"Base timeframe {cfg.data.base_timeframe}"},
    {"label": "Last Price", "value": fmt_num(last_price, 3), "sub": fmt_pct(session_change) + " last bar"},
    {"label": "Volatility Regime", "value": str(vol_regime), "sub": "ATR / realized-vol percentile"},
    {"label": "Trend Regime", "value": str(trend_bucket), "sub": str(regime_label)},
    {"label": "Open Gap", "value": "Yes" if gap_open else "No", "sub": "tracked to closure horizon"},
])

st.write("")
left, right = st.columns([2.4, 1])

with left:
    section_title(f"Price & Bollinger Bands — {cfg.data.base_timeframe}")
    n_bars = st.slider("Bars shown", 100, min(1500, len(dataset)), min(400, len(dataset)), step=50, key="dash_bars")
    view = dataset.tail(n_bars)
    price_cols = ["price_open", "price_high", "price_low", "price_close"]
    ohlc = view[price_cols].rename(columns=lambda c: c.replace("price_", ""))
    bb = compute_bollinger(ohlc, cfg.bollinger.period, cfg.bollinger.std_multiplier)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=ohlc.index, open=ohlc["open"], high=ohlc["high"], low=ohlc["low"], close=ohlc["close"],
        increasing_line_color=SERIES["aqua"], decreasing_line_color=SERIES["red"],
        increasing_fillcolor=SERIES["aqua"], decreasing_fillcolor=SERIES["red"], name="Price",
    ))
    fig.add_trace(go.Scatter(x=bb.index, y=bb["bb_upper"], line=dict(color=INK_MUTED, width=1), name="BB Upper"))
    fig.add_trace(go.Scatter(x=bb.index, y=bb["bb_lower"], line=dict(color=INK_MUTED, width=1), name="BB Lower",
                              fill="tonexty", fillcolor="rgba(137,135,129,0.08)"))
    fig.add_trace(go.Scatter(x=bb.index, y=bb["bb_middle"], line=dict(color=SERIES["blue"], width=1.5, dash="dot"), name="BB Middle"))

    if "sr_support_price" in view.columns and view["sr_support_price"].notna().any():
        sup = view["sr_support_price"].iloc[-1]
        if pd.notna(sup):
            fig.add_hline(y=sup, line=dict(color=SERIES["green"], width=1, dash="dash"), annotation_text="Support")
    if "sr_resistance_price" in view.columns and view["sr_resistance_price"].notna().any():
        res = view["sr_resistance_price"].iloc[-1]
        if pd.notna(res):
            fig.add_hline(y=res, line=dict(color=SERIES["orange"], width=1, dash="dash"), annotation_text="Resistance")

    fig.update_layout(height=460, xaxis_rangeslider_visible=False, showlegend=True,
                       legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)

with right:
    section_title("Latest AI Output")
    result = get_training_result()
    preds = get_predictions()
    if result is None or preds is None:
        st.info("No trained model yet. Head to **AI / Model** or **Training** to fit XGBoost + Temporal Transformer on the current dataset.")
    else:
        last = preds.iloc[-1]
        st.markdown(f"""
        <div class="stat-card" style="margin-bottom:10px;">
            <div class="label">Continuation Probability</div>
            <div class="value">{last['continuation_probability']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-card" style="margin-bottom:10px;">
            <div class="label">Reversal Probability</div>
            <div class="value">{last['reversal_probability']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        gcp = last.get("gap_closure_probability", np.nan)
        gcp_str = f"{gcp*100:.1f}%" if pd.notna(gcp) else "n/a (no open gap)"
        st.markdown(f"""
        <div class="stat-card" style="margin-bottom:10px;">
            <div class="label">Gap Closure Probability</div>
            <div class="value">{gcp_str}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-card" style="margin-bottom:10px;">
            <div class="label">Expected Return ({cfg.model.forward_horizon_bars} bars)</div>
            <div class="value">{fmt_pct(last['expected_return_pct'])}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-card">
            <div class="label">Model Confidence</div>
            <div class="value">{last['confidence']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")
section_title("Multi-Timeframe Trend Snapshot")
tf_cols = st.columns(len(cfg.trend.timeframes) or 1)
for col, tf in zip(tf_cols, cfg.trend.timeframes):
    colname = f"tf{tf}_trend_direction"
    strcol = f"tf{tf}_trend_strength"
    with col:
        if colname in dataset.columns:
            d = dataset[colname].iloc[-1]
            s = dataset[strcol].iloc[-1] if strcol in dataset.columns else np.nan
            label = "Up" if d > 0 else ("Down" if d < 0 else "Flat")
            status = "good" if d > 0 else ("critical" if d < 0 else "neutral")
            st.markdown(f"**{tf}**  \n{badge_html(label, status)}  \n<span style='color:{INK_MUTED};font-size:0.8rem'>strength {fmt_num(s,2)}%</span>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"**{tf}**  \n—")

st.write("")
st.caption(
    "Data shown is generated by a built-in synthetic multi-session market simulator "
    "(regime-switching volatility + realistic overnight/weekend gaps) so the whole app "
    "runs standalone. Swap in a real feed via Settings → Data Source without touching "
    "any downstream feature, model, or backtest code."
)
