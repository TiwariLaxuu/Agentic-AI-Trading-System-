import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ui.theme import apply_theme, SERIES
from ui.components import section_title, stat_row, badge_html, fmt_num, fmt_pct
from ui.state import get_config, get_ohlcv, get_dataset
from core.features.trend_momentum import compute_trend_momentum

st.set_page_config(page_title="Trend & Momentum — AI Trading Engine", page_icon="🚀", layout="wide")
apply_theme("Trend & Momentum", "Direction, strength, returns, momentum, and higher-timeframe alignment")

cfg = get_config()
tfs, sr_by_tf = get_ohlcv()
dataset, events = get_dataset()

trend_tfs = [t for t in cfg.trend.timeframes if t in tfs]
tf = st.selectbox("Timeframe", trend_tfs, index=0)
df_tf = tfs[tf]
trend = compute_trend_momentum(df_tf, cfg.trend.fast_ma, cfg.trend.slow_ma, cfg.trend.roc_period)
view_n = st.slider("Bars shown", 80, min(1000, len(df_tf)), min(300, len(df_tf)), step=20)
trend_view = trend.tail(view_n)
last = trend.iloc[-1]

dir_label = "Up" if last["trend_direction"] > 0 else ("Down" if last["trend_direction"] < 0 else "Flat")
dir_status = "good" if last["trend_direction"] > 0 else ("critical" if last["trend_direction"] < 0 else "neutral")

stat_row([
    {"label": "Trend Direction", "value": dir_label, "sub": f"strength {fmt_num(last['trend_strength'], 2)}%"},
    {"label": "Momentum (ROC)", "value": fmt_pct(last["roc"])},
    {"label": "Return (5 bars)", "value": fmt_pct(last["return_5"])},
    {"label": "Momentum Accel.", "value": fmt_pct(last["momentum_accel"])},
])
st.write("")
if "htf_trend_alignment" in dataset.columns:
    align = dataset["htf_trend_alignment"].iloc[-1]
    st.markdown(f"{badge_html(dir_label, dir_status)} &nbsp;&nbsp; Higher-timeframe alignment: "
                f"{badge_html(f'{align*100:.0f}%', 'good' if align >= 0.7 else ('warning' if align >= 0.4 else 'critical'))}",
                unsafe_allow_html=True)

st.write("")
section_title(f"Trend strength & momentum — {tf}")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.08,
                     subplot_titles=("EMA-spread trend strength (%)", "Rate of Change (%)"))
fig.add_trace(go.Bar(x=trend_view.index, y=trend_view["trend_strength"],
                       marker_color=[SERIES["aqua"] if v >= 0 else SERIES["red"] for v in trend_view["trend_strength"]],
                       name="Trend strength"), row=1, col=1)
fig.add_trace(go.Scatter(x=trend_view.index, y=trend_view["roc"], line=dict(color=SERIES["blue"], width=1.5), name="ROC %"), row=2, col=1)
fig.update_layout(height=520, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.write("")
section_title("Multi-timeframe direction agreement")
cols = st.columns(len(trend_tfs) or 1)
for col, t in zip(cols, trend_tfs):
    dcol = f"tf{t}_trend_direction"
    with col:
        if dcol in dataset.columns:
            d = dataset[dcol].iloc[-1]
            label = "Up" if d > 0 else ("Down" if d < 0 else "Flat")
            status = "good" if d > 0 else ("critical" if d < 0 else "neutral")
            st.markdown(f"**{t}**  \n{badge_html(label, status)}", unsafe_allow_html=True)
        else:
            st.markdown(f"**{t}**  \n—")
