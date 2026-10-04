import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ui.theme import apply_theme, SERIES, STATUS
from ui.components import section_title, stat_row, badge_html, fmt_num, fmt_pct, regime_status
from ui.state import get_config, get_ohlcv
from core.features.volatility import compute_volatility

st.set_page_config(page_title="Volatility — AI Trading Engine", page_icon="🌪️", layout="wide")
apply_theme("Volatility", "ATR, realized volatility, percentile rank, and regime — a descriptive AI feature, not a signal")

cfg = get_config()
tfs, sr_by_tf = get_ohlcv()

tf = st.selectbox("Timeframe", list(tfs.keys()), index=0)
df_tf = tfs[tf]
vol = compute_volatility(df_tf, cfg.volatility.atr_period, cfg.volatility.realized_vol_window,
                           cfg.volatility.percentile_lookback, cfg.volatility.low_pct,
                           cfg.volatility.high_pct, cfg.volatility.extreme_pct)
view_n = st.slider("Bars shown", 80, min(1000, len(df_tf)), min(300, len(df_tf)), step=20)
vol_view = vol.tail(view_n)
last = vol.iloc[-1]

exp_label = "Expanding" if last["vol_expansion"] > 0 else ("Contracting" if last["vol_expansion"] < 0 else "Stable")

stat_row([
    {"label": "ATR %", "value": fmt_num(last["atr_pct"], 3)},
    {"label": "Realized Vol %", "value": fmt_num(last["realized_vol_pct"], 3), "sub": fmt_pct(last["vol_change"]) + " vs prior window"},
    {"label": "Vol Percentile", "value": fmt_num(last["vol_percentile"], 1)},
    {"label": "Regime", "value": str(last["vol_regime"])},
])
st.write("")
st.markdown(f"{badge_html(str(last['vol_regime']), regime_status(last['vol_regime']))} &nbsp;&nbsp; {badge_html(exp_label, 'warning' if exp_label=='Expanding' else 'neutral')}",
            unsafe_allow_html=True)

st.write("")
section_title(f"ATR % & Realized Volatility — {tf}")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.08,
                     subplot_titles=("ATR %", "Realized Volatility % & Percentile"))
fig.add_trace(go.Scatter(x=vol_view.index, y=vol_view["atr_pct"], line=dict(color=SERIES["blue"], width=1.5), name="ATR %"), row=1, col=1)
fig.add_trace(go.Scatter(x=vol_view.index, y=vol_view["realized_vol_pct"], line=dict(color=SERIES["orange"], width=1.5), name="Realized Vol %"), row=2, col=1)
fig.add_trace(go.Scatter(x=vol_view.index, y=vol_view["vol_percentile"] / 100 * vol_view["realized_vol_pct"].max(),
                          line=dict(color=SERIES["violet"], width=1, dash="dot"), name="Percentile (scaled)"), row=2, col=1)
fig.update_layout(height=520, legend=dict(orientation="h", y=1.06))
st.plotly_chart(fig, use_container_width=True)

st.write("")
section_title("Regime distribution (visible window)")
counts = vol_view["vol_regime"].value_counts().reindex(["Low", "Normal", "High", "Extreme"]).fillna(0)
colors = [STATUS["good"], SERIES["blue"], STATUS["warning"], STATUS["critical"]]
fig2 = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=colors))
fig2.update_layout(height=260, yaxis_title="Bars")
st.plotly_chart(fig2, use_container_width=True)

st.caption(
    "Low/Normal/High/Extreme is a descriptive percentile bucket, not a trading rule — the AI models "
    "consume the continuous ATR/realized-vol/percentile values directly and learn their significance."
)
