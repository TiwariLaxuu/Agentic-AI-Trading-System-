import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ui.theme import apply_theme, SERIES, INK_MUTED
from ui.components import section_title, stat_row, badge_html, fmt_num, fmt_pct
from ui.state import get_config, get_ohlcv
from core.features.bollinger import compute_bollinger

st.set_page_config(page_title="Bollinger Bands — AI Trading Engine", page_icon="📉", layout="wide")
apply_theme("Bollinger Bands", "Middle/Upper/Lower bands, width dynamics, and breakout/mean-reversion state")

cfg = get_config()
tfs, sr_by_tf = get_ohlcv()

tf = st.selectbox("Timeframe", [t for t in cfg.bollinger.timeframes if t in tfs], index=0)
df_tf = tfs[tf]
bb = compute_bollinger(df_tf, cfg.bollinger.period, cfg.bollinger.std_multiplier)
view_n = st.slider("Bars shown", 80, min(1000, len(df_tf)), min(300, len(df_tf)), step=20)
df_view = df_tf.tail(view_n)
bb_view = bb.tail(view_n)
last = bb.iloc[-1]

exp_label = "Expanding" if last["bb_expansion"] > 0 else ("Contracting" if last["bb_expansion"] < 0 else "Stable")
exp_status = "warning" if last["bb_expansion"] > 0 else ("neutral" if last["bb_expansion"] == 0 else "good")
breakout_label = "Upper breakout" if last["bb_breakout_upper"] else ("Lower breakout" if last["bb_breakout_lower"] else "Inside bands")
breakout_status = "critical" if last["bb_breakout_upper"] else ("warning" if last["bb_breakout_lower"] else "neutral")

stat_row([
    {"label": "BB Position", "value": fmt_num(last["bb_position"], 2), "sub": "0 = lower band, 1 = upper band"},
    {"label": "Width %", "value": fmt_num(last["bb_width_pct"], 2), "sub": fmt_pct(last["bb_width_change"]) + " change"},
    {"label": "Slope", "value": fmt_pct(last["bb_slope"]), "sub": "middle-band, 5-bar"},
    {"label": "Dist to Upper / Lower", "value": f"{fmt_pct(last['bb_dist_upper_pct'])} / {fmt_pct(last['bb_dist_lower_pct'])}"},
])
st.write("")
st.markdown(f"{badge_html(exp_label, exp_status)} &nbsp;&nbsp; {badge_html(breakout_label, breakout_status)} &nbsp;&nbsp; "
            f"{badge_html('Mean-reversion zone', 'warning') if last['bb_mean_reversion_zone'] else ''}",
            unsafe_allow_html=True)

st.write("")
section_title(f"Price & Bands — {tf}")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
fig.add_trace(go.Candlestick(x=df_view.index, open=df_view["open"], high=df_view["high"],
                               low=df_view["low"], close=df_view["close"],
                               increasing_line_color=SERIES["aqua"], decreasing_line_color=SERIES["red"],
                               increasing_fillcolor=SERIES["aqua"], decreasing_fillcolor=SERIES["red"], name="Price"),
              row=1, col=1)
fig.add_trace(go.Scatter(x=bb_view.index, y=bb_view["bb_upper"], line=dict(color=INK_MUTED, width=1), name="Upper"), row=1, col=1)
fig.add_trace(go.Scatter(x=bb_view.index, y=bb_view["bb_lower"], line=dict(color=INK_MUTED, width=1), fill="tonexty",
                          fillcolor="rgba(137,135,129,0.08)", name="Lower"), row=1, col=1)
fig.add_trace(go.Scatter(x=bb_view.index, y=bb_view["bb_middle"], line=dict(color=SERIES["blue"], width=1.5, dash="dot"), name="Middle"), row=1, col=1)
fig.add_trace(go.Scatter(x=bb_view.index, y=bb_view["bb_width_pct"], line=dict(color=SERIES["violet"], width=1.5), name="Width %"), row=2, col=1)
fig.update_layout(height=560, xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.06))
fig.update_yaxes(title_text="Width %", row=2, col=1)
st.plotly_chart(fig, use_container_width=True)

st.write("")
section_title("Configuration")
st.caption(f"Period: **{cfg.bollinger.period}** · Std multiplier: **{cfg.bollinger.std_multiplier}** — change these in Settings.")
