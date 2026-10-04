import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.theme import apply_theme, SERIES, INK_MUTED
from ui.components import section_title, stat_row, badge_html, fmt_num, fmt_pct, format_df
from ui.state import get_config, get_dataset, get_ohlcv

st.set_page_config(page_title="Support & Resistance — AI Trading Engine", page_icon="🛡️", layout="wide")
apply_theme("Support & Resistance", "External levels ingested as AI features — never recomputed inside the AI engine")

cfg = get_config()
dataset, events = get_dataset()
tfs, sr_by_tf = get_ohlcv()
base_tf = cfg.data.base_timeframe if cfg.data.base_timeframe in tfs else next(iter(tfs))

st.info(
    f"Per spec, Support & Resistance levels come from the existing trading robot. This app's "
    f"`generate_external_sr_levels()` is a stand-in external feed (swing-pivot detection) so the "
    f"dashboard runs standalone — currently wired for the **{base_tf}** base timeframe. Replace it "
    f"with the real S/R API and every feature/model below keeps working unchanged.",
    icon="ℹ️",
)

last = dataset.iloc[-1]
sup_status = "good" if last.get("sr_breakout_support", 0) == 1 else ("warning" if last.get("sr_rejection_support", 0) == 1 else "neutral")
res_status = "critical" if last.get("sr_breakout_resistance", 0) == 1 else ("warning" if last.get("sr_rejection_resistance", 0) == 1 else "neutral")

stat_row([
    {"label": "Nearest Support", "value": fmt_num(last.get("sr_support_price"), 3),
     "sub": f"{fmt_pct(last.get('sr_support_dist_pct'))} away"},
    {"label": "Support Strength", "value": fmt_num(last.get("sr_support_strength"), 2),
     "sub": f"{int(last.get('sr_support_touch_count', 0) or 0)} touches · age {int(last.get('sr_support_age', 0) or 0)} bars"},
    {"label": "Nearest Resistance", "value": fmt_num(last.get("sr_resistance_price"), 3),
     "sub": f"{fmt_pct(last.get('sr_resistance_dist_pct'))} away"},
    {"label": "Resistance Strength", "value": fmt_num(last.get("sr_resistance_strength"), 2),
     "sub": f"{int(last.get('sr_resistance_touch_count', 0) or 0)} touches · age {int(last.get('sr_resistance_age', 0) or 0)} bars"},
])

st.write("")
st.markdown(f"Support status: {badge_html('Breakout' if sup_status=='good' else ('Rejection' if sup_status=='warning' else 'Holding'), sup_status)}"
            f"&nbsp;&nbsp;&nbsp;Resistance status: {badge_html('Breakout' if res_status=='critical' else ('Rejection' if res_status=='warning' else 'Holding'), res_status)}",
            unsafe_allow_html=True)

st.write("")
section_title(f"Price with Support / Resistance — {base_tf}")
n_bars = st.slider("Bars shown", 100, min(1500, len(dataset)), min(500, len(dataset)), step=50)
view = dataset.tail(n_bars)

fig = go.Figure()
fig.add_trace(go.Scatter(x=view.index, y=view["price_close"], line=dict(color=INK_MUTED, width=1.5), name="Close"))
fig.add_trace(go.Scatter(x=view.index, y=view["sr_support_price"], mode="lines",
                          line=dict(color=SERIES["green"], width=1.5, shape="hv"), name="Support"))
fig.add_trace(go.Scatter(x=view.index, y=view["sr_resistance_price"], mode="lines",
                          line=dict(color=SERIES["orange"], width=1.5, shape="hv"), name="Resistance"))
fig.update_layout(height=420, legend=dict(orientation="h", y=1.08))
st.plotly_chart(fig, use_container_width=True)

sr_levels = sr_by_tf.get(base_tf)
if sr_levels is not None and not sr_levels.empty:
    section_title("External S/R level feed (raw)")
    sr_display = format_df(sr_levels.sort_values("timestamp", ascending=False), {"level": "{:.3f}", "strength": "{:.2f}"})
    st.dataframe(sr_display, use_container_width=True, height=320)
