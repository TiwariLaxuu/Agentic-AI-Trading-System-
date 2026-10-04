import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ui.theme import apply_theme, SERIES, STATUS
from ui.components import section_title, stat_row, badge_html, fmt_num, fmt_pct, format_df
from ui.state import get_config, get_dataset

st.set_page_config(page_title="Gap Analysis — AI Trading Engine", page_icon="🕳️", layout="wide")
apply_theme("Multi-Timeframe Gap Analysis", "Intraday, overnight, weekend & holiday gaps — tracked to closure")

cfg = get_config()
dataset, events = get_dataset()

tf = st.selectbox("Timeframe", [t for t in cfg.gaps.timeframes if t in events], index=0)
ev = events.get(tf, pd.DataFrame())

if ev.empty:
    st.warning(f"No qualifying gaps detected on {tf} at the current minimum-gap threshold "
               f"({cfg.gaps.min_gap_pct}%). Lower it in Settings → Gaps to see more events.")
    st.stop()

closed_rate = ev["closed_within_horizon"].mean() * 100
avg_gap = ev["gap_pct"].abs().mean()
avg_ttc = ev.loc[ev["closed_within_horizon"], "time_to_closure_bars"].mean()

stat_row([
    {"label": "Gap Events", "value": str(len(ev))},
    {"label": "Avg |Gap Size|", "value": fmt_num(avg_gap, 3) + "%"},
    {"label": "Closed Within Horizon", "value": f"{closed_rate:.1f}%"},
    {"label": "Avg Bars to Closure", "value": fmt_num(avg_ttc, 1) if pd.notna(avg_ttc) else "—"},
])

st.write("")
left, right = st.columns(2)
with left:
    section_title("Gaps by category")
    cat_counts = ev["category"].value_counts()
    colors = {"overnight": SERIES["blue"], "weekend": SERIES["violet"], "holiday": SERIES["orange"], "intraday": SERIES["aqua"]}
    fig = go.Figure(go.Bar(x=cat_counts.index, y=cat_counts.values,
                             marker_color=[colors.get(c, SERIES["blue"]) for c in cat_counts.index]))
    fig.update_layout(height=300, yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

with right:
    section_title("Closure probability by category")
    prob_by_cat = ev.groupby("category")["closed_within_horizon"].mean() * 100
    fig2 = go.Figure(go.Bar(x=prob_by_cat.index, y=prob_by_cat.values,
                              marker_color=[colors.get(c, SERIES["blue"]) for c in prob_by_cat.index]))
    fig2.update_layout(height=300, yaxis_title="Closed within horizon (%)")
    st.plotly_chart(fig2, use_container_width=True)

st.write("")
section_title("Gap size vs. maximum favorable / adverse movement")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=ev["gap_pct"], y=ev["max_favorable_pct"], mode="markers", name="Max favorable",
                            marker=dict(color=SERIES["aqua"], size=8)))
fig3.add_trace(go.Scatter(x=ev["gap_pct"], y=ev["max_adverse_pct"], mode="markers", name="Max adverse",
                            marker=dict(color=SERIES["red"], size=8)))
fig3.update_layout(height=360, xaxis_title="Gap size (%)", yaxis_title="Movement (%)", legend=dict(orientation="h", y=1.08))
st.plotly_chart(fig3, use_container_width=True)

if tf == cfg.data.base_timeframe:
    st.write("")
    section_title(f"Rolling gap behavior — {tf}")
    view_n = st.slider("Bars shown", 100, min(1500, len(dataset)), min(500, len(dataset)), step=50)
    view = dataset.tail(view_n)
    fig4 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.08,
                          subplot_titles=("Cumulative & weighted cumulative gap (%)", "Gap clustering / directional consistency"))
    fig4.add_trace(go.Scatter(x=view.index, y=view.get("tf5m_cumulative_gap", pd.Series(dtype=float)),
                                line=dict(color=SERIES["blue"], width=1.5), name="Cumulative gap"), row=1, col=1)
    fig4.add_trace(go.Scatter(x=view.index, y=view.get("tf5m_weighted_cumulative_gap", pd.Series(dtype=float)),
                                line=dict(color=SERIES["orange"], width=1.5), name="Weighted cumulative gap"), row=1, col=1)
    fig4.add_trace(go.Scatter(x=view.index, y=view.get("tf5m_gap_clustering", pd.Series(dtype=float)),
                                line=dict(color=SERIES["violet"], width=1.5), name="Clustering"), row=2, col=1)
    fig4.add_trace(go.Scatter(x=view.index, y=view.get("tf5m_gap_directional_consistency", pd.Series(dtype=float)),
                                line=dict(color=SERIES["aqua"], width=1.5), name="Directional consistency"), row=2, col=1)
    fig4.update_layout(height=480, legend=dict(orientation="h", y=1.06))
    st.plotly_chart(fig4, use_container_width=True)

st.write("")
section_title("Gap + Bollinger Band interaction (base timeframe)")
interaction_cols = ["gap_bb_near_lower", "gap_bb_near_upper", "gap_bb_outside_bands",
                     "gap_bb_expanding", "gap_bb_contracting", "gap_after_bb_breakout",
                     "gap_with_htf_trend", "gap_against_htf_trend"]
present = [c for c in interaction_cols if c in dataset.columns]
if present:
    counts = dataset[present].sum().sort_values(ascending=False)
    fig5 = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=SERIES["magenta"]))
    fig5.update_layout(height=320, yaxis_title="Occurrences", xaxis_tickangle=-30)
    st.plotly_chart(fig5, use_container_width=True)

st.write("")
section_title("Gap events")
show_cols = ["timestamp", "category", "gap_pct", "direction", "gap_closure_pct", "closed_within_horizon",
             "time_to_closure_bars", "same_session_closure", "max_favorable_pct", "max_adverse_pct",
             "historical_closure_probability"]
ev_display = format_df(
    ev[[c for c in show_cols if c in ev.columns]].sort_values("timestamp", ascending=False),
    {"gap_pct": "{:+.3f}", "gap_closure_pct": "{:.1f}", "max_favorable_pct": "{:.2f}",
     "max_adverse_pct": "{:.2f}", "historical_closure_probability": "{:.2f}"},
)
st.dataframe(ev_display, use_container_width=True, height=360)

st.caption(
    "historical_closure_probability is computed causally — for each event, only using events "
    "strictly before it in the same category — so it is safe to use as a live feature, not just an "
    "after-the-fact statistic."
)
