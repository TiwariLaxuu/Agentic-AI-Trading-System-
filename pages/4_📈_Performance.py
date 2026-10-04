import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.theme import apply_theme, SERIES, STATUS
from ui.components import section_title, stat_row, fmt_pct, fmt_num, format_df
from ui.state import get_dataset, get_backtest_result

st.set_page_config(page_title="Performance — AI Trading Engine", page_icon="📈", layout="wide")
apply_theme("Performance", "Deeper analytics on the most recent backtest run")

dataset, events = get_dataset()
bt = get_backtest_result()

if bt is None:
    st.info("Run a backtest on the **Backtesting** page first — this view breaks down that run in more detail.")
    st.stop()

m = bt.metrics
stat_row([
    {"label": "Total Return", "value": fmt_pct(m["total_return_pct"])},
    {"label": "Profit Factor", "value": fmt_num(m["profit_factor"], 2)},
    {"label": "Avg Win", "value": fmt_pct(m["avg_win_pct"])},
    {"label": "Avg Loss", "value": fmt_pct(m["avg_loss_pct"])},
])

st.write("")
section_title("Underwater curve (drawdown)")
running_max = bt.equity_curve.cummax()
drawdown = (bt.equity_curve - running_max) / running_max * 100
fig = go.Figure(go.Scatter(x=drawdown.index, y=drawdown, line=dict(color=SERIES["red"], width=1.5),
                             fill="tozeroy", fillcolor="rgba(230,103,103,0.15)"))
fig.update_layout(height=260, yaxis_title="Drawdown (%)")
st.plotly_chart(fig, use_container_width=True)

trades = bt.trades
if trades.empty:
    st.warning("No trades in the current backtest run to analyze.")
    st.stop()

left, right = st.columns(2)
with left:
    section_title("By direction")
    by_dir = trades.groupby("direction")["pnl_pct"].agg(["count", "mean", "sum"])
    by_dir.columns = ["trades", "avg pnl %", "total pnl %"]
    st.dataframe(format_df(by_dir, {"avg pnl %": "{:+.3f}", "total pnl %": "{:+.3f}"}), use_container_width=True)

    fig_dir = go.Figure(go.Bar(x=by_dir.index, y=by_dir["total pnl %"],
                                 marker_color=[SERIES["aqua"] if v >= 0 else SERIES["red"] for v in by_dir["total pnl %"]]))
    fig_dir.update_layout(height=260, yaxis_title="Total PnL (%)")
    st.plotly_chart(fig_dir, use_container_width=True)

with right:
    section_title("By market regime at entry")
    if "regime_label" in dataset.columns:
        entry_regime = trades["entry_time"].map(lambda t: dataset["regime_label"].asof(t) if t in dataset.index or t >= dataset.index[0] else np.nan)
        trades_r = trades.assign(regime=entry_regime)
        by_regime = trades_r.groupby("regime")["pnl_pct"].agg(["count", "mean", "sum"]).sort_values("sum", ascending=False)
        by_regime.columns = ["trades", "avg pnl %", "total pnl %"]
        st.dataframe(format_df(by_regime, {"avg pnl %": "{:+.3f}", "total pnl %": "{:+.3f}"}), use_container_width=True, height=260)
    else:
        st.caption("Regime labels not available in the current dataset.")

st.write("")
section_title("Holding period vs. outcome")
fig_scatter = go.Figure(go.Scatter(
    x=trades["bars_held"], y=trades["pnl_pct"], mode="markers",
    marker=dict(color=[SERIES["aqua"] if v >= 0 else SERIES["red"] for v in trades["pnl_pct"]], size=8,
                line=dict(width=1, color="rgba(255,255,255,0.15)")),
))
fig_scatter.add_hline(y=0, line=dict(color="#383835", width=1))
fig_scatter.update_layout(height=320, xaxis_title="Bars held", yaxis_title="Trade PnL (%)")
st.plotly_chart(fig_scatter, use_container_width=True)

section_title("Cumulative trade PnL")
cum = trades.sort_values("exit_time")["pnl_pct"].cumsum()
fig_cum = go.Figure(go.Scatter(x=trades.sort_values("exit_time")["exit_time"], y=cum,
                                 line=dict(color=SERIES["violet"], width=2)))
fig_cum.update_layout(height=280, yaxis_title="Cumulative PnL (%)")
st.plotly_chart(fig_cum, use_container_width=True)
