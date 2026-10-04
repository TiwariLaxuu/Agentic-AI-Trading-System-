import plotly.graph_objects as go
import streamlit as st

from ui.theme import apply_theme, SERIES
from ui.components import section_title, stat_row, fmt_pct, fmt_num, format_df
from ui.state import get_config, get_dataset, get_training_result, get_predictions, set_backtest_result, get_backtest_result
from core.backtest.engine import run_backtest

st.set_page_config(page_title="Backtesting — AI Trading Engine", page_icon="📊", layout="wide")
apply_theme("Backtesting", "Simulate the model's signal on held-out (test) data")

cfg = get_config()
dataset, events = get_dataset()
result = get_training_result()
preds = get_predictions()

if result is None or preds is None:
    st.info("Train a model first on the **Training** page — the backtest runs on its held-out test split.")
    st.stop()

section_title("Backtest parameters")
c1, c2, c3, c4 = st.columns(4)
entry_threshold = c1.slider("Entry threshold (P(up) − P(down))", 0.02, 0.5, 0.10, step=0.02)
min_confidence = c2.slider("Minimum confidence", 0.2, 0.9, 0.40, step=0.02)
holding_bars = c3.number_input("Holding period (bars)", 2, 200, cfg.model.forward_horizon_bars)
cost_bps = c4.number_input("Round-trip cost (bps, one side)", 0.0, 20.0, 2.0, step=0.5)

run_clicked = st.button("▶ Run Backtest", type="primary")
bt = get_backtest_result()
if run_clicked:
    with st.spinner("Simulating trades on the test window..."):
        bt = run_backtest(dataset, preds, result.splits.test, entry_threshold=entry_threshold,
                            min_confidence=min_confidence, holding_bars=int(holding_bars), cost_bps=cost_bps)
        set_backtest_result(bt)

if bt is None:
    st.stop()

st.write("")
m = bt.metrics
stat_row([
    {"label": "Total Return", "value": fmt_pct(m["total_return_pct"])},
    {"label": "Max Drawdown", "value": fmt_pct(m["max_drawdown_pct"])},
    {"label": "Sharpe Ratio", "value": fmt_num(m["sharpe_ratio"], 2)},
    {"label": "Win Rate", "value": f"{m['win_rate']*100:.1f}%" if m["win_rate"] == m["win_rate"] else "—"},
    {"label": "Trades", "value": str(m["num_trades"])},
])

st.write("")
section_title("Equity curve (test window)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=bt.equity_curve.index, y=(bt.equity_curve - 1) * 100,
                          line=dict(color=SERIES["blue"], width=2), fill="tozeroy",
                          fillcolor="rgba(57,135,229,0.10)", name="Equity"))
fig.add_hline(y=0, line=dict(color="#383835", width=1))
fig.update_layout(height=380, yaxis_title="Cumulative return (%)")
st.plotly_chart(fig, use_container_width=True)

if not bt.trades.empty:
    section_title("Trade log")
    trades_display = format_df(bt.trades.sort_values("exit_time", ascending=False),
                                 {"entry_price": "{:.3f}", "exit_price": "{:.3f}", "pnl_pct": "{:+.3f}"})
    st.dataframe(trades_display, use_container_width=True, height=320)

    section_title("PnL distribution")
    fig2 = go.Figure(go.Histogram(x=bt.trades["pnl_pct"], nbinsx=30, marker_color=SERIES["blue"]))
    fig2.update_layout(height=280, xaxis_title="Trade PnL (%)", yaxis_title="Count")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("No trades were triggered at this threshold/confidence — try lowering the entry threshold.")

st.caption(
    "Signals are evaluated on data through bar t and executed at bar t+1's open — no same-bar "
    "lookahead. This is a research backtest (single position, fixed sizing) to sanity-check the "
    "model's edge, not a production execution simulator."
)
