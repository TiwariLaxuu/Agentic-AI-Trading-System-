import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.theme import apply_theme, SERIES
from ui.components import section_title, stat_row, badge_html, fmt_pct, fmt_num, format_df
from ui.state import get_config, get_dataset, get_training_result, get_predictions

st.set_page_config(page_title="Trading — AI Trading Engine", page_icon="💹", layout="wide")
apply_theme("Trading", "Paper-trading simulator driven by live model output — steps forward bar by bar")

cfg = get_config()
dataset, events = get_dataset()
result = get_training_result()
preds = get_predictions()

if result is None or preds is None:
    st.info("Train a model on the **Training** page first — this simulator trades its predictions bar by bar.")
    st.stop()

section_title("Simulator controls")
c1, c2, c3, c4 = st.columns(4)
entry_threshold = c1.slider("Entry threshold", 0.02, 0.5, 0.10, step=0.02, key="live_thr")
min_confidence = c2.slider("Min confidence", 0.2, 0.9, 0.40, step=0.02, key="live_conf")
holding_bars = c3.number_input("Holding period (bars)", 2, 200, cfg.model.forward_horizon_bars, key="live_hold")
risk_per_trade = c4.slider("Risk per trade (%)", 0.1, 5.0, 1.0, step=0.1, key="live_risk")

test_start = result.splits.test[0] if len(result.splits.test) else dataset.index[-200]

if "paper" not in st.session_state or st.session_state["paper"]["start"] != test_start:
    st.session_state["paper"] = {
        "start": test_start,
        "cursor": dataset.index.get_loc(test_start) + 1,
        "position": 0, "entry_price": None, "entry_time": None, "entry_i": None,
        "equity": 1.0, "equity_history": [(test_start, 1.0)], "blotter": [],
    }
paper = st.session_state["paper"]

bcol1, bcol2, bcol3, bcol4 = st.columns(4)
step1 = bcol1.button("▶ Step 1 bar")
step10 = bcol2.button("▶▶ Step 10 bars")
step50 = bcol3.button("▶▶▶ Step 50 bars")
reset = bcol4.button("↺ Reset simulation")

if reset:
    del st.session_state["paper"]
    st.rerun()


def _step_once():
    i = paper["cursor"]
    if i >= len(dataset.index):
        return False
    t = dataset.index[i]
    prev_t = dataset.index[i - 1]
    close = dataset["price_close"]
    r = close.loc[t] / close.loc[prev_t] - 1
    cost = 0.0002

    if paper["position"] != 0:
        paper["equity"] *= (1 + paper["position"] * r)
        held = i - paper["entry_i"]
        sig = preds.loc[prev_t]
        opposing = (
            (paper["position"] > 0 and sig["p_down"] - sig["p_up"] > entry_threshold and sig["confidence"] >= min_confidence) or
            (paper["position"] < 0 and sig["p_up"] - sig["p_down"] > entry_threshold and sig["confidence"] >= min_confidence)
        )
        if held >= holding_bars or opposing:
            exit_price = close.loc[t]
            pnl_pct = paper["position"] * (exit_price / paper["entry_price"] - 1) - 2 * cost
            paper["equity"] *= (1 - cost)
            paper["blotter"].append({
                "entry_time": paper["entry_time"], "exit_time": t,
                "direction": "long" if paper["position"] > 0 else "short",
                "entry_price": paper["entry_price"], "exit_price": exit_price,
                "bars_held": held, "pnl_pct": pnl_pct * 100,
            })
            paper["position"] = 0
            paper["entry_price"] = None
            paper["entry_time"] = None
            paper["entry_i"] = None
    else:
        sig = preds.loc[prev_t]
        if pd.notna(sig.get("p_up")):
            diff = sig["p_up"] - sig["p_down"]
            if diff > entry_threshold and sig["confidence"] >= min_confidence:
                paper["position"] = 1
                paper["entry_price"] = close.loc[t]
                paper["entry_time"] = t
                paper["entry_i"] = i
                paper["equity"] *= (1 - cost)
            elif -diff > entry_threshold and sig["confidence"] >= min_confidence:
                paper["position"] = -1
                paper["entry_price"] = close.loc[t]
                paper["entry_time"] = t
                paper["entry_i"] = i
                paper["equity"] *= (1 - cost)

    paper["equity_history"].append((t, paper["equity"]))
    paper["cursor"] = i + 1
    return True


n_steps = 1 if step1 else (10 if step10 else (50 if step50 else 0))
for _ in range(n_steps):
    if not _step_once():
        st.warning("Reached the end of the dataset.")
        break

st.write("")
cur_i = min(paper["cursor"], len(dataset.index) - 1)
cur_t = dataset.index[cur_i]
last_sig = preds.loc[cur_t]

stat_row([
    {"label": "Simulated bar", "value": cur_t.strftime("%Y-%m-%d %H:%M"), "sub": f"bar {cur_i - dataset.index.get_loc(test_start)} into test window"},
    {"label": "Price", "value": fmt_num(dataset['price_close'].loc[cur_t], 3)},
    {"label": "Position", "value": "Long" if paper["position"] > 0 else ("Short" if paper["position"] < 0 else "Flat")},
    {"label": "Paper Equity", "value": fmt_pct((paper["equity"] - 1) * 100)},
])

st.write("")
left, right = st.columns([2, 1])
with left:
    section_title("Paper equity curve")
    eq_df = pd.DataFrame(paper["equity_history"], columns=["time", "equity"]).set_index("time")
    fig = go.Figure(go.Scatter(x=eq_df.index, y=(eq_df["equity"] - 1) * 100,
                                 line=dict(color=SERIES["blue"], width=2), fill="tozeroy",
                                 fillcolor="rgba(57,135,229,0.10)"))
    fig.update_layout(height=340, yaxis_title="Cumulative return (%)")
    st.plotly_chart(fig, use_container_width=True)

with right:
    section_title("Current AI signal")
    diff = last_sig["p_up"] - last_sig["p_down"]
    would_trade = abs(diff) > entry_threshold and last_sig["confidence"] >= min_confidence
    direction_label = "Long" if diff > 0 else "Short"
    st.markdown(badge_html(f"{'Would enter ' + direction_label if would_trade else 'No signal'}",
                             "good" if would_trade and diff > 0 else ("critical" if would_trade else "neutral")),
                unsafe_allow_html=True)
    st.write("")
    st.metric("P(up)", f"{last_sig['p_up']*100:.1f}%")
    st.metric("P(down)", f"{last_sig['p_down']*100:.1f}%")
    st.metric("Confidence", f"{last_sig['confidence']*100:.1f}%")

    if paper["position"] != 0:
        vol_col = "tf5m_atr_pct" if "tf5m_atr_pct" in dataset.columns else None
        atr_pct = dataset[vol_col].loc[cur_t] if vol_col else np.nan
        if pd.notna(atr_pct) and atr_pct > 0:
            suggested_size = min(1.0, risk_per_trade / atr_pct)
            st.caption(f"Risk-normalized size suggestion: **{suggested_size*100:.0f}%** of max (risk budget {risk_per_trade}% ÷ ATR {atr_pct:.2f}%).")

if paper["blotter"]:
    st.write("")
    section_title("Simulated trade blotter")
    blotter_df = pd.DataFrame(paper["blotter"]).sort_values("exit_time", ascending=False)
    blotter_display = format_df(blotter_df, {"entry_price": "{:.3f}", "exit_price": "{:.3f}", "pnl_pct": "{:+.3f}"})
    st.dataframe(blotter_display, use_container_width=True, height=280)

st.caption(
    "This is a research paper-trading simulator, not a live broker connection — it steps through "
    "the held-out test window bar by bar using the same signal rules as the Backtesting page, so "
    "you can watch the model's decisions unfold sequentially instead of vectorized all at once."
)
