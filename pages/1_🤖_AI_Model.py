import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.theme import apply_theme, SERIES, SURFACE, BORDER, INK_MUTED, INK_PRIMARY
from ui.components import section_title, stat_row, fmt_pct, fmt_num, format_df
from ui.state import get_config, get_dataset, get_training_result, get_predictions

st.set_page_config(page_title="AI / Model — AI Trading Engine", page_icon="🤖", layout="wide")
apply_theme("AI / Model", "Architecture, feature importance, and live model output")

cfg = get_config()
dataset, events = get_dataset()
result = get_training_result()
preds = get_predictions()

section_title("Architecture")


def _flow_box(label: str, fill: str = SURFACE, text_color: str = INK_PRIMARY) -> str:
    return (f'<div style="background:{fill};border:1px solid {BORDER};border-radius:8px;'
            f'padding:8px 12px;color:{text_color};font-size:0.8rem;font-weight:600;'
            f'text-align:center;white-space:nowrap;">{label}</div>')


def _arrow_down() -> str:
    return f'<div style="text-align:center;color:{INK_MUTED};font-size:1.1rem;line-height:1;">↓</div>'


feature_blocks = ["Bollinger Bands", "Volatility", "Trend / Momentum", "External Support &amp; Resistance",
                   "Intraday Gap Analysis", "Out-of-Hours Gap Analysis", "Gap Closure", "Market Regime",
                   "Time / Session"]
blocks_html = "".join(
    f'<div style="flex:1 1 100px;min-width:100px;">{_flow_box(b, fill="#141413", text_color=INK_MUTED)}</div>'
    for b in feature_blocks
)

st.markdown(f"""
<div style="display:flex;flex-direction:column;gap:4px;max-width:760px;margin:0 auto;">
    {_flow_box("Market Data")}
    {_arrow_down()}
    {_flow_box("Data Processing")}
    {_arrow_down()}
    {_flow_box("Multi-Timeframe Feature Engine", fill="#1a1a19")}
    <div style="display:flex;gap:4px;flex-wrap:wrap;margin:2px 0;">{blocks_html}</div>
    {_arrow_down()}
    {_flow_box("Unified Feature Dataset", fill="#184f95")}
    {_arrow_down()}
    {_flow_box("XGBoost&nbsp;&nbsp;+&nbsp;&nbsp;Temporal Transformer", fill="#256abf")}
    {_arrow_down()}
    {_flow_box("Probabilities / Expected Return", fill="#199e70")}
    {_arrow_down()}
    {_flow_box("Risk Management", fill="#d95926")}
</div>
""", unsafe_allow_html=True)

if result is None or preds is None:
    st.info("No trained model yet. Go to **Training** to fit both models on the current feature dataset.")
else:
    st.write("")
    section_title("Feature importance (XGBoost — direction head)")
    top_n = st.slider("Top N features", 5, 40, 20)
    importance = result.xgb_suite.feature_importance_.head(top_n).sort_values()
    fig = go.Figure(go.Bar(
        x=importance.values, y=importance.index, orientation="h",
        marker_color=SERIES["blue"],
    ))
    fig.update_layout(height=max(320, 22 * top_n), margin=dict(l=10, r=10, t=10, b=10),
                       xaxis_title="Gain-based importance")
    st.plotly_chart(fig, use_container_width=True)

    st.write("")
    section_title("Latest AI output")
    last = preds.iloc[-1]
    stat_row([
        {"label": "Continuation Probability", "value": f"{last['continuation_probability']*100:.1f}%"},
        {"label": "Reversal Probability", "value": f"{last['reversal_probability']*100:.1f}%"},
        {"label": "Expected Return", "value": fmt_pct(last["expected_return_pct"])},
        {"label": "Confidence", "value": f"{last['confidence']*100:.1f}%"},
    ])

    st.write("")
    section_title("Model agreement — XGBoost vs Temporal Transformer")
    recent = preds.tail(200)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=recent.index, y=recent["p_up_xgb"], name="P(up) XGBoost",
                                line=dict(color=SERIES["blue"], width=2)))
    fig2.add_trace(go.Scatter(x=recent.index, y=recent["p_up_transformer"], name="P(up) Transformer",
                                line=dict(color=SERIES["orange"], width=2)))
    fig2.update_layout(height=320, yaxis_title="P(up)", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig2, use_container_width=True)

    section_title("Recent predictions")
    show_cols = ["price_close", "continuation_probability", "reversal_probability",
                 "gap_closure_probability", "expected_return_pct", "confidence"]
    display_df = pd.concat([dataset[["price_close"]], preds.drop(columns=["price_close"], errors="ignore")], axis=1)
    display_df = display_df[[c for c in show_cols if c in display_df.columns]].tail(25).sort_index(ascending=False)
    st.dataframe(format_df(display_df, {c: "{:.3f}" for c in display_df.select_dtypes("number").columns}),
                 use_container_width=True)

st.write("")
section_title("Why two models")
st.caption(
    "**XGBoost** learns from the standardized tabular feature set directly — good at picking up "
    "threshold effects and interactions between BB/volatility/gap/regime features. "
    "**Temporal Transformer** consumes the same features as a rolling window of bars and learns "
    "chronological relationships (how the sequence of recent bars, not just their latest values, "
    "shapes the outcome). Both are trained on an identical, leakage-safe feature matrix; the app "
    "ensembles their outputs by simple averaging."
)
