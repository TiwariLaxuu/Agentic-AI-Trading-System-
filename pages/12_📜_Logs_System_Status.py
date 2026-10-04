import sys

import numpy as np
import pandas as pd
import streamlit as st

from ui.theme import apply_theme, STATUS
from ui.components import section_title, stat_row, badge_html
from ui.state import get_config, get_dataset, get_training_result, get_backtest_result
from core.logging_utils import get_logs, clear_logs

st.set_page_config(page_title="Logs / System Status — AI Trading Engine", page_icon="📜", layout="wide")
apply_theme("Logs / System Status", "Recent activity and a health snapshot of the current session")

cfg = get_config()
dataset, events = get_dataset()
result = get_training_result()
bt = get_backtest_result()

section_title("System status")
nan_frac = float(dataset.tail(200).isna().mean().mean()) * 100
health = "good" if nan_frac < 15 else ("warning" if nan_frac < 35 else "critical")

stat_row([
    {"label": "Dataset", "value": f"{dataset.shape[0]} x {dataset.shape[1]}", "sub": "bars x features"},
    {"label": "Recent NaN ratio", "value": f"{nan_frac:.1f}%", "sub": "last 200 bars, all columns"},
    {"label": "Model Trained", "value": "Yes" if result else "No"},
    {"label": "Backtest Run", "value": "Yes" if bt else "No"},
])
st.write("")
st.markdown(badge_html("Feature pipeline healthy" if health == "good" else "Elevated NaN ratio — check warmup window", health),
            unsafe_allow_html=True)

st.write("")
section_title("Environment")
try:
    import torch, xgboost, sklearn, plotly, streamlit as st_mod
    versions = {
        "python": sys.version.split()[0], "streamlit": st_mod.__version__, "pandas": pd.__version__,
        "numpy": np.__version__, "xgboost": xgboost.__version__, "torch": torch.__version__,
        "scikit-learn": sklearn.__version__, "plotly": plotly.__version__,
    }
except Exception:
    versions = {}
st.dataframe(pd.DataFrame(versions.items(), columns=["package", "version"]), use_container_width=True, hide_index=True)

st.write("")
section_title("Active configuration")
cfg_summary = {
    "Symbol": cfg.data.symbol, "Base timeframe": cfg.data.base_timeframe,
    "Enabled timeframes": ", ".join(cfg.enabled_timeframes), "Lookback bars": cfg.data.lookback_bars,
    "BB period / std": f"{cfg.bollinger.period} / {cfg.bollinger.std_multiplier}",
    "Gap threshold": f"{cfg.gaps.min_gap_pct}%", "Forward horizon": f"{cfg.model.forward_horizon_bars} bars",
}
cfg_summary_df = pd.DataFrame([(k, str(v)) for k, v in cfg_summary.items()], columns=["setting", "value"])
st.dataframe(cfg_summary_df, use_container_width=True, hide_index=True)

st.write("")
section_title("Activity log")
c1, c2 = st.columns([3, 1])
level_filter = c1.multiselect("Levels", ["DEBUG", "INFO", "WARNING", "ERROR"], default=["INFO", "WARNING", "ERROR"])
if c2.button("Clear log"):
    clear_logs()
    st.rerun()

logs = get_logs()
if logs:
    log_df = pd.DataFrame(logs)
    if level_filter:
        log_df = log_df[log_df["level"].isin(level_filter)]
    st.dataframe(log_df.sort_index(ascending=False), use_container_width=True, height=420, hide_index=True)
else:
    st.caption("No log entries yet this session — actions like saving Settings or running Training will appear here.")
