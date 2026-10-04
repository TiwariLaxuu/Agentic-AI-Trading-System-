import time

import pandas as pd
import streamlit as st

from ui.theme import apply_theme
from ui.components import section_title, fmt_num
from ui.state import get_config, get_dataset, persist_config, set_training_result, set_predictions
from core.models.trainer import train_all_models
from core.models.trainer import predict_ensemble
from core.logging_utils import get_logger

logger = get_logger("training")

st.set_page_config(page_title="Training — AI Trading Engine", page_icon="🎯", layout="wide")
apply_theme("Training", "Fit XGBoost + Temporal Transformer on the unified feature dataset")

cfg = get_config()
dataset, events = get_dataset()

section_title("Training configuration")
with st.form("training_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**XGBoost**")
        n_estimators = st.number_input("n_estimators", 50, 1000, cfg.model.xgb_n_estimators, step=50)
        max_depth = st.number_input("max_depth", 2, 10, cfg.model.xgb_max_depth)
        lr = st.number_input("learning_rate", 0.01, 0.5, cfg.model.xgb_learning_rate, step=0.01, format="%.2f")
    with c2:
        st.markdown("**Temporal Transformer**")
        seq_len = st.number_input("sequence length (bars)", 8, 128, cfg.model.transformer_seq_len, step=4)
        d_model = st.number_input("d_model", 16, 256, cfg.model.transformer_d_model, step=16)
        nhead = st.selectbox("attention heads", [2, 4, 8], index=[2, 4, 8].index(cfg.model.transformer_nhead) if cfg.model.transformer_nhead in [2,4,8] else 1)
        layers = st.number_input("encoder layers", 1, 6, cfg.model.transformer_layers)
        epochs = st.number_input("epochs", 1, 100, cfg.model.transformer_epochs)
    with c3:
        st.markdown("**Labels / split**")
        horizon = st.number_input("forward horizon (bars)", 2, 100, cfg.model.forward_horizon_bars)
        train_split = st.slider("train fraction", 0.4, 0.9, cfg.model.train_split, step=0.05)
        val_split = st.slider("validation fraction", 0.05, 0.4, cfg.model.val_split, step=0.05)
        st.caption(f"Test fraction: {max(0.0, 1 - train_split - val_split):.2f} — chronological split with an embargo of {horizon} bars at each boundary to prevent label leakage.")

    submitted = st.form_submit_button("Save configuration", use_container_width=False)
    if submitted:
        cfg.model.xgb_n_estimators = int(n_estimators)
        cfg.model.xgb_max_depth = int(max_depth)
        cfg.model.xgb_learning_rate = float(lr)
        cfg.model.transformer_seq_len = int(seq_len)
        cfg.model.transformer_d_model = int(d_model)
        cfg.model.transformer_nhead = int(nhead)
        cfg.model.transformer_layers = int(layers)
        cfg.model.transformer_epochs = int(epochs)
        cfg.model.forward_horizon_bars = int(horizon)
        cfg.model.train_split = float(train_split)
        cfg.model.val_split = float(val_split)
        persist_config(cfg)
        st.success("Training configuration saved.")
        st.rerun()

st.write("")
train_clicked = st.button("🚀 Train Now", type="primary")

if train_clicked:
    progress_bar = st.progress(0.0, text="Preparing training data...")
    status_text = st.empty()

    def cb(epoch, total, loss):
        progress_bar.progress(epoch / total, text=f"Temporal Transformer — epoch {epoch}/{total} (loss {loss:.4f})")

    t0 = time.time()
    with st.spinner("Fitting XGBoost..."):
        pass
    result = train_all_models(dataset, cfg.model, progress_cb=cb)
    progress_bar.progress(1.0, text="Done")
    preds = predict_ensemble(dataset, result)
    set_training_result(result)
    set_predictions(preds)
    logger.info(f"Training complete in {time.time()-t0:.1f}s — "
                f"{len(result.splits.train)} train / {len(result.splits.val)} val / {len(result.splits.test)} test bars")
    st.success(f"Training complete in {time.time()-t0:.1f}s.")

result = st.session_state.get("training_result")
if result is not None:
    st.write("")
    section_title("Split summary")
    s = result.splits
    c1, c2, c3 = st.columns(3)
    c1.metric("Train bars", len(s.train), help=f"{s.train[0]} → {s.train[-1]}" if len(s.train) else "")
    c2.metric("Validation bars", len(s.val), help=f"{s.val[0]} → {s.val[-1]}" if len(s.val) else "")
    c3.metric("Test bars", len(s.test), help=f"{s.test[0]} → {s.test[-1]}" if len(s.test) else "")

    section_title("Held-out metrics")
    rows = []
    for split_model, m in result.metrics.items():
        split, model = split_model.split("_", 1)
        row = {"split": split, "model": model}
        row.update(m)
        rows.append(row)
    metrics_df = pd.DataFrame(rows)
    if not metrics_df.empty:
        for col in metrics_df.columns:
            if col in ("split", "model"):
                continue
            metrics_df[col] = metrics_df[col].apply(lambda v: fmt_num(v, 4) if pd.notna(v) else "—")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    st.caption(
        "direction_accuracy: 3-class (down/flat/up) accuracy over the forward horizon · "
        "return_mae_pct: mean absolute error of the expected-return regression · "
        "return_direction_corr: correlation between predicted and realized return · "
        "gap_closure_auc: ROC-AUC restricted to bars with an open gap."
    )
else:
    st.info("No model trained yet this session. Configure hyperparameters above and click **Train Now**.")
