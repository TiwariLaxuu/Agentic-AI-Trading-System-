"""Training orchestration: builds the feature/label split, fits both the
XGBoost and Temporal Transformer heads on the identical standardized feature
matrix, evaluates each on held-out validation/test windows, and combines
their outputs into the spec's AI Output (section 13): continuation
probability, reversal probability, gap closure probability, expected return,
confidence.
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score

from core.models.labels import build_all_labels
from core.models.xgb_model import XGBSuite
from core.models.transformer_model import TemporalTransformerSuite


def select_feature_columns(dataset: pd.DataFrame) -> list[str]:
    numeric = dataset.select_dtypes(include=[np.number, bool]).columns.tolist()
    return [c for c in numeric if not c.startswith("price_")]


@dataclasses.dataclass
class SplitIndex:
    train: pd.Index
    val: pd.Index
    test: pd.Index


def time_split(index: pd.Index, train_split: float, val_split: float, embargo: int) -> SplitIndex:
    n = len(index)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))
    train_idx = index[:max(train_end - embargo, 1)]
    val_idx = index[train_end:max(val_end - embargo, train_end + 1)]
    test_idx = index[val_end:]
    return SplitIndex(train_idx, val_idx, test_idx)


@dataclasses.dataclass
class TrainingResult:
    xgb_suite: XGBSuite
    transformer_suite: TemporalTransformerSuite
    feature_cols: list[str]
    splits: SplitIndex
    metrics: dict
    labels: pd.DataFrame


def _classification_metrics(y_true_dir: pd.Series, p_down, p_flat, p_up) -> dict:
    valid = y_true_dir.notna()
    if valid.sum() == 0:
        return {}
    pred = np.argmax(np.column_stack([p_down, p_flat, p_up])[valid.values], axis=1)
    dir_map = {-1: 0, 0: 1, 1: 2}
    true_enc = y_true_dir[valid].map(dir_map).values
    return {"direction_accuracy": float(accuracy_score(true_enc, pred))}


def _regression_metrics(y_true_ret: pd.Series, pred_ret) -> dict:
    valid = y_true_ret.notna() & pd.notna(pred_ret)
    if valid.sum() == 0:
        return {}
    mae = mean_absolute_error(y_true_ret[valid], np.asarray(pred_ret)[valid.values])
    corr = float(np.corrcoef(y_true_ret[valid], np.asarray(pred_ret)[valid.values])[0, 1]) if valid.sum() > 2 else np.nan
    return {"return_mae_pct": float(mae), "return_direction_corr": corr}


def _gap_metrics(y_true_gap: pd.Series, pred_gap) -> dict:
    mask = y_true_gap.notna() & pd.notna(pred_gap)
    if mask.sum() < 10 or y_true_gap[mask].nunique() < 2:
        return {}
    try:
        auc = float(roc_auc_score(y_true_gap[mask], np.asarray(pred_gap)[mask.values]))
    except ValueError:
        auc = np.nan
    return {"gap_closure_auc": auc}


def train_all_models(dataset: pd.DataFrame, cfg_model, progress_cb=None) -> TrainingResult:
    labels = build_all_labels(dataset, cfg_model)
    feature_cols = select_feature_columns(dataset)
    X = dataset[feature_cols]

    splits = time_split(dataset.index, cfg_model.train_split, cfg_model.val_split,
                          embargo=cfg_model.forward_horizon_bars)

    xgb_suite = XGBSuite(cfg_model.xgb_n_estimators, cfg_model.xgb_max_depth, cfg_model.xgb_learning_rate)
    xgb_suite.fit(X.loc[splits.train], labels.loc[splits.train, "y_direction"],
                  labels.loc[splits.train, "y_return"], labels.loc[splits.train, "y_gap_closure"])

    transformer_suite = TemporalTransformerSuite(
        seq_len=cfg_model.transformer_seq_len, d_model=cfg_model.transformer_d_model,
        nhead=cfg_model.transformer_nhead, n_layers=cfg_model.transformer_layers,
        epochs=cfg_model.transformer_epochs,
    )
    transformer_suite.fit(X.loc[splits.train], labels.loc[splits.train, "y_direction"],
                            labels.loc[splits.train, "y_return"], labels.loc[splits.train, "y_gap_closure"],
                            progress_cb=progress_cb)

    metrics = {}
    for split_name, idx in (("val", splits.val), ("test", splits.test)):
        if len(idx) == 0:
            continue
        xgb_pred = xgb_suite.predict(X.loc[idx])
        tr_pred = transformer_suite.predict(X.loc[idx])
        for model_name, pred in (("xgboost", xgb_pred), ("transformer", tr_pred)):
            m = {}
            m.update(_classification_metrics(labels.loc[idx, "y_direction"], pred["p_down"], pred["p_flat"], pred["p_up"]))
            m.update(_regression_metrics(labels.loc[idx, "y_return"], pred["expected_return"]))
            m.update(_gap_metrics(labels.loc[idx, "y_gap_closure"], pred["p_gap_closure"]))
            metrics[f"{split_name}_{model_name}"] = m

    return TrainingResult(xgb_suite, transformer_suite, feature_cols, splits, metrics, labels)


def predict_ensemble(dataset: pd.DataFrame, result: TrainingResult) -> pd.DataFrame:
    X = dataset[result.feature_cols]
    xgb_pred = result.xgb_suite.predict(X)
    tr_pred = result.transformer_suite.predict(X)

    out = pd.DataFrame(index=dataset.index)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for key in ["p_down", "p_flat", "p_up", "expected_return", "p_gap_closure"]:
            a = np.asarray(xgb_pred[key], dtype=float)
            b = np.asarray(tr_pred[key], dtype=float)
            stacked = np.vstack([a, b])
            out[key] = np.nanmean(stacked, axis=0)
            out[f"{key}_xgb"] = a
            out[f"{key}_transformer"] = b

    trend_col = "trendbase_trend_direction"
    trend_dir = dataset[trend_col] if trend_col in dataset.columns else pd.Series(0, index=dataset.index)
    trend_sign = np.sign(trend_dir.fillna(0))
    out["continuation_probability"] = np.where(trend_sign >= 0, out["p_up"], out["p_down"])
    out["reversal_probability"] = np.where(trend_sign >= 0, out["p_down"], out["p_up"])
    out["expected_return_pct"] = out["expected_return"]
    out["confidence"] = out[["p_down", "p_flat", "p_up"]].max(axis=1)
    if "gap_still_open" in dataset.columns:
        gap_open_mask = dataset["gap_still_open"].fillna(0).astype(bool)
        out["gap_closure_probability"] = out["p_gap_closure"].where(gap_open_mask)
    else:
        out["gap_closure_probability"] = out["p_gap_closure"]
    return out
