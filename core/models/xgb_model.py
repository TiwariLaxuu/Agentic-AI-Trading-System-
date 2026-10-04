"""XGBoost head: structured/tabular learning over the unified feature set."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb


class XGBSuite:
    """Three XGBoost models sharing the same feature matrix:
    - direction: 3-class (down/flat/up) classifier
    - regressor: forward return (%) regressor
    - gap_closure: binary classifier, fit only on rows with an open gap
    """

    def __init__(self, n_estimators: int = 300, max_depth: int = 4, learning_rate: float = 0.05):
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                             tree_method="hist", n_jobs=-1)
        self.direction_model: xgb.XGBClassifier | None = None
        self.return_model: xgb.XGBRegressor | None = None
        self.gap_closure_model: xgb.XGBClassifier | None = None
        self.feature_names: list[str] = []
        self.feature_importance_: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y_direction: pd.Series, y_return: pd.Series,
            y_gap_closure: pd.Series | None = None) -> "XGBSuite":
        self.feature_names = list(X.columns)
        dir_map = {-1: 0, 0: 1, 1: 2}
        y_dir_enc = y_direction.map(dir_map)

        self.direction_model = xgb.XGBClassifier(**self.params, objective="multi:softprob", num_class=3)
        self.direction_model.fit(X, y_dir_enc)

        self.return_model = xgb.XGBRegressor(**self.params, objective="reg:squarederror")
        self.return_model.fit(X, y_return)

        if y_gap_closure is not None:
            mask = y_gap_closure.notna()
            if mask.sum() >= 30 and y_gap_closure[mask].nunique() > 1:
                self.gap_closure_model = xgb.XGBClassifier(**self.params, objective="binary:logistic")
                self.gap_closure_model.fit(X[mask], y_gap_closure[mask])

        self.feature_importance_ = pd.Series(
            self.direction_model.feature_importances_, index=self.feature_names
        ).sort_values(ascending=False)
        return self

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        X = X[self.feature_names]
        dir_proba = self.direction_model.predict_proba(X)  # columns: down, flat, up
        expected_return = self.return_model.predict(X)
        out = {
            "p_down": dir_proba[:, 0],
            "p_flat": dir_proba[:, 1],
            "p_up": dir_proba[:, 2],
            "expected_return": expected_return,
        }
        if self.gap_closure_model is not None:
            out["p_gap_closure"] = self.gap_closure_model.predict_proba(X)[:, 1]
        else:
            out["p_gap_closure"] = np.full(len(X), np.nan)
        return out
