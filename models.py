"""Final model: LightGBM for the lane price + a small linear layer for the date effect.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from features import LANE_FEATURES

QS_NEUTRAL = 2.05  #quote_signal level where its effect changes sign

GBM_PARAMS = dict(
    n_estimators=600, learning_rate=0.04, num_leaves=31, min_child_samples=40,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=4,
)


def date_matrix(X: pd.DataFrame, terms: tuple) -> np.ndarray:
    cols = {}
    qs = X["quote_signal"] - QS_NEUTRAL
    qs_day = X["quote_signal_date"] - QS_NEUTRAL
    if "market" in terms:
        cols["market_index"] = X["market_index"]
        cols["market_index_date"] = X["market_index_date"]
    if "trend" in terms:
        cols["t_days"] = X["t_days"] / 100.0
    if "quote" in terms:
        cols["qs"] = qs
        cols["qs_sq"] = qs ** 2           
        cols["qs_x_day"] = qs * qs_day    #sign of its effect follows the day's level
        cols["qs_day"] = qs_day
    if "dow" in terms:
        for d in range(1, 7):
            cols[f"dow{d}"] = (X["dow"] == d).astype(float)
    return pd.DataFrame(cols, index=X.index).to_numpy(dtype=float)


class LaneDateModel:
    def __init__(self, terms=("market", "trend", "quote"), rounds: int = 3, gbm_params=None):
        self.terms = tuple(terms)
        self.rounds = rounds
        self.gbm_params = gbm_params or GBM_PARAMS

    def fit(self, X: pd.DataFrame, log_rate: pd.Series):
        y = log_rate.to_numpy()
        Z = date_matrix(X, self.terms)
        effect = np.zeros(len(y))
        for _ in range(self.rounds):
            self.gbm = lgb.LGBMRegressor(**self.gbm_params).fit(X[LANE_FEATURES], y - effect)
            lane = self.gbm.predict(X[LANE_FEATURES])
            self.linear = Ridge(alpha=1e-3).fit(Z, y - lane)
            effect = self.linear.predict(Z)
        return self

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        return self.gbm.predict(X[LANE_FEATURES]) + self.linear.predict(date_matrix(X, self.terms))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(X))