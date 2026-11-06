"""Validation experiments: how the data is split and which model is chosen.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES, build_features, flag_rate_outliers, load_raw
from models import LaneDateModel

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
SEED = 42

LGB_PARAMS = dict(
    n_estimators=700, learning_rate=0.03, num_leaves=31, min_child_samples=40,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=SEED, verbose=-1, n_jobs=4,
)
NO_MARKET = [c for c in FEATURES if not c.startswith(("market_index", "quote_signal"))]


def metrics(y_true, y_pred) -> dict:
    err = y_pred - y_true
    ape = np.abs(err) / y_true
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_%": float(100 * ape.mean()),
        "MedAPE_%": float(100 * np.median(ape)),
    }



class LocalMedianBaseline:
    """Simplest sensible baseline: median rate per mile for the same equipment and distance slice."""

    def fit(self, X, df, y):
        self.edges = np.unique(np.quantile(X["distance"], np.linspace(0, 1, 21)))
        slice_ = np.clip(np.searchsorted(self.edges, X["distance"], side="right") - 1, 0, len(self.edges) - 2)
        rpm = pd.Series(y.values / X["distance"].values, index=X.index)
        self.table = rpm.groupby([df["equipment"].values, slice_]).median()
        self.fallback = float(rpm.median())
        return self

    def predict(self, X, df):
        slice_ = np.clip(np.searchsorted(self.edges, X["distance"], side="right") - 1, 0, len(self.edges) - 2)
        keys = list(zip(df["equipment"].values, slice_))
        rpm = np.array([self.table.get(k, self.fallback) for k in keys])
        return rpm * X["distance"].values


def linear_matrix(X, df):
    M = pd.DataFrame(index=X.index)
    M["log_distance"] = X["log_distance"]
    M["log_distance_sq"] = X["log_distance"] ** 2
    M["weight"] = X["weight_clean"] / 1e4
    M["log_market_index"] = np.log(X["market_index"])
    M["quote_signal"] = X["quote_signal"]
    for e in ["Flatbed", "Reefer"]:
        M[e] = (df["equipment"] == e).astype(float)
    for d in range(1, 7):
        M[f"dow{d}"] = (X["dow"] == d).astype(float)
    return M


HYBRIDS = {
    "hybrid_market_trend": ("market", "trend"),
    "hybrid_final": ("market", "trend", "quote"),
    "hybrid_final_plus_dow": ("market", "trend", "quote", "dow"),
}


def fit_predict(name, X_tr, df_tr, y_tr, X_te, df_te):
    """Fit one named model on the cleaned training rows and predict dollars for the test rows."""
    if name == "baseline_local_median":
        return LocalMedianBaseline().fit(X_tr, df_tr, y_tr).predict(X_te, df_te)
    if name == "linear_regression":
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        m.fit(linear_matrix(X_tr, df_tr), np.log(y_tr))
        return np.exp(m.predict(linear_matrix(X_te, df_te)))
    if name in HYBRIDS:
        return LaneDateModel(terms=HYBRIDS[name]).fit(X_tr, np.log(y_tr)).predict(X_te)
    cols = NO_MARKET if name == "lightgbm_no_market" else FEATURES
    m = lgb.LGBMRegressor(**LGB_PARAMS).fit(X_tr[cols], np.log(y_tr))
    return np.exp(m.predict(X_te[cols]))


MODELS = ["baseline_local_median", "linear_regression", "lightgbm_no_market", "lightgbm_all_features", *HYBRIDS]


def evaluate_split(train, X, outlier, tr_idx, te_idx, label):
    """Train on cleaned rows of tr_idx, score on te_idx (raw and cleaned)."""
    tr_idx = tr_idx[~outlier.loc[tr_idx].values]
    clean_te = te_idx[~outlier.loc[te_idx].values]
    rows = []
    for name in MODELS:
        pred = pd.Series(
            fit_predict(name, X.loc[tr_idx], train.loc[tr_idx], train.loc[tr_idx, "posted_rate"],
                        X.loc[te_idx], train.loc[te_idx]), index=te_idx)
        for scope, idx in [("clean_holdout", clean_te), ("raw_holdout", te_idx)]:
            rows.append({"split": label, "model": name, "scope": scope, "n_test": len(idx),
                         **metrics(train.loc[idx, "posted_rate"].values, pred.loc[idx].values)})
    return rows


FOLD_LAST_TRAIN_MONTHS = [4, 5, 6, 7, 8]  #train on Jan.., test on the next two months


def month_name(m):
    return pd.Timestamp(2025, m, 1).strftime("%b")


LAST_FOLD = "time: Jan-Aug -> Sep-Oct"  #closest copy of the real task
FOLD_ORDER = [f"time: Jan-{month_name(k)} -> {month_name(k + 1)}-{month_name(k + 2)}" for k in FOLD_LAST_TRAIN_MONTHS]


def summarize(table: pd.DataFrame):
    pd.set_option("display.width", 220)
    clean = table[table.scope == "clean_holdout"]
    pivot = clean.pivot(index="model", columns="split", values="MAPE_%").reindex(MODELS)
    pivot = pivot[FOLD_ORDER + ["RANDOM 80/20 (not used)"]]
    on_time = clean[clean.split.isin(FOLD_ORDER)]
    last = clean[clean.split == LAST_FOLD].set_index("model").reindex(MODELS)
    summary = pd.DataFrame({
        "mean_MAPE_%_time_folds": pivot[FOLD_ORDER].mean(axis=1),
        "mean_MAE_time_folds": on_time.groupby("model").MAE.mean().reindex(MODELS),
        "last_fold_MAPE_%": last["MAPE_%"],
        "last_fold_MAE": last["MAE"],
        "random_split_MAPE_%": pivot["RANDOM 80/20 (not used)"],
    }).round(3)
    summary.to_csv(RESULTS / "model_summary.csv")
    print("\n=== MAPE % by split (cleaned hold-out rows) ===")
    print(pivot.round(2).to_string())
    print("\n=== Summary (cleaned hold-out rows) ===")
    print(summary.to_string())
    raw = table[(table.scope == "raw_holdout") & table.split.isin(FOLD_ORDER)]
    print("\n=== Raw hold-out (still contains the corrupted rows), mean over time folds ===")
    print(raw.groupby("model")[["MAE", "MAPE_%", "MedAPE_%"]].mean().reindex(MODELS).round(3).to_string())


def main():
    train, _ = load_raw()
    X = build_features(train)
    outlier = flag_rate_outliers(train)
    print(f"Rows: {len(train):,}  flagged as corrupted rates: {outlier.sum()} ({outlier.mean():.2%})")

    month = train["date"].dt.month
    all_rows = []

    #always train on the past only, test on the next two unseen months.
    for k in FOLD_LAST_TRAIN_MONTHS:
        tr = train.index[month <= k]
        te = train.index[month.isin((k + 1, k + 2))]
        label = f"time: Jan-{month_name(k)} -> {month_name(k + 1)}-{month_name(k + 2)}"
        all_rows += evaluate_split(train, X, outlier, tr, te, label)

    #a random split mixes dates, so it flatters the model
    tr_r, te_r = train_test_split(train.index, test_size=0.2, random_state=SEED)
    all_rows += evaluate_split(train, X, outlier, pd.Index(tr_r), pd.Index(te_r), "RANDOM 80/20 (not used)")

    table = pd.DataFrame(all_rows)
    table.to_csv(RESULTS / "validation_table.csv", index=False)
    summarize(table)
    (RESULTS / "metrics.json").write_text(json.dumps(
        {"rows": len(train), "outliers_flagged": int(outlier.sum()), "results": all_rows}, indent=2))


if __name__ == "__main__":
    main()