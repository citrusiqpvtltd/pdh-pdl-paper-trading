"""
Retrain the ML entry filter on the FULL ~9-year dataset
(data/ml_training_data_full.csv, from build_full_history_dataset.py)
instead of just 2022-onward, to test whether the model actually needs to
have SEEN a market regime (e.g. 2018's crypto winter) to handle it well -
see backtest_5yr.py's 10-year run, which found the currently-deployed
model (trained only on 2022-2025-06) actively hurt performance in 2018.

Keeps the SAME held-out test period (>= TRAIN_END, 2025-06-01) as
train_ml_model.py so this is an apples-to-apples comparison against the
currently-deployed model on the ONE period neither model has ever
trained on - that comparison is what actually matters for a deployment
decision. The full 2017-2025-06 backtest_5yr-style check is a separate,
weaker check (2018 becomes in-sample for this model, so improvement there
mostly shows the model CAN fit that regime once given the data, not that
it generalizes to a regime it hasn't seen).

Saves to ml_entry_filter_full_history.joblib - a SEPARATE file, does not
touch the currently-deployed ml_entry_filter.joblib.

DIAGNOSED ISSUE (see README): training on the full ~9 years performed
worse, out-of-sample, than training on just 2022-2025.06, despite a
slightly higher raw AUC. Root cause: BTC's ATR% (volatility relative to
price) has shrunk roughly 4x over this span (2017 ~1.0% -> 2025-2026
~0.25%) - real market maturation, not a data bug. `atr_pct` (and other
absolute-scale features) feed into the tree model as raw numbers, so
training equally across eras with very different volatility regimes
teaches splits calibrated to conditions that barely exist anymore,
diluting calibration for the regime that actually matters for live
trading now.

FIX: exponential recency weighting (sample_weight, HALF_LIFE_DAYS) - the
model still sees all 9 years of patterns, but a row's influence on the
fit decays with age, so it can't get equally swayed by an outdated
volatility regime. Pass --half-life-days to try a different decay (default
365 - one year).
"""
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from train_ml_model import FEATURE_COLS, TRAIN_END, add_rule_based_score

IN_FILE = "data/ml_training_data_full.csv"
OUT_FILE = "ml_entry_filter_full_history.joblib"
DEFAULT_HALF_LIFE_DAYS = 365


def main():
    half_life_days = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_HALF_LIFE_DAYS

    df = pd.read_csv(IN_FILE, parse_dates=["time"])
    df = add_rule_based_score(df)
    df["side_sell"] = (df["side"] == "sell").astype(int)
    feats = FEATURE_COLS + ["side_sell"]

    train = df[df["time"] < TRAIN_END].copy()
    test = df[df["time"] >= TRAIN_END].copy()
    print(f"Train: {len(train)} rows ({train['time'].min()} - {train['time'].max()})", file=sys.stderr)
    print(f"Test:  {len(test)} rows ({test['time'].min()} - {test['time'].max()})", file=sys.stderr)

    X_train, y_train = train[feats], train["win"]
    X_test, y_test = test[feats], test["win"]
    medians = X_train.median()
    X_train_f = X_train.fillna(medians)
    X_test_f = X_test.fillna(medians)

    age_days = (pd.Timestamp(TRAIN_END) - train["time"]).dt.total_seconds() / 86400
    sample_weight = 0.5 ** (age_days / half_life_days)
    print(f"Recency weighting: half-life={half_life_days} days. "
          f"Oldest row weight={sample_weight.min():.5f}, newest row weight={sample_weight.max():.5f}, "
          f"effective sample size={sample_weight.sum():.0f} (of {len(train)} raw rows)", file=sys.stderr)

    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                            class_weight="balanced", random_state=42)
    model.fit(X_train_f, y_train, sample_weight=sample_weight.values)
    auc = roc_auc_score(y_test, model.predict_proba(X_test_f)[:, 1])
    print(f"\nTest AUC (full-population, same as before's 0.5197 for comparison): {auc:.4f}", file=sys.stderr)

    rule_subset = test[test["rule_qualifies"]]
    if len(rule_subset):
        proba = model.predict_proba(rule_subset[feats].fillna(medians))[:, 1]
        auc_subset = roc_auc_score(rule_subset["win"], proba) if rule_subset["win"].nunique() > 1 else float("nan")
        print(f"Test AUC (within rule-qualifying subset, same as before's 0.6254): {auc_subset:.4f}", file=sys.stderr)

    joblib.dump(dict(model=model, features=feats, medians=medians.to_dict()), OUT_FILE)
    print(f"\nSaved to {OUT_FILE} (trained on {len(train)} rows spanning {train['time'].min()} to {train['time'].max()}, "
          f"half-life={half_life_days}d)", file=sys.stderr)


if __name__ == "__main__":
    main()
