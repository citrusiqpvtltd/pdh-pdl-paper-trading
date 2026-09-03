"""
Diagnostic follow-up to train_full_history_model.py: soft recency-weighting
(sample_weight decay) did NOT fix the full-history model's underperformance
- subset AUC got worse (0.54-0.57) than the deployed model's 0.625, likely
because HistGradientBoostingClassifier bins continuous features (atr_pct
etc.) into quantiles computed from the FULL unweighted training array, so
old rows still dilute bin resolution for today's much narrower volatility
range even when down-weighted in the loss.

This sweeps HARD training-window cutoffs instead (actually excluding old
rows, like the currently-deployed model already does at ~3.4 years) to
find whether a different window length beats what's deployed - not
whether "more data with decay" can rescue the full-history approach.
"""
import sys

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from train_ml_model import FEATURE_COLS, TRAIN_END, add_rule_based_score

IN_FILE = "data/ml_training_data_full.csv"
WINDOWS_DAYS = [365, 545, 730, 1095, 1260, 1460, None]  # None = deployed's exact ~1246 days (2022-01-01 to TRAIN_END)


def main():
    df = pd.read_csv(IN_FILE, parse_dates=["time"])
    df = add_rule_based_score(df)
    df["side_sell"] = (df["side"] == "sell").astype(int)
    feats = FEATURE_COLS + ["side_sell"]

    test = df[df["time"] >= TRAIN_END].copy()
    X_test, y_test = test[feats], test["win"]
    rule_subset = test[test["rule_qualifies"]]

    train_end_ts = pd.Timestamp(TRAIN_END)
    for window in WINDOWS_DAYS:
        train_start = pd.Timestamp("2022-01-01") if window is None else train_end_ts - pd.Timedelta(days=window)
        train = df[(df["time"] >= train_start) & (df["time"] < train_end_ts)]
        X_train, y_train = train[feats], train["win"]
        medians = X_train.median()
        X_train_f = X_train.fillna(medians)
        X_test_f = X_test.fillna(medians)

        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                                class_weight="balanced", random_state=42)
        model.fit(X_train_f, y_train)
        auc = roc_auc_score(y_test, model.predict_proba(X_test_f)[:, 1])
        auc_subset = float("nan")
        if len(rule_subset):
            proba = model.predict_proba(rule_subset[feats].fillna(medians))[:, 1]
            if rule_subset["win"].nunique() > 1:
                auc_subset = roc_auc_score(rule_subset["win"], proba)

        label = "deployed-equivalent (2022-01-01 cutoff)" if window is None else f"{window}d ({window/365.25:.2f}yr)"
        print(f"window={label:42s} n_train={len(train):6d}  AUC(full)={auc:.4f}  AUC(subset)={auc_subset:.4f}")


if __name__ == "__main__":
    main()
