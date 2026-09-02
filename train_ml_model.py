"""
Train an ML entry filter on data/ml_training_data.csv (built by
build_ml_dataset.py) to replace the hand-tuned "score >= 3" rule.

Chronological split (not random) - train on the earlier period, test on
the later one, since shuffling would leak future information into
training via overlapping/correlated nearby setups.

Evaluates two ways:
  1. Standard ML metrics (ROC-AUC) on the held-out test set.
  2. The real test: simulate an actual out-of-sample backtest using the
     trained model's predicted probability as the entry filter, and
     compare CAGR/profit-factor/drawdown against the existing rule-based
     "score >= 3" threshold restricted to the SAME test period - a model
     that only looks good on AUC but doesn't beat the simple rule
     out-of-sample is not worth deploying.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "bearish_candle", "bullish_candle", "double_top", "double_bottom",
    "head_shoulders", "inv_head_shoulders", "rising_wedge", "falling_wedge",
    "bull_flag", "bear_flag", "failed_breakout_pdh", "failed_breakdown_pdl",
    "structure_bull", "structure_bear", "vol_confirm", "vol_ratio",
    "momentum_bull", "momentum_bear", "htf_trend_up", "rsi", "atr_pct",
    "hour", "dow",
]
# A book-informed expansion of this set (~30 more indicators drawn from Nison,
# Elder, and Murphy - see build_ml_dataset.py/backtest_pdh_pdl.py comments and
# README) was tried and found to NOT improve AUC (0.495/0.522 vs 0.4925/0.5197
# here) or the sequential out-of-sample validation (noisier, non-monotonic
# across thresholds vs this simpler set's clean monotonic improvement) - see
# README for the full comparison. This leaner set is what's actually deployed.

TRAIN_END = "2025-06-01"  # ~3.4 years train, ~1.25 years held-out test


def add_rule_based_score(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the live engine's actual 0-6 score from the raw feature
    columns, so the ML filter can be compared against the REAL current
    baseline (score >= 3) on the identical test-period rows - not just an
    unfiltered 'take everything' strawman."""
    chart_bear = (df["double_top"] | df["head_shoulders"] | df["rising_wedge"] | df["bear_flag"] | df["failed_breakout_pdh"]).astype(int)
    chart_bull = (df["double_bottom"] | df["inv_head_shoulders"] | df["falling_wedge"] | df["bull_flag"] | df["failed_breakdown_pdl"]).astype(int)
    sell_score = 1 + df["bearish_candle"] + chart_bear + df["vol_confirm"] + df["structure_bear"] + df["momentum_bear"]
    buy_score = 1 + df["bullish_candle"] + chart_bull + df["vol_confirm"] + df["structure_bull"] + df["momentum_bull"]
    df["score"] = np.where(df["side"] == "sell", sell_score, buy_score)
    # the live engine also hard-gates on HTF trend direction (use_htf_filter=True)
    # and breakout protection, separate from the score itself - reconstruct both
    # for a faithful comparison. NOTE: this still does NOT reconstruct the live
    # engine's per-approach "fired" one-shot flag (stateful, can't be derived from
    # a single row) - so `rule_qualifies` here is necessarily a superset of what
    # the live engine would actually enter. Use validate_ml_filter.py (which
    # drives the real backtest_pdh_pdl.step_bar via its ml_gate hook) for any
    # claim about beating the actual deployed system.
    htf_ok = np.where(df["side"] == "sell", ~df["htf_trend_up"].astype(bool), df["htf_trend_up"].astype(bool))
    breakout_ok = np.where(df["side"] == "sell", ~df["strong_break_up"].astype(bool), ~df["strong_break_down"].astype(bool))
    df["rule_qualifies"] = (df["score"] >= 3) & htf_ok & breakout_ok
    return df


def load_and_split():
    df = pd.read_csv("data/ml_training_data.csv", parse_dates=["time"])
    df = add_rule_based_score(df)
    df["side_sell"] = (df["side"] == "sell").astype(int)
    feats = FEATURE_COLS + ["side_sell"]

    train = df[df["time"] < TRAIN_END].copy()
    test = df[df["time"] >= TRAIN_END].copy()
    print(f"Train: {len(train)} rows ({train['time'].min()} - {train['time'].max()})", file=sys.stderr)
    print(f"Test:  {len(test)} rows ({test['time'].min()} - {test['time'].max()})", file=sys.stderr)
    return train, test, feats


def summarize_r(sub: pd.DataFrame) -> dict:
    """Non-compounding summary in R-multiples - deliberately NOT an equity
    simulation. The labeled dataset contains many overlapping/simultaneous
    touches (this labeling has no is_flat constraint - see
    build_ml_dataset.py), so naively compounding them in row order as if
    they were sequential, non-overlapping trades is invalid and produces
    nonsense (an earlier version of this function did exactly that and
    produced fake 50,000%+ 'returns' - a bug, not a result). Mean/median R
    and win rate are valid regardless of overlap; a real equity curve is
    not, and isn't computed here."""
    if not len(sub):
        return dict(trades=0)
    r = sub["realized_r"]
    wins = r[r > 0]
    losses = r[r <= 0]
    pf = wins.sum() / -losses.sum() if losses.sum() != 0 else float("inf")
    return dict(trades=len(sub), win_rate=(r > 0).mean() * 100, mean_r=r.mean(),
                median_r=r.median(), profit_factor=pf)


def main():
    train, test, feats = load_and_split()

    X_train, y_train = train[feats], train["win"]
    X_test, y_test = test[feats], test["win"]

    # median-impute NaNs (vol_ratio/rsi can be NaN early in warm-up windows)
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    print("\n=== Logistic Regression (interpretable baseline) ===", file=sys.stderr)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(X_train_s, y_train)
    auc_lr = roc_auc_score(y_test, logreg.predict_proba(X_test_s)[:, 1])
    print(f"Test AUC: {auc_lr:.4f}", file=sys.stderr)
    coefs = pd.Series(logreg.coef_[0], index=feats).sort_values()
    print("Feature coefficients (negative = predicts loss, positive = predicts win):", file=sys.stderr)
    print(coefs.to_string(), file=sys.stderr)

    print("\n=== HistGradientBoostingClassifier ===", file=sys.stderr)
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                          class_weight="balanced", random_state=42)
    hgb.fit(X_train, y_train)
    proba_test = hgb.predict_proba(X_test)[:, 1]
    auc_hgb = roc_auc_score(y_test, proba_test)
    print(f"Test AUC: {auc_hgb:.4f}", file=sys.stderr)

    test = test.copy()
    test["hgb_proba"] = proba_test

    print("\n=== OUT-OF-SAMPLE COMPARISON (test period only, mean-R basis) ===")
    print(f"Test period: {test['time'].min()} to {test['time'].max()}\n")
    print("NOTE: these are non-compounding R-multiple summaries, not equity")
    print("simulations - the labeled touches overlap in time (no is_flat")
    print("constraint), so a real equity curve isn't valid here. Compare")
    print("mean_r / win_rate / profit_factor only.\n")

    baseline_result = summarize_r(test)
    print("All touches, no filter (strawman):", baseline_result)

    rule_result = summarize_r(test[test["rule_qualifies"]])
    print("\nCURRENT LIVE RULE (score>=3 + HTF-aligned) - the real baseline to beat:")
    print(rule_result)

    print("\nML filter at various selectivity levels (whole test set):")
    for pct in [50, 60, 70, 80, 90, 95]:
        thresh = np.percentile(test["hgb_proba"], pct)
        sub = test[test["hgb_proba"] >= thresh]
        result = summarize_r(sub)
        print(f"  HGB top {100-pct}% by predicted probability (thresh={thresh:.3f}): {result}")

    # also: does the ML filter add value ON TOP of the rule (i.e. as a stricter
    # secondary filter), vs. replacing it outright?
    rule_subset = test[test["rule_qualifies"]]
    print("\nML filter applied ONLY within the rule-qualifying subset (secondary filter):")
    for pct in [30, 50, 70]:
        if len(rule_subset) < 10:
            break
        thresh = np.percentile(rule_subset["hgb_proba"], pct)
        sub = rule_subset[rule_subset["hgb_proba"] >= thresh]
        result = summarize_r(sub)
        print(f"  top {100-pct}% of rule-qualifying by ML probability (thresh={thresh:.3f}): {result}")

    # decile calibration check: does higher predicted probability actually
    # correlate with better realized outcomes? this is the cleanest signal
    # of whether the model found anything real, independent of any
    # simulation methodology.
    print("\nCalibration check - mean realized_r by predicted-probability decile (whole test set):")
    test["decile"] = pd.qcut(test["hgb_proba"], 10, labels=False, duplicates="drop")
    decile_stats = test.groupby("decile").agg(
        n=("realized_r", "size"), mean_r=("realized_r", "mean"), win_rate=("win", "mean")
    )
    print(decile_stats.to_string())

    import joblib
    joblib.dump(dict(model=hgb, features=feats, medians=medians.to_dict()), "ml_entry_filter.joblib")
    print("\nSaved model to ml_entry_filter.joblib", file=sys.stderr)


if __name__ == "__main__":
    main()
