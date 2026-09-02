"""
Scheduled (monthly) retraining for the ML entry filter - NOT online/
incremental learning. Each run does the same offline batch process
train_ml_model.py does by hand, but:
  1. rebuilds the labeled dataset from ALL data currently on disk (so it
     includes bars that closed since the last retrain)
  2. re-splits train/holdout using a ROLLING holdout window (the most
     recent RETRAIN_HOLDOUT_DAYS, not a fixed calendar date), so the
     holdout always means "the most recent stretch," however often this
     runs
  3. trains a candidate model on everything before the holdout
  4. validates the candidate SEQUENTIALLY and POSITION-EXCLUSIVELY on the
     holdout, through the real backtest_pdh_pdl.step_bar engine (via the
     same ml_filter.make_ml_gate used live) - not a naive R-multiple
     average, for the same reason validate_ml_filter.py doesn't use one
  5. deploys the candidate ONLY if it clears a minimum sample size and a
     minimum profit-factor floor on that holdout; otherwise keeps the
     currently-deployed model and logs why

The gate threshold (paper_trade.ML_GATE_THRESHOLD) is deliberately NOT
re-searched here - only the model is retrained, at the fixed, already-
validated threshold. Re-searching the threshold on a small rolling holdout
every month would be a second, compounding source of overfitting on top of
whatever the model itself picks up.

Writes RETRAIN_LOG.md with the outcome of every run (deployed or skipped
and why) so retraining stays auditable, not a silent background process.
"""
import sys
from datetime import datetime, timedelta, timezone

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import backtest_pdh_pdl as bt
import ml_filter
from build_ml_dataset import HTF_RULE, ZONE_PCT, build_dataset
from paper_trade import ML_FILTER_FILE, ML_GATE_THRESHOLD, QTY_PCT_OF_EQUITY, STEP_KWARGS
from train_ml_model import FEATURE_COLS
from validate_ml_filter import evaluate

RETRAIN_HOLDOUT_DAYS = 270  # ~9 months, rolling - not a fixed calendar date. A
# 6-month window was tried first and only produced ~27 holdout trades at the
# deployed threshold - right at the noise floor. 9 months tracks the
# originally-validated 1.25-year holdout's ~75-trade sample closely enough
# (scaled: ~44) to actually clear MIN_HOLDOUT_TRADES on a normal run, while
# still being recent enough to catch real drift.
MIN_HOLDOUT_TRADES = 30     # below this, a PF reading is too noisy to act on
DEPLOY_PF_FLOOR = 0.90      # candidate must clear this on its OWN holdout to deploy
CANDIDATE_FILE = "ml_entry_filter_candidate.joblib"
LOG_FILE = "RETRAIN_LOG.md"


def train_candidate(dataset: pd.DataFrame, holdout_start: pd.Timestamp):
    dataset = dataset.copy()
    dataset["side_sell"] = (dataset["side"] == "sell").astype(int)
    feats = FEATURE_COLS + ["side_sell"]
    train = dataset[dataset["time"] < holdout_start]
    X_train, y_train = train[feats], train["win"]
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                            class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    return model, feats, medians, len(train)


def main():
    df = pd.read_parquet("data/btcusdt_15m.parquet").sort_values("open_time").reset_index(drop=True)
    df = bt.compute_indicators(df, zone_pct=ZONE_PCT, htf_rule=HTF_RULE)

    print("Rebuilding labeled dataset from current data...", file=sys.stderr)
    dataset = build_dataset(df)
    dataset.to_csv("data/ml_training_data.csv", index=False)
    dataset["time"] = pd.to_datetime(dataset["time"])

    last_bar_time = dataset["time"].max()
    holdout_start = last_bar_time - timedelta(days=RETRAIN_HOLDOUT_DAYS)
    print(f"Rolling holdout: {holdout_start} to {last_bar_time}", file=sys.stderr)

    model, feats, medians, n_train = train_candidate(dataset, holdout_start)
    joblib.dump(dict(model=model, features=feats, medians=medians.to_dict()), CANDIDATE_FILE)
    print(f"Candidate trained on {n_train} rows before {holdout_start}", file=sys.stderr)

    bt.QTY_PCT_OF_EQUITY = QTY_PCT_OF_EQUITY
    gate = ml_filter.make_ml_gate(model, feats, medians, ML_GATE_THRESHOLD)
    trades, _ = bt.run_backtest(df, **STEP_KWARGS, ml_gate=gate)
    result = evaluate(trades, test_start=holdout_start)
    print(f"Candidate holdout result: {result}", file=sys.stderr)

    n_trades = result.get("trades", 0)
    pf = float(result.get("profit_factor", 0))
    deployed = n_trades >= MIN_HOLDOUT_TRADES and pf >= DEPLOY_PF_FLOOR
    if deployed:
        joblib.dump(dict(model=model, features=feats, medians=medians.to_dict()), ML_FILTER_FILE)
        reason = f"cleared floor (PF {pf:.3f} >= {DEPLOY_PF_FLOOR}, {n_trades} >= {MIN_HOLDOUT_TRADES} trades)"
    elif n_trades < MIN_HOLDOUT_TRADES:
        reason = f"too few holdout trades to judge ({n_trades} < {MIN_HOLDOUT_TRADES}) - kept existing model"
    else:
        reason = f"PF {pf:.3f} below floor {DEPLOY_PF_FLOOR} on its own holdout - kept existing model"
    print(("DEPLOYED: " if deployed else "SKIPPED: ") + reason, file=sys.stderr)

    import os
    if os.path.exists(CANDIDATE_FILE):
        os.remove(CANDIDATE_FILE)

    if n_trades:
        result_str = (f"{n_trades} trades, win rate {float(result['win_rate']):.1f}%, PF {pf:.3f}, "
                      f"CAGR {float(result['cagr_pct']):.2f}%, max DD {float(result['max_dd_pct']):.2f}%")
    else:
        result_str = "0 trades in holdout window"
    log_entry = (
        f"## {datetime.now(timezone.utc).isoformat()}\n\n"
        f"- Holdout window: {holdout_start.date()} to {last_bar_time.date()} ({RETRAIN_HOLDOUT_DAYS} days, rolling)\n"
        f"- Trained on {n_train} rows before the holdout\n"
        f"- Holdout result at threshold {ML_GATE_THRESHOLD}: {result_str}\n"
        f"- **{'DEPLOYED' if deployed else 'SKIPPED'}**: {reason}\n\n"
    )
    header = "# ML Entry Filter - Retrain Log\n\nEach entry is one scheduled retrain run (see `.github/workflows/retrain_ml_filter.yml` and `retrain_pipeline.py`). A skipped run keeps the previously deployed model - retraining never deploys a model that fails its own out-of-sample floor.\n\n"
    existing = open(LOG_FILE).read() if os.path.exists(LOG_FILE) else header
    with open(LOG_FILE, "w") as f:
        f.write(existing + log_entry)


if __name__ == "__main__":
    main()
