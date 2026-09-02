"""
Sequential, position-exclusive validation of the ML secondary filter -
drives the REAL backtest_pdh_pdl.step_bar (via its optional `ml_gate` hook)
so every mechanic other than the ML gate itself (fired-flag state machine,
breakout protection, HTF filter, Swing SL, 2R/4R partials, commission,
slippage, position exclusivity) is byte-for-byte identical to the live
engine - no separate reconstruction of step_bar's logic, which is what
caused the previous version of this script to silently diverge from the
real system (134 trades/24.6% win rate reconstructed vs. the real engine's
159 trades/45.3% win rate on the same period).

Runs the full 2022-present history once per variant (needed since pivot/ATR
state must stay continuous), then reports metrics restricted to trades
whose entry falls in the test period (>= TRAIN_END, genuinely out-of-sample
for the model, which was trained only on data before that date).
"""
import sys

import joblib
import pandas as pd

import backtest_pdh_pdl as bt
from build_ml_dataset import HTF_RULE, ZONE_PCT
from ml_filter import make_ml_gate
from paper_trade import QTY_PCT_OF_EQUITY, STEP_KWARGS
from train_ml_model import TRAIN_END

INITIAL_CAPITAL = 100.0


def evaluate(trades, initial_capital=INITIAL_CAPITAL, test_start=TRAIN_END):
    tdf = pd.DataFrame(trades)
    if tdf.empty:
        return dict(trades=0)
    tdf["entry_time"] = pd.to_datetime(tdf["entry_time"])
    tdf = tdf.sort_values("exit_time").reset_index(drop=True)
    tdf["equity"] = initial_capital + tdf["pnl"].cumsum()

    test_mask = tdf["entry_time"] >= pd.Timestamp(test_start)
    if not test_mask.any():
        return dict(trades=0)
    first_idx = test_mask.idxmax()
    equity_start = tdf["equity"].iloc[first_idx - 1] if first_idx > 0 else initial_capital
    sub = tdf[test_mask]

    wins = sub[sub["pnl"] > 0]["pnl"]
    losses = sub[sub["pnl"] <= 0]["pnl"]
    pf = wins.sum() / -losses.sum() if losses.sum() != 0 else float("inf")
    eq_path = pd.concat([pd.Series([equity_start]), sub["equity"]]).reset_index(drop=True)
    max_dd = ((eq_path - eq_path.cummax()) / eq_path.cummax()).min() * 100
    days = (sub["exit_time"].max() - sub["entry_time"].min()).days
    years = max(days / 365.25, 0.01)
    total_return_pct = (eq_path.iloc[-1] / equity_start - 1) * 100
    cagr = ((eq_path.iloc[-1] / equity_start) ** (1 / years) - 1) * 100 if eq_path.iloc[-1] > 0 else float("-inf")
    return dict(
        trades=len(sub), win_rate=(sub["pnl"] > 0).mean() * 100, profit_factor=pf,
        total_return_pct=total_return_pct, cagr_pct=cagr, max_dd_pct=max_dd, years=round(years, 2),
    )


def main():
    df = pd.read_parquet("data/btcusdt_15m.parquet").sort_values("open_time").reset_index(drop=True)
    df = bt.compute_indicators(df, zone_pct=ZONE_PCT, htf_rule=HTF_RULE)
    d = joblib.load("ml_entry_filter.joblib")
    model, feats, medians = d["model"], d["features"], d["medians"]
    bt.QTY_PCT_OF_EQUITY = QTY_PCT_OF_EQUITY

    print(f"Sizing: {QTY_PCT_OF_EQUITY*100:.0f}% of equity/trade (matches live deployment)", file=sys.stderr)
    print(f"Test period: entries >= {TRAIN_END} (out-of-sample for the model)\n", file=sys.stderr)

    trades_a, _ = bt.run_backtest(df, **STEP_KWARGS, ml_gate=None)
    print("A) LIVE RULE, no ML gate (real step_bar, exact current deployment):")
    print(" ", evaluate(trades_a))

    for thresh in [0.44, 0.49, 0.52, 0.55, 0.60]:
        gate = make_ml_gate(model, feats, medians, thresh)
        trades_b, _ = bt.run_backtest(df, **STEP_KWARGS, ml_gate=gate)
        print(f"\nB) LIVE RULE + ML gate (proba >= {thresh}):")
        print(" ", evaluate(trades_b))


if __name__ == "__main__":
    main()
