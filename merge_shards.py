"""
Merge per-shard trade/decision CSVs from the sharded LLM backtest into one
aggregate report.

CAVEAT worth understanding before trusting the aggregate PnL%: each shard
runs independently starting from bt.new_state() - i.e. its own fresh $100
baseline - because position sizing (10% of CURRENT equity) needs a
continuous account to compound correctly, and shards run in parallel with
no way to share state. So the aggregate dollar PnL sums N independent
$100-starting segments, which is NOT the same as one continuously
compounding one-year equity curve (a true compounding curve run serially
would end up on a different base each subsequent trade). Win rate, profit
factor, and total dollar PnL are still meaningful for judging whether the
strategy has an edge; "aggregate PnL %" specifically should be read as
"average return per independent ~week-or-two segment", not one number you
can extrapolate as a single year's compounded growth.

Usage:
    python merge_shards.py --artifacts-dir downloaded_artifacts/
"""
import argparse
import glob
import os

import pandas as pd

import backtest_pdh_pdl as bt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-dir", required=True)
    args = ap.parse_args()

    trade_files = sorted(glob.glob(os.path.join(args.artifacts_dir, "**", "backtest_llm_trades*.csv"), recursive=True))
    decision_files = sorted(glob.glob(os.path.join(args.artifacts_dir, "**", "backtest_llm_decisions*.csv"), recursive=True))

    print(f"Found {len(trade_files)} trade files, {len(decision_files)} decision files")

    def safe_read(f):
        try:
            return pd.read_csv(f)
        except pd.errors.EmptyDataError:
            return None

    trades = [df for df in (safe_read(f) for f in trade_files) if df is not None and len(df)]
    decisions = [df for df in (safe_read(f) for f in decision_files) if df is not None and len(df)]

    tdf = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    ddf = pd.concat(decisions, ignore_index=True) if decisions else pd.DataFrame()

    if len(tdf):
        tdf["entry_time"] = pd.to_datetime(tdf["entry_time"])
        tdf = tdf.sort_values("entry_time").reset_index(drop=True)
    if len(ddf):
        ddf["time"] = pd.to_datetime(ddf["time"])
        ddf = ddf.sort_values("time").reset_index(drop=True)

    tdf.to_csv("backtest_llm_merged_trades.csv", index=False)
    ddf.to_csv("backtest_llm_merged_decisions.csv", index=False)

    print("\n=== MERGED RESULTS (see caveat in this script's docstring re: aggregate PnL%) ===")
    if len(ddf):
        print(f"Date range: {ddf['time'].min()} -> {ddf['time'].max()}")
        print(f"Total LLM decisions: {len(ddf)}")
        print(ddf["action"].value_counts())
    if len(tdf):
        wins = tdf[tdf["pnl"] > 0]
        gross_loss = -tdf[tdf["pnl"] <= 0]["pnl"].sum()
        gross_profit = wins["pnl"].sum()
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        print(f"\nTotal trades: {len(tdf)}  Win rate: {len(wins)/len(tdf)*100:.2f}%")
        print(f"Gross profit: {gross_profit:.2f}  Gross loss: {gross_loss:.2f}")
        print(f"Total PnL (sum across independent shard segments): {tdf['pnl'].sum():.2f} USDT")
        print(f"Profit factor: {pf:.3f}")
        print(f"\nBy exit reason:")
        print(tdf.groupby("reason")["pnl"].agg(["count", "sum", "mean"]))
        tdf["year_month"] = tdf["entry_time"].dt.to_period("M")
        print(f"\nBy month:")
        print(tdf.groupby("year_month")["pnl"].agg(["count", "sum"]))
    else:
        print("No trades closed across any shard.")


if __name__ == "__main__":
    main()
