"""
Rebuild the labeled touch dataset from the FULL available history
(2017-08-17 onward, reusing the backfill fetched by backtest_5yr.py's
10-year run) instead of just the live bot's own 2022-onward dataset.
Does NOT touch data/btcusdt_15m.parquet or data/state.json (same reason as
backtest_5yr.py - the live bot's integer bar-index state would break).

Output: data/ml_training_data_full.csv
"""
import sys

import pandas as pd

import backtest_pdh_pdl as bt
from build_ml_dataset import HTF_RULE, ZONE_PCT, build_dataset

BACKFILL_FILE = "data/btcusdt_15m_backfill_10.0yr.parquet"
LIVE_FILE = "data/btcusdt_15m.parquet"
OUT_FILE = "data/ml_training_data_full.csv"


def main():
    backfill = pd.read_parquet(BACKFILL_FILE)
    live = pd.read_parquet(LIVE_FILE)
    df_raw = (pd.concat([backfill, live], ignore_index=True)
              .drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True))
    print(f"Full history: {len(df_raw)} bars, {df_raw['open_time'].min()} to {df_raw['open_time'].max()}", file=sys.stderr)

    df = bt.compute_indicators(df_raw, zone_pct=ZONE_PCT, htf_rule=HTF_RULE)
    out = build_dataset(df)
    out.to_csv(OUT_FILE, index=False)
    print(f"\nDone: {len(out)} labeled touches -> {OUT_FILE}", file=sys.stderr)
    print(out["win"].value_counts(normalize=True), file=sys.stderr)
    print(f"mean realized_r: {out['realized_r'].mean():.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
