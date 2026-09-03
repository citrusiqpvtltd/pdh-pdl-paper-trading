"""
Standalone 5-year backtest report - does NOT touch the live bot's dataset
or state (data/btcusdt_15m.parquet, data/state.json), since paper_trade.py
tracks position with an absolute integer bar index (last_processed_index);
prepending older history to that file would silently corrupt it. Instead
this fetches the missing older stretch into its own file and works from a
combined, in-memory copy only.

Runs BOTH the pure rule engine and the currently-deployed rule+ML-gate
engine over the full ~5-year span, with a year-by-year breakdown, so the
existing "4.63 years, 489 trades" numbers can be seen extended and the ML
gate's effect checked over a longer, partly-unseen-by-the-model stretch
(the model was only ever trained on 2022-2025-06 data, so 2021-09..2021-12
here is additional, out-of-sample-by-construction data it has never
touched in any form).
"""
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

import backtest_pdh_pdl as bt
import ml_filter
from build_ml_dataset import HTF_RULE, ZONE_PCT
from paper_trade import BASE_URL, COLUMNS, ML_FILTER_FILE, ML_GATE_THRESHOLD, QTY_PCT_OF_EQUITY, STEP_KWARGS, SYMBOL, INTERVAL

YEARS_BACK = float(sys.argv[1]) if len(sys.argv) > 1 else 5
EARLIEST_AVAILABLE = pd.Timestamp("2017-08-17", tz="UTC")  # Binance BTCUSDT history starts here - no data before this
BACKFILL_FILE = f"data/btcusdt_15m_backfill_{YEARS_BACK}yr.parquet"
LIVE_FILE = "data/btcusdt_15m.parquet"


def fetch_klines(start_ms: int, end_ms: int) -> list:
    rows = []
    cur = start_ms
    limit = 1000
    while cur < end_ms:
        params = {"symbol": SYMBOL, "interval": INTERVAL, "startTime": cur, "endTime": end_ms, "limit": limit}
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        rows.extend(data)
        cur = data[-1][0] + 1
        if len(data) < limit:
            break
        time.sleep(0.15)
    return rows


def load_5yr_data() -> pd.DataFrame:
    live = pd.read_parquet(LIVE_FILE)
    cutoff = datetime.now(timezone.utc) - timedelta(days=365 * YEARS_BACK)
    if cutoff < EARLIEST_AVAILABLE.to_pydatetime():
        print(f"Requested {YEARS_BACK} years back would be {cutoff}, but Binance BTCUSDT "
              f"history only goes back to {EARLIEST_AVAILABLE} - clamping to that.", file=sys.stderr)
        cutoff = EARLIEST_AVAILABLE.to_pydatetime()
    earliest_needed_ms = int(cutoff.timestamp() * 1000)
    earliest_have_ms = int(live["open_time"].min().timestamp() * 1000)

    if earliest_needed_ms < earliest_have_ms:
        print(f"Backfilling {live['open_time'].min()} back to {cutoff}...", file=sys.stderr)
        rows = fetch_klines(earliest_needed_ms, earliest_have_ms - 1)
        backfill = pd.DataFrame(rows, columns=COLUMNS)
        backfill["open_time"] = pd.to_datetime(backfill["open_time"], unit="ms", utc=True)
        backfill["close_time"] = pd.to_datetime(backfill["close_time"], unit="ms", utc=True)
        for c in ["open", "high", "low", "close", "volume"]:
            backfill[c] = backfill[c].astype(float)
        backfill = backfill[["open_time", "close_time", "open", "high", "low", "close", "volume"]]
        backfill.to_parquet(BACKFILL_FILE, index=False)
        print(f"Backfilled {len(backfill)} bars -> {BACKFILL_FILE} (separate file, live dataset untouched)", file=sys.stderr)
        df = pd.concat([backfill, live], ignore_index=True)
    else:
        df = live.copy()

    df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)
    return df[df["open_time"] >= pd.Timestamp(cutoff)].reset_index(drop=True)


def yearly_breakdown(trades: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    trades = trades.sort_values("exit_time").copy()
    trades["year"] = pd.to_datetime(trades["exit_time"]).dt.year
    rows = []
    equity = initial_capital
    for year, grp in trades.groupby("year"):
        start_eq = equity
        equity += grp["pnl"].sum()
        wins = grp[grp["pnl"] > 0]
        pf = wins["pnl"].sum() / -grp[grp["pnl"] <= 0]["pnl"].sum() if grp[grp["pnl"] <= 0]["pnl"].sum() != 0 else float("inf")
        rows.append(dict(year=year, trades=len(grp), win_rate=(grp["pnl"] > 0).mean() * 100,
                          pnl_pct=(equity - start_eq) / start_eq * 100, profit_factor=pf, end_equity=equity))
    return pd.DataFrame(rows)


def summarize(trades: list, initial_capital: float, label: str):
    tdf = pd.DataFrame(trades)
    if tdf.empty:
        print(f"\n=== {label}: no trades ===")
        return
    tdf = tdf.sort_values("exit_time").reset_index(drop=True)
    wins = tdf[tdf["pnl"] > 0]
    losses = tdf[tdf["pnl"] <= 0]
    pf = wins["pnl"].sum() / -losses["pnl"].sum() if losses["pnl"].sum() != 0 else float("inf")
    final_equity = initial_capital + tdf["pnl"].sum()
    eq_curve = initial_capital + tdf["pnl"].cumsum()
    max_dd = ((eq_curve - eq_curve.cummax()) / eq_curve.cummax()).min() * 100
    span_days = (pd.to_datetime(tdf["exit_time"].iloc[-1]) - pd.to_datetime(tdf["entry_time"].iloc[0])).days
    years = span_days / 365.25
    cagr = ((final_equity / initial_capital) ** (1 / years) - 1) * 100

    print(f"\n=== {label} ===")
    print(f"Span: {tdf['entry_time'].iloc[0]} to {tdf['exit_time'].iloc[-1]} ({years:.2f} years)")
    print(f"Trades: {len(tdf)}  Win rate: {(tdf['pnl']>0).mean()*100:.2f}%")
    print(f"Total PnL: {tdf['pnl'].sum():.2f} USDT ({tdf['pnl'].sum()/initial_capital*100:.2f}%)")
    print(f"Profit factor: {pf:.3f}  CAGR: {cagr:.2f}%/yr  Max drawdown: {max_dd:.2f}%")
    print(f"Final equity (from {initial_capital:.2f}): {final_equity:.2f}")
    print("\nYear-by-year:")
    print(yearly_breakdown(tdf, initial_capital).to_string(index=False))


def main():
    df_raw = load_5yr_data()
    print(f"\nBacktesting {len(df_raw)} bars, {df_raw['open_time'].min()} to {df_raw['open_time'].max()} "
          f"({(df_raw['open_time'].max()-df_raw['open_time'].min()).days/365.25:.2f} years)", file=sys.stderr)

    df = bt.compute_indicators(df_raw, zone_pct=ZONE_PCT, htf_rule=HTF_RULE)
    bt.QTY_PCT_OF_EQUITY = QTY_PCT_OF_EQUITY

    trades_rule, _ = bt.run_backtest(df, **STEP_KWARGS, ml_gate=None)
    summarize(trades_rule, bt.INITIAL_CAPITAL, "A) Rule-only (no ML gate)")

    gate, meta = ml_filter.load_gate(ML_FILTER_FILE, ML_GATE_THRESHOLD)
    trades_ml, _ = bt.run_backtest(df, **STEP_KWARGS, ml_gate=gate)
    summarize(trades_ml, bt.INITIAL_CAPITAL, f"B) Rule + ML gate (threshold {ML_GATE_THRESHOLD}, currently deployed)")


if __name__ == "__main__":
    main()
