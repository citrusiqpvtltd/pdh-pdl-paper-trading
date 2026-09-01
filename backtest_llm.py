"""
Backtest the LLM-decided engine (paper_trade.step_bar_llm) over a bounded
recent window of real historical data, using a REAL local Ollama model for
every decision (not a stub) - this is the actual decision logic that runs
live, replayed against the past.

Scope is deliberately bounded to a recent window, not the full 4.67-year
history: near-PDH/PDL touches occur ~23/day, so a full replay would mean
tens of thousands of real Ollama calls - likely many hours to days on a
CPU-only local model. This script runs over --days (default 14) days of
the most recent history and reports timing as it goes, so the actual cost
of a longer run can be judged from real numbers instead of guessed.

Usage:
    python backtest_llm.py --days 14
"""
import argparse
import sys
import time

import pandas as pd

import backtest_pdh_pdl as bt
import llm_decide
import paper_trade as pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=14, help="How many most-recent days of history to replay")
    args = ap.parse_args()

    print(f"Loading data and computing indicators...", file=sys.stderr)
    df = pd.read_parquet("data/btcusdt_15m.parquet")
    df = bt.compute_indicators(df, zone_pct=pt.ZONE_PCT, htf_rule=pt.HTF_RULE)
    arr = bt.prepare_arrays(df)

    cutoff = df["open_time"].max() - pd.Timedelta(days=args.days)
    start_i = int((df["open_time"] >= cutoff).idxmax())
    print(f"Replaying from {df['open_time'].iloc[start_i]} to {df['open_time'].iloc[-1]} "
          f"({arr['n'] - start_i} bars, ~{args.days} days)", file=sys.stderr)

    # warm up pivot/state arrays using history BEFORE start_i, without calling the LLM,
    # so pivots/structure are realistic at the start of the replay window (not empty)
    state = bt.new_state()
    for i in range(max(0, start_i - 200), start_i):
        if arr["ph_confirmed"][i]:
            state["pivot_high_vals"].append(arr["ph_peak_val"][i])
            state["pivot_high_bars"].append(i - bt.PIVOT_RIGHT)
            if len(state["pivot_high_vals"]) > bt.MAX_PIVOT_HISTORY:
                state["pivot_high_vals"].pop(0); state["pivot_high_bars"].pop(0)
        if arr["pl_confirmed"][i]:
            state["pivot_low_vals"].append(arr["pl_peak_val"][i])
            state["pivot_low_bars"].append(i - bt.PIVOT_RIGHT)
            if len(state["pivot_low_vals"]) > bt.MAX_PIVOT_HISTORY:
                state["pivot_low_vals"].pop(0); state["pivot_low_bars"].pop(0)

    all_trades, all_decisions = [], []
    t_start = time.time()
    n_calls = 0
    for i in range(start_i, arr["n"]):
        state, trades, decisions = pt.step_bar_llm(i, arr, state)
        all_trades.extend(trades)
        all_decisions.extend(decisions)
        if decisions:
            n_calls += len(decisions)
            elapsed = time.time() - t_start
            rate = elapsed / n_calls
            remaining_bars = arr["n"] - i
            print(f"  [{n_calls} calls, {elapsed:.0f}s elapsed, {rate:.1f}s/call] "
                  f"{decisions[-1]['time']} {decisions[-1]['side']} -> {decisions[-1]['action']}: "
                  f"{decisions[-1]['reasoning'][:80]}", file=sys.stderr)

    elapsed = time.time() - t_start
    print(f"\nDone: {n_calls} LLM calls in {elapsed:.0f}s ({elapsed/max(n_calls,1):.1f}s/call avg)", file=sys.stderr)

    tdf = pd.DataFrame(all_trades)
    ddf = pd.DataFrame(all_decisions)
    tdf.to_csv("backtest_llm_trades.csv", index=False)
    ddf.to_csv("backtest_llm_decisions.csv", index=False)

    print("\n=== RESULTS ===")
    if len(ddf):
        print(f"Total LLM decisions: {len(ddf)}")
        print(ddf["action"].value_counts())
    if len(tdf):
        wins = tdf[tdf["pnl"] > 0]
        gross_loss = -tdf[tdf["pnl"] <= 0]["pnl"].sum()
        pf = wins["pnl"].sum() / gross_loss if gross_loss > 0 else float("inf")
        print(f"\nTrades: {len(tdf)}  Win rate: {len(wins)/len(tdf)*100:.2f}%")
        print(f"Total PnL: {tdf['pnl'].sum():.4f} USDT ({tdf['pnl'].sum()/bt.INITIAL_CAPITAL*100:.2f}%)")
        print(f"Profit factor: {pf:.3f}")
        print(f"Final equity: {state['equity']:.4f}")
    else:
        print("No trades closed in this window.")


if __name__ == "__main__":
    main()
