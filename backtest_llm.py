"""
Backtest the LLM-decided engine (paper_trade.step_bar_llm) over a bounded
window of real historical data, using a REAL local Ollama model for every
decision (not a stub) - this is the actual decision logic that runs live,
replayed against the past.

Two modes:
  --days N          replay the most recent N days (quick probes)
  --start / --end   replay an explicit [start, end) window (used by the
                     sharded 1+ year pipeline - see plan_shards.py and
                     .github/workflows/backtest_llm_sharded.yml)

Near-PDH/PDL touches occur ~25-30/day at current settings, so a full
multi-year replay means tens of thousands of real Ollama calls - the
sharded pipeline exists because a single job cannot run long enough
(GitHub Actions hard-caps every job at 6 hours) to do this serially.

Market status uses the REAL point-in-time Fear & Greed value for whichever
historical date is being replayed (see market_status.fetch_fear_greed_history)
- not "today's" value, which would be lookahead bias at this scale. BTC
dominance/total market cap are omitted here (CoinGecko's historical global
data requires a paid plan); the live bot still uses current-value versions
of all three.

Usage:
    python backtest_llm.py --days 14
    python backtest_llm.py --start 2025-01-01 --end 2025-01-15 --out-suffix shard03
"""
import argparse
import sys
import time

import pandas as pd

import backtest_pdh_pdl as bt
import llm_decide
import market_status
import paper_trade as pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=None, help="Replay the most recent N days (quick probes)")
    ap.add_argument("--start", type=str, default=None, help="ISO date/datetime - start of an explicit window (inclusive)")
    ap.add_argument("--end", type=str, default=None, help="ISO date/datetime - end of an explicit window (exclusive)")
    ap.add_argument("--out-suffix", type=str, default="", help="Suffix for output CSV filenames (for sharded runs)")
    args = ap.parse_args()

    print(f"Loading data and computing indicators...", file=sys.stderr)
    df = pd.read_parquet("data/btcusdt_15m.parquet")
    df = bt.compute_indicators(df, zone_pct=pt.ZONE_PCT, htf_rule=pt.HTF_RULE)
    arr = bt.prepare_arrays(df)

    if args.start is not None:
        start_ts = pd.Timestamp(args.start, tz="UTC")
        start_i = int((df["open_time"] >= start_ts).idxmax())
        if args.end is not None:
            end_ts = pd.Timestamp(args.end, tz="UTC")
            end_i = int((df["open_time"] < end_ts).values[::-1].argmax())
            end_i = arr["n"] - 1 - end_i
        else:
            end_i = arr["n"] - 1
    else:
        days = args.days if args.days is not None else 14
        cutoff = df["open_time"].max() - pd.Timedelta(days=days)
        start_i = int((df["open_time"] >= cutoff).idxmax())
        end_i = arr["n"] - 1

    print(f"Replaying from {df['open_time'].iloc[start_i]} to {df['open_time'].iloc[end_i]} "
          f"({end_i - start_i + 1} bars)", file=sys.stderr)

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

    fg_history = market_status.fetch_fear_greed_history()
    open_time = arr["open_time"]

    def market_status_for_bar(i):
        date_str = pd.Timestamp(open_time[i]).strftime("%Y-%m-%d")
        return market_status.format_for_context_historical(date_str, fg_history)

    all_trades, all_decisions = [], []
    t_start = time.time()
    n_calls = 0
    for i in range(start_i, end_i + 1):
        state, trades, decisions = pt.step_bar_llm(i, arr, state, market_status_for_bar)
        all_trades.extend(trades)
        all_decisions.extend(decisions)
        if decisions:
            n_calls += len(decisions)
            elapsed = time.time() - t_start
            rate = elapsed / n_calls
            print(f"  [{n_calls} calls, {elapsed:.0f}s elapsed, {rate:.1f}s/call] "
                  f"{decisions[-1]['time']} {decisions[-1]['side']} -> {decisions[-1]['action']}: "
                  f"{decisions[-1]['reasoning'][:80]}", file=sys.stderr)

    elapsed = time.time() - t_start
    print(f"\nDone: {n_calls} LLM calls in {elapsed:.0f}s ({elapsed/max(n_calls,1):.1f}s/call avg)", file=sys.stderr)

    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    # Explicit columns even when empty (a shard can legitimately have zero
    # trades/decisions) - pd.DataFrame([]).to_csv() writes a headerless blank
    # line that later chokes pd.read_csv with EmptyDataError.
    TRADE_COLS = ["side", "entry_time", "exit_time", "reason", "qty", "entry_px", "exit_px", "pnl"]
    DECISION_COLS = ["time", "side", "level_price", "action", "reasoning"]
    tdf = pd.DataFrame(all_trades, columns=TRADE_COLS)
    ddf = pd.DataFrame(all_decisions, columns=DECISION_COLS)
    tdf.to_csv(f"backtest_llm_trades{suffix}.csv", index=False)
    ddf.to_csv(f"backtest_llm_decisions{suffix}.csv", index=False)

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
