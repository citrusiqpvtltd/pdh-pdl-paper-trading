"""
Live paper-trading runner for the PDH/PDL confluence strategy.

Meant to be invoked on a schedule (see .github/workflows/paper_trade.yml,
every 15 minutes) with no human present. Each run:

  1. Appends any newly-closed 15m BTCUSDT candles from Binance's free public
     API to data/btcusdt_15m.parquet (a growing, git-committed dataset that
     starts 2022-01-01 - the same one the strategy was tuned and validated
     against over 4.67 years).
  2. Recomputes indicators over the FULL dataset (cheap, ~0.3s even at this
     size) so pivots/ATR/RSI/HTF-trend are exactly as accurate as a from-
     scratch backtest would give - no drift from only ever seeing a short
     rolling window.
  3. Replays the strategy (backtest_pdh_pdl.step_bar - the same function
     used for backtesting, so live and backtested behavior cannot diverge)
     over only the bars closed since the last run, continuing from
     persisted state (data/state.json): equity, any open paper position,
     pivot history, sellFired/buyFired.
  4. Appends any trades that closed to data/trades.csv, and rewrites
     STATUS.md with a human-readable summary.

On the very first run (no state.json yet), this replays the entire 2022-
present history once to arrive at a "live" state - so trades.csv starts as
exactly the validated backtest, and everything after `live_since` in
state.json is genuine forward paper trading with no lookahead.

This is a SIMULATOR ONLY. No real orders, no exchange account, no API keys.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

import backtest_pdh_pdl as bt

DATA_DIR = "data"
PARQUET_FILE = os.path.join(DATA_DIR, "btcusdt_15m.parquet")
STATE_FILE = os.path.join(DATA_DIR, "state.json")
TRADES_FILE = os.path.join(DATA_DIR, "trades.csv")
STATUS_FILE = "STATUS.md"

SYMBOL = "BTCUSDT"
INTERVAL = "15m"
# data-api.binance.vision is Binance's dedicated, geo-unrestricted public
# market-data mirror - api.binance.com returns HTTP 451 from US-hosted
# infrastructure (including GitHub Actions' ubuntu-latest runners).
BASE_URL = "https://data-api.binance.vision/api/v3/klines"
HISTORY_START = datetime(2022, 1, 1, tzinfo=timezone.utc)

# Tuned, live-validated config (see README.md for how these were chosen)
STRATEGY_PARAMS = dict(
    zone_pct=0.3, htf_rule="4h",
    score_threshold=5, rr1=1.5, rr2=3.0, sl_method="Swing", tp2_enabled=True,
    use_htf_filter=True, enable_breakout_protection=True,
)

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


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
        time.sleep(0.2)
    return rows


def update_dataset() -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(PARQUET_FILE):
        df = pd.read_parquet(PARQUET_FILE)
        start_ms = int(df["open_time"].max().timestamp() * 1000) + 1
    else:
        df = pd.DataFrame(columns=["open_time", "close_time", "open", "high", "low", "close", "volume"])
        start_ms = int(HISTORY_START.timestamp() * 1000)

    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    rows = fetch_klines(start_ms, end_ms)
    if rows:
        new = pd.DataFrame(rows, columns=COLUMNS)
        new["open_time"] = pd.to_datetime(new["open_time"], unit="ms", utc=True)
        new["close_time"] = pd.to_datetime(new["close_time"], unit="ms", utc=True)
        for c in ["open", "high", "low", "close", "volume"]:
            new[c] = new[c].astype(float)
        new = new[["open_time", "close_time", "open", "high", "low", "close", "volume"]]
        df = pd.concat([df, new], ignore_index=True)
        df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)
        df.to_parquet(PARQUET_FILE, index=False)
        print(f"Fetched {len(new)} new bar(s); dataset now {len(df)} bars, up to {df['open_time'].max()}", file=sys.stderr)
    else:
        print("No new closed bars available yet.", file=sys.stderr)
    return df


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            raw = json.load(f)
        position = raw["position"]
        if position is not None:
            position["entry_time"] = pd.Timestamp(position["entry_time"])
        return dict(
            pivot_high_vals=raw["pivot_high_vals"], pivot_high_bars=raw["pivot_high_bars"],
            pivot_low_vals=raw["pivot_low_vals"], pivot_low_bars=raw["pivot_low_bars"],
            sell_fired=raw["sell_fired"], buy_fired=raw["buy_fired"], equity=raw["equity"],
            position=position,
        ), raw["last_processed_index"], raw["live_since"]
    return bt.new_state(), -1, None


def save_state(state, last_processed_index, live_since):
    position = dict(state["position"]) if state["position"] is not None else None
    if position is not None:
        position["entry_time"] = pd.Timestamp(position["entry_time"]).isoformat()
    raw = dict(
        pivot_high_vals=state["pivot_high_vals"], pivot_high_bars=state["pivot_high_bars"],
        pivot_low_vals=state["pivot_low_vals"], pivot_low_bars=state["pivot_low_bars"],
        sell_fired=state["sell_fired"], buy_fired=state["buy_fired"], equity=state["equity"],
        position=position, last_processed_index=last_processed_index, live_since=live_since,
    )
    with open(STATE_FILE, "w") as f:
        json.dump(raw, f, indent=2, default=str)


def append_trades(new_trades):
    if not new_trades:
        return
    tdf = pd.DataFrame(new_trades)
    header = not os.path.exists(TRADES_FILE)
    tdf.to_csv(TRADES_FILE, mode="a", header=header, index=False)


def write_status(state, last_bar_time, signals_this_run):
    all_trades = pd.read_csv(TRADES_FILE, parse_dates=["entry_time", "exit_time"]) if os.path.exists(TRADES_FILE) else pd.DataFrame()
    lines = []
    lines.append("# PDH/PDL Confluence Reversal — Live Paper Trading\n")
    lines.append(f"_Simulator only. No real money, no exchange account, no API keys. Last updated: {datetime.now(timezone.utc).isoformat()}_\n")
    lines.append("## Current State\n")
    lines.append(f"- Equity: **{state['equity']:.2f} USDT** (started at {bt.INITIAL_CAPITAL:.2f})")
    lines.append(f"- Last processed bar: {last_bar_time}")
    pos = state["position"]
    if pos is None:
        lines.append("- Position: **flat**")
    else:
        lines.append(f"- Position: **{pos['side'].upper()}** {pos['qty_remaining']:.6f} BTC @ {pos['entry_px']:.2f} "
                      f"(SL {pos['sl']:.2f}, TP1 {pos['tp1']:.2f}, TP2 {pos['tp2']:.2f}, TP1 filled: {pos['tp1_filled']})")
    if signals_this_run:
        lines.append(f"- Signal(s) this run: {signals_this_run}")
    lines.append("")

    if len(all_trades):
        wins = all_trades[all_trades["pnl"] > 0]
        pf = wins["pnl"].sum() / -all_trades[all_trades["pnl"] <= 0]["pnl"].sum() if (all_trades["pnl"] <= 0).any() else float("inf")
        lines.append("## All-Time Stats (backtest + live combined)\n")
        lines.append(f"- Total trades: {len(all_trades)}")
        lines.append(f"- Win rate: {len(wins)/len(all_trades)*100:.2f}%")
        lines.append(f"- Total PnL: {all_trades['pnl'].sum():.2f} USDT ({all_trades['pnl'].sum()/bt.INITIAL_CAPITAL*100:.2f}%)")
        lines.append(f"- Profit factor: {pf:.3f}\n")

        lines.append("## Most Recent Trades\n")
        lines.append("| Entry time | Side | Exit reason | Entry | Exit | PnL |")
        lines.append("|---|---|---|---|---|---|")
        for _, t in all_trades.tail(15).iloc[::-1].iterrows():
            lines.append(f"| {t['entry_time']} | {t['side']} | {t['reason']} | {t['entry_px']:.2f} | {t['exit_px']:.2f} | {t['pnl']:.2f} |")
    else:
        lines.append("No trades yet.\n")

    with open(STATUS_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    df_raw = update_dataset()
    df = bt.compute_indicators(df_raw.copy(), zone_pct=STRATEGY_PARAMS["zone_pct"], htf_rule=STRATEGY_PARAMS["htf_rule"])
    arr = bt.prepare_arrays(df)

    state, last_processed_index, live_since = load_state()
    is_first_run = live_since is None
    if is_first_run:
        live_since = str(df["open_time"].iloc[-1]) if len(df) else None
        print("First run: replaying full history to build up live state...", file=sys.stderr)

    now = datetime.now(timezone.utc)
    step_kwargs = {k: v for k, v in STRATEGY_PARAMS.items() if k not in ("zone_pct", "htf_rule")}

    new_trades = []
    signals = []
    processed_upto = last_processed_index
    for i in range(last_processed_index + 1, arr["n"]):
        close_time = df["close_time"].iloc[i]
        if close_time.to_pydatetime() > now:
            break  # still-forming bar, not closed yet
        state, trades, signal_info = bt.step_bar(i, arr, state, **step_kwargs)
        new_trades.extend(trades)
        if signal_info["buy_signal"]:
            signals.append(f"BUY score {signal_info['buy_score']}/6 @ {arr['close'][i]:.2f} ({arr['open_time'][i]})")
        if signal_info["sell_signal"]:
            signals.append(f"SELL score {signal_info['sell_score']}/6 @ {arr['close'][i]:.2f} ({arr['open_time'][i]})")
        processed_upto = i

    # Save state FIRST: if the process dies partway, we must never re-replay
    # bars whose signals already fired - a duplicate trade-log write is a
    # cosmetic, recoverable loss; a duplicate strategy replay is not.
    save_state(state, processed_upto, live_since)
    append_trades(new_trades)
    last_bar_time = df["open_time"].iloc[processed_upto] if processed_upto >= 0 else "none yet"
    write_status(state, last_bar_time, signals)

    print(f"Processed through bar index {processed_upto} ({last_bar_time}); "
          f"{len(new_trades)} trade(s) closed this run; equity={state['equity']:.2f}; "
          f"position={'flat' if state['position'] is None else state['position']['side']}", file=sys.stderr)
    for s in signals:
        print(f"SIGNAL: {s}", file=sys.stderr)


if __name__ == "__main__":
    main()
