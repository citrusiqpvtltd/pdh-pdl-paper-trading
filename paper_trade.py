"""
Live paper-trading runner for the PDH/PDL confluence strategy - rule-based
engine (deterministic scoring, no LLM).

Meant to be invoked on a schedule (see .github/workflows/paper_trade.yml)
with no human present. Each run:

  1. Appends any newly-closed 15m BTCUSDT candles from Binance's free public
     API to data/btcusdt_15m.parquet (starts 2022-01-01).
  2. Recomputes indicators over the full dataset and walks forward over
     bars closed since the last run, continuing from persisted state
     (data/state.json), using backtest_pdh_pdl.step_bar - the SAME function
     used for backtesting, so live and backtested behavior cannot diverge.
  3. Appends any trades that closed to data/trades.csv, and rewrites
     STATUS.md with a human-readable summary.

On the very first run (no state.json yet), this replays the entire 2022-
present history once to arrive at a live state - trades.csv starts as
exactly the validated backtest, and everything after `live_since` in
state.json is genuine forward paper trading with no lookahead.

PARAMETERS (see README.md for how these were chosen - a broader sweep than
the original tuning, ranked by CAGR rather than raw profit factor, since
the actual goal is a 12%/year return target, not just PF>1):
  - Zone 0.4% of level, 1H HTF EMA-50 trend filter, min score 3/6,
    Swing-based SL, 2R/4R partial take-profits.
  - Backtested: PF 1.501, CAGR 2.02%/year, max DD -2.80% at 10% sizing.
  - Position size scaled to 60% of equity per trade to hit the stated
    12%/year target - backtested at that size: CAGR 12.05%/year, max
    drawdown -15.82%. This is a real, quantified risk tradeoff, not a free
    lunch: the higher return comes with a much rougher equity curve.
  - ML secondary filter (ml_entry_filter.joblib, via ml_gate): a
    HistGradientBoostingClassifier trained on 49,619 historical near-PDH/PDL
    touches (build_ml_dataset.py / train_ml_model.py) gates entries ON TOP
    of the score>=3+HTF rule above - it does not replace it. Sequential,
    position-exclusive, out-of-sample validation (validate_ml_filter.py)
    found the rule alone losing money over the most recent ~1.25 years
    (CAGR -2.53%, PF 0.91) while adding this gate at threshold 0.55 turned
    that into a small gain (CAGR +0.15%, PF 1.02) at a real cost to trade
    frequency (159 -> 75 trades) and no guarantee it holds (small sample;
    see README for the full investigation, including why a much larger,
    book-informed feature set was tried and did NOT do better).

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
from ml_filter import load_gate

DATA_DIR = "data"
PARQUET_FILE = os.path.join(DATA_DIR, "btcusdt_15m.parquet")
STATE_FILE = os.path.join(DATA_DIR, "state.json")
TRADES_FILE = os.path.join(DATA_DIR, "trades.csv")
STATUS_FILE = "STATUS.md"

SYMBOL = "BTCUSDT"
INTERVAL = "15m"
BASE_URL = "https://data-api.binance.vision/api/v3/klines"  # geo-unrestricted Binance market-data mirror
HISTORY_START = datetime(2022, 1, 1, tzinfo=timezone.utc)

# Tuned config - see module docstring for how these were chosen.
ZONE_PCT, HTF_RULE = 0.4, "1h"
STEP_KWARGS = dict(
    score_threshold=3, rr1=2.0, rr2=4.0, sl_method="Swing", tp2_enabled=True,
    use_htf_filter=True, enable_breakout_protection=True,
)
QTY_PCT_OF_EQUITY = 0.60  # see module docstring - the real lever for the 12%/year target

ML_FILTER_FILE = "ml_entry_filter.joblib"
ML_GATE_THRESHOLD = 0.55  # see module docstring - the validated breakeven-or-better threshold

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
        engine="rule-based+ml-gate", zone_pct=ZONE_PCT, htf_rule=HTF_RULE, qty_pct_of_equity=QTY_PCT_OF_EQUITY,
        ml_gate_threshold=ML_GATE_THRESHOLD,
        **STEP_KWARGS,
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
    lines.append("# PDH/PDL Confluence Reversal — Live Paper Trading (rule-based + ML secondary filter)\n")
    lines.append(f"_Simulator only. No real money, no exchange account, no API keys. Deterministic score>=3+HTF "
                 f"rule, entries additionally gated by a trained ML filter (ml_entry_filter.joblib, threshold "
                 f"{ML_GATE_THRESHOLD}) - see README for how these parameters, the 60% position size, and the "
                 f"ML gate were chosen (target: 12%/year). Last updated: {datetime.now(timezone.utc).isoformat()}_\n")
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
        shown = signals_this_run[-10:]
        prefix = f"(showing last 10 of {len(signals_this_run)}) " if len(signals_this_run) > 10 else ""
        lines.append(f"- Signal(s) this run: {prefix}{shown}")
    lines.append("")

    if len(all_trades):
        wins = all_trades[all_trades["pnl"] > 0]
        gross_loss = -all_trades[all_trades["pnl"] <= 0]["pnl"].sum()
        pf = wins["pnl"].sum() / gross_loss if gross_loss > 0 else float("inf")
        years = max((all_trades["exit_time"].max() - all_trades["entry_time"].min()).days / 365.25, 0.01)
        cagr = ((state["equity"] / bt.INITIAL_CAPITAL) ** (1 / years) - 1) * 100
        eq_curve = bt.INITIAL_CAPITAL + all_trades.sort_values("exit_time")["pnl"].cumsum()
        max_dd = ((eq_curve - eq_curve.cummax()) / eq_curve.cummax()).min() * 100
        lines.append("## All-Time Stats (backtest + live combined)\n")
        lines.append(f"- Total trades: {len(all_trades)}")
        lines.append(f"- Win rate: {len(wins)/len(all_trades)*100:.2f}%")
        lines.append(f"- Total PnL: {all_trades['pnl'].sum():.2f} USDT ({all_trades['pnl'].sum()/bt.INITIAL_CAPITAL*100:.2f}%)")
        lines.append(f"- Profit factor: {pf:.3f}")
        lines.append(f"- CAGR: {cagr:.2f}%/year (target: 12%/year)")
        lines.append(f"- Max drawdown: {max_dd:.2f}%\n")

        lines.append("## Most Recent Trades\n")
        lines.append("| Entry time | Side | Exit reason | Entry | Exit | PnL |")
        lines.append("|---|---|---|---|---|---|")
        for _, t in all_trades.tail(15).iloc[::-1].iterrows():
            lines.append(f"| {t['entry_time']} | {t['side']} | {t['reason']} | {t['entry_px']:.2f} | {t['exit_px']:.2f} | {t['pnl']:.2f} |")
    else:
        lines.append("No trades yet.\n")

    lines.append("\n## Reference: prior LLM-decided engine (archived)\n")
    lines.append("See `data/trades_llm_archive.csv` / `data/llm_decisions_archive.csv` - the LLM-judgment engine "
                 "this replaced. Tested at real scale (~2 years, multiple configurations) and consistently showed "
                 "profit factor 0.50-0.67 (net losing) with no configuration found that produced a real edge. "
                 "See README for the full investigation.")

    with open(STATUS_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    bt.QTY_PCT_OF_EQUITY = QTY_PCT_OF_EQUITY  # step_bar reads this module global at call time
    ml_gate, ml_meta = load_gate(ML_FILTER_FILE, ML_GATE_THRESHOLD)
    print(f"Loaded ML secondary filter ({ML_FILTER_FILE}, threshold={ML_GATE_THRESHOLD}, "
          f"{len(ml_meta['features'])} features)", file=sys.stderr)

    df_raw = update_dataset()
    df = bt.compute_indicators(df_raw.copy(), zone_pct=ZONE_PCT, htf_rule=HTF_RULE)
    arr = bt.prepare_arrays(df)

    state, last_processed_index, live_since = load_state()
    is_first_run = live_since is None
    if is_first_run:
        live_since = str(df["open_time"].iloc[-1]) if len(df) else None
        print("First run: replaying full history to build up live state...", file=sys.stderr)

    now = datetime.now(timezone.utc)
    new_trades = []
    signals = []
    processed_upto = last_processed_index
    for i in range(last_processed_index + 1, arr["n"]):
        close_time = df["close_time"].iloc[i]
        if close_time.to_pydatetime() > now:
            break  # still-forming bar, not closed yet
        state, trades, signal_info = bt.step_bar(i, arr, state, ml_gate=ml_gate, **STEP_KWARGS)
        new_trades.extend(trades)
        if signal_info["buy_signal"]:
            signals.append(f"BUY score {signal_info['buy_score']}/6 @ {arr['close'][i]:.2f} ({arr['open_time'][i]})")
        if signal_info["sell_signal"]:
            signals.append(f"SELL score {signal_info['sell_score']}/6 @ {arr['close'][i]:.2f} ({arr['open_time'][i]})")
        processed_upto = i

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
