"""
Live paper-trading runner for the PDH/PDL reversal strategy - LLM-decided
entries (replaces the fixed 6-point confluence score threshold).

Meant to be invoked on a schedule (see .github/workflows/paper_trade.yml)
with no human present. Each run:

  1. Appends any newly-closed 15m BTCUSDT candles from Binance's free public
     API to data/btcusdt_15m.parquet (starts 2022-01-01).
  2. Recomputes indicators over the full dataset (ATR, RSI, MACD, pivots,
     candlestick/chart patterns, market structure, 4H HTF trend) using the
     same backtest_pdh_pdl.compute_indicators used for the archived
     rule-based backtest.
  3. Walks forward over bars closed since the last run, continuing from
     persisted state (data/state.json). Whenever price is testing PDH or
     PDL and the bot is flat, it asks a local LLM (Ollama, see
     llm_decide.py) whether this specific setup is worth taking - the
     model sees the same technical context a human would (patterns,
     structure, volume, momentum, HTF trend, recent candles) and decides
     enter/skip with reasoning. Stop-loss/take-profit levels and exits, if
     it enters, are still computed by the same deterministic Swing/RR math
     as the archived rule-based engine - only the entry judgment is LLM-
     driven.
  4. Appends any trades that closed to data/trades.csv, and rewrites
     STATUS.md with a human-readable summary, including the LLM's stated
     reasoning for its most recent decisions.

IMPORTANT: this decision logic has NOT been backtested (see llm_decide.py's
module docstring) - it is a live experiment, not a validated strategy. The
prior rule-based engine's 4.67-year-validated results are preserved in
data/trades_rulebased_archive.csv for reference/comparison.

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
import llm_decide
import market_status

DATA_DIR = "data"
PARQUET_FILE = os.path.join(DATA_DIR, "btcusdt_15m.parquet")
STATE_FILE = os.path.join(DATA_DIR, "state.json")
TRADES_FILE = os.path.join(DATA_DIR, "trades.csv")
DECISIONS_FILE = os.path.join(DATA_DIR, "llm_decisions.csv")
STATUS_FILE = "STATUS.md"

SYMBOL = "BTCUSDT"
INTERVAL = "15m"
BASE_URL = "https://data-api.binance.vision/api/v3/klines"  # geo-unrestricted Binance market-data mirror
HISTORY_START = datetime(2022, 1, 1, tzinfo=timezone.utc)

ZONE_PCT, HTF_RULE = 0.3, "4h"
SL_METHOD, RR1, RR2, TP2_ENABLED = "Swing", 1.5, 3.0, True
# Real 2-year backtest (data/llm_2year_backtest_trades.csv, ~Sep 2024-Sep 2026):
# short round-trips won 16.7% (4/24, -$0.80 total) vs long 28.9% (13/45, -$0.13
# total) - shorts lost in EVERY sub-breakdown checked (including split by
# whether the entry fought or aligned with the 4H trend), while one long
# sub-bucket was genuinely profitable (+$0.45, 40% win rate, buying against a
# downtrend). This is the largest, most consistent pattern in the data -
# disabling shorts removes the clear loser without touching the side that at
# least has a shot. Re-enable only after fixing sell-side judgment and
# re-validating at real scale.
ALLOW_SHORTS = False
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
        engine="llm-ollama", model=llm_decide.OLLAMA_MODEL,
    )
    with open(STATE_FILE, "w") as f:
        json.dump(raw, f, indent=2, default=str)


def append_csv(path, rows):
    if not rows:
        return
    tdf = pd.DataFrame(rows)
    header = not os.path.exists(path)
    tdf.to_csv(path, mode="a", header=header, index=False)


def compute_sl_tp(side: str, entry_close: float, atr: float, pdh: float, pdl: float,
                   pivot_high_vals: list, pivot_low_vals: list, buffer_ticks: int = 5,
                   sl_atr_mult: float = 1.5) -> dict:
    """Same deterministic Swing-SL / RR-TP math as backtest_pdh_pdl.step_bar."""
    tick_buffer = bt.MINTICK * buffer_ticks
    if side == "short":
        valid_swing = len(pivot_high_vals) >= 1 and pivot_high_vals[-1] > entry_close
        sl = (pivot_high_vals[-1] + tick_buffer) if valid_swing else (entry_close + sl_atr_mult * atr)
        risk = sl - entry_close
        tp1, tp2 = entry_close - risk * RR1, entry_close - risk * RR2
    else:
        valid_swing = len(pivot_low_vals) >= 1 and pivot_low_vals[-1] < entry_close
        sl = (pivot_low_vals[-1] - tick_buffer) if valid_swing else (entry_close - sl_atr_mult * atr)
        risk = entry_close - sl
        tp1, tp2 = entry_close + risk * RR1, entry_close + risk * RR2
    return dict(sl=sl, tp1=tp1, tp2=tp2, risk=risk)


def step_bar_llm(i, arr, state, market_status_text=""):
    """One bar of the LLM-decided engine. Entry judgment is delegated to
    llm_decide.decide_trade; pivot maintenance, structure/pattern context,
    and exit management reuse the same logic/constants as the validated
    rule-based engine."""
    trades, decisions = [], []
    tick_buffer = bt.MINTICK * bt.BUFFER_TICKS

    if arr["ph_confirmed"][i]:
        state["pivot_high_vals"].append(arr["ph_peak_val"][i])
        state["pivot_high_bars"].append(i - bt.PIVOT_RIGHT)
        if len(state["pivot_high_vals"]) > bt.MAX_PIVOT_HISTORY:
            state["pivot_high_vals"].pop(0)
            state["pivot_high_bars"].pop(0)
    if arr["pl_confirmed"][i]:
        state["pivot_low_vals"].append(arr["pl_peak_val"][i])
        state["pivot_low_bars"].append(i - bt.PIVOT_RIGHT)
        if len(state["pivot_low_vals"]) > bt.MAX_PIVOT_HISTORY:
            state["pivot_low_vals"].pop(0)
            state["pivot_low_bars"].pop(0)

    phv, plv = state["pivot_high_vals"], state["pivot_low_vals"]
    structure_bear = len(phv) >= 2 and phv[-1] < phv[-2]
    structure_bull = len(plv) >= 2 and plv[-1] > plv[-2]
    structure_desc = "Higher-high/higher-low (bullish)" if structure_bull else \
                      "Lower-high/lower-low (bearish)" if structure_bear else "No clear structure"

    pdh_i, pdl_i = arr["pdh"][i], arr["pdl"][i]
    zone_pdh_i, zone_pdl_i = arr["zone_dist_pdh"][i], arr["zone_dist_pdl"][i]

    if arr["new_day"][i]:
        state["sell_fired"], state["buy_fired"] = False, False
    if arr["high"][i] < pdh_i - zone_pdh_i * 1.5:
        state["sell_fired"] = False
    if arr["low"][i] > pdl_i + zone_pdl_i * 1.5:
        state["buy_fired"] = False

    position = state["position"]

    # ---- manage an already-open position: same deterministic exit logic as backtest_pdh_pdl.step_bar ----
    if position is not None:
        target = position["tp2"] if (position["tp1_filled"] and TP2_ENABLED) else position["tp1"]
        sl, side = position["sl"], position["side"]
        bo, bh, bl = arr["open"][i], arr["high"][i], arr["low"][i]
        hit_sl, hit_tp = (bl <= sl, bh >= target) if side == "long" else (bh >= sl, bl <= target)
        if hit_sl or hit_tp:
            sl_first = (abs(bo - sl) < abs(bo - target)) if (hit_sl and hit_tp) else hit_sl
            if sl_first:
                exit_px = sl - bt.MINTICK * bt.SLIPPAGE_TICKS if side == "long" else sl + bt.MINTICK * bt.SLIPPAGE_TICKS
                qty_exit, reason = position["qty_remaining"], "SL"
            else:
                exit_px = target
                if position["tp1_filled"] or not TP2_ENABLED:
                    qty_exit, reason = position["qty_remaining"], ("TP2" if position["tp1_filled"] else "TP1(full)")
                else:
                    qty_exit, reason = position["qty_total"] * 0.5, "TP1"
            price_pnl = (exit_px - position["entry_px"]) * qty_exit if side == "long" else (position["entry_px"] - exit_px) * qty_exit
            exit_comm = exit_px * qty_exit * bt.COMMISSION_PCT
            state["equity"] += price_pnl - exit_comm
            entry_comm_share = position["entry_comm_total"] * (qty_exit / position["qty_total"])
            pnl = price_pnl - exit_comm - entry_comm_share
            trades.append(dict(side=side, entry_time=position["entry_time"], exit_time=arr["open_time"][i],
                                reason=reason, qty=qty_exit, entry_px=position["entry_px"],
                                exit_px=exit_px, pnl=pnl))
            position["qty_remaining"] -= qty_exit
            if reason == "SL" or reason in ("TP2", "TP1(full)") or position["qty_remaining"] <= 1e-12:
                position = None
            else:
                position["tp1_filled"] = True
            state["position"] = position

    is_flat = state["position"] is None
    near_pdh, near_pdl = bool(arr["near_pdh"][i]), bool(arr["near_pdl"][i])
    if not ALLOW_SHORTS:
        near_pdh = False  # see ALLOW_SHORTS docstring above - shorts disabled pending sell-side fix

    if is_flat and (near_pdh or near_pdl):
        side_key = "sell" if near_pdh else "buy"
        level_price = pdh_i if near_pdh else pdl_i
        already_asked = state["sell_fired"] if side_key == "sell" else state["buy_fired"]
        if not already_asked and pd.notna(arr["atr"][i]) and pd.notna(pdh_i) and pd.notna(pdl_i):
            patterns = {
                "bearish candlestick pattern": bool(arr["bearish_candle"][i]),
                "bullish candlestick pattern": bool(arr["bullish_candle"][i]),
                "bullish chart pattern (double bottom/inv H&S/falling wedge/bull flag/failed breakdown)": bool(arr["bull_flag"][i] or arr["failed_breakdown_pdl"][i]),
                "bearish chart pattern (double top/H&S/rising wedge/bear flag/failed breakout)": bool(arr["bear_flag"][i] or arr["failed_breakout_pdh"][i]),
                "volume above 1.2x its 40-bar average": bool(arr["vol_confirm"][i]),
                "momentum turn (RSI or MACD histogram)": bool(arr["momentum_bear"][i] if side_key == "sell" else arr["momentum_bull"][i]),
            }
            start = max(0, i - 10)
            recent = [dict(time=str(arr["open_time"][j]), o=arr["open"][j], h=arr["high"][j],
                            l=arr["low"][j], c=arr["close"][j]) for j in range(start, i + 1)]
            vol_ratio = (arr["volume"][i] / arr["vol_ma"][i]) if pd.notna(arr["vol_ma"][i]) and arr["vol_ma"][i] > 0 else float("nan")
            # market_status_text is either a fixed string (live bot: one fetch
            # per run) or a callable taking the bar index (backtests: point-
            # in-time-correct lookup per historical date - see backtest_llm.py)
            ms_text = market_status_text(i) if callable(market_status_text) else market_status_text
            context_text = llm_decide.build_context(
                side=side_key, level_price=level_price, close=arr["close"][i], atr=arr["atr"][i],
                rsi=arr["rsi"][i], vol_ratio=vol_ratio, patterns=patterns, structure=structure_desc,
                htf_trend_up=bool(arr["htf_trend_up"][i]), recent_candles=recent,
                market_status_text=ms_text,
            )
            decision = llm_decide.decide_trade(context_text)
            decisions.append(dict(time=str(arr["open_time"][i]), side=side_key, level_price=level_price,
                                   action=decision["action"], reasoning=decision["reasoning"]))

            if side_key == "sell":
                state["sell_fired"] = True
            else:
                state["buy_fired"] = True

            if decision["action"] == "enter":
                entry_close = arr["close"][i]
                calc = compute_sl_tp("short" if side_key == "sell" else "long", entry_close, arr["atr"][i],
                                      pdh_i, pdl_i, phv, plv)
                if calc["risk"] > 0:
                    slip = -bt.MINTICK * bt.SLIPPAGE_TICKS if side_key == "sell" else bt.MINTICK * bt.SLIPPAGE_TICKS
                    fill_px = entry_close + slip
                    qty = (state["equity"] * bt.QTY_PCT_OF_EQUITY) / fill_px
                    entry_comm = fill_px * qty * bt.COMMISSION_PCT
                    state["position"] = dict(
                        side="short" if side_key == "sell" else "long", entry_time=arr["open_time"][i],
                        entry_px=fill_px, qty_total=qty, qty_remaining=qty, sl=calc["sl"], tp1=calc["tp1"],
                        tp2=calc["tp2"], tp1_filled=False, entry_comm_total=entry_comm,
                        llm_reasoning=decision["reasoning"],
                    )
                    state["equity"] -= entry_comm

    return state, trades, decisions


def write_status(state, last_bar_time, recent_decisions):
    all_trades = pd.read_csv(TRADES_FILE, parse_dates=["entry_time", "exit_time"]) if os.path.exists(TRADES_FILE) else pd.DataFrame()
    all_decisions = pd.read_csv(DECISIONS_FILE) if os.path.exists(DECISIONS_FILE) else pd.DataFrame()
    lines = []
    lines.append("# PDH/PDL Reversal — Live Paper Trading (LLM-decided entries)\n")
    lines.append(f"_Simulator only. No real money, no exchange account, no API keys. Entries are judged live by a local "
                 f"Ollama model ({llm_decide.OLLAMA_MODEL}), not the fixed rule-based score. "
                 f"**This decision logic is unvalidated / experimental** - see README. "
                 f"Last updated: {datetime.now(timezone.utc).isoformat()}_\n")
    lines.append("## Current State\n")
    lines.append(f"- Equity: **{state['equity']:.2f} USDT** (started at {bt.INITIAL_CAPITAL:.2f})")
    lines.append(f"- Last processed bar: {last_bar_time}")
    pos = state["position"]
    if pos is None:
        lines.append("- Position: **flat**")
    else:
        lines.append(f"- Position: **{pos['side'].upper()}** {pos['qty_remaining']:.6f} BTC @ {pos['entry_px']:.2f} "
                      f"(SL {pos['sl']:.2f}, TP1 {pos['tp1']:.2f}, TP2 {pos['tp2']:.2f})")
        lines.append(f"  - LLM's reasoning at entry: _{pos.get('llm_reasoning', 'n/a')}_")
    lines.append("")

    if len(all_trades):
        wins = all_trades[all_trades["pnl"] > 0]
        gross_loss = -all_trades[all_trades["pnl"] <= 0]["pnl"].sum()
        pf = wins["pnl"].sum() / gross_loss if gross_loss > 0 else float("inf")
        lines.append("## Live Stats (since switching to LLM-decided entries)\n")
        lines.append(f"- Total trades: {len(all_trades)}")
        lines.append(f"- Win rate: {len(wins)/len(all_trades)*100:.2f}%")
        lines.append(f"- Total PnL: {all_trades['pnl'].sum():.2f} USDT ({all_trades['pnl'].sum()/bt.INITIAL_CAPITAL*100:.2f}%)")
        lines.append(f"- Profit factor: {pf:.3f}\n")
    else:
        lines.append("No trades closed yet under the LLM-decided engine.\n")

    if len(all_decisions):
        lines.append("## Most Recent LLM Decisions\n")
        lines.append("| Time | Side | Level | Action | Reasoning |")
        lines.append("|---|---|---|---|---|")
        for _, d in all_decisions.tail(15).iloc[::-1].iterrows():
            lines.append(f"| {d['time']} | {d['side']} | {d['level_price']:.2f} | {d['action']} | {d['reasoning']} |")

    lines.append("\n## Reference: prior validated rule-based backtest\n")
    lines.append("See `data/trades_rulebased_archive.csv` - the fixed-rule engine this replaced, "
                 "validated over 4.67 years of history (355 trades, profit factor 1.483, +4.08%, "
                 "profitable in all 5 calendar years). That validation does **not** carry over to "
                 "this LLM-decided engine.")

    with open(STATUS_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    df_raw = update_dataset()
    df = bt.compute_indicators(df_raw.copy(), zone_pct=ZONE_PCT, htf_rule=HTF_RULE)
    arr = bt.prepare_arrays(df)

    state, last_processed_index, live_since = load_state()
    if live_since is None:
        # First run under the LLM engine: start fresh from "now", not a full
        # historical replay (that would mean thousands of real LLM calls).
        live_since = str(df["open_time"].iloc[-1]) if len(df) else None
        last_processed_index = arr["n"] - 1
        print("First run under LLM-decided engine: starting fresh from the latest bar "
              "(no historical replay - see llm_decide.py docstring for why).", file=sys.stderr)

    # Fetched once per run, not once per decision - this is slow-moving,
    # market-wide context, not something worth a fresh call per bar.
    market_status_text = market_status.format_for_context(market_status.fetch_market_status())
    print(market_status_text, file=sys.stderr)

    now = datetime.now(timezone.utc)
    new_trades, new_decisions = [], []
    processed_upto = last_processed_index
    for i in range(last_processed_index + 1, arr["n"]):
        if df["close_time"].iloc[i].to_pydatetime() > now:
            break
        state, trades, decisions = step_bar_llm(i, arr, state, market_status_text)
        new_trades.extend(trades)
        new_decisions.extend(decisions)
        processed_upto = i

    save_state(state, processed_upto, live_since)
    append_csv(TRADES_FILE, new_trades)
    append_csv(DECISIONS_FILE, new_decisions)
    last_bar_time = df["open_time"].iloc[processed_upto] if processed_upto >= 0 else "none yet"
    write_status(state, last_bar_time, new_decisions)

    print(f"Processed through bar index {processed_upto} ({last_bar_time}); "
          f"{len(new_trades)} trade(s) closed, {len(new_decisions)} LLM decision(s) made this run; "
          f"equity={state['equity']:.2f}; position={'flat' if state['position'] is None else state['position']['side']}",
          file=sys.stderr)
    for d in new_decisions:
        print(f"DECISION: {d['side']} @ {d['level_price']:.2f} -> {d['action']} ({d['reasoning']})", file=sys.stderr)


if __name__ == "__main__":
    main()
