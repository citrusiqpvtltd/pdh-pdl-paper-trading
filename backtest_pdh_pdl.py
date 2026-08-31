"""
Python port of pdh-pdl-confluence-strategy.pine, backtested against free,
unlimited-history 15m BTCUSDT data from Binance (see fetch_binance_klines.py).

This exists ONLY to get a larger-sample read on the strategy's performance,
since TradingView Basic's 5,000-bar cap limits a real 15m backtest there to
~2 months / ~29 trades. It mirrors the Pine logic as closely as practical in
pandas; it is a second implementation for research, not a drop-in replacement
for the live Pine script, and the two should be kept in sync by hand.

Config below matches what was actually live-tested on TradingView (some of
which was set via the Settings UI, not the .pine file's own defaults):
  - Entry timeframe: 15m (this is exactly what's being backtested)
  - HTF filter timeframe: 4 hours (the .pine file's own default is 60/1h;
    the live-tested chart had it overridden to 240/4h)
  - Everything else matches pdh-pdl-confluence-strategy.pine's current inputs.

Known simplifications vs. the real Pine/TradingView engine:
  - Pivot ties (two bars with the exact equal high/low in a window) are not
    specially handled; at BTC's price precision this is effectively never hit.
  - When both the stop and a take-profit level fall inside the same bar's
    high/low range, we assume whichever price is closer to that bar's OPEN
    was touched first (a documented, standard backtest approximation -
    TradingView does the same absent its paid "bar magnifier" feature).
  - A position still open at the end of the data is excluded from closed-
    trade stats (matches TradingView's own behavior).
"""
import sys

import numpy as np
import pandas as pd

DATA_FILE = "btcusdt_15m.parquet"
MINTICK = 0.01

# ---- Inputs (mirroring pdh-pdl-confluence-strategy.pine, current live-tested config) ----
ZONE_PCT = 0.3
ATR_LEN = 14
PIVOT_LEFT = 10
PIVOT_RIGHT = 10
MAX_PIVOT_HISTORY = 10
PATTERN_TOL_PCT = 0.3
PATTERN_LOOKBACK = 20
FLAG_IMPULSE_PCT = 3.0
FLAG_CONSOL_ATR_MULT = 1.0
VOL_MA_LEN = 40
VOL_MULT = 1.2
RSI_LEN = 21
RSI_OB = 70
RSI_OS = 30
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
SCORE_THRESHOLD = 5
BREAKOUT_ATR_MULT = 0.5
BREAKOUT_VOL_MULT = 1.5
SL_ATR_MULT = 1.5
BUFFER_TICKS = 5
RR1, RR2 = 1.5, 3.0
TP2_ENABLED = True
HTF_RULE = "4h"
HTF_EMA_LEN = 50
INITIAL_CAPITAL = 10000.0
QTY_PCT_OF_EQUITY = 0.10
COMMISSION_PCT = 0.05 / 100
SLIPPAGE_TICKS = 1


def ema(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(span=length, adjust=False).mean()


def rma(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(alpha=1.0 / length, adjust=False).mean()


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_FILE)
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def compute_indicators(df: pd.DataFrame, zone_pct=ZONE_PCT, htf_rule=HTF_RULE, htf_ema_len=HTF_EMA_LEN) -> pd.DataFrame:
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # ---- ATR (Wilder) ----
    prev_close = c.shift(1)
    tr = pd.concat([h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    df["atr"] = rma(tr, ATR_LEN)

    # ---- PDH / PDL: previous COMPLETED calendar (UTC) day's high/low ----
    date = df["open_time"].dt.floor("D")
    daily = df.groupby(date).agg(day_high=("high", "max"), day_low=("low", "min"))
    daily = daily.shift(1)  # previous day's values
    day_map = daily.reindex(date).reset_index(drop=True)
    df["pdh"] = day_map["day_high"].values
    df["pdl"] = day_map["day_low"].values
    df["new_day"] = date.ne(date.shift(1)).values
    df.loc[0, "new_day"] = True

    df["zone_dist_pdh"] = df["pdh"] * zone_pct / 100
    df["zone_dist_pdl"] = df["pdl"] * zone_pct / 100
    df["near_pdh"] = (h >= df["pdh"] - df["zone_dist_pdh"]) & (l <= df["pdh"] + df["zone_dist_pdh"])
    df["near_pdl"] = (l <= df["pdl"] + df["zone_dist_pdl"]) & (h >= df["pdl"] - df["zone_dist_pdl"])

    # ---- Pivot highs / lows (confirmed pivotRight bars late) ----
    w = PIVOT_LEFT + PIVOT_RIGHT + 1
    rmax = h.rolling(w).max()
    rmin = l.rolling(w).min()
    rmax_aligned = rmax.shift(-PIVOT_RIGHT)
    rmin_aligned = rmin.shift(-PIVOT_RIGHT)
    n = len(df)
    valid = np.zeros(n, dtype=bool)
    valid[PIVOT_LEFT: n - PIVOT_RIGHT] = True
    is_ph = valid & (h.values == rmax_aligned.values)
    is_pl = valid & (l.values == rmin_aligned.values)
    # confirmed at bar (peak_index + PIVOT_RIGHT)
    ph_confirmed_at = np.zeros(n, dtype=bool)
    pl_confirmed_at = np.zeros(n, dtype=bool)
    ph_confirmed_at[PIVOT_RIGHT:] = is_ph[: n - PIVOT_RIGHT]
    pl_confirmed_at[PIVOT_RIGHT:] = is_pl[: n - PIVOT_RIGHT]
    df["ph_confirmed"] = ph_confirmed_at
    df["pl_confirmed"] = pl_confirmed_at
    df["ph_peak_val"] = np.where(ph_confirmed_at, h.shift(PIVOT_RIGHT).values, np.nan)
    df["pl_peak_val"] = np.where(pl_confirmed_at, l.shift(PIVOT_RIGHT).values, np.nan)

    # ---- Candlestick patterns ----
    body0 = (c - o).abs()
    body1 = body0.shift(1)
    body2 = body0.shift(2)
    range1 = (h - l).shift(1)
    upper0 = h - pd.concat([c, o], axis=1).max(axis=1)
    lower0 = pd.concat([c, o], axis=1).min(axis=1) - l
    is_bull0, is_bear0 = c > o, c < o
    is_bull1, is_bear1 = (c > o).shift(1), (c < o).shift(1)
    is_bull2, is_bear2 = (c > o).shift(2), (c < o).shift(2)
    avg_body = body0.rolling(14).mean()
    o1, c1 = o.shift(1), c.shift(1)
    o2, c2 = o.shift(2), c.shift(2)
    h1, l1 = h.shift(1), l.shift(1)

    bear_engulf = is_bull1 & is_bear0 & (o >= c1) & (c <= o1) & (body0 > body1)
    shooting_star = (upper0 >= 2 * body0) & (lower0 <= body0 * 0.3) & (body0 <= avg_body * 0.8)
    evening_star = is_bull2 & (body2 > avg_body) & (body1 < body2 * 0.5) & is_bear0 & (c < (o2 + c2) / 2)
    bear_pin = (upper0 >= body0 * 2) & (lower0 <= body0 * 0.5) & is_bear0
    doji_bear = (body1 <= range1 * 0.1) & is_bear0 & (c < l1)

    bull_engulf = is_bear1 & is_bull0 & (o <= c1) & (c >= o1) & (body0 > body1)
    hammer = (lower0 >= 2 * body0) & (upper0 <= body0 * 0.3) & (body0 <= avg_body * 0.8)
    morning_star = is_bear2 & (body2 > avg_body) & (body1 < body2 * 0.5) & is_bull0 & (c > (o2 + c2) / 2)
    bull_pin = (lower0 >= body0 * 2) & (upper0 <= body0 * 0.5) & is_bull0
    doji_bull = (body1 <= range1 * 0.1) & is_bull0 & (c > h1)

    df["bearish_candle"] = (bear_engulf | shooting_star | evening_star | bear_pin | doji_bear).fillna(False)
    df["bullish_candle"] = (bull_engulf | hammer | morning_star | bull_pin | doji_bull).fillna(False)

    # ---- Flag proxy (non-pivot-based, purely offset-based like the Pine source) ----
    prior_low_flag = l.shift(PATTERN_LOOKBACK + 5)
    prior_high_flag = h.shift(PATTERN_LOOKBACK + 5)
    close_lb = c.shift(PATTERN_LOOKBACK)
    impulse_up_pct = (close_lb - prior_low_flag) / prior_low_flag * 100
    impulse_down_pct = (prior_high_flag - close_lb) / prior_high_flag * 100
    recent_range = h.rolling(5).max() - l.rolling(5).min()
    tight_consolidation = recent_range <= df["atr"] * FLAG_CONSOL_ATR_MULT
    df["bull_flag"] = ((impulse_up_pct > FLAG_IMPULSE_PCT) & tight_consolidation).fillna(False)
    df["bear_flag"] = ((impulse_down_pct > FLAG_IMPULSE_PCT) & tight_consolidation).fillna(False)

    df["failed_breakout_pdh"] = ((h >= df["pdh"] + df["zone_dist_pdh"] * 0.5) & (c < df["pdh"])).fillna(False)
    df["failed_breakdown_pdl"] = ((l <= df["pdl"] - df["zone_dist_pdl"] * 0.5) & (c > df["pdl"])).fillna(False)

    # ---- Volume & momentum ----
    vol_ma = v.rolling(VOL_MA_LEN).mean()
    df["vol_ma"] = vol_ma
    df["vol_confirm"] = (v > vol_ma * VOL_MULT).fillna(False)

    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = rma(gain, RSI_LEN)
    avg_loss = rma(loss, RSI_LEN)
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    df["rsi"] = rsi
    rsi_prev = rsi.shift(1)
    df["rsi_bear_turn"] = ((rsi_prev >= RSI_OB) & (rsi < rsi_prev)).fillna(False)
    df["rsi_bull_turn"] = ((rsi_prev <= RSI_OS) & (rsi > rsi_prev)).fillna(False)

    macd_line = ema(c, MACD_FAST) - ema(c, MACD_SLOW)
    signal_line = ema(macd_line, MACD_SIGNAL)
    hist = macd_line - signal_line
    hist1, hist2 = hist.shift(1), hist.shift(2)
    df["macd_bear_turn"] = ((hist < hist1) & (hist1 >= hist2)).fillna(False)
    df["macd_bull_turn"] = ((hist > hist1) & (hist1 <= hist2)).fillna(False)

    # Momentum Source = "Both-Any"
    df["momentum_bear"] = df["rsi_bear_turn"] | df["macd_bear_turn"]
    df["momentum_bull"] = df["rsi_bull_turn"] | df["macd_bull_turn"]

    # ---- HTF trend filter (previous CLOSED HTF bar only) ----
    htf = df.set_index("open_time")["close"].resample(htf_rule).agg(["last"]).dropna()
    htf["ema"] = ema(htf["last"], htf_ema_len)
    htf = htf.reset_index().rename(columns={"open_time": "htf_close_time", "last": "htf_close"})
    merged = pd.merge_asof(
        df[["open_time"]], htf, left_on="open_time", right_on="htf_close_time",
        direction="backward", allow_exact_matches=False,
    )
    df["htf_trend_up"] = (merged["htf_close"] > merged["ema"]).fillna(False).values

    # ---- Breakout protection ----
    df["strong_break_up"] = ((c > df["pdh"] + BREAKOUT_ATR_MULT * df["atr"]) & (v > vol_ma * BREAKOUT_VOL_MULT) & df["momentum_bull"]).fillna(False)
    df["strong_break_down"] = ((c < df["pdl"] - BREAKOUT_ATR_MULT * df["atr"]) & (v > vol_ma * BREAKOUT_VOL_MULT) & df["momentum_bear"]).fillna(False)

    return df


def prepare_arrays(df: pd.DataFrame) -> dict:
    """Pull every column run/step needs into plain numpy arrays once, up front."""
    cols = ["open", "high", "low", "close", "pdh", "pdl", "zone_dist_pdh", "zone_dist_pdl",
            "atr", "near_pdh", "near_pdl", "bearish_candle", "bullish_candle", "bull_flag",
            "bear_flag", "failed_breakout_pdh", "failed_breakdown_pdl", "vol_confirm",
            "momentum_bear", "momentum_bull", "htf_trend_up", "strong_break_up",
            "strong_break_down", "new_day", "ph_confirmed", "pl_confirmed", "ph_peak_val",
            "pl_peak_val"]
    arr = {c: df[c].values for c in cols}
    arr["open_time"] = df["open_time"].values
    arr["n"] = len(df)
    return arr


def new_state(initial_capital=INITIAL_CAPITAL) -> dict:
    return dict(pivot_high_vals=[], pivot_high_bars=[], pivot_low_vals=[], pivot_low_bars=[],
                sell_fired=False, buy_fired=False, equity=initial_capital, position=None)


def step_bar(i, arr, state, score_threshold=SCORE_THRESHOLD, rr1=RR1, rr2=RR2,
             sl_method="Swing", tp_method="RR", tp2_enabled=TP2_ENABLED,
             use_htf_filter=True, enable_breakout_protection=True,
             sl_atr_mult=SL_ATR_MULT, buffer_ticks=BUFFER_TICKS, tp_atr_mult=2.0,
             sl_fixed_pct=1.0):
    """
    Advance the strategy by exactly one bar, mutating and returning `state`.

    Every bar updates pivot arrays, market structure, scoring, and the
    sellFired/buyFired reset logic UNCONDITIONALLY - matching Pine, where
    these are plain script-level computations that run every bar regardless
    of position state. (An earlier version of this backtest skipped all of
    this while a position was open, jumping straight from the entry bar to
    the exit bar - silently going stale on pivots/fired-flags for the
    entire life of any open trade. Fixed here.)

    Returns (state, trades_closed_this_bar: list[dict], signal_info: dict).
    """
    tick_buffer = MINTICK * buffer_ticks
    trades = []

    # ---- maintain pivot arrays (every bar, unconditionally) ----
    if arr["ph_confirmed"][i]:
        state["pivot_high_vals"].append(arr["ph_peak_val"][i])
        state["pivot_high_bars"].append(i - PIVOT_RIGHT)
        if len(state["pivot_high_vals"]) > MAX_PIVOT_HISTORY:
            state["pivot_high_vals"].pop(0)
            state["pivot_high_bars"].pop(0)
    if arr["pl_confirmed"][i]:
        state["pivot_low_vals"].append(arr["pl_peak_val"][i])
        state["pivot_low_bars"].append(i - PIVOT_RIGHT)
        if len(state["pivot_low_vals"]) > MAX_PIVOT_HISTORY:
            state["pivot_low_vals"].pop(0)
            state["pivot_low_bars"].pop(0)

    phv, phb = state["pivot_high_vals"], state["pivot_high_bars"]
    plv, plb = state["pivot_low_vals"], state["pivot_low_bars"]
    ph_size, pl_size = len(phv), len(plv)

    # ---- market structure ----
    structure_bear = structure_bull = False
    if ph_size >= 2 and pl_size >= 2:
        structure_bear = phv[-1] < phv[-2]
        structure_bull = plv[-1] > plv[-2]

    # ---- chart pattern proxies (pivot-based) ----
    double_top = double_bottom = head_shoulders = inv_head_shoulders = False
    rising_wedge = falling_wedge = False
    if ph_size >= 2:
        d1, d2, dbar2 = phv[-1], phv[-2], phb[-2]
        double_top = abs(d1 - d2) <= d2 * PATTERN_TOL_PCT / 100 and (i - dbar2) <= PATTERN_LOOKBACK
    if pl_size >= 2:
        d1, d2, dbar2 = plv[-1], plv[-2], plb[-2]
        double_bottom = abs(d1 - d2) <= d2 * PATTERN_TOL_PCT / 100 and (i - dbar2) <= PATTERN_LOOKBACK
    if ph_size >= 3:
        left_, head_, right_, bar1 = phv[-3], phv[-2], phv[-1], phb[-3]
        head_shoulders = head_ > left_ and head_ > right_ and abs(left_ - right_) <= left_ * PATTERN_TOL_PCT / 100 and (i - bar1) <= PATTERN_LOOKBACK
    if pl_size >= 3:
        left_, head_, right_, bar1 = plv[-3], plv[-2], plv[-1], plb[-3]
        inv_head_shoulders = head_ < left_ and head_ < right_ and abs(left_ - right_) <= left_ * PATTERN_TOL_PCT / 100 and (i - bar1) <= PATTERN_LOOKBACK
    if ph_size >= 3 and pl_size >= 3:
        wh1, wh3, wb1, wb3 = phv[-1], phv[-3], phb[-1], phb[-3]
        wl1, wl3, lb1, lb3 = plv[-1], plv[-3], plb[-1], plb[-3]
        if wb1 != wb3 and lb1 != lb3:
            slope_high = (wh1 - wh3) / (wb1 - wb3)
            slope_low = (wl1 - wl3) / (lb1 - lb3)
            rising_wedge = slope_high > 0 and slope_low > 0 and slope_low > slope_high
            falling_wedge = slope_high < 0 and slope_low < 0 and slope_high < slope_low

    chart_pattern_bear = double_top or head_shoulders or rising_wedge or arr["bear_flag"][i] or arr["failed_breakout_pdh"][i]
    chart_pattern_bull = double_bottom or inv_head_shoulders or falling_wedge or arr["bull_flag"][i] or arr["failed_breakdown_pdl"][i]

    # ---- scoring ----
    buy_score = int(arr["near_pdl"][i]) + int(arr["bullish_candle"][i]) + int(chart_pattern_bull) + int(arr["vol_confirm"][i]) + int(structure_bull) + int(arr["momentum_bull"][i])
    sell_score = int(arr["near_pdh"][i]) + int(arr["bearish_candle"][i]) + int(chart_pattern_bear) + int(arr["vol_confirm"][i]) + int(structure_bear) + int(arr["momentum_bear"][i])

    buy_qualified = buy_score >= score_threshold
    sell_qualified = sell_score >= score_threshold

    if enable_breakout_protection:
        sell_qualified = sell_qualified and not arr["strong_break_up"][i]
        buy_qualified = buy_qualified and not arr["strong_break_down"][i]

    if use_htf_filter:
        sell_qualified = sell_qualified and not arr["htf_trend_up"][i]
        buy_qualified = buy_qualified and arr["htf_trend_up"][i]

    # ---- state machine resets (every bar, unconditionally) ----
    if arr["new_day"][i]:
        state["sell_fired"] = False
        state["buy_fired"] = False
    pdh_i, pdl_i, zone_pdh_i, zone_pdl_i = arr["pdh"][i], arr["pdl"][i], arr["zone_dist_pdh"][i], arr["zone_dist_pdl"][i]
    if arr["high"][i] < pdh_i - zone_pdh_i * 1.5:
        state["sell_fired"] = False
    if arr["low"][i] > pdl_i + zone_pdl_i * 1.5:
        state["buy_fired"] = False

    position = state["position"]
    is_flat = position is None

    # ---- if a position was ALREADY open coming into this bar, check this bar's
    #      range for a fill first (mirrors Pine: entry fills at bar i's close,
    #      so the earliest an exit can trigger is bar i+1) ----
    if position is not None:
        target = position["tp2"] if (position["tp1_filled"] and tp2_enabled) else position["tp1"]
        sl = position["sl"]
        side = position["side"]
        bo, bh, bl = arr["open"][i], arr["high"][i], arr["low"][i]

        if side == "long":
            hit_sl, hit_tp = bl <= sl, bh >= target
        else:
            hit_sl, hit_tp = bh >= sl, bl <= target

        if hit_sl or hit_tp:
            sl_first = (abs(bo - sl) < abs(bo - target)) if (hit_sl and hit_tp) else hit_sl
            exit_time = arr["open_time"][i]
            if sl_first:
                exit_px = sl - MINTICK * SLIPPAGE_TICKS if side == "long" else sl + MINTICK * SLIPPAGE_TICKS
                qty_exit, reason = position["qty_remaining"], "SL"
            else:
                exit_px = target
                if position["tp1_filled"] or not tp2_enabled:
                    qty_exit = position["qty_remaining"]
                    reason = "TP2" if position["tp1_filled"] else "TP1(full)"
                else:
                    qty_exit, reason = position["qty_total"] * 0.5, "TP1"

            if side == "long":
                price_pnl = (exit_px - position["entry_px"]) * qty_exit
            else:
                price_pnl = (position["entry_px"] - exit_px) * qty_exit
            exit_comm = exit_px * qty_exit * COMMISSION_PCT
            state["equity"] += price_pnl - exit_comm
            entry_comm_share = position["entry_comm_total"] * (qty_exit / position["qty_total"])
            pnl = price_pnl - exit_comm - entry_comm_share

            trades.append(dict(side=side, entry_time=position["entry_time"], exit_time=exit_time,
                                reason=reason, qty=qty_exit, entry_px=position["entry_px"],
                                exit_px=exit_px, pnl=pnl))

            position["qty_remaining"] -= qty_exit
            if reason == "SL" or reason in ("TP2", "TP1(full)") or position["qty_remaining"] <= 1e-12:
                position = None
            else:
                position["tp1_filled"] = True
            state["position"] = position
            is_flat = position is None

    sell_signal = buy_signal = False
    # ---- entry check (only if flat - either already flat, or just closed above) ----
    if is_flat:
        sell_signal = sell_qualified and arr["near_pdh"][i] and not state["sell_fired"]
        buy_signal = buy_qualified and arr["near_pdl"][i] and not state["buy_fired"]
        if sell_signal:
            state["sell_fired"] = True
        if buy_signal:
            state["buy_fired"] = True

        atr_i = arr["atr"][i]
        if (sell_signal or buy_signal) and pd.notna(atr_i) and pd.notna(pdh_i) and pd.notna(pdl_i):
            entry_close = arr["close"][i]
            if sell_signal:
                if sl_method == "Swing":
                    valid_swing = ph_size >= 1 and phv[-1] > entry_close
                    sl_price = (phv[-1] + tick_buffer) if valid_swing else (entry_close + sl_atr_mult * atr_i)
                elif sl_method == "ATR":
                    sl_price = entry_close + sl_atr_mult * atr_i
                elif sl_method == "PDH/PDL":
                    sl_price = (pdh_i + tick_buffer) if pdh_i > entry_close else (entry_close + sl_atr_mult * atr_i)
                else:
                    sl_price = entry_close * (1 + sl_fixed_pct / 100)
                risk = sl_price - entry_close
                if tp_method == "RR":
                    tp1, tp2 = entry_close - risk * rr1, entry_close - risk * rr2
                elif tp_method == "OppositeLevel":
                    tp1 = tp2 = pdl_i
                else:
                    tp1 = tp2 = entry_close - tp_atr_mult * atr_i
                if risk > 0:
                    fill_px = entry_close - MINTICK * SLIPPAGE_TICKS
                    qty = (state["equity"] * QTY_PCT_OF_EQUITY) / fill_px
                    entry_comm = fill_px * qty * COMMISSION_PCT
                    state["position"] = dict(side="short", entry_time=arr["open_time"][i], entry_px=fill_px,
                                              qty_total=qty, qty_remaining=qty, sl=sl_price, tp1=tp1, tp2=tp2,
                                              tp1_filled=False, entry_comm_total=entry_comm)
                    state["equity"] -= entry_comm
            else:
                if sl_method == "Swing":
                    valid_swing = pl_size >= 1 and plv[-1] < entry_close
                    sl_price = (plv[-1] - tick_buffer) if valid_swing else (entry_close - sl_atr_mult * atr_i)
                elif sl_method == "ATR":
                    sl_price = entry_close - sl_atr_mult * atr_i
                elif sl_method == "PDH/PDL":
                    sl_price = (pdl_i - tick_buffer) if pdl_i < entry_close else (entry_close - sl_atr_mult * atr_i)
                else:
                    sl_price = entry_close * (1 - sl_fixed_pct / 100)
                risk = entry_close - sl_price
                if tp_method == "RR":
                    tp1, tp2 = entry_close + risk * rr1, entry_close + risk * rr2
                elif tp_method == "OppositeLevel":
                    tp1 = tp2 = pdh_i
                else:
                    tp1 = tp2 = entry_close + tp_atr_mult * atr_i
                if risk > 0:
                    fill_px = entry_close + MINTICK * SLIPPAGE_TICKS
                    qty = (state["equity"] * QTY_PCT_OF_EQUITY) / fill_px
                    entry_comm = fill_px * qty * COMMISSION_PCT
                    state["position"] = dict(side="long", entry_time=arr["open_time"][i], entry_px=fill_px,
                                              qty_total=qty, qty_remaining=qty, sl=sl_price, tp1=tp1, tp2=tp2,
                                              tp1_filled=False, entry_comm_total=entry_comm)
                    state["equity"] -= entry_comm

    signal_info = dict(buy_signal=buy_signal, sell_signal=sell_signal, buy_score=buy_score, sell_score=sell_score)
    return state, trades, signal_info


def run_backtest(df: pd.DataFrame, score_threshold=SCORE_THRESHOLD, rr1=RR1, rr2=RR2,
                  sl_method="Swing", tp_method="RR", tp2_enabled=TP2_ENABLED,
                  use_htf_filter=True, enable_breakout_protection=True,
                  sl_atr_mult=SL_ATR_MULT, buffer_ticks=BUFFER_TICKS, tp_atr_mult=2.0,
                  sl_fixed_pct=1.0):
    arr = prepare_arrays(df)
    state = new_state()
    all_trades = []
    kwargs = dict(score_threshold=score_threshold, rr1=rr1, rr2=rr2, sl_method=sl_method,
                  tp_method=tp_method, tp2_enabled=tp2_enabled, use_htf_filter=use_htf_filter,
                  enable_breakout_protection=enable_breakout_protection, sl_atr_mult=sl_atr_mult,
                  buffer_ticks=buffer_ticks, tp_atr_mult=tp_atr_mult, sl_fixed_pct=sl_fixed_pct)
    for i in range(arr["n"]):
        state, trades, _ = step_bar(i, arr, state, **kwargs)
        all_trades.extend(trades)
    return all_trades, state["equity"]


def report(trades, final_equity):
    if not trades:
        print("No trades.")
        return
    tdf = pd.DataFrame(trades)
    total_pnl = tdf["pnl"].sum()
    wins = tdf[tdf["pnl"] > 0]
    losses = tdf[tdf["pnl"] <= 0]
    win_rate = len(wins) / len(tdf) * 100
    gross_profit = wins["pnl"].sum()
    gross_loss = -losses["pnl"].sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    equity_curve = INITIAL_CAPITAL + tdf["pnl"].cumsum()
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max)
    max_dd = drawdown.min()
    max_dd_pct = (drawdown / running_max).min() * 100

    print(f"Period: {tdf['entry_time'].min()} -> {tdf['exit_time'].max()}")
    print(f"Total trades (TV-style, partial exits counted separately): {len(tdf)}")
    print(f"Win rate: {win_rate:.2f}% ({len(wins)}/{len(tdf)})")
    print(f"Total PnL: {total_pnl:.2f} USDT ({total_pnl/INITIAL_CAPITAL*100:.2f}%)")
    print(f"Final equity: {final_equity:.2f} USDT  (reconciliation check: initial+PnL = {INITIAL_CAPITAL+total_pnl:.2f}, should equal final equity above)")
    print(f"Gross profit: {gross_profit:.2f}  Gross loss: {gross_loss:.2f}")
    print(f"Profit factor: {pf:.3f}")
    print(f"Max drawdown: {max_dd:.2f} USDT ({max_dd_pct:.2f}%)")
    print()
    print("By side:")
    for side in ("long", "short"):
        sd = tdf[tdf["side"] == side]
        if len(sd):
            print(f"  {side}: {len(sd)} trades, PnL {sd['pnl'].sum():.2f}, win rate {(sd['pnl']>0).mean()*100:.2f}%")
    print()
    print("By exit reason:")
    print(tdf.groupby("reason")["pnl"].agg(["count", "sum", "mean"]))
    print()
    print("By year:")
    tdf["year"] = tdf["exit_time"].dt.year
    print(tdf.groupby("year")["pnl"].agg(["count", "sum"]))

    tdf.to_csv("backtest_trades.csv", index=False)
    print("\nFull trade log written to backtest_trades.csv")


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} bars: {df['open_time'].min()} -> {df['open_time'].max()}", file=sys.stderr)
    df = compute_indicators(df)
    trades, final_equity = run_backtest(df)
    report(trades, final_equity)
