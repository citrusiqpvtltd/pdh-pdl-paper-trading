"""
Build a labeled training dataset for an ML entry filter, to replace the
hand-tuned "score >= 3" threshold with a learned one.

For EVERY near-PDH/PDL touch (not just the 489 that already passed the
score filter and got taken live), computes:
  - features: the same raw signals the rule-based score is built from
    (individual candlestick/chart pattern flags, structure, volume,
    momentum, HTF trend, ATR, RSI - ungated, not collapsed into a 0-6 sum)
  - label: what would have happened if this setup had been taken in
    isolation (same Swing-SL / 2R-4R-TP mechanics as the live engine),
    independent of whether a real position was open elsewhere at the time.

This gives ~10x+ more labeled examples than the filtered trade log, since
most touches never reach score>=3. Output: data/ml_training_data.csv.
"""
import sys

import numpy as np
import pandas as pd

import backtest_pdh_pdl as bt

ZONE_PCT, HTF_RULE = 0.4, "1h"
RR1, RR2 = 2.0, 4.0
SL_ATR_MULT, BUFFER_TICKS = 1.5, 5


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    arr = bt.prepare_arrays(df)
    n = arr["n"]
    tick_buffer = bt.MINTICK * BUFFER_TICKS

    pivot_high_vals, pivot_high_bars = [], []
    pivot_low_vals, pivot_low_bars = [], []
    rows = []

    for i in range(n):
        if arr["ph_confirmed"][i]:
            pivot_high_vals.append(arr["ph_peak_val"][i])
            pivot_high_bars.append(i - bt.PIVOT_RIGHT)
            if len(pivot_high_vals) > bt.MAX_PIVOT_HISTORY:
                pivot_high_vals.pop(0); pivot_high_bars.pop(0)
        if arr["pl_confirmed"][i]:
            pivot_low_vals.append(arr["pl_peak_val"][i])
            pivot_low_bars.append(i - bt.PIVOT_RIGHT)
            if len(pivot_low_vals) > bt.MAX_PIVOT_HISTORY:
                pivot_low_vals.pop(0); pivot_low_bars.pop(0)

        near_pdh, near_pdl = bool(arr["near_pdh"][i]), bool(arr["near_pdl"][i])
        if not (near_pdh or near_pdl):
            continue

        phv, phb = pivot_high_vals, pivot_high_bars
        plv, plb = pivot_low_vals, pivot_low_bars
        ph_size, pl_size = len(phv), len(plv)

        # NOTE: matches backtest_pdh_pdl.step_bar exactly - structure requires
        # BOTH pivot-high and pivot-low history to exist, not each independently
        # (an earlier version of this file checked them independently, which
        # silently diverged from the live engine's actual gating).
        structure_bear = structure_bull = False
        if ph_size >= 2 and pl_size >= 2:
            structure_bear = phv[-1] < phv[-2]
            structure_bull = plv[-1] > plv[-2]

        double_top = double_bottom = head_shoulders = inv_head_shoulders = False
        rising_wedge = falling_wedge = False
        if ph_size >= 2:
            d1, d2, dbar2 = phv[-1], phv[-2], phb[-2]
            double_top = abs(d1 - d2) <= d2 * bt.PATTERN_TOL_PCT / 100 and (i - dbar2) <= bt.PATTERN_LOOKBACK
        if pl_size >= 2:
            d1, d2, dbar2 = plv[-1], plv[-2], plb[-2]
            double_bottom = abs(d1 - d2) <= d2 * bt.PATTERN_TOL_PCT / 100 and (i - dbar2) <= bt.PATTERN_LOOKBACK
        if ph_size >= 3:
            l_, h_, r_, b1 = phv[-3], phv[-2], phv[-1], phb[-3]
            head_shoulders = h_ > l_ and h_ > r_ and abs(l_ - r_) <= l_ * bt.PATTERN_TOL_PCT / 100 and (i - b1) <= bt.PATTERN_LOOKBACK
        if pl_size >= 3:
            l_, h_, r_, b1 = plv[-3], plv[-2], plv[-1], plb[-3]
            inv_head_shoulders = h_ < l_ and h_ < r_ and abs(l_ - r_) <= l_ * bt.PATTERN_TOL_PCT / 100 and (i - b1) <= bt.PATTERN_LOOKBACK
        if ph_size >= 3 and pl_size >= 3:
            wh1, wh3, wb1, wb3 = phv[-1], phv[-3], phb[-1], phb[-3]
            wl1, wl3, lb1, lb3 = plv[-1], plv[-3], plb[-1], plb[-3]
            if wb1 != wb3 and lb1 != lb3:
                sh = (wh1 - wh3) / (wb1 - wb3)
                sl_ = (wl1 - wl3) / (lb1 - lb3)
                rising_wedge = sh > 0 and sl_ > 0 and sl_ > sh
                falling_wedge = sh < 0 and sl_ < 0 and sh < sl_

        bearish_divergence = ph_size >= 2 and phv[-1] > phv[-2] and arr["rsi"][i] < arr["rsi"][phb[-1]]
        bullish_divergence = pl_size >= 2 and plv[-1] < plv[-2] and arr["rsi"][i] > arr["rsi"][plb[-1]]

        for side_key in (["sell"] if near_pdh else []) + (["buy"] if near_pdl else []):
            pdh_i, pdl_i, atr_i = arr["pdh"][i], arr["pdl"][i], arr["atr"][i]
            if not (pd.notna(atr_i) and pd.notna(pdh_i) and pd.notna(pdl_i)):
                continue
            entry_close = arr["close"][i]

            if side_key == "sell":
                valid_swing = ph_size >= 1 and phv[-1] > entry_close
                sl = (phv[-1] + tick_buffer) if valid_swing else (entry_close + SL_ATR_MULT * atr_i)
                risk = sl - entry_close
                tp1, tp2 = entry_close - risk * RR1, entry_close - risk * RR2
            else:
                valid_swing = pl_size >= 1 and plv[-1] < entry_close
                sl = (plv[-1] - tick_buffer) if valid_swing else (entry_close - SL_ATR_MULT * atr_i)
                risk = entry_close - sl
                tp1, tp2 = entry_close + risk * RR1, entry_close + risk * RR2
            if risk <= 0:
                continue

            # forward-scan the counterfactual outcome, independent of any real position
            tp1_filled = False
            realized_r = None
            for j in range(i + 1, min(i + 2000, n)):
                bo, bh, bl = arr["open"][j], arr["high"][j], arr["low"][j]
                target = tp2 if tp1_filled else tp1
                if side_key == "sell":
                    hit_sl, hit_tp = bh >= sl, bl <= target
                else:
                    hit_sl, hit_tp = bl <= sl, bh >= target
                if not (hit_sl or hit_tp):
                    continue
                sl_first = (abs(bo - sl) < abs(bo - target)) if (hit_sl and hit_tp) else hit_sl
                if sl_first:
                    realized_r = -0.5 if tp1_filled else -1.0
                    break
                else:
                    if tp1_filled:
                        realized_r = 0.5 * RR1 + 0.5 * RR2
                        break
                    else:
                        tp1_filled = True
                        continue
            if realized_r is None:
                continue  # ran off the end of history without resolving - drop (rare, only near the very end)

            vol_ratio = arr["volume"][i] / arr["vol_ma"][i] if pd.notna(arr["vol_ma"][i]) and arr["vol_ma"][i] > 0 else np.nan
            rows.append(dict(
                time=arr["open_time"][i], side=side_key,
                bearish_candle=int(arr["bearish_candle"][i]), bullish_candle=int(arr["bullish_candle"][i]),
                double_top=int(double_top), double_bottom=int(double_bottom),
                head_shoulders=int(head_shoulders), inv_head_shoulders=int(inv_head_shoulders),
                rising_wedge=int(rising_wedge), falling_wedge=int(falling_wedge),
                bull_flag=int(arr["bull_flag"][i]), bear_flag=int(arr["bear_flag"][i]),
                failed_breakout_pdh=int(arr["failed_breakout_pdh"][i]), failed_breakdown_pdl=int(arr["failed_breakdown_pdl"][i]),
                structure_bull=int(structure_bull), structure_bear=int(structure_bear),
                vol_confirm=int(arr["vol_confirm"][i]), vol_ratio=vol_ratio,
                momentum_bull=int(arr["momentum_bull"][i]), momentum_bear=int(arr["momentum_bear"][i]),
                htf_trend_up=int(arr["htf_trend_up"][i]),
                strong_break_up=int(arr["strong_break_up"][i]), strong_break_down=int(arr["strong_break_down"][i]),
                rsi=arr["rsi"][i],
                atr_pct=atr_i / entry_close * 100,
                hour=pd.Timestamp(arr["open_time"][i]).hour, dow=pd.Timestamp(arr["open_time"][i]).dayofweek,
                # --- Nison (Japanese Candlestick Charting Techniques): granular patterns ---
                bear_engulf=int(arr["bear_engulf"][i]), bull_engulf=int(arr["bull_engulf"][i]),
                shooting_star=int(arr["shooting_star"][i]), hammer_candle=int(arr["hammer_candle"][i]),
                evening_star=int(arr["evening_star"][i]), morning_star=int(arr["morning_star"][i]),
                bear_pin=int(arr["bear_pin"][i]), bull_pin=int(arr["bull_pin"][i]),
                doji_bear=int(arr["doji_bear"][i]), doji_bull=int(arr["doji_bull"][i]),
                harami_bear=int(arr["harami_bear"][i]), harami_bull=int(arr["harami_bull"][i]),
                harami_cross_bear=int(arr["harami_cross_bear"][i]), harami_cross_bull=int(arr["harami_cross_bull"][i]),
                dark_cloud_cover=int(arr["dark_cloud_cover"][i]), piercing_pattern=int(arr["piercing_pattern"][i]),
                three_black_crows=int(arr["three_black_crows"][i]), three_white_soldiers=int(arr["three_white_soldiers"][i]),
                tweezer_top=int(arr["tweezer_top"][i]), tweezer_bottom=int(arr["tweezer_bottom"][i]),
                # --- Elder (New Trading for a Living): Force Index, Impulse System, divergence ---
                force_index_2=arr["force_index_2"][i], force_index_13=arr["force_index_13"][i],
                impulse_green=int(arr["impulse_green"][i]), impulse_red=int(arr["impulse_red"][i]),
                bearish_divergence=int(bearish_divergence), bullish_divergence=int(bullish_divergence),
                # --- Murphy (Technical Analysis of the Financial Markets): Stochastic, Bollinger, ADX ---
                stoch_k=arr["stoch_k"][i], stoch_d=arr["stoch_d"][i],
                stoch_bull_turn=int(arr["stoch_bull_turn"][i]), stoch_bear_turn=int(arr["stoch_bear_turn"][i]),
                bb_pctb=arr["bb_pctb"][i], bb_squeeze=int(arr["bb_squeeze"][i]),
                adx=arr["adx"][i], adx_strong_trend=int(arr["adx_strong_trend"][i]),
                realized_r=realized_r, win=int(realized_r > 0),
            ))

        if i % 20000 == 0:
            print(f"  bar {i}/{n}, {len(rows)} labeled touches so far...", file=sys.stderr)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_parquet("data/btcusdt_15m.parquet").sort_values("open_time").reset_index(drop=True)
    df = bt.compute_indicators(df, zone_pct=ZONE_PCT, htf_rule=HTF_RULE)
    print(f"Building labeled dataset over {len(df)} bars...", file=sys.stderr)
    out = build_dataset(df)
    out.to_csv("data/ml_training_data.csv", index=False)
    print(f"\nDone: {len(out)} labeled touches -> data/ml_training_data.csv", file=sys.stderr)
    print(out["win"].value_counts(normalize=True), file=sys.stderr)
    print(f"mean realized_r: {out['realized_r'].mean():.4f}", file=sys.stderr)
