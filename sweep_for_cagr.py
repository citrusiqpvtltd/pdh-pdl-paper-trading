"""
Broader parameter sweep of the validated rule-based engine, ranked by CAGR
(annualized return) rather than raw profit factor - the user's actual goal
is a 12%/year (1%/month) target, and the current tuned config only manages
~0.87%/year despite a real, positive profit factor (1.483).

Structural params (zone_pct, htf_rule) require recomputing indicators;
behavioral params (score_threshold, RR, SL method, TP2, HTF filter) reuse
the same precomputed dataframe.
"""
import itertools
import time

import numpy as np
import pandas as pd

import backtest_pdh_pdl as bt

STRUCTURAL_GRID = [
    dict(zone_pct=0.2, htf_rule="4h"),
    dict(zone_pct=0.3, htf_rule="4h"),
    dict(zone_pct=0.4, htf_rule="4h"),
    dict(zone_pct=0.2, htf_rule="1h"),
    dict(zone_pct=0.3, htf_rule="1h"),
    dict(zone_pct=0.4, htf_rule="1h"),
]

BEHAVIORAL_GRID = dict(
    score_threshold=[3, 4, 5],
    rr=[(1.0, 2.0), (1.5, 3.0), (2.0, 4.0)],
    sl_method=["Swing", "ATR"],
    tp2_enabled=[True, False],
    use_htf_filter=[True, False],
)

MIN_TRADES = 80


def summarize(trades, final_equity):
    if not trades or len(trades) < 5:
        return None
    tdf = pd.DataFrame(trades)
    total_pnl = tdf["pnl"].sum()
    wins = tdf[tdf["pnl"] > 0]
    losses = tdf[tdf["pnl"] <= 0]
    win_rate = len(wins) / len(tdf) * 100
    gross_profit = wins["pnl"].sum()
    gross_loss = -losses["pnl"].sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    years = (tdf["exit_time"].max() - tdf["entry_time"].min()).days / 365.25
    years = max(years, 0.1)
    final_ratio = final_equity / bt.INITIAL_CAPITAL
    cagr = (final_ratio ** (1 / years) - 1) * 100 if final_ratio > 0 else -100.0

    equity_curve = bt.INITIAL_CAPITAL + tdf["pnl"].cumsum()
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    max_dd_pct = (drawdown / running_max).min() * 100

    tdf["year"] = tdf["exit_time"].dt.year
    year_pnl = tdf.groupby("year")["pnl"].sum()
    positive_years = int((year_pnl > 0).sum())
    years_present = int(year_pnl.shape[0])

    rt = tdf.groupby(["side", "entry_time"]).agg(pnl=("pnl", "sum")).reset_index()

    return dict(
        num_trades=len(tdf), round_trips=len(rt), win_rate=win_rate,
        total_pnl_pct=total_pnl / bt.INITIAL_CAPITAL * 100, profit_factor=pf,
        cagr_pct=cagr, max_dd_pct=max_dd_pct, positive_years=positive_years,
        years_present=years_present, years=years,
    )


def main():
    keys = list(BEHAVIORAL_GRID.keys())
    combos = list(itertools.product(*BEHAVIORAL_GRID.values()))
    total_runs = len(STRUCTURAL_GRID) * len(combos)
    print(f"Structural variants: {len(STRUCTURAL_GRID)}, behavioral combos: {len(combos)}, total runs: {total_runs}")

    df_raw = pd.read_parquet("data/btcusdt_15m.parquet").sort_values("open_time").reset_index(drop=True)
    results = []
    t_start = time.time()
    run_idx = 0

    for struct in STRUCTURAL_GRID:
        df = bt.compute_indicators(df_raw.copy(), zone_pct=struct["zone_pct"], htf_rule=struct["htf_rule"])
        for combo in combos:
            params = dict(zip(keys, combo))
            rr1, rr2 = params.pop("rr")
            run_idx += 1
            trades, final_equity = bt.run_backtest(
                df, score_threshold=params["score_threshold"], rr1=rr1, rr2=rr2,
                sl_method=params["sl_method"], tp2_enabled=params["tp2_enabled"],
                use_htf_filter=params["use_htf_filter"], enable_breakout_protection=True,
            )
            summ = summarize(trades, final_equity)
            if summ is None or summ["num_trades"] < MIN_TRADES:
                continue
            row = dict(struct, rr1=rr1, rr2=rr2, **params, **summ)
            results.append(row)
            if run_idx % 50 == 0:
                elapsed = time.time() - t_start
                print(f"  [{run_idx}/{total_runs}] elapsed {elapsed:.0f}s", flush=True)

    rdf = pd.DataFrame(results)
    rdf.to_csv("sweep_cagr_results.csv", index=False)
    print(f"\nSaved {len(rdf)} results ({time.time()-t_start:.0f}s total)")

    print(f"\n=== Top 20 by CAGR (min {MIN_TRADES} trades) ===")
    print(rdf.sort_values("cagr_pct", ascending=False).head(20).to_string(index=False))

    robust = rdf[(rdf["positive_years"] >= rdf["years_present"] - 1) & (rdf["profit_factor"] >= 1.1)]
    print(f"\n=== Robust (PF>=1.1, positive in all-but-1 years present): {len(robust)} of {len(rdf)} ===")
    if len(robust):
        print(robust.sort_values("cagr_pct", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
