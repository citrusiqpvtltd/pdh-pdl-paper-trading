"""
Shared ML secondary-filter gate for backtest_pdh_pdl.step_bar's `ml_gate`
hook. Used by BOTH paper_trade.py (live) and validate_ml_filter.py
(validation) so there is exactly one implementation of the feature-row
construction - no risk of live and validation code silently diverging the
way the standalone rule-reconstruction in an earlier version of
train_ml_model.py did (see that file's history / README for the story).

The feature list here must match train_ml_model.FEATURE_COLS exactly (the
"simple" set - see that file for why the book-informed expansion was tried
and NOT adopted). If the model is ever retrained on a different feature
set, this function needs to change with it.
"""
import numpy as np
import pandas as pd


def make_ml_gate(model, feats, medians, threshold):
    def gate(side, i, arr, common):
        atr_i, entry_close = arr["atr"][i], arr["close"][i]
        if not (pd.notna(atr_i) and entry_close):
            return False
        vol_ma_i = arr["vol_ma"][i]
        vol_ratio = arr["volume"][i] / vol_ma_i if pd.notna(vol_ma_i) and vol_ma_i > 0 else np.nan
        row = pd.DataFrame([dict(
            bearish_candle=int(arr["bearish_candle"][i]), bullish_candle=int(arr["bullish_candle"][i]),
            double_top=int(common["double_top"]), double_bottom=int(common["double_bottom"]),
            head_shoulders=int(common["head_shoulders"]), inv_head_shoulders=int(common["inv_head_shoulders"]),
            rising_wedge=int(common["rising_wedge"]), falling_wedge=int(common["falling_wedge"]),
            bull_flag=int(arr["bull_flag"][i]), bear_flag=int(arr["bear_flag"][i]),
            failed_breakout_pdh=int(arr["failed_breakout_pdh"][i]), failed_breakdown_pdl=int(arr["failed_breakdown_pdl"][i]),
            structure_bull=int(common["structure_bull"]), structure_bear=int(common["structure_bear"]),
            vol_confirm=int(arr["vol_confirm"][i]), vol_ratio=vol_ratio,
            momentum_bull=int(arr["momentum_bull"][i]), momentum_bear=int(arr["momentum_bear"][i]),
            htf_trend_up=int(arr["htf_trend_up"][i]), rsi=arr["rsi"][i],
            atr_pct=atr_i / entry_close * 100,
            hour=pd.Timestamp(arr["open_time"][i]).hour, dow=pd.Timestamp(arr["open_time"][i]).dayofweek,
            side_sell=int(side == "sell"),
        )])[feats].fillna(pd.Series(medians))
        proba = model.predict_proba(row)[0, 1]
        return proba >= threshold
    return gate


def load_gate(path, threshold):
    import joblib
    d = joblib.load(path)
    return make_ml_gate(d["model"], d["features"], d["medians"], threshold), d
