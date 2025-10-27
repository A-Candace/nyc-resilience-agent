# forecasting/models/linear_residual.py
import numpy as np
import pandas as pd

def forecast_linear_residual(precip_df, multiplier_by_board, alpha=0.15):
    """
    Baseline + small ‘persistence’ term:
      flood_index = (p * multiplier) + alpha * (p_prev * multiplier)
    If no previous step, the residual term is 0.
    """
    out = []
    prev_p = None
    for _, r in precip_df.iterrows():
        p = float(r["precip_mm_hr"]); ts = r["timestamp"]
        core = multiplier_by_board * p
        if prev_p is None:
            vals = core
        else:
            vals = core + alpha * (multiplier_by_board * prev_p)
        out.append(pd.DataFrame({"timestamp": ts, "unit_id": vals.index, "flood_index": vals.values}))
        prev_p = p
    return pd.concat(out, ignore_index=True)
