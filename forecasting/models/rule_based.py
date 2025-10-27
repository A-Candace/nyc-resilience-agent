# forecasting/models/rule_based.py
import numpy as np
import pandas as pd

def forecast_rule_based(precip_df,  # DataFrame: columns=['timestamp','precip_mm_hr']
                        multiplier_by_board  # Series indexed by unit_id
                        ):
    """
    Simple forecast: flood_index = precip_mm_hr * multiplier(board)
    For each timestamp, produce a row per board.
    Returns long DataFrame with columns ['timestamp','unit_id','flood_index'].
    """
    out = []
    for _, r in precip_df.iterrows():
        p = float(r["precip_mm_hr"])
        ts = r["timestamp"]
        vals = multiplier_by_board * p
        df = pd.DataFrame({"timestamp": ts,
                           "unit_id": vals.index,
                           "flood_index": vals.values})
        out.append(df)
    return pd.concat(out, ignore_index=True)
