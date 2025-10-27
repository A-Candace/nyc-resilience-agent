# forecasting/static_modifiers.py
import numpy as np
import pandas as pd

def compute_static_matrix(boards, attr_map):
    """
    Build a DataFrame: index=unit_id, columns = attr_map.values()
    Pulls attributes from boards[i]['attrs'] (as your app already stores).
    Missing -> NaN.
    """
    rows = []
    for b in boards:
        uid = b["unit_id"]
        attrs = b.get("attrs", {})
        row = {"unit_id": uid}
        for code, col in attr_map.items():
            row[col] = attrs.get(col, np.nan)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("unit_id")
    return df

def robust_scale(df, q_low=0.05, q_high=0.95):
    """
    Robust min-max scaling using the 5th–95th percentiles to reduce outlier effects.
    Result is ~[0,1] per column; NaNs remain NaN.
    """
    scaled = df.copy()
    for c in scaled.columns:
        col = scaled[c].astype(float)
        lo = col.quantile(q_low)
        hi = col.quantile(q_high)
        if pd.isna(lo) or pd.isna(hi) or hi <= lo:
            # fallback to simple min–max if needed
            lo, hi = col.min(), col.max()
        if pd.isna(lo) or pd.isna(hi) or hi <= lo:
            scaled[c] = 0.0  # all missing or constant
        else:
            scaled[c] = (col - lo) / (hi - lo)
            scaled[c] = scaled[c].clip(0, 1)
    return scaled

def static_risk_weight_per_board(static_df_scaled, weight_dict, attr_map):
    """
    Returns a pd.Series (index=unit_id) with a weighted score in [0,1].
    We only combine the columns referenced in attr_map (six features).
    """
    cols = [attr_map[k] for k in weight_dict.keys()]
    # Ensure all columns exist
    for c in cols:
        if c not in static_df_scaled.columns:
            static_df_scaled[c] = 0.0
    # Weighted sum
    out = pd.Series(0.0, index=static_df_scaled.index, dtype=float)
    for code, w in weight_dict.items():
        col = attr_map[code]
        out = out + w * static_df_scaled[col].fillna(0.0)
    # stays in [0,1] because weights sum to 1 and columns are [0,1]
    return out

def to_multiplier(score_series, low=0.7, high=1.5):
    """
    Convert risk score in [0,1] into a multiplicative factor [low, high].
    e.g. score=0 -> 0.7x; score=1 -> 1.5x. Adjust as you like.
    """
    return low + (high - low) * score_series
