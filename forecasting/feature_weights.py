# forecasting/feature_weights.py
# Static-variable weights from the user’s Random Forest paper.
# We keep (1) raw model weights, and (2) a normalized average across the two.

MODEL1 = {
    "BLD":  9.43,
    "FP":   6.10,
    "SLPE": 7.61,
    "ELEV": 5.10,
    "IMPV": 3.68,
    "FPBD": 3.22,
}

MODEL2 = {
    "BLD":  3.47,
    "FP":   3.43,
    "SLPE": 6.73,
    "ELEV": 2.12,
    "IMPV": 4.37,
    "FPBD": 4.79,
}

SELECTED_KEYS = ["BLD", "FP", "SLPE", "ELEV", "IMPV", "FPBD"]

def _normalize(d):
    s = sum(d[k] for k in SELECTED_KEYS)
    return {k: (d[k] / s) for k in SELECTED_KEYS}

WEIGHTS_MODEL1_NORM = _normalize(MODEL1)
WEIGHTS_MODEL2_NORM = _normalize(MODEL2)

# Average the normalized weights, then renormalize for exact sum=1
_avg = {k: 0.5 * (WEIGHTS_MODEL1_NORM[k] + WEIGHTS_MODEL2_NORM[k]) for k in SELECTED_KEYS}
s = sum(_avg.values())
WEIGHTS_AVG_NORM = {k: (_avg[k] / s) for k in SELECTED_KEYS}

# Human-friendly labels used in-app
WEIGHT_SETS = {
    "Random Forest (Model 1)": WEIGHTS_MODEL1_NORM,
    "Random Forest (Model 2)": WEIGHTS_MODEL2_NORM,
    "Average (Model 1 & 2)":   WEIGHTS_AVG_NORM,
}

# Map from our six feature codes to your merged board attribute names
# (from your merged CSV step in app.py)
ATTR_MAP = {
    "BLD":  "Buildings",      # number of buildings
    "FP":   "Footprint",      # sum of building footprints
    "SLPE": "Slope",          # mean percent rise
    "ELEV": "Elevation",      # mean elevation
    "IMPV": "Imperv",         # percent impervious cover
    "FPBD": "FTPperArea",     # footprint per unit area
}
