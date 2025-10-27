# forecasting/models/graph_diffusion.py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def _knn_adjacency(coords, k=6):
    """
    Build a row-stochastic kNN adjacency (numpy array NxN) from (lat,lon) coords.
    """
    pts = np.array(coords)  # [[lat,lon],...]
    tree = cKDTree(pts)
    dists, idxs = tree.query(pts, k=k+1)  # first is itself
    N = len(pts)
    W = np.zeros((N, N), dtype=float)
    for i in range(N):
        neigh = idxs[i][1:]     # drop self
        nd    = dists[i][1:]
        # inverse distance weights
        w = 1.0 / np.maximum(nd, 1e-6)
        w = w / w.sum()
        W[i, neigh] = w
    return W

def _static_similarity(static_matrix, eps=1e-6):
    """
    Cosine-similarity-based matrix in [0,1]; row-normalized.
    """
    X = np.nan_to_num(static_matrix.values.astype(float))
    # cosine sim
    norms = np.linalg.norm(X, axis=1, keepdims=True) + eps
    Xn = X / norms
    S = np.clip(Xn @ Xn.T, 0.0, 1.0)
    # row normalize
    S = S / (S.sum(axis=1, keepdims=True) + eps)
    return S

def _blend_adjacency(W_geo, W_static, beta=0.5):
    """
    Blend geographic and static-similarity graphs: W = (1-beta)*W_geo + beta*W_static
    """
    return (1.0 - beta) * W_geo + beta * W_static

def forecast_graph_diffusion(precip_df,         # ['timestamp','precip_mm_hr']
                             multiplier_by_board,  # Series indexed by unit_id
                             board_centroids_df,   # DataFrame: index=unit_id, cols=['lat','lon']
                             static_matrix,        # DataFrame: index=unit_id, static columns
                             k_geo=6, beta=0.5, diffusion_steps=2, gamma=0.3):
    """
    Graph diffusion forecaster (fast, numpy-only):
      1) base = precip * multiplier(board)
      2) diffuse base across blended graph W for 'diffusion_steps' with decay gamma
         X_{t}^{(0)} = base
         X_{t}^{(s+1)} = (1-gamma)*X_{t}^{(s)} + gamma*W @ X_{t}^{(s)}
    """
    unit_ids = multiplier_by_board.index.to_list()
    uid_to_ix = {u:i for i,u in enumerate(unit_ids)}

    # Build W once
    coords = board_centroids_df.loc[unit_ids, ["lat","lon"]].values
    W_geo = _knn_adjacency(coords, k=k_geo)
    W_static = _static_similarity(static_matrix.loc[unit_ids])
    W = _blend_adjacency(W_geo, W_static, beta=beta)

    out = []
    for _, r in precip_df.iterrows():
        p = float(r["precip_mm_hr"]); ts = r["timestamp"]
        base = (multiplier_by_board * p).values  # Nx1
        X = base.copy()
        for _ in range(diffusion_steps):
            X = (1.0 - gamma) * X + gamma * (W @ X)
        df = pd.DataFrame({"timestamp": ts, "unit_id": unit_ids, "flood_index": X})
        out.append(df)
    return pd.concat(out, ignore_index=True)
