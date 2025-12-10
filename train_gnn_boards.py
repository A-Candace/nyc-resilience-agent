# -*- coding: utf-8 -*-
"""
Train a GNN to predict community-board hourly flood depth from aggregated precipitation + statics.

Pipeline:
  1) Load Community Board polygons.
  2) Load radar precipitation (Excel w/ multiple sheets) + metadata (lat/lon per sheet).
  3) Load FloodNet water depth CSVs + deployment metadata (lat/lon).
  4) Aggregate BOTH signals to hourly, board-level:
        - average of sensors/radars within polygon
        - if none inside, use nearest to centroid
  5) Merge static variables -> risk score -> static multiplier m_b
  6) Build graph over boards (k-NN on centroids) -> edge_index
  7) Build temporal graph snapshots (one per hour): X = [precip, m_b] (+ optional extras)
  8) 80/20 time split; masked loss for nodes w/ labels
  9) Train 2-layer GCN; report R^2 on holdout

Requires: shapely>=2, pyproj, torch, torch_geometric, pandas, numpy, openpyxl
"""

import os
import math
import glob
import json
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from shapely.geometry import shape as shp_shape, Point as ShpPoint
from shapely.ops import transform as shp_transform

from pyproj import Transformer, CRS

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from sklearn.metrics import r2_score

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------
# CONFIG
# -----------------------
CONFIG = {
    # Community Boards geometry (prefer GeoJSON). If SHAPEFILE_ZIP is set, script will try it first.
    "BOARDS_GEOJSON": "nyc_community_boards.geojson",
    "SHAPEFILE_ZIP": None,  # e.g., "NYC_Community_Boards.zip"  (set to None to skip)

    # Radar precipitation
    "RADAR_EXCEL": "radar_hourly.xlsx",       # one sheet per radar
    "RADAR_META_CSV": "radar_metadata.csv",   # columns: sheet, lat, lon
    "RADAR_TIME_COL": "time",
    "RADAR_VALUE_COLS_TRY": ["precip_mm_hr", "precip"],

    # FloodNet depths (CSV export root + deployment metadata CSV in root)
    "FLOODNET_ROOT": "floodnet_csv_root",     # contains subfolders [slug]/[slug]+YYYY-MM.csv
    "FLOODNET_DEPLOYMENT_CSV": "floodnet_csv_root/deployment-info.csv",
    "FLOODNET_TIME_COL": "time",              # ISO8601 UTC
    "FLOODNET_DEPTH_COL": "depth_proc_mm",    # recommended field

    # Static attributes (per board)
    "STATIC_CSV": "DataForBoxPlots.csv",
    "STATIC_CB_COL": "CB_id",  # to match to board boro_cd/ID after normalization
    # A minimal Model 1 weight set (edit to your paper's values)
    "STATIC_WEIGHTS": {        # name -> weight (positive means risk-raising)
        "Buildings": 0.25,
        "Footprint": 0.25,
        "Imperv": 0.25,
        "Slope": -0.10,
        "Elevation": -0.10,
        "Commuting": 0.15
    },
    "STATIC_MULT_RANGE": (0.7, 1.5),  # to_multiplier(low, high)

    # Time window (inclusive start, inclusive end for selection; end hour included)
    "START": "2023-01-01 00:00:00",
    "END":   "2025-06-01 23:00:00",

    # Graph construction
    "K_NEIGHBORS": 6,           # kNN on centroids; set to 0 to use polygon adjacency (slower)
    "USE_POLY_ADJ": False,      # True to build edges by polygon touches()

    # Training
    "BATCH_SIZE": 16,           # batch of hourly snapshots
    "EPOCHS": 40,
    "LR": 1e-3,
    "HIDDEN_DIM": 64,
    "SEED": 42,

    # Outlier handling (winsorize)
    "PRECIP_UPPER_Q": 0.999,
    "DEPTH_UPPER_Q": 0.999,

    # Device
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}


# -----------------------
# Helpers
# -----------------------
def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_cb_id(val) -> str:
    if val is None:
        return ""
    try:
        return str(int(float(val)))
    except Exception:
        s = str(val)
        digits = "".join(c for c in s if c.isdigit())
        return digits or s


def load_boards(config: dict):
    """
    Return: boards list of dicts:
      {unit_id, name, geom(Shapely), feature_geom(geojson), attrs{}, centroid(lat,lon)}
    """
    # 1) Try shapefile .zip if provided
    shp_zip = config.get("SHAPEFILE_ZIP")
    boards = []
    if shp_zip and os.path.exists(shp_zip):
        boards = _load_boards_from_shapefile_zip(shp_zip)
    # 2) Fallback to GeoJSON
    if (not boards) and os.path.exists(config["BOARDS_GEOJSON"]):
        with open(config["BOARDS_GEOJSON"], "r", encoding="utf-8") as f:
            gj = json.load(f)
        for i, feat in enumerate(gj.get("features", [])):
            props = feat.get("properties", {}) or {}
            unit_id = props.get("BoroCD") or props.get("boro_cd") or props.get("cd") or f"CB_{i:03d}"
            name = props.get("cd_name") or props.get("name") or str(unit_id)
            geom = shp_shape(feat.get("geometry"))
            if geom.is_empty:
                continue
            c = geom.representative_point()
            boards.append({
                "unit_id": str(unit_id),
                "name": str(name),
                "geom": geom,
                "feature_geom": feat.get("geometry"),
                "attrs": dict(props),
                "centroid": (float(c.y), float(c.x))
            })
    if not boards:
        raise FileNotFoundError("No boards geometry found. Provide BOARDS_GEOJSON or SHAPEFILE_ZIP.")
    # Normalize id + compute centroid
    out = []
    for b in boards:
        uid = normalize_cb_id(b["attrs"].get("BoroCD") or b["attrs"].get("boro_cd") or b["unit_id"])
        c = b["geom"].representative_point()
        out.append({**b, "unit_id": uid, "centroid": (float(c.y), float(c.x))})
    return out


def _load_boards_from_shapefile_zip(zip_path: str):
    from zipfile import ZipFile
    import tempfile
    import shapefile as pyshp  # pyshp

    tmp = tempfile.mkdtemp(prefix="cb_shp_")
    with ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp)

    shp_paths = []
    for root, _, files in os.walk(tmp):
        for f in files:
            if f.lower().endswith(".shp"):
                shp_paths.append(os.path.join(root, f))
    if not shp_paths:
        return []

    shp_path = shp_paths[0]
    prj_path = os.path.splitext(shp_path)[0] + ".prj"
    src_crs = None
    if os.path.exists(prj_path):
        try:
            with open(prj_path, "r") as f:
                wkt = f.read()
            src_crs = CRS.from_wkt(wkt)
        except Exception:
            src_crs = None

    transformer = None
    if src_crs and (src_crs.to_epsg() != 4326):
        transformer = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)

    r = pyshp.Reader(shp_path)
    fields = [f[0] for f in r.fields if f[0] != "DeletionFlag"]

    boards = []
    for idx, sr in enumerate(r.shapeRecords()):
        rec = {fields[i]: sr.record[i] for i in range(len(fields))}
        unit_id = rec.get("BoroCD") or rec.get("boro_cd") or rec.get("cd") or f"CB_{idx:03d}"
        name = rec.get("cd_name") or rec.get("name") or str(unit_id)
        geom_geojson = sr.shape.__geo_interface__
        geom = shp_shape(geom_geojson)
        if transformer is not None:
            geom = shp_transform(lambda x, y, z=None: transformer.transform(x, y), geom)
        if geom.is_empty:
            continue
        c = geom.representative_point()
        boards.append({
            "unit_id": str(unit_id),
            "name": str(name),
            "geom": geom,
            "feature_geom": geom_geojson,
            "attrs": dict(rec),
            "centroid": (float(c.y), float(c.x))
        })
    return boards


def load_radar_hourly(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      radar_meta: DataFrame columns [sheet, lat, lon]
      radar_long: DataFrame columns [time (UTC naive), sheet, precip_mm_hr]
    """
    meta = pd.read_csv(config["RADAR_META_CSV"])
    meta["sheet"] = meta["sheet"].astype(str)

    xls = pd.ExcelFile(config["RADAR_EXCEL"])
    frames = []
    for sheet in xls.sheet_names:
        if sheet not in set(meta["sheet"]):
            # allow, but warn
            print(f"[radar] WARNING: sheet '{sheet}' missing from radar_metadata.csv; skipping.")
            continue
        df = xls.parse(sheet)
        # Normalize columns
        time_col = config["RADAR_TIME_COL"]
        value_col = next((c for c in config["RADAR_VALUE_COLS_TRY"] if c in df.columns), None)
        if value_col is None:
            raise ValueError(f"Sheet '{sheet}' missing precip column, tried {config['RADAR_VALUE_COLS_TRY']}.")

        df = df[[time_col, value_col]].copy()
        df[time_col] = pd.to_datetime(df[time_col], utc=True)  # assume times are UTC or tag as UTC
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        # Resample to hourly (mean), keep continuous index
        df = df.set_index(time_col).sort_index().resample("1H").mean()
        df.columns = ["precip_mm_hr"]
        df["sheet"] = sheet
        frames.append(df.reset_index())

    if not frames:
        raise ValueError("No radar sheets loaded.")
    radar = pd.concat(frames, ignore_index=True)
    # clip outliers (upper tail)
    upper = radar["precip_mm_hr"].quantile(CONFIG["PRECIP_UPPER_Q"])
    radar["precip_mm_hr"] = radar["precip_mm_hr"].clip(lower=0, upper=upper)
    return meta, radar


def load_floodnet_hourly(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      deploy: DataFrame with [slug, lat, lon, date_deployed, ...]
      depth:  DataFrame with columns [time (UTC), slug, depth_m]
    """
    deploy = pd.read_csv(config["FLOODNET_DEPLOYMENT_CSV"])
    # normalize required cols
    if "slug" not in deploy.columns:
        # sometimes 'deployment_id' is used; use name as slug if needed
        if "deployment_id" in deploy.columns:
            deploy = deploy.rename(columns={"deployment_id": "slug"})
        else:
            raise ValueError("deployment-info.csv must have a 'slug' (or 'deployment_id') column.")
    for c in ("latitude", "longitude"):
        if c not in deploy.columns:
            raise ValueError("deployment-info.csv must include 'latitude' and 'longitude'.")

    deploy["slug"] = deploy["slug"].astype(str)

    # Load all monthly CSVs for each slug
    depth_frames = []
    root = config["FLOODNET_ROOT"].rstrip("/")

    for slug in deploy["slug"].unique():
        pfx = os.path.join(root, slug)
        if not os.path.isdir(pfx):
            # not all slugs in metadata exist locally; skip
            continue
        # pattern: [slug]/[slug]+YYYY-MM.csv
        pattern = os.path.join(pfx, f"{slug}+*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            continue
        for f in files:
            df = pd.read_csv(f)
            # Keep columns
            tcol = config["FLOODNET_TIME_COL"]
            dcol = config["FLOODNET_DEPTH_COL"]
            if tcol not in df.columns or dcol not in df.columns:
                # skip month if malformed
                continue
            df = df[[tcol, dcol]].copy()
            df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
            df[dcol] = pd.to_numeric(df[dcol], errors="coerce")
            df = df.dropna(subset=[tcol])
            # Resample to hourly mean; convert mm -> m
            df = df.set_index(tcol).sort_index().resample("1H").mean()
            df["slug"] = slug
            df["depth_m"] = (df[dcol] / 1000.0).clip(lower=0)  # no negative depth
            df = df.drop(columns=[dcol]).reset_index()
            depth_frames.append(df)

    if not depth_frames:
        raise ValueError("No FloodNet monthly CSVs found under FLOODNET_ROOT.")
    depth = pd.concat(depth_frames, ignore_index=True)

    # Winsorize outliers
    upper = depth["depth_m"].quantile(CONFIG["DEPTH_UPPER_Q"])
    depth["depth_m"] = depth["depth_m"].clip(lower=0, upper=upper)

    return deploy, depth


def build_point_to_board_mapper(boards: List[dict]) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    """
    Precompute:
      - 'inside': for each board uid, which point indices are inside (computed per set of points later)
      We can't precompute for unknown points, so we return helpers instead.
    """
    # We’ll use the boards list for direct containment checks later, but we can precompute centroids.
    centroids = {b["unit_id"]: b["centroid"] for b in boards}
    return centroids


def match_points_to_boards(points_df: pd.DataFrame, boards: List[dict], id_col: str) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, str]]:
    """
    points_df: columns [id_col, lat, lon]
    Returns:
      points_df (same order),
      inside_map: board_uid -> list of ids inside,
      nearest_map: board_uid -> single nearest id (fallback)
    """
    # Build shapely point objects
    pt_geom = {row[id_col]: ShpPoint(float(row["lon"]), float(row["lat"])) for _, row in points_df.iterrows()}
    inside_map = {b["unit_id"]: [] for b in boards}
    for b in boards:
        geom = b["geom"]
        for pid, P in pt_geom.items():
            if geom.contains(P):
                inside_map[b["unit_id"]].append(pid)

    # Nearest fallback per board (to centroid)
    nearest_map = {}
    for b in boards:
        if inside_map[b["unit_id"]]:
            nearest_map[b["unit_id"]] = inside_map[b["unit_id"]][0]
            continue
        latc, lonc = b["centroid"]
        best, bestd = None, 1e18
        for _, r in points_df.iterrows():
            d = haversine_km(latc, lonc, float(r["lat"]), float(r["lon"]))
            if d < bestd:
                bestd = d; best = r[id_col]
        nearest_map[b["unit_id"]] = best
    return points_df, inside_map, nearest_map


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def aggregate_points_to_board_hourly(
    times: pd.DatetimeIndex,
    value_long: pd.DataFrame,   # columns [time, point_id, value]
    points_df: pd.DataFrame,    # columns [point_id, lat, lon]
    boards: List[dict],
    id_col: str,
    value_col: str
) -> pd.DataFrame:
    """
    Returns board-hour table: columns [time, unit_id, value]
    Averaging all points inside; if none, use nearest point for that board.
    """
    # Map points to boards
    _, inside_map, nearest_map = match_points_to_boards(points_df, boards, id_col=id_col)

    # Pivot point series for fast lookup: index=time, columns=point_id
    pivot = value_long.pivot_table(index="time", columns=id_col, values=value_col, aggfunc="mean")
    pivot = pivot.reindex(times).sort_index()  # ensure full hourly index

    rows = []
    for uid in [b["unit_id"] for b in boards]:
        inside = inside_map[uid]
        if inside:
            vals = pivot[inside].mean(axis=1)
        else:
            nearest_id = nearest_map[uid]
            vals = pivot[nearest_id]
        out = pd.DataFrame({"time": pivot.index, "unit_id": uid, value_col: vals.to_numpy()})
        rows.append(out)
    agg = pd.concat(rows, ignore_index=True)
    return agg


def load_and_scale_static(config: dict, boards: List[dict]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Returns:
      static_scaled: index=unit_id, scaled columns as in CSV intersecting the weight keys
      multiplier:    index=unit_id float in [low, high]
    """
    df = pd.read_csv(config["STATIC_CSV"])
    df[config["STATIC_CB_COL"]] = df[config["STATIC_CB_COL"]].apply(normalize_cb_id)
    df = df.set_index(config["STATIC_CB_COL"])

    # Align to boards
    uids = [normalize_cb_id(b["attrs"].get("BoroCD") or b["attrs"].get("boro_cd") or b["unit_id"]) for b in boards]
    sub = df.loc[df.index.intersection(uids)].copy()

    # pick weighted columns
    weight_keys = list(CONFIG["STATIC_WEIGHTS"].keys())
    cols = [c for c in weight_keys if c in sub.columns]
    if not cols:
        # fallback: empty static
        static_scaled = pd.DataFrame(0.0, index=np.unique(uids), columns=["_zeros_"])
        mult = pd.Series(1.0, index=np.unique(uids))
        return static_scaled, mult

    # robust scale to ~[0,1] (median/IQR)
    def robust_minmax(col: pd.Series):
        q1, q3 = col.quantile(0.25), col.quantile(0.75)
        iqr = (q3 - q1) if (q3 > q1) else 1.0
        z = (col - col.median()) / (iqr if iqr != 0 else 1.0)
        # squash to [0,1] via logistic
        return 1 / (1 + np.exp(-z))

    scaled = sub[cols].apply(robust_minmax, axis=0)

    # risk score = sum(w_i * x_i)
    weights = pd.Series({k: CONFIG["STATIC_WEIGHTS"][k] for k in cols})
    risk = (scaled * weights).sum(axis=1)
    # normalize risk to [0,1]
    rmin, rmax = risk.min(), risk.max()
    if rmax > rmin:
        r01 = (risk - rmin) / (rmax - rmin)
    else:
        r01 = pd.Series(0.5, index=risk.index)

    low, high = CONFIG["STATIC_MULT_RANGE"]
    mult = low + (high - low) * r01
    # reindex to all boards; fill missing to neutral 1.0
    mult = mult.reindex(uids).fillna(1.0)
    scaled = scaled.reindex(uids).fillna(0.0)

    static_scaled = scaled.copy()
    static_scaled.index.name = "unit_id"
    static_scaled = static_scaled.reset_index().rename(columns={static_scaled.index.name: "unit_id"}).set_index("unit_id")
    mult.index = static_scaled.index
    return static_scaled, mult


def build_knn_edges(boards: List[dict], k: int = 6) -> np.ndarray:
    """
    Return edge_index (2,E) for k-NN over board centroids (undirected).
    """
    coords = np.array([[b["centroid"][0], b["centroid"][1]] for b in boards])  # lat, lon
    n = len(boards)
    dmat = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = haversine_km(coords[i,0], coords[i,1], coords[j,0], coords[j,1])
            dmat[i,j] = dmat[j,i] = d
    edge_set = set()
    for i in range(n):
        idx = np.argsort(dmat[i])[:k+1]  # include self; drop later
        for j in idx:
            if i == j: continue
            edge_set.add((i, j)); edge_set.add((j, i))
    edges = np.array(list(edge_set)).T  # shape (2, E)
    return edges


def build_polygon_adjacency_edges(boards: List[dict]) -> np.ndarray:
    """
    Edge when polygons touch. (Slower)
    """
    n = len(boards)
    edge_set = set()
    for i in range(n):
        gi = boards[i]["geom"]
        for j in range(i+1, n):
            if gi.touches(boards[j]["geom"]):
                edge_set.add((i, j)); edge_set.add((j, i))
    edges = np.array(list(edge_set)).T if edge_set else np.zeros((2,0), dtype=int)
    return edges


# -----------------------
# Model
# -----------------------
class GNNModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x.squeeze(-1)  # (N,) predicted depth_m


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    diff = (pred - target)**2
    diff = diff[mask]
    return diff.mean() if diff.numel() > 0 else torch.tensor(0.0, device=pred.device)


# -----------------------
# Main
# -----------------------
def main():
    cfg = CONFIG
    set_seed(cfg["SEED"])

    # 1) Load boards
    boards = load_boards(cfg)
    board_ids = [b["unit_id"] for b in boards]
    uid_to_idx = {uid: i for i, uid in enumerate(board_ids)}

    # 2) Load radar (hourly per radar)
    radar_meta, radar_long = load_radar_hourly(cfg)
    radar_meta = radar_meta.rename(columns={"sheet": "radar_id"})
    radar_long = radar_long.rename(columns={"sheet": "radar_id"})
    # Constrain to global window
    t_start = pd.Timestamp(cfg["START"], tz="UTC")
    t_end   = pd.Timestamp(cfg["END"], tz="UTC")
    full_hours = pd.date_range(t_start, t_end, freq="1H")
    radar_long = radar_long[(radar_long["time"] >= t_start) & (radar_long["time"] <= t_end)]

    # 3) Load FloodNet (hourly per slug)
    deploy, depth_long = load_floodnet_hourly(cfg)
    depth_long = depth_long[(depth_long["time"] >= t_start) & (depth_long["time"] <= t_end)]

    # 4) Aggregate to boards (hourly)
    radar_points = radar_meta[["radar_id", "lat", "lon"]].copy()
    radar_points = radar_points.rename(columns={"radar_id": "point_id"})

    radar_agg = aggregate_points_to_board_hourly(
        full_hours,
        value_long=radar_long.rename(columns={"radar_id": "point_id", "precip_mm_hr": "value"}),
        points_df=radar_points,
        boards=boards,
        id_col="point_id",
        value_col="value"
    ).rename(columns={"value": "precip_mm_hr"})

    # FloodNet points (use deployment lat/lon)
    if not {"slug", "latitude", "longitude"}.issubset(deploy.columns):
        raise ValueError("deployment-info.csv must include slug, latitude, longitude.")

    flood_points = deploy[["slug", "latitude", "longitude"]].rename(columns={"latitude": "lat", "longitude": "lon"}).copy()
    flood_points = flood_points.dropna(subset=["lat", "lon"])
    flood_points["slug"] = flood_points["slug"].astype(str)
    # Only keep slugs we actually loaded
    used_slugs = set(depth_long["slug"].unique().tolist())
    flood_points = flood_points[flood_points["slug"].isin(used_slugs)].rename(columns={"slug": "point_id"})

    depth_agg = aggregate_points_to_board_hourly(
        full_hours,
        value_long=depth_long.rename(columns={"slug": "point_id", "depth_m": "value"}),
        points_df=flood_points,
        boards=boards,
        id_col="point_id",
        value_col="value"
    ).rename(columns={"value": "depth_m"})

    # 5) Merge static + multiplier m_b
    static_scaled, mult = load_and_scale_static(cfg, boards)  # index=unit_id
    mult = mult.reindex(board_ids).fillna(1.0)

    # 6) Join precip + depth by time,unit_id
    df = radar_agg.merge(depth_agg, on=["time", "unit_id"], how="left")
    # Attach multiplier as a column
    df["m_b"] = df["unit_id"].map(mult.to_dict())
    # Optional: features — you can expand here
    # Handle missing precip (should be rare after nearest fallback)
    df["precip_mm_hr"] = df["precip_mm_hr"].fillna(0.0)

    # Respect “don’t train before a sensor existed”:
    # For each board/hour, if its board-level depth is NaN (no sensor coverage that hour), we simply mask it out.
    # That naturally avoids pre-start periods or downtime.

    # 7) Build graph edges (stable across time)
    if cfg["USE_POLY_ADJ"]:
        edge_index_np = build_polygon_adjacency_edges(boards)
    else:
        edge_index_np = build_knn_edges(boards, k=cfg["K_NEIGHBORS"])
    if edge_index_np.size == 0:
        raise RuntimeError("Empty graph edges. Check geometry or switch K_NEIGHBORS/USE_POLY_ADJ.")
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    # 8) Create temporal snapshots (one Data per hour)
    #    Node order must match board_ids order for every snapshot.
    #    X = [precip, precip*m_b, m_b]  (precip-multiplied channel helps the model emphasize precip)
    #    y = depth_m
    hours = pd.to_datetime(sorted(df["time"].astype(str).unique())).to_pydatetime()
    hour_to_df = {h: df[df["time"] == pd.Timestamp(h, tz="UTC")].copy() for h in hours}

    snapshots: List[Data] = []
    for h in hours:
        d = hour_to_df[h]
        # build node features in board order
        m_lookup = d.set_index("unit_id")["m_b"]
        p_lookup = d.set_index("unit_id")["precip_mm_hr"]
        y_lookup = d.set_index("unit_id")["depth_m"]  # may have NaN

        precip = np.array([p_lookup.get(uid, 0.0) for uid in board_ids], dtype=np.float32)
        mb     = np.array([m_lookup.get(uid, 1.0) for uid in board_ids], dtype=np.float32)
        xm     = precip * mb
        X = np.stack([precip, xm, mb], axis=1)  # (N,3)

        y = np.array([y_lookup.get(uid, np.nan) for uid in board_ids], dtype=np.float32)
        mask = ~np.isnan(y)
        y[np.isnan(y)] = 0.0  # filler; won’t count in loss via mask

        data = Data(
            x=torch.tensor(X, dtype=torch.float32),
            edge_index=edge_index,
            y=torch.tensor(y, dtype=torch.float32),
        )
        data.train_mask = torch.tensor(mask, dtype=torch.bool)  # both train/test use masks per snapshot
        data.time_index = h  # for split
        snapshots.append(data)

    # 9) 80/20 time split (by hours)
    n = len(snapshots)
    split_idx = int(0.8 * n)
    train_snaps = snapshots[:split_idx]
    test_snaps  = snapshots[split_idx:]

    train_loader = DataLoader(train_snaps, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    test_loader  = DataLoader(test_snaps,  batch_size=CONFIG["BATCH_SIZE"], shuffle=False)

    # 10) Train
    device = torch.device(CONFIG["DEVICE"])
    model = GNNModel(in_dim=3, hidden_dim=CONFIG["HIDDEN_DIM"], out_dim=1, dropout=0.1).to(device)
    opt = optim.Adam(model.parameters(), lr=CONFIG["LR"])

    for epoch in range(CONFIG["EPOCHS"]):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            pred = model(batch)  # (batch*N,) due to PyG batching with shared edge_index per snapshot
            # PyG concatenates graphs; train_mask refers to concatenated nodes
            loss = masked_mse(pred, batch.y, batch.train_mask)
            loss.backward()
            opt.step()
            total_loss += loss.item(); batches += 1
        avg_loss = total_loss / max(batches, 1)
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d} | Train MSE: {avg_loss:.6f}")

    # 11) Evaluate R^2 on test (concatenate all available labels)
    model.eval()
    y_true_all = []
    y_pred_all = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            mask = batch.train_mask  # reuse mask as "has label"
            y_true_all.append(batch.y[mask].cpu().numpy())
            y_pred_all.append(pred[mask].cpu().numpy())
    if y_true_all:
        y_true = np.concatenate(y_true_all, axis=0)
        y_pred = np.concatenate(y_pred_all, axis=0)
        r2 = r2_score(y_true, y_pred) if y_true.size > 0 else np.nan
        print(f"\nTest R² (node-hours with labels): {r2:.4f}  | Count: {y_true.size}")
    else:
        print("\nNo labeled test samples found (check sensor coverage/time window).")

    # 12) Small report
    print(f"\nSamples: hours={len(hours)}, boards={len(board_ids)}, train_hours={len(train_snaps)}, test_hours={len(test_snaps)}")
    print(f"Graph: nodes={len(board_ids)}, edges={edge_index.size(1)} (directed pairs)")
    print(f"Features: X=[precip, precip*m_b, m_b]; Static multiplier range = ({mult.min():.3f}, {mult.max():.3f})")


if __name__ == "__main__":
    main()
