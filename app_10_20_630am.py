from io import BytesIO
import os
import math
import json
from datetime import datetime, timedelta, timezone, date

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from dotenv import load_dotenv

# System helpers
import tempfile
from zipfile import ZipFile

# Optional: ArcGIS Python API (for hosted feature layers — not required here)
try:
    from arcgis.gis import GIS
    ARC_GIS_AVAILABLE = True
except Exception:
    ARC_GIS_AVAILABLE = False

# ---------- ZIP helper that fixes non-seekable blobs (CRITICAL CHANGE) ----------
def _zipfile_from_any(obj):
    """
    Return a ZipFile that works for: path string/PathLike, UploadedFile/_Blob, or raw bytes.
    """
    # Path on disk
    if isinstance(obj, (str, os.PathLike)):
        return ZipFile(obj)

    # File-like / blob: read to bytes and wrap in BytesIO
    if hasattr(obj, "read"):
        try:
            obj.seek(0)
        except Exception:
            pass
        data = obj.read()
        return ZipFile(BytesIO(data))

    # Already bytes-like (edge case)
    if isinstance(obj, (bytes, bytearray)):
        return ZipFile(BytesIO(obj))

    raise TypeError("Unsupported ZIP input type for shapefile loader.")

# -----------------------------
# Environment
# -----------------------------
load_dotenv()
MAPBOX_API_KEY = os.getenv("MAPBOX_API_KEY")
if MAPBOX_API_KEY:
    import pydeck.settings as pdk_settings
    pdk_settings.mapbox_api_key = MAPBOX_API_KEY

ARCGIS_PORTAL = os.getenv("ARCGIS_PORTAL", "https://www.arcgis.com")
ARCGIS_USER   = os.getenv("ARCGIS_USERNAME", "")
ARCGIS_PASS   = os.getenv("ARCGIS_PASSWORD", "")

# Auto-load local ZIP filename (same folder as this app)
LOCAL_CB_ZIP = os.getenv("LOCAL_CB_ZIP", "NYC Community Boards_20251006.zip")

# ADD THE DEBUG FUNCTION HERE:
def debug_shapefile_contents(zip_path):
    """Debug: show all files in the ZIP"""
    if not os.path.exists(zip_path):
        st.warning(f"File not found: {zip_path}")
        return
    
    st.write(f"**Contents of {zip_path}:**")
    try:
        with _zipfile_from_any(zip_path) as zf:
            files = zf.namelist()
            shp_files = [f for f in files if f.lower().endswith('.shp')]
            st.write(f"All files: {files}")
            st.write(f"Shapefiles found: {shp_files}")
            
            # Try to read the first shapefile and check its type
            if shp_files:
                import tempfile
                tmpdir = tempfile.mkdtemp()
                zf.extractall(tmpdir)
                
                for shp in shp_files:
                    shp_path = os.path.join(tmpdir, shp)
                    try:
                        import shapefile
                        r = shapefile.Reader(shp_path)
                        st.write(f"\n**{shp}:**")
                        st.write(f"- Shape type: {r.shapeType} ({r.shapeTypeName})")
                        st.write(f"- Number of records: {len(r)}")
                        if len(r) > 0:
                            st.write(f"- First shape type: {r.shape(0).shapeType}")
                            st.write(f"- Fields: {[f[0] for f in r.fields if f[0] != 'DeletionFlag']}")
                    except Exception as e:
                        st.write(f"Error reading {shp}: {e}")
    except Exception as e:
        st.error(f"Could not read ZIP: {e}")

def load_and_merge_board_data(boards: list, csv_path: str = "DataForBoxPlots.csv") -> list:
    """
    Load CSV data and merge with board attributes.
    Matches on boro_cd (from shapefile) with CB_id (from CSV).
    
    Returns updated boards list with merged attributes.
    """
    if not os.path.exists(csv_path):
        st.warning(f"CSV file not found: {csv_path}")
        return boards
    
    try:
        # Load the CSV
        csv_df = pd.read_csv(csv_path)
        
        # Columns to pull from CSV
        csv_cols = ['CB_id', 'Buildings', 'Elevation', 'Slope', 'Commuting', 
                    'Imperv', 'Footprint', 'BLDperArea', 'FTPperArea']
        
        # Check if all required columns exist
        missing_cols = [col for col in csv_cols if col not in csv_df.columns]
        if missing_cols:
            st.warning(f"Missing columns in CSV: {missing_cols}")
            return boards
        
        # Select only needed columns
        csv_df = csv_df[csv_cols]
        
        # Normalize CB_id: convert to string and remove decimals if present
        def normalize_id(val):
            if pd.isna(val):
                return ''
            # Convert to float first to handle any format, then to int to remove decimals, then to string
            try:
                return str(int(float(val)))
            except:
                return str(val)
        
        csv_df['CB_id'] = csv_df['CB_id'].apply(normalize_id)
        
        # Create lookup dictionary
        csv_lookup = csv_df.set_index('CB_id').to_dict('index')
        
        # Debug: show what keys are in the lookup
        st.write(f"**Debug:** CSV has {len(csv_lookup)} unique CB_ids")
        st.write(f"**Sample CSV keys:** {list(csv_lookup.keys())[:5]}")
        
        # Update boards with merged data
        updated_boards = []
        matched_count = 0
        unmatched_ids = []
        
        for board in boards:
            # Get boro_cd from board attributes and normalize it
            boro_cd_raw = board.get('attrs', {}).get('boro_cd', '')
            boro_cd = normalize_id(boro_cd_raw)
            
            # Filter shapefile attributes to keep only what we need
            filtered_attrs = {
                'boro_cd': boro_cd,  # Use normalized version
                'shape_area': board['attrs'].get('shape_area'),
                'shape_leng': board['attrs'].get('shape_leng'),
                'Area': board['attrs'].get('Area'),
            }
            
            # Try to match with CSV data
            if boro_cd in csv_lookup:
                # Merge CSV data into attributes
                csv_data = csv_lookup[boro_cd]
                filtered_attrs.update(csv_data)
                matched_count += 1
            else:
                # If no match, add None values for CSV columns
                for col in csv_cols[1:]:  # Skip CB_id
                    filtered_attrs[col] = None
                unmatched_ids.append(f"{boro_cd} (raw: {boro_cd_raw})")
            
            # Update board with filtered and merged attributes
            board['attrs'] = filtered_attrs
            updated_boards.append(board)
        
        st.success(f"✅ Merged data for {matched_count}/{len(boards)} community boards.")
        
        if unmatched_ids:
            with st.expander(f"⚠️ {len(unmatched_ids)} boards not matched", expanded=False):
                st.write("These boro_cd values from shapefile were not found in CSV CB_id:")
                st.write(unmatched_ids[:10])  # Show first 10
        
        return updated_boards
        
    except Exception as e:
        st.error(f"Error merging CSV data: {e}")
        import traceback
        st.code(traceback.format_exc())
        return boards
# -----------------------------
# App config
# -----------------------------
#st.set_page_config(page_title="NYC Resilience AI Agent", page_icon="🌆", layout="wide")
# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="NYC Resilience AI Agent", page_icon="🌆", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "landing"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# NYC bounds (for mock precip patterning)
# -----------------------------
NYC_BOUNDS = {
    "min_lon": -74.2556,
    "min_lat":  40.4960,
    "max_lon": -73.7004,
    "max_lat":  40.9153,
}

def km_to_deg_lat(km: float) -> float:
    return km / 111.32

def km_to_deg_lon(km: float, at_lat_deg: float) -> float:
    return km / (111.32 * math.cos(math.radians(at_lat_deg)))

def build_grid(bounds: dict, cell_km: float = 3.0, target_cells: int = 60) -> pd.DataFrame:
    lat_c = (bounds["min_lat"] + bounds["max_lat"]) / 2.0
    dlat = km_to_deg_lat(cell_km)
    dlon = km_to_deg_lon(cell_km, lat_c)

    lats = np.arange(bounds["min_lat"], bounds["max_lat"], dlat)
    lons = np.arange(bounds["min_lon"], bounds["max_lon"], dlon)

    cells = []
    for i in range(len(lats) - 1):
        for j in range(len(lons) - 1):
            lat0, lat1 = float(lats[i]), float(lats[i+1])
            lon0, lon1 = float(lons[j]), float(lons[j+1])
            poly = [
                [lon0, lat0],
                [lon1, lat0],
                [lon1, lat1],
                [lon0, lat1],
                [lon0, lat0],
            ]
            cells.append({
                "cell_id": f"{i}-{j}",
                "lat": (lat0 + lat1) / 2.0,
                "lon": (lon0 + lon1) / 2.0,
                "polygon": poly,
            })

    df = pd.DataFrame(cells)
    if len(df) > target_cells:
        step = max(1, int(len(df) / target_cells))
        df = df.iloc[::step].reset_index(drop=True)
    return df

# Fallback grid (only used if boards aren’t loaded for some reason)
GRID_DF = build_grid(NYC_BOUNDS, cell_km=3.0, target_cells=60)

# -----------------------------
# Mock precipitation simulators
# -----------------------------
def simulate_hourly_precip(dt: datetime, points_df: pd.DataFrame) -> pd.DataFrame:
    """Realistic-looking hourly precip (mm/hr) at point locations."""
    seed = int(dt.replace(tzinfo=timezone.utc).timestamp()) // 3600
    rng = np.random.default_rng(seed)

    lat = points_df["lat"].to_numpy()
    lon = points_df["lon"].to_numpy()
    lat_n = (lat - NYC_BOUNDS["min_lat"]) / (NYC_BOUNDS["max_lat"] - NYC_BOUNDS["min_lat"])
    lon_n = (lon - NYC_BOUNDS["min_lon"]) / (NYC_BOUNDS["max_lon"] - NYC_BOUNDS["min_lon"])

    hour = dt.hour
    diurnal = 0.5 + 0.5 * np.sin((hour - 15) / 24 * 2 * np.pi)
    month = dt.month
    seasonal = 0.6 + 0.4 * np.sin((month - 7) / 12 * 2 * np.pi)

    base = 0.3 + 3.0 * seasonal * diurnal
    coastal = 2.0 * (0.6 * lon_n + 0.4 * lat_n)

    def gaussian_blob(cx, cy, sx, sy):
        return np.exp(-(((lon_n - cx) ** 2) / (2 * sx ** 2) + ((lat_n - cy) ** 2) / (2 * sy ** 2)))

    cx1, cy1 = rng.uniform(0.3, 0.8), rng.uniform(0.2, 0.8)
    cx2, cy2 = rng.uniform(0.2, 0.7), rng.uniform(0.2, 0.7)
    blob = 8.0 * gaussian_blob(cx1, cy1, 0.10, 0.08) + 5.0 * gaussian_blob(cx2, cy2, 0.12, 0.10)

    noise = rng.normal(0, 0.6, size=len(points_df))
    precip = np.clip(base + coastal + blob + noise, 0, None)

    out = points_df.copy()
    out["precip_mm_hr"] = precip.round(2)
    out["timestamp"] = dt.isoformat()
    return out

def simulate_hourly_precip_range(start_dt: datetime, end_dt: datetime, points_df: pd.DataFrame):
    start_utc = pd.Timestamp(start_dt, tz="UTC")
    end_utc = pd.Timestamp(end_dt, tz="UTC")
    hours = pd.date_range(start=start_utc, end=end_utc, freq="H", inclusive="left", tz="UTC")

    lat = points_df["lat"].to_numpy()
    lon = points_df["lon"].to_numpy()
    lat_n = (lat - NYC_BOUNDS["min_lat"]) / (NYC_BOUNDS["max_lat"] - NYC_BOUNDS["min_lat"])
    lon_n = (lon - NYC_BOUNDS["min_lon"]) / (NYC_BOUNDS["max_lon"] - NYC_BOUNDS["min_lon"])

    hours_of_day = hours.hour.to_numpy()
    months = hours.month.to_numpy()

    diurnal = 0.5 + 0.5 * np.sin((hours_of_day - 15) / 24.0 * 2.0 * math.pi)
    seasonal = 0.6 + 0.4 * np.sin((months - 7) / 12.0 * 2.0 * math.pi)
    base = 0.3 + 3.0 * seasonal * diurnal

    base_2d = np.repeat(base[:, None], len(points_df), axis=1)

    coastal = 2.0 * (0.6 * lon_n + 0.4 * lat_n)
    coastal_2d = np.repeat(coastal[None, :], len(hours), axis=0)

    blob_total = np.zeros_like(base_2d)
    noise_total = np.zeros_like(base_2d)
    for h_idx, ts in enumerate(hours):
        seed = int(ts.timestamp()) // 3600
        rng = np.random.default_rng(seed)
        cx1, cy1 = rng.uniform(0.3, 0.8), rng.uniform(0.2, 0.8)
        cx2, cy2 = rng.uniform(0.2, 0.7), rng.uniform(0.2, 0.7)

        def gaussian_blob(cx, cy, sx, sy):
            return np.exp(-(((lon_n - cx) ** 2) / (2 * sx ** 2) + ((lat_n - cy) ** 2) / (2 * sy ** 2)))

        blob = 8.0 * gaussian_blob(cx1, cy1, 0.10, 0.08) + 5.0 * gaussian_blob(cx2, cy2, 0.12, 0.10)
        blob_total[h_idx, :] = blob
        noise_total[h_idx, :] = rng.normal(0, 0.6, size=len(points_df))

    precip = np.clip(base_2d + coastal_2d + blob_total + noise_total, 0, None)
    return hours, precip

def compute_daily_metrics(hours_idx: pd.DatetimeIndex, precip_matrix: np.ndarray, points_df: pd.DataFrame):
    hourly_df = pd.DataFrame(precip_matrix, index=hours_idx, columns=points_df["cell_id"].tolist())
    by_day = hourly_df.groupby(hourly_df.index.floor('D'))
    daily_max = by_day.max()
    daily_mean = by_day.mean()
    daily_total = by_day.sum()

    avg_of_daily_max = daily_max.mean(axis=0)
    avg_of_daily_mean = daily_mean.mean(axis=0)
    avg_of_daily_total = daily_total.mean(axis=0)

    def to_long(df_daily: pd.DataFrame, value_name: str) -> pd.DataFrame:
        long_df = df_daily.stack().reset_index()
        long_df.columns = ["date", "cell_id", value_name]
        long_df = long_df.merge(points_df[["cell_id", "lat", "lon"]], on="cell_id", how="left")
        return long_df[["date", "cell_id", "lat", "lon", value_name]]

    daily_max_long   = to_long(daily_max,   "daily_max_mm_hr")
    daily_mean_long  = to_long(daily_mean,  "daily_avg_mm_hr")
    daily_total_long = to_long(daily_total, "daily_total_mm")

    return daily_max_long, daily_mean_long, daily_total_long, avg_of_daily_max, avg_of_daily_mean, avg_of_daily_total

# -----------------------------
# Colors / breaks
# -----------------------------
BLUES = [
    [239, 243, 255],
    [198, 219, 239],
    [158, 202, 225],
    [107, 174, 214],
    [66, 146, 198],
    [33, 113, 181],
    [8, 81, 156],
]

def compute_breaks(values: pd.Series, k: int = 7) -> list:
    v = values.dropna().to_numpy()
    if len(v) == 0:
        return [0] * (k - 1)
    qs = np.linspace(0, 1, k)
    b = np.quantile(v, qs)
    thresholds = [float(b[i]) for i in range(1, k)]
    return thresholds

def color_for_value_dynamic(val: float, breaks: list) -> list:
    idx = 0
    for b in breaks:
        if val > b:
            idx += 1
        else:
            break
    idx = min(idx, len(BLUES) - 1)
    return BLUES[idx]

# -----------------------------
# Shapely helpers + loaders
# -----------------------------
def try_import_shapely():
    try:
        import shapely.geometry as sg
        from shapely.geometry import shape as shp_shape, mapping as shp_mapping
        from shapely.ops import transform as shp_transform
        return sg, shp_shape, shp_mapping, shp_transform
    except Exception:
        return None, None, None, None

def load_community_boards_from_json(js: dict):
    sg, shp_shape, _, _ = try_import_shapely()
    if not sg:
        st.error("Community Board aggregation requires 'shapely'.")
        return None

    feats = js.get("features", [])
    polys = []
    for idx, f in enumerate(feats):
        props = f.get("properties", {}) or {}
        unit_id = props.get("BoroCD") or props.get("boro_cd") or props.get("cd") or f"CB_{idx:03d}"
        name = props.get("cd_name") or props.get("name") or unit_id

        geom_geojson = f.get("geometry")
        if not geom_geojson:
            continue
        try:
            geom = shp_shape(geom_geojson)
        except Exception:
            continue
        if geom.is_empty:
            continue
        polys.append({
            "unit_id": str(unit_id),
            "name": str(name),
            "geom": geom,
            "feature_geom": json.loads(json.dumps(geom_geojson)),
            "attrs": dict(props),  # <- keep attributes
        })
    if not polys:
        st.error("No valid Polygon/MultiPolygon features found in the GeoJSON.")
    return polys or None

def load_boards_from_shapefile_zip_streamlit():
    """UI fallback: user uploads a shapefile .zip"""
    zip_file = st.file_uploader("Upload Community Board/District **Shapefile (.zip)**", type=["zip"], key="shpzip_upl")
    if zip_file is None:
        return None
    return _load_boards_from_shapefile_zipfilelike(zip_file)

def load_boards_from_shapefile_path(zip_path: str):
    """Auto-load from a local ZIP path (same folder as app)."""
    if not os.path.exists(zip_path):
        return None
    # CRITICAL CHANGE: pass the path directly; the callee will handle wrapping
    return _load_boards_from_shapefile_zipfilelike(zip_path)

def _load_boards_from_shapefile_zipfilelike(zip_filelike):
    try:
        import shapefile  # pyshp
    except ImportError:
        st.error("Missing dependency: 'pyshp' (pip install pyshp).")
        return None

    sg, shp_shape, shp_mapping, shp_transform = try_import_shapely()
    if not sg:
        st.error("Shapefile support requires 'shapely>=2.0'.")
        return None
    try:
        from pyproj import CRS, Transformer
    except Exception:
        st.error("Shapefile reprojection requires 'pyproj>=3.6'.")
        return None

    def to_pure_geojson(geom):
        """Convert shapely geometry to pure GeoJSON dict"""
        m = shp_mapping(geom)
        return json.loads(json.dumps(m))

    # Extract ZIP
    tmpdir = tempfile.mkdtemp(prefix="cb_shp_")
    try:
        with _zipfile_from_any(zip_filelike) as zf:
            zf.extractall(tmpdir)
    except Exception as e:
        st.error(f"Could not extract shapefile ZIP: {e}")
        return None

    shp_paths = []
    for root, _, files in os.walk(tmpdir):
        for f in files:
            if f.lower().endswith(".shp"):
                shp_paths.append(os.path.join(root, f))
    if not shp_paths:
        st.error("ZIP does not contain a .shp file.")
        return None
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
        try:
            transformer = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)
        except Exception:
            transformer = None

    try:
        r = shapefile.Reader(shp_path)
    except Exception as e:
        st.error(f"Could not read shapefile: {e}")
        return None

    fields = [f[0] for f in r.fields if f[0] != "DeletionFlag"]

    def looks_like_lonlat(x, y):
        return (-180.0 <= x <= 180.0) and (-90.0 <= y <= 90.0)

    polys = []
    guessed_2263 = False
    
    for idx, sr in enumerate(r.shapeRecords()):
        rec = {fields[i]: sr.record[i] for i in range(len(fields))}
        unit_id = rec.get("BoroCD") or rec.get("boro_cd") or rec.get("cd") or f"CB_{idx:03d}"
        name = rec.get("cd_name") or rec.get("name") or unit_id

        # Get geometry from pyshp
        geom_geojson = sr.shape.__geo_interface__
        
        # Convert to shapely
        try:
            geom = shp_shape(geom_geojson)
        except Exception as e:
            st.warning(f"Could not parse geometry for {unit_id}: {e}")
            continue
            
        if geom.is_empty:
            continue

        # Check if we need to reproject
        if transformer is None:
            # Try to detect coordinate system
            try:
                if geom.geom_type == 'Polygon':
                    x0, y0 = list(geom.exterior.coords)[0]
                elif geom.geom_type == 'MultiPolygon':
                    x0, y0 = list(list(geom.geoms)[0].exterior.coords)[0]
                else:
                    x0, y0 = (0, 0)
                    
                if not looks_like_lonlat(x0, y0):
                    # Likely State Plane (EPSG:2263 for NYC)
                    transformer = Transformer.from_crs(
                        CRS.from_epsg(2263), 
                        CRS.from_epsg(4326), 
                        always_xy=True
                    )
                    guessed_2263 = True
            except Exception:
                pass

        # Apply transformation if needed
        if transformer is not None:
            try:
                geom = shp_transform(
                    lambda x, y, z=None: transformer.transform(x, y), 
                    geom
                )
            except Exception as e:
                st.warning(f"Could not transform geometry for {unit_id}: {e}")
                continue

        # Verify we have a valid polygon geometry
        if geom.geom_type not in ['Polygon', 'MultiPolygon']:
            st.warning(f"Skipping {unit_id}: geometry is {geom.geom_type}, not Polygon/MultiPolygon")
            continue

        # Convert to GeoJSON dict
        feature_geom = to_pure_geojson(geom)

        polys.append({
            "unit_id": str(unit_id),
            "name": str(name),
            "geom": geom,  # Keep shapely geometry
            "feature_geom": feature_geom,  # GeoJSON dict
            "attrs": dict(rec),
        })

    if not polys:
        st.error("No valid polygon features found in the shapefile.")
        return None

    msg = f"Loaded {len(polys)} Community Board/District polygons (Shapefile"
    if src_crs:
        try:
            msg += f", source EPSG:{src_crs.to_epsg()}"
        except Exception:
            pass
    if guessed_2263:
        msg += "; guessed EPSG:2263 → WGS84"
    msg += ")."
    st.success(msg)
    return polys

def load_boards_from_arcgis():
    if not ARC_GIS_AVAILABLE:
        return None
    with st.expander("Or sign in to ArcGIS and provide a Feature Layer item id", expanded=False):
        portal = st.text_input("Portal URL", value=ARCGIS_PORTAL)
        user = st.text_input("ArcGIS username", value=ARCGIS_USER)
        pwd  = st.text_input("ArcGIS password", value=ARCGIS_PASS, type="password")
        item_id = st.text_input("Feature Layer item id", value="")
        go = st.button("Load from ArcGIS")
        if go and item_id.strip():
            try:
                gis = GIS(portal, user, pwd)
                item = gis.content.get(item_id.strip())
                lyr = item.layers[0]
                df = lyr.query(where="1=1", out_sr=4326).sdf
                feats = json.loads(df.to_geojson())["features"]
                js = {"type": "FeatureCollection", "features": feats}
                return load_community_boards_from_json(js)
            except Exception as e:
                st.error(f"ArcGIS load failed: {e}")
    return None

# -----------------------------
# Radar points from boards
# -----------------------------
def radars_from_boards(boards: list) -> pd.DataFrame:
    sg, _, _, _ = try_import_shapely()
    rows = []
    for b in boards:
        p = b["geom"].representative_point()  # always inside
        rows.append({
            "cell_id": b["unit_id"],
            "lat": float(p.y),
            "lon": float(p.x),
        })
    return pd.DataFrame(rows)

# -----------------------------
# Aggregation points -> boards
# -----------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def aggregate_series_to_boards(series_by_cell: pd.Series, points_df: pd.DataFrame, boards: list):
    sg, _, _, _ = try_import_shapely()
    if not sg:
        st.error("Community Board aggregation requires 'shapely'.")
        return None, None

    val_lookup = series_by_cell.to_dict()
    pt_lookup = {r["cell_id"]: (r["lat"], r["lon"]) for _, r in points_df.iterrows()}

    board_vals = {}
    for b in boards:
        geom = b["geom"]
        inside_vals = []
        for cid, (plat, plon) in pt_lookup.items():
            if geom.contains(sg.Point(plon, plat)):
                v = val_lookup.get(cid, np.nan)
                if pd.notna(v):
                    inside_vals.append(v)
        if inside_vals:
            board_vals[b["unit_id"]] = float(np.mean(inside_vals))
        else:
            c = geom.representative_point()
            latc, lonc = float(c.y), float(c.x)
            best, bestd = None, 1e18
            for cid, (plat, plon) in pt_lookup.items():
                d = haversine_km(latc, lonc, plat, plon)
                if d < bestd:
                    bestd = d
                    best = val_lookup.get(cid, np.nan)
            board_vals[b["unit_id"]] = float(best) if pd.notna(best) else 0.0

    s = pd.Series(board_vals).sort_index()
    breaks = compute_breaks(s, k=len(BLUES))

    feats = []
    for b in boards:
        v = float(s.get(b["unit_id"], 0.0))
        feats.append({
            "type": "Feature",
            "properties": {
                "unit_id": b["unit_id"],
                "name": b.get("name", b["unit_id"]),
                "value": v,
                "fill_color": color_for_value_dynamic(v, breaks),
                "attrs": b.get("attrs", {}),  # expose attributes in tooltip if needed
            },
            "geometry": b["feature_geom"]
        })
    fc = {"type": "FeatureCollection", "features": feats}
    return s, fc

# -----------------------------
# Basemap + view helpers
# -----------------------------
def esri_light_gray_basemap():
    return pdk.Layer(
        "TileLayer",
        data="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        minZoom=0, maxZoom=19, tileSize=256
    )

def fc_center_and_zoom(fc):
    minx, miny, maxx, maxy = 999, 999, -999, -999
    def update_bbox(coords, is_multi):
        nonlocal minx, miny, maxx, maxy
        if is_multi:
            for poly in coords:
                for ring in poly:
                    for x, y in ring:
                        minx = min(minx, x); maxx = max(maxx, x)
                        miny = min(miny, y); maxy = max(maxy, y)
        else:
            for ring in coords:
                for x, y in ring:
                    minx = min(minx, x); maxx = max(maxx, x)
                    miny = min(miny, y); maxy = max(maxy, y)
    for f in fc.get("features", []):
        geom = f.get("geometry", {})
        if not geom: continue
        t = geom.get("type"); c = geom.get("coordinates", [])
        if t == "Polygon": update_bbox(c, False)
        elif t == "MultiPolygon": update_bbox(c, True)
    if minx > maxx or miny > maxy:
        return 40.7128, -74.0060, 10.0
    lon = (minx + maxx) / 2.0
    lat = (miny + maxy) / 2.0
    span = max(maxx - minx, maxy - miny)
    zoom = 10.5
    if span > 2.0: zoom = 9.0
    elif span > 1.0: zoom = 9.8
    elif span > 0.5: zoom = 10.2
    return lat, lon, zoom

# -----------------------------
# Claude (Bedrock) helper
# -----------------------------
def bedrock_enabled() -> bool:
    return all([
        os.getenv("AWS_ACCESS_KEY_ID"),
        os.getenv("AWS_SECRET_ACCESS_KEY"),
        os.getenv("AWS_REGION")
    ])

def call_claude(prompt: str, system: str = None, temperature: float = 0.2, max_tokens: int = 600) -> str:
    import boto3
    client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
    model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }
    if system:
        body["system"] = system
    resp = client.invoke_model(modelId=model_id, contentType="application/json", accept="application/json", body=json.dumps(body))
    payload = json.loads(resp["body"].read())
    parts = []
    for blk in payload.get("content", []):
        if blk.get("type") == "text":
            parts.append(blk.get("text", ""))
    return "".join(parts).strip() or "(no text returned)"

# -----------------------------
# Sidebar + landing
# -----------------------------
def sidebar_agents():
    with st.sidebar:
        st.markdown("### NYC Resilience AI Agent")
        st.caption("Demo – Flooding & UHI")
        st.divider()
        st.checkbox("Mapping Agent", value=True, disabled=True)
        st.checkbox("Forecasting Agent", value=False, disabled=True)
        st.checkbox("Optimization Agent", value=False, disabled=True)
        st.checkbox("Green Infrastructure Agent", value=False, disabled=True)
        st.divider()
        if bedrock_enabled():
            st.success("Claude (Bedrock) connected.")
        else:
            st.info("Claude: set AWS creds in .env to enable chat.")

def landing_page():
    st.title("🏙️ NYC Resilience AI Agent")
    st.subheader("Choose a design problem to explore")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌊 Flooding", use_container_width=True, type="primary"):
            st.session_state.page = "flooding"; st.rerun()
    with col2:
        st.button("🌡️ Urban Heat Island (coming soon)", use_container_width=True, disabled=True)
    st.info("Pick a scenario and visualize variables. Claude chat optional.")

# -----------------------------
# Mapping Agent
# -----------------------------
def flooding_mapping_agent():
    st.title("🌊 Flooding → Mapping Agent (NYC)")
    st.caption("We first display Community Board polygons. Then we place one radar point per board, simulate precipitation, and color the boards (graduated blues).")

    # ADD THIS DEBUG CALL HERE:
    debug_shapefile_contents(LOCAL_CB_ZIP)
    
    # ---- Step 1: Load Community Boards automatically from local ZIP ----
    boards = load_boards_from_shapefile_path(LOCAL_CB_ZIP)
    
    # Merge with CSV data if boards were loaded
    if boards:
        boards = load_and_merge_board_data(boards, "DataForBoxPlots.csv")

    # If not found locally, offer fallbacks (upload / GeoJSON / ArcGIS)
    if not boards:
        st.warning(f"Local shapefile ZIP not found at: `{LOCAL_CB_ZIP}`. Upload or paste a GeoJSON URL, or sign into ArcGIS (optional).")
        col_u1, col_u2, col_u3 = st.columns(3)
        with col_u1:
            uploaded_geojson = st.file_uploader("Upload Community Board/District **GeoJSON**", type=["geojson", "json"], key="geojson_upl")
            js = None
            if uploaded_geojson is not None:
                b = uploaded_geojson.read()
                try:
                    js = json.loads(b.decode("utf-8"))
                except UnicodeDecodeError:
                    js = json.loads(b.decode("latin-1"))
            if js:
                boards = load_community_boards_from_json(js)
        with col_u2:
            boards = boards or load_boards_from_shapefile_zip_streamlit()
        with col_u3:
            boards = boards or load_boards_from_arcgis()

    if not boards:
        st.stop()

    # ---- Draw polygons immediately (outline on Esri Light Gray) ----
    fc_outlines = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": b.get("name", b["unit_id"])}, "geometry": b["feature_geom"]} for b in boards
    ]}
    lat_c, lon_c, zoom_c = fc_center_and_zoom(fc_outlines)
    
    # Debug the feature collection
    st.write("**Debug Info:**")
    st.write(f"Number of features in fc_outlines: {len(fc_outlines.get('features', []))}")
    if fc_outlines.get("features"):
        sample_feat = fc_outlines["features"][0]
        st.write(f"Sample feature geometry type: {sample_feat.get('geometry', {}).get('type')}")
        st.write(f"Sample feature properties: {sample_feat.get('properties')}")
        # Show first few coordinates
        geom = sample_feat.get('geometry', {})
        if geom.get('type') == 'Polygon':
            st.write(f"First ring has {len(geom['coordinates'][0])} points")
            st.write(f"Sample coords: {geom['coordinates'][0][:3]}")
    outline = pdk.Layer(
        "GeoJsonLayer",
        data=fc_outlines,
        pickable=True,
        stroked=True,
        filled=False,
        extruded=False,
        wireframe=True,
        get_line_color=[255, 0, 0, 255],  # Bright red, fully opaque
        get_line_width=200,  # Width in meters (adjust based on your scale)
        line_width_min_pixels=3,
        line_width_max_pixels=10,
    )
    deck0 = pdk.Deck(
        layers=[esri_light_gray_basemap(), outline],
        initial_view_state=pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom_c),
        tooltip={"html": "<b>Board:</b> {name}", "style": {"backgroundColor": "white", "color": "black"}}
    )
    st.pydeck_chart(deck0, use_container_width=True)
    st.caption("Community Boards/Districts (outline).")

    # Show a quick sample of attributes (so the table is “available” right away)
    # Show a quick sample of attributes (so the table is "available" right away)
    try:
        # Create a more readable attributes table
        attrs_list = []
        for b in boards:
            row = {
                "unit_id": b["unit_id"],
                "name": b.get("name", ""),
                **b.get("attrs", {})
            }
            attrs_list.append(row)
        
        attrs_df = pd.DataFrame(attrs_list)
        
        # Reorder columns to show key attributes first
        priority_cols = ["unit_id", "name", "boro_cd", "Buildings", "Elevation", 
                        "Slope", "Commuting", "Imperv", "Footprint", 
                        "BLDperArea", "FTPperArea", "shape_area", "shape_leng", "Area"]
        
        # Only include columns that exist
        display_cols = [col for col in priority_cols if col in attrs_df.columns]
        other_cols = [col for col in attrs_df.columns if col not in display_cols]
        final_cols = display_cols + other_cols
        
        attrs_df = attrs_df[final_cols]
        
        st.write("**Community Board attributes (with merged CSV data):**")
        st.dataframe(attrs_df.head(10), hide_index=True, use_container_width=True)
        
        # Option to download full attributes table
        st.download_button(
            "Download Full Attributes Table (CSV)",
            data=attrs_df.to_csv(index=False).encode("utf-8"),
            file_name="nyc_boards_full_attributes.csv",
            mime="text/csv",
            use_container_width=True,
        )
    except Exception as e:
        st.warning(f"Could not display attributes table: {e}")

    # ---- Step 2: Place Radar Points (one per board) ----
    RADAR_DF = radars_from_boards(boards)

    # ---- Step 3: Single-hour precipitation choropleth ----
    st.subheader("Single-hour precipitation (board choropleth)")
    with st.expander("Time controls", expanded=True):
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        end = now + timedelta(days=365 * 10)
        date_sel = st.date_input("Forecast date", value=now.date(), min_value=now.date(), max_value=end.date(), key="single_date")
        hour = st.slider("Hour (local)", 0, 23, value=15, key="single_hour")
        dt_selected = datetime.combine(date_sel, datetime.min.time()) + timedelta(hours=hour)
    opacity = st.slider("Polygon fill opacity", 0.2, 1.0, 0.8)
    show_points = st.checkbox("Show radar points", value=True)

    hour_df = simulate_hourly_precip(dt_selected, RADAR_DF)
    series = hour_df.set_index("cell_id")["precip_mm_hr"]
    board_series, fc_hour = aggregate_series_to_boards(series, RADAR_DF, boards)

    cb_layer = pdk.Layer(
        "GeoJsonLayer",
        data=fc_hour,
        pickable=True, stroked=True, filled=True,
        get_fill_color="properties.fill_color",
        get_line_color=[50, 50, 50], lineWidthMinPixels=2,
        opacity=opacity
    )
    layers1 = [esri_light_gray_basemap(), cb_layer]
    if show_points:
        layers1.append(
            pdk.Layer("ScatterplotLayer", data=RADAR_DF, get_position='[lon, lat]', get_radius=100, get_fill_color=[20,20,20])
        )
    deck1 = pdk.Deck(
        layers=layers1,
        initial_view_state=pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom_c),
        tooltip={"html": "<b>Board:</b> {name}<br/><b>Precip:</b> {value} mm/hr", "style": {"backgroundColor":"white","color":"black"}}
    )
    st.pydeck_chart(deck1, use_container_width=True)

    # Download of hourly, board-aggregated values
    dl_df = board_series.rename("precip_mm_hr").reset_index()
    dl_df.columns = ["unit_id", "precip_mm_hr"]
    st.download_button(
        "Download this hour (board-aggregated) CSV",
        data=dl_df.to_csv(index=False).encode("utf-8"),
        file_name=f"nyc_boards_precip_{dt_selected:%Y%m%d_%H}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption(f"Hourly precip range: min={board_series.min():.2f} mm/hr, max={board_series.max():.2f} mm/hr.")

    # ---- Step 3b: Community Board Attributes Visualization ----
    st.divider()
    st.subheader("Community Board Attributes Visualization")
    st.caption("Visualize key attributes from the merged data across community boards.")
    
    # Check if we have merged attributes
    has_merged_data = any(board.get('attrs', {}).get('Buildings') is not None for board in boards)
    
    if not has_merged_data:
        st.warning("No merged CSV data available. Please ensure DataForBoxPlots.csv is loaded.")
    else:
        # Attribute selection
        attribute_options = {
            "Number of Buildings": "Buildings",
            "Percentage Impervious Cover": "Imperv",
            "Building Footprint": "Footprint",
            "Elevation": "Elevation",
            "Slope": "Slope"
        }
        
        col_attr1, col_attr2 = st.columns([2, 1])
        with col_attr1:
            selected_attr_label = st.selectbox(
                "Select attribute to visualize:",
                options=list(attribute_options.keys()),
                index=0
            )
        with col_attr2:
            opacity_attr = st.slider("Fill opacity", 0.2, 1.0, 0.75, key="attr_opacity")
        
        selected_attr = attribute_options[selected_attr_label]
        
        # Create series from board attributes
        attr_values = {}
        for b in boards:
            val = b.get('attrs', {}).get(selected_attr)
            if val is not None and pd.notna(val):
                try:
                    attr_values[b["unit_id"]] = float(val)
                except (ValueError, TypeError):
                    attr_values[b["unit_id"]] = 0.0
            else:
                attr_values[b["unit_id"]] = 0.0
        
        attr_series = pd.Series(attr_values).sort_index()
        
        # Check if we have valid data
        if attr_series.max() == 0 and attr_series.min() == 0:
            st.warning(f"No valid data found for {selected_attr_label}.")
        else:
            # Compute breaks for color scaling
            breaks_attr = compute_breaks(attr_series, k=len(BLUES))
            
            # Create GeoJSON features with colors
            feats_attr = []
            for b in boards:
                v = float(attr_series.get(b["unit_id"], 0.0))
                feats_attr.append({
                    "type": "Feature",
                    "properties": {
                        "unit_id": b["unit_id"],
                        "name": b.get("name", b["unit_id"]),
                        "value": v,
                        "attribute": selected_attr_label,
                        "fill_color": color_for_value_dynamic(v, breaks_attr),
                    },
                    "geometry": b["feature_geom"]
                })
            fc_attr = {"type": "FeatureCollection", "features": feats_attr}
            
            # Create map layer
            cb_attr_layer = pdk.Layer(
                "GeoJsonLayer",
                data=fc_attr,
                pickable=True,
                stroked=True,
                filled=True,
                get_fill_color="properties.fill_color",
                get_line_color=[50, 50, 50],
                lineWidthMinPixels=2,
                opacity=opacity_attr
            )
            
            # Create deck
            deck_attr = pdk.Deck(
                layers=[esri_light_gray_basemap(), cb_attr_layer],
                initial_view_state=pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom_c),
                tooltip={
                    "html": "<b>Board:</b> {name}<br/><b>{attribute}:</b> {value}",
                    "style": {"backgroundColor": "white", "color": "black"}
                }
            )
            
            # Display map
            st.pydeck_chart(deck_attr, use_container_width=True)
            
            # Show statistics
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("Minimum", f"{attr_series.min():.2f}")
            with col_stat2:
                st.metric("Maximum", f"{attr_series.max():.2f}")
            with col_stat3:
                st.metric("Mean", f"{attr_series.mean():.2f}")
            with col_stat4:
                st.metric("Median", f"{attr_series.median():.2f}")
            
            # Download button for this attribute
            dl_attr_df = attr_series.rename(selected_attr).reset_index()
            dl_attr_df.columns = ["unit_id", selected_attr]
            st.download_button(
                f"Download {selected_attr_label} Data (CSV)",
                data=dl_attr_df.to_csv(index=False).encode("utf-8"),
                file_name=f"nyc_boards_{selected_attr}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            
            # Optional: Show color legend
            with st.expander("Color Legend", expanded=False):
                st.write("**Value ranges and colors:**")
                legend_data = []
                breaks_with_bounds = [attr_series.min()] + breaks_attr + [attr_series.max()]
                for i in range(len(BLUES)):
                    lower = breaks_with_bounds[i]
                    upper = breaks_with_bounds[i + 1]
                    color_rgb = BLUES[i]
                    legend_data.append({
                        "Range": f"{lower:.2f} - {upper:.2f}",
                        "Color (RGB)": f"rgb({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]})"
                    })
                st.dataframe(pd.DataFrame(legend_data), hide_index=True, use_container_width=True)
                
    # ---- Step 4: Date-range generation + daily metrics ----
    st.divider()
    st.subheader("Generate range & compute daily metrics (board choropleth)")
    with st.expander("Date range (inclusive end date) and generation", expanded=True):
        default_start = date(2025, 10, 6)
        default_end = date(2025, 11, 6)
        start_date, end_date = st.date_input(
            "Select date range (inclusive of end date)",
            value=(default_start, default_end),
            min_value=date(2025, 10, 6),
            max_value=date(2030, 10, 6)
        )
    if isinstance(start_date, tuple):
        start_date, end_date = start_date
    if start_date >= end_date:
        st.error("End date must be after start date.")
        st.stop()

    generate = st.button("Generate mock precipitation for selected range", type="primary", use_container_width=True)

    if generate:
        with st.spinner("Generating hourly precipitation and computing daily metrics..."):
            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time())
            hours_idx, precip_matrix = simulate_hourly_precip_range(start_dt, end_dt, RADAR_DF)
            (daily_max_long,
             daily_mean_long,
             daily_total_long,
             avg_of_daily_max,
             avg_of_daily_mean,
             avg_of_daily_total) = compute_daily_metrics(hours_idx, precip_matrix, RADAR_DF)

            st.session_state["daily_max_long"] = daily_max_long
            st.session_state["daily_mean_long"] = daily_mean_long
            st.session_state["daily_total_long"] = daily_total_long
            st.session_state["avg_of_daily_max"] = avg_of_daily_max
            st.session_state["avg_of_daily_mean"] = avg_of_daily_mean
            st.session_state["avg_of_daily_total"] = avg_of_daily_total
            st.session_state["range_str"] = f"{start_date.isoformat()}_to_{end_date.isoformat()}"
        st.success("Precipitation generated and daily metrics computed.")

    if all(k in st.session_state for k in ["daily_max_long", "daily_mean_long", "daily_total_long"]):
        rstr = st.session_state.get("range_str", "range")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "Daily Max (mm/hr) CSV",
                data=st.session_state["daily_max_long"].to_csv(index=False).encode("utf-8"),
                file_name=f"nyc_daily_max_{rstr}.csv", mime="text/csv", use_container_width=True
            )
        with c2:
            st.download_button(
                "Daily Average (mm/hr) CSV",
                data=st.session_state["daily_mean_long"].to_csv(index=False).encode("utf-8"),
                file_name=f"nyc_daily_avg_{rstr}.csv", mime="text/csv", use_container_width=True
            )
        with c3:
            st.download_button(
                "Daily Total (mm) CSV",
                data=st.session_state["daily_total_long"].to_csv(index=False).encode("utf-8"),
                file_name=f"nyc_daily_total_{rstr}.csv", mime="text/csv", use_container_width=True
            )

        st.divider()
        st.subheader("Summary map across the selected period")

        metric = st.radio(
            "Metric to visualize (average across days):",
            ["Average of daily Max (mm/hr)", "Average of daily Average (mm/hr)", "Average of daily Total (mm/day)"],
            index=0
        )
        opacity2 = st.slider("Polygon fill opacity (summary map)", 0.2, 1.0, 0.8, key="sum_opacity")

        if metric.startswith("Average of daily Max"):
            series = st.session_state["avg_of_daily_max"]; label = "avg_daily_max_mm_hr"
        elif metric.startswith("Average of daily Average"):
            series = st.session_state["avg_of_daily_mean"]; label = "avg_daily_avg_mm_hr"
        else:
            series = st.session_state["avg_of_daily_total"]; label = "avg_daily_total_mm"

        board_series2, fc_sum = aggregate_series_to_boards(series, RADAR_DF, boards)
        cb_sum = pdk.Layer(
            "GeoJsonLayer",
            data=fc_sum,
            pickable=True, stroked=True, filled=True,
            get_fill_color="properties.fill_color",
            get_line_color=[40,40,40], lineWidthMinPixels=2,
            opacity=opacity2
        )
        deck2 = pdk.Deck(
            layers=[esri_light_gray_basemap(), cb_sum],
            initial_view_state=pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom_c),
            tooltip={"html": "<b>Board:</b> {name}<br/><b>Value:</b> {value}",
                     "style": {"backgroundColor":"white","color":"black"}}
        )
        st.pydeck_chart(deck2, use_container_width=True)
        st.caption(f"Summary range: min={board_series2.min():.2f}, max={board_series2.max():.2f} ({label}).")

    # -------------------------
    # Claude chat (unchanged)
    # -------------------------
    st.subheader("🤖 Claude Agent (Bedrock)")
    default_hint = (
        "Explain the precipitation map above for New York City at the selected time. "
        "Point out hotspots, likely causes, and suggest green infrastructure ideas."
    )
    user_prompt = st.text_area("Ask a question about the map or flood planning:", value=default_hint, height=120)
    col_chat1, col_chat2 = st.columns([1,1])
    with col_chat1:
        ask = st.button("Ask Claude", type="primary", use_container_width=True)
    with col_chat2:
        describe = st.button("Describe Current Map", use_container_width=True)

    hour_ctx = simulate_hourly_precip(dt_selected, RADAR_DF)
    if ask or describe:
        top_cells = hour_ctx.sort_values("precip_mm_hr", ascending=False).head(5)
        context = {
            "timestamp": hour_ctx["timestamp"].iloc[0],
            "top5_points": top_cells[["cell_id","lat","lon","precip_mm_hr"]].to_dict(orient="records"),
            "mean_precip_mm_hr": float(hour_ctx["precip_mm_hr"].mean()),
            "max_precip_mm_hr": float(hour_ctx["precip_mm_hr"].max()),
            "min_precip_mm_hr": float(hour_ctx["precip_mm_hr"].min()),
        }
        system_msg = (
            "You are an assistant for a flood risk hackathon. Explain findings clearly for business users. "
            "Suggest practical green infrastructure options. Keep it concise."
        )
        final_prompt = user_prompt
        if describe:
            final_prompt = (
                "Use the following JSON context to describe the NYC precipitation map and planning suggestions:\n"
                f"{json.dumps(context, indent=2)}\n\n{user_prompt}"
            )
        if bedrock_enabled():
            try:
                answer = call_claude(final_prompt, system=system_msg)
                st.session_state.chat_history += [{"role":"user","text":user_prompt},{"role":"assistant","text":answer}]
                st.success("Claude responded:"); st.write(answer)
            except Exception as e:
                st.error(f"Claude/Bedrock error: {e}")
        else:
            mock = ("Peak hourly precipitation is concentrated along the southeast coastal band with a few convective cells inland. "
                    "Expect rapid ponding in low-lying areas. Consider bioswales and permeable pavement pilots near hotspots.")
            st.session_state.chat_history += [{"role":"user","text":user_prompt},{"role":"assistant","text":mock}]
            st.write(mock)

    if st.session_state.chat_history:
        st.markdown("#### Conversation")
        for turn in st.session_state.chat_history[-10:]:
            st.markdown(("**You:** " if turn["role"]=="user" else "**Claude:** ") + turn["text"])

# -----------------------------
# Router + sidebar
# -----------------------------
def sidebar_and_route():
    with st.sidebar:
        st.markdown("### Agents")
        st.button("🏠 Home", on_click=lambda: st.session_state.update(page="landing"))
        st.button("🌊 Flooding", on_click=lambda: st.session_state.update(page="flooding"))
    if st.session_state.page == "landing":
        landing_page()
    elif st.session_state.page == "flooding":
        flooding_mapping_agent()

sidebar_agents()
if st.session_state.page == "landing":
    landing_page()
elif st.session_state.page == "flooding":
    flooding_mapping_agent()
