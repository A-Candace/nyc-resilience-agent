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

# === Forecasting Agent imports ===
from forecasting.feature_weights import WEIGHT_SETS, ATTR_MAP
from forecasting.static_modifiers import (
    compute_static_matrix,
    robust_scale,
    static_risk_weight_per_board,
    to_multiplier
)
from forecasting.models.rule_based import forecast_rule_based
from forecasting.models.linear_residual import forecast_linear_residual
from forecasting.models.graph_diffusion import forecast_graph_diffusion

# Optional: ArcGIS Python API (for hosted feature layers — not required here)
try:
    from arcgis.gis import GIS
    ARC_GIS_AVAILABLE = True
except Exception:
    ARC_GIS_AVAILABLE = False


# =========================
# Feature flags
# =========================
# Deactivate the sidebar "Community Board Agent (beta)" without deleting code:
ENABLE_CB_AGENT_SIDEBAR = False


# =========================
# Shared helpers for static multiplier
# =========================
def build_board_multiplier(boards, weight_set_key="Random Forest (Model 1)"):
    # Restrict to Model 1 only
    weights = WEIGHT_SETS[weight_set_key]
    static_df = compute_static_matrix(boards, ATTR_MAP)         # index=unit_id
    static_df_scaled = robust_scale(static_df)                  # ~[0,1]
    risk_score = static_risk_weight_per_board(static_df_scaled, weights, ATTR_MAP)  # [0,1]
    mult = to_multiplier(risk_score, low=0.7, high=1.5)         # tweakable range
    return mult, static_df_scaled


# ---------- ZIP helper that fixes non-seekable blobs ----------
def _zipfile_from_any(obj):
    """
    Return a ZipFile that works for: path string/PathLike, UploadedFile/_Blob, or raw bytes.
    """
    if isinstance(obj, (str, os.PathLike)):
        return ZipFile(obj)
    if hasattr(obj, "read"):
        try:
            obj.seek(0)
        except Exception:
            pass
        data = obj.read()
        return ZipFile(BytesIO(data))
    if isinstance(obj, (bytes, bytearray)):
        return ZipFile(BytesIO(obj))
    raise TypeError("Unsupported ZIP input type for shapefile loader.")

def _seed_env_from_secrets():
    try:
        import streamlit as st  # already imported above in your file
        keys = [
            "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION",
            "BEDROCK_MODEL_ID", "ARCGIS_PORTAL", "ARCGIS_USERNAME", "ARCGIS_PASSWORD",
            "MAPBOX_API_KEY", "LOCAL_CB_ZIP"
        ]
        for k in keys:
            if k in st.secrets and st.secrets[k]:
                os.environ[k] = str(st.secrets[k])
    except Exception:
        pass

_seed_env_from_secrets()
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


def normalize_cb_id(val):
    """
    Normalize community board id to a clean string of digits (e.g., '305'), never '305.0'.
    """
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    try:
        return str(int(float(val)))
    except Exception:
        return str(val)


def get_board_display_id(board):
    """
    Choose the best display id for the board: prefer attrs['boro_cd'], else unit_id.
    Always return a string without decimals.
    """
    attrs = board.get("attrs", {}) or {}
    boro_cd = attrs.get("boro_cd")
    if boro_cd is not None and str(boro_cd) != "":
        return normalize_cb_id(boro_cd)
    return normalize_cb_id(board.get("unit_id", ""))


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
        csv_df = pd.read_csv(csv_path)

        csv_cols = ['CB_id', 'Buildings', 'Elevation', 'Slope', 'Commuting',
                    'Imperv', 'Footprint', 'BLDperArea', 'FTPperArea']

        missing_cols = [col for col in csv_cols if col not in csv_df.columns]
        if missing_cols:
            st.warning(f"Missing columns in CSV: {missing_cols}")
            return boards

        csv_df = csv_df[csv_cols]
        csv_df['CB_id'] = csv_df['CB_id'].apply(normalize_cb_id)

        csv_lookup = csv_df.set_index('CB_id').to_dict('index')

        updated_boards = []
        for board in boards:
            boro_cd_raw = board.get('attrs', {}).get('boro_cd', '')
            boro_cd = normalize_cb_id(boro_cd_raw)

            filtered_attrs = {
                'boro_cd': boro_cd,  # normalized
                'shape_area': board['attrs'].get('shape_area'),
                'shape_leng': board['attrs'].get('shape_leng'),
                'Area': board['attrs'].get('Area'),
            }

            if boro_cd in csv_lookup:
                csv_data = csv_lookup[boro_cd]
                filtered_attrs.update(csv_data)
            else:
                for col in csv_cols[1:]:
                    filtered_attrs[col] = None

            board['attrs'] = filtered_attrs
            updated_boards.append(board)

        return updated_boards

    except Exception as e:
        st.error(f"Error merging CSV data: {e}")
        import traceback
        st.code(traceback.format_exc())
        return boards


# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="NYC Resilience AI Agent", page_icon="🌆", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "landing"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Session keys we’ll share between agents
SESSION_KEYS = {
    "boards": "boards",
    "lat": "lat_c",
    "lon": "lon_c",
    "zoom": "zoom_c",
    "precip_df": "precip_forecast_df",         # required by Forecasting agent
    "precip_meta": "precip_meta",              # dict with notes on how precip was produced
    "static_scaled": "static_scaled_cache",    # store scaled static table from last run
    "weights_used": "weights_used_cache"       # name of weight set used when building multipliers
}


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

# Fallback grid (only used if boards aren't loaded for some reason)
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
# Info bars (papers-aware)
# -----------------------------
def attribute_info(selected_attr_label: str) -> str:
    P1 = "A machine learning approach to evaluate the spatial variability of NYC’s 311 street flooding complaints (CEUS, 2022)"
    P2 = "A review of recent advances in urban flood research (Water Security, 2023)"
    P3 = "Structured exploration of ML model complexity for spatio-temporal forecasting of urban flooding (manuscript)"

    if selected_attr_label.startswith("Buildings per Area"):
        return (
            "**Why it matters.** Higher **building density** concentrates impervious surfaces and obstructs overland flow "
            "paths, increasing runoff and localized ponding. In NYC analyses, building-related variables were among the "
            "strongest spatial predictors of reported street flooding. — *" + P1 + "*"
        )
    if selected_attr_label.startswith("Footprint per Area"):
        return (
            "**Why it matters.** Greater **built footprint per unit area** means more continuous impervious cover and "
            "fewer infiltration/retention opportunities, which elevates pluvial flood susceptibility and prolongs "
            "drainage times. Building footprint metrics performed strongly in explaining complaint variability. — *" + P1 + "*"
        )
    if selected_attr_label == "Percentage Impervious Cover":
        return (
            "**Why it matters.** Impervious surfaces (streets, sidewalks, roofs) limit infiltration, raising runoff "
            "coefficients and peak discharges. Higher imperviousness generally elevates pluvial flood risk. — *" + P2 + "*"
        )
    if selected_attr_label == "Slope":
        return (
            "**Why it matters.** Flatter areas retain water and show higher flood susceptibility; steeper slopes reduce "
            "ponding depth and duration. — *" + P2 + "*"
        )
    if selected_attr_label == "Elevation":
        return (
            "**Why it matters.** Low-lying areas face elevated risk due to ponding and potential surge/tide interactions. — *" + P2 + "*"
        )
    return "Context coming soon."


def precip_metric_info(metric_label: str) -> str:
    P2 = "A review of recent advances in urban flood research (Water Security, 2023)"
    P3 = "Structured exploration of ML model complexity for spatio-temporal forecasting of urban flooding (manuscript)"

    if metric_label.startswith("Average of daily Max"):
        return (
            "**Why it matters.** The day’s **maximum hourly intensity** proxies design-critical peaks used in drainage "
            "methods. Intensification trends increase pluvial exceedance risk. — *" + P2 + "*"
        )
    if metric_label.startswith("Average of daily Average"):
        return (
            "**Why it matters.** The day’s **mean rate** captures sustained loading on sewers—useful when longer, "
            "moderate rain still overwhelms capacity. — *" + P2 + "*"
        )
    if metric_label.startswith("Average of daily Total"):
        return (
            "**Why it matters.** **Total daily accumulation** highlights prolonged events that drive basement seepage "
            "and widespread ponding even without sharp peaks. — *" + P3 + "*"
        )
    return "Context coming soon."


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
            "attrs": dict(props),
        })
    if not polys:
        st.error("No valid Polygon/MultiPolygon features found in the GeoJSON.")
    return polys or None

def load_boards_from_shapefile_zip_streamlit():
    zip_file = st.file_uploader("Upload Community Board/District **Shapefile (.zip)**", type=["zip"], key="shpzip_upl")
    if zip_file is None:
        return None
    return _load_boards_from_shapefile_zipfilelike(zip_file)

def load_boards_from_shapefile_path(zip_path: str):
    if not os.path.exists(zip_path):
        return None
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
        m = shp_mapping(geom)
        return json.loads(json.dumps(m))

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
    for idx, sr in enumerate(r.shapeRecords()):
        rec = {fields[i]: sr.record[i] for i in range(len(fields))}
        unit_id = rec.get("BoroCD") or rec.get("boro_cd") or rec.get("cd") or f"CB_{idx:03d}"
        name = rec.get("cd_name") or rec.get("name") or unit_id

        geom_geojson = sr.shape.__geo_interface__

        try:
            geom = shp_shape(geom_geojson)
        except Exception as e:
            st.warning(f"Could not parse geometry for {unit_id}: {e}")
            continue

        if geom.is_empty:
            continue

        if transformer is None:
            try:
                if geom.geom_type == 'Polygon':
                    x0, y0 = list(geom.exterior.coords)[0]
                elif geom.geom_type == 'MultiPolygon':
                    x0, y0 = list(list(geom.geoms)[0].exterior.coords)[0]
                else:
                    x0, y0 = (0, 0)

                if not looks_like_lonlat(x0, y0):
                    transformer = Transformer.from_crs(
                        2263,
                        4326,
                        always_xy=True
                    )
            except Exception:
                pass

        if transformer is not None:
            try:
                geom = shp_transform(
                    lambda x, y, z=None: transformer.transform(x, y),
                    geom
                )
            except Exception as e:
                st.warning(f"Could not transform geometry for {unit_id}: {e}")
                continue

        if geom.geom_type not in ['Polygon', 'MultiPolygon']:
            st.warning(f"Skipping {unit_id}: geometry is {geom.geom_type}, not Polygon/MultiPolygon")
            continue

        feature_geom = to_pure_geojson(geom)

        polys.append({
            "unit_id": str(unit_id),
            "name": str(name),
            "geom": geom,
            "feature_geom": feature_geom,
            "attrs": dict(rec),
        })

    if not polys:
        st.error("No valid polygon features found in the shapefile.")
    return polys or None


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
        p = b["geom"].representative_point()
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
                "cb": get_board_display_id(b),
                "name": b.get("name", b["unit_id"]),
                "value": v,
                "fill_color": color_for_value_dynamic(v, breaks),
                "attrs": b.get("attrs", {}),
            },
            "geometry": b["feature_geom"]
        })
    fc = {"type": "FeatureCollection", "features": feats}
    return s, fc


# -----------------------------
# Basemap + view helpers + labels
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

def cb_label_layer_from_boards(boards, text_size=12):
    """
    Returns a pydeck TextLayer placing the community board number ('cb') at each board's centroid.
    """
    sg, _, _, _ = try_import_shapely()
    rows = []
    for b in boards:
        cb_num = get_board_display_id(b)
        if not cb_num:
            continue
        p = b["geom"].representative_point() if sg else None
        if p is None:
            continue
        rows.append({"lon": float(p.x), "lat": float(p.y), "cb": cb_num})
    if not rows:
        return None
    return pdk.Layer(
        "TextLayer",
        data=rows,
        get_position='[lon, lat]',
        get_text='cb',
        get_size=text_size,
        get_angle=0,
        get_color=[20, 20, 20],
        get_alignment_baseline='"center"'
    )


# -----------------------------
# NYC CB prefix rules & helpers
# -----------------------------
BORO_PREFIX = {
    "1": "Manhattan",
    "2": "Bronx",
    "3": "Brooklyn",
    "4": "Queens",
    "5": "Staten Island",
}

def interpret_cb_code(code_str: str):
    """
    Interpret a 3-digit community board code like '308' -> ('Brooklyn', 8).
    Returns (borough_name, board_number) or (None, None) if not parseable.
    """
    if not code_str:
        return None, None
    s = "".join([c for c in str(code_str) if c.isdigit()])
    if len(s) < 3:
        return None, None
    prefix = s[0]
    borough = BORO_PREFIX.get(prefix)
    try:
        num = int(s[1:3])
    except Exception:
        num = None
    if borough and num and 1 <= num <= 18:
        return borough, num
    if borough and num:
        return borough, num
    return None, None


# -----------------------------
# Claude (Bedrock) helper
# -----------------------------
def bedrock_enabled() -> bool:
    return all([
        (os.getenv("AWS_ACCESS_KEY_ID") or ""),
        (os.getenv("AWS_SECRET_ACCESS_KEY") or ""),
        (os.getenv("AWS_REGION") or "")
    ])

def call_claude(prompt: str, system: str = None, temperature: float = 0.2, max_tokens: int = 600) -> str:
    import json
    import boto3
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError, ClientError

    cfg = Config(
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        read_timeout=20,
        connect_timeout=5,
        retries={"max_attempts": 2, "mode": "standard"},
    )
    client = boto3.client("bedrock-runtime", config=cfg)
    model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }
    if system:
        body["system"] = system

    try:
        resp = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        payload = json.loads(resp["body"].read())
        parts = []
        for blk in payload.get("content", []):
            if blk.get("type") == "text":
                parts.append(blk.get("text", ""))
        text = "".join(parts).strip()
        return text or "(no text returned)"
    except (BotoCoreError, ClientError) as e:
        # surface a friendly, visible error in the UI
        st.error(f"Claude/Bedrock error ({type(e).__name__}): {e}")
        return "(Bedrock error)"

# -----------------------------
# Per-map / per-result assistant UI helper
# -----------------------------
def render_map_assistant(block_key: str, context: dict, default_hint: str, describe_by_cb: bool = False):
    chat_key = f"chat_{block_key}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    st.markdown("##### 🧠 Describe this map & ask follow-ups")

    cols = st.columns([1, 2])
    with cols[0]:
        if st.button("Describe this map", key=f"describe_{block_key}"):
            system_msg = (
                "You are an assistant for a flood risk hackathon. Explain findings clearly for business users. "
                "Be concise and point out hotspots, drivers, and practical green-infrastructure ideas."
            )
            numbering_rules = (
                "\n\nNYC Community Board numbering:\n"
                "- Prefix 1=Manhattan, 2=Bronx, 3=Brooklyn, 4=Queens, 5=Staten Island.\n"
                "- A three-digit code XYZ means borough X and board YZ (e.g., 308 → Brooklyn CB 8, 402 → Queens CB 2)."
            )
            extra = "\n\nIMPORTANT: Refer to locations by their community board NUMBER only (e.g., 305)."
            prompt = (
                "Use the following JSON context to describe the current map and give planning suggestions. "
                "Focus on what stands out, why, and what to do next.\n\n"
                + json.dumps(context, indent=2)
                + numbering_rules
            )
            if describe_by_cb:
                prompt += extra
            try:
                if bedrock_enabled():
                    answer = call_claude(prompt, system=system_msg)
                else:
                    answer = (
                        "Boards with the highest index cluster near low-lying, coastal areas. "
                        "Prioritize inlet cleaning and curb-extension bioswales at boards such as 305 and 311."
                    )
                st.session_state[chat_key].append({"role": "assistant", "text": answer})
            except Exception as e:
                st.error(f"Claude/Bedrock error: {e}")

    with cols[1]:
        q = st.text_input("Follow-up question", value="", placeholder=default_hint, key=f"q_{block_key}")
        if st.button("Ask", key=f"ask_{block_key}"):
            system_msg = (
                "Helpful, technically accurate, but succinct flood-planning assistant. "
                "If uncertain, say so briefly and suggest a next step."
            )
            prior = st.session_state[chat_key]
            transcript = []
            for turn in prior[-6:]:
                transcript.append(f"{turn['role'].capitalize()}: {turn['text']}")
            transcript_text = "\n".join(transcript)

            numbering_rules = (
                "\n\nNYC Community Board numbering:\n"
                "- Prefix 1=Manhattan, 2=Bronx, 3=Brooklyn, 4=Queens, 5=Staten Island.\n"
                "- A three-digit code XYZ means borough X and board YZ."
            )
            prompt = (
                "Context JSON for the current map:\n"
                + json.dumps(context, indent=2)
                + numbering_rules
                + "\n\nRecent exchange:\n"
                + transcript_text
                + "\n\nUser follow-up:\n"
                + q
            )
            if describe_by_cb:
                prompt += "\n\nAgain, reference community boards by NUMBER only."
            try:
                if bedrock_enabled():
                    answer = call_claude(prompt, system=system_msg)
                else:
                    answer = (
                        "Hotspots align with high imperviousness. "
                        "Target curbside bio-retention near boards 301 and 302."
                    )
                st.session_state[chat_key].append({"role": "user", "text": q})
                st.session_state[chat_key].append({"role": "assistant", "text": answer})
            except Exception as e:
                st.error(f"Claude/Bedrock error: {e}")

    if st.session_state[chat_key]:
        with st.expander("Conversation for this map", expanded=True):
            for turn in st.session_state[chat_key][-8:]:
                speaker = "You" if turn["role"] == "user" else "Assistant"
                st.markdown(f"**{speaker}:** {turn['text']}")


# -----------------------------
# Sidebar + landing
# -----------------------------
def sidebar_agents():
    with st.sidebar:
        st.markdown("### NYC Resilience AI Agent")
        st.caption("Demo – Flooding & UHI")
        st.divider()
        # Status checkboxes
        st.checkbox("Mapping Agent", value=True, disabled=True)
        st.checkbox("Forecasting Agent", value=True, disabled=True)
        st.checkbox("Optimization Agent", value=False, disabled=True)
        st.checkbox("Green Infrastructure Agent", value=False, disabled=True)
        st.divider()
        # Navigation buttons
        st.button("🏠 Home", use_container_width=True, on_click=lambda: st.session_state.update(page="landing"))
        st.button("🌊 Flooding", use_container_width=True, on_click=lambda: st.session_state.update(page="flooding"))
        st.button("📈 Forecasting", use_container_width=True, on_click=lambda: st.session_state.update(page="forecasting"))
        st.divider()

        # 🛑 Community Board Agent (beta) is deactivated by flag
        if ENABLE_CB_AGENT_SIDEBAR:
            with st.expander("🧭 Community Board Agent (beta)", expanded=False):
                st.write("Browse official NYC Community Boards info:")
                st.link_button(
                    "Open NYC CAU Community Boards",
                    "https://www.nyc.gov/site/cau/community-boards/community-boards.page",
                    use_container_width=True
                )
                st.caption("Tip: find the borough & board you care about, then come back and ask questions here.")

                code_in = st.text_input("Enter a 3-digit CB code (e.g., 308 → Brooklyn CB 8)", "")
                if code_in.strip():
                    boro, num = interpret_cb_code(code_in.strip())
                    if boro and num:
                        st.success(f"Interpreted: **{boro} Community Board {num}**")
                    else:
                        st.warning("Could not interpret that code. Expected a 3-digit like 308, 402, 115, etc.")

                q = st.text_area("Ask about a Community Board (e.g., 'What do you know about CB 308?')", "")
                if st.button("Ask Claude about a CB"):
                    if not q.strip():
                        st.warning("Please type a question first.")
                    else:
                        try:
                            numbering_rules = (
                                "When the user mentions a 3-digit code XYZ, interpret it as borough prefix X and board YZ.\n"
                                "Prefixes: 1=Manhattan, 2=Bronx, 3=Brooklyn, 4=Queens, 5=Staten Island.\n"
                                "Examples: 308 → Brooklyn CB 8; 402 → Queens CB 2; 115 → Manhattan CB 15."
                            )
                            base_prompt = (
                                "You are helping with NYC Community Board context for planning. "
                                "Answer briefly and clearly. If unsure, say so and suggest the official NYC CB page "
                                "(https://www.nyc.gov/site/cau/community-boards/community-boards.page).\n\n"
                                + numbering_rules
                                + "\n\nUser question:\n"
                                + q
                            )
                            if bedrock_enabled():
                                answer = call_claude(base_prompt, system="NYC CB assistant", temperature=0.2)
                            else:
                                answer = (
                                    "If Claude is connected, I’ll interpret codes like 308 as Brooklyn CB 8 and summarize what’s generally known. "
                                    "For official details, use the NYC CB page linked above."
                                )
                            st.info(answer)
                        except Exception as e:
                            st.error(f"Claude/Bedrock error: {e}")

        if bedrock_enabled():
            st.success("Claude (Bedrock) connected.")
        else:
            missing = [k for k in ("AWS_ACCESS_KEY_ID","AWS_SECRET_ACCESS_KEY","AWS_REGION") if not os.getenv(k)]
            st.info("Claude: missing " + ", ".join(missing) + " — set in Streamlit Secrets.")


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

    st.divider()
    st.markdown("#### Agentic flow (human-in-the-loop)")
    st.graphviz_chart('''
        digraph G {
            rankdir=LR;
            node [shape=box, style=rounded];
            User -> "Community Board Agent (beta)" [style=dashed];
            User -> "Flood Mapping Agent";
            "Flood Mapping Agent" -> "Forecasting Agent" [label="precip forecast (shared)"];
            "Community Board Agent (beta)" -> "Forecasting Agent" [style=dashed, label="context (CB directory/info)"];
            "Forecasting Agent" -> User [label="Explain + QA"];
            "Forecasting Agent" -> "Optimization Agent" [style=dashed, label="(soon)"];
            "Optimization Agent" -> "Green Infrastructure Agent" [style=dashed, label="(soon)"];
        }
    ''')


# -----------------------------
# Mapping Agent
# -----------------------------
def flooding_mapping_agent():
    st.title("🌊 Flooding → Mapping Agent (NYC)")
    st.caption("Displays Community Board polygons, places one radar point per board, simulates precipitation, and colors the boards. Saves the forecast for downstream agents.")

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

    # Cache boards & their map extents for downstream agents
    st.session_state[SESSION_KEYS["boards"]] = boards

    # ---- Draw polygons immediately (outline on Esri Light Gray) ----
    fc_outlines = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": get_board_display_id(b)}, "geometry": b["feature_geom"]} for b in boards
    ]}
    lat_c, lon_c, zoom_c = fc_center_and_zoom(fc_outlines)
    st.session_state[SESSION_KEYS["lat"]] = lat_c
    st.session_state[SESSION_KEYS["lon"]] = lon_c
    st.session_state[SESSION_KEYS["zoom"]] = zoom_c

    outline = pdk.Layer(
        "GeoJsonLayer",
        data=fc_outlines,
        pickable=True,
        stroked=True,
        filled=False,
        extruded=False,
        wireframe=True,
        get_line_color=[255, 0, 0, 255],
        get_line_width=200,
        line_width_min_pixels=3,
        line_width_max_pixels=10,
    )
    label_layer = cb_label_layer_from_boards(boards, text_size=12)
    deck0 = pdk.Deck(
        layers=[esri_light_gray_basemap(), outline] + ([label_layer] if label_layer else []),
        initial_view_state=pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom_c),
        tooltip={"html": "<b>Board #:</b> {name}", "style": {"backgroundColor": "white", "color": "black"}}
    )
    st.pydeck_chart(deck0, use_container_width=True)
    st.caption("Community Boards/Districts (outline).")

    # Prepare attributes table for download only (not displayed)
    try:
        attrs_list = []
        for b in boards:
            row = {
                "unit_id": b["unit_id"],
                "cb_number": get_board_display_id(b),
                **b.get("attrs", {})
            }
            attrs_list.append(row)

        attrs_df = pd.DataFrame(attrs_list)

        # Reorder columns to show key attributes first
        priority_cols = ["unit_id", "cb_number", "boro_cd", "Buildings", "Elevation",
                         "Slope", "Commuting", "Imperv", "Footprint",
                         "BLDperArea", "FTPperArea", "shape_area", "shape_leng", "Area"]

        display_cols = [col for col in priority_cols if col in attrs_df.columns]
        other_cols = [col for col in attrs_df.columns if col not in display_cols]
        final_cols = display_cols + other_cols
        attrs_df = attrs_df[final_cols]

        # Drop rows where ALL tracked static attributes are None/NaN
        tracked = ["Buildings","Elevation","Slope","Commuting","Imperv","Footprint","BLDperArea","FTPperArea"]
        if any(col in attrs_df.columns for col in tracked):
            sub = [c for c in tracked if c in attrs_df.columns]
            all_missing_mask = attrs_df[sub].isna().all(axis=1)
            attrs_df = attrs_df.loc[~all_missing_mask].copy()

            # Push partial-missing rows to bottom
            some_missing_mask = attrs_df[sub].isna().any(axis=1)
            attrs_df["__missing_rank__"] = np.where(some_missing_mask, 1, 0)
            attrs_df = attrs_df.sort_values(["__missing_rank__", "cb_number"]).drop(columns="__missing_rank__")

        st.download_button(
            "Download Full Attributes Table (CSV)",
            data=attrs_df.to_csv(index=False).encode("utf-8"),
            file_name="nyc_boards_full_attributes.csv",
            mime="text/csv",
            use_container_width=True,
        )
    except Exception as e:
        st.warning(f"Could not prepare attributes table: {e}")

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
    label_layer = cb_label_layer_from_boards(boards, text_size=12)
    if label_layer:
        layers1.append(label_layer)
    deck1 = pdk.Deck(
        layers=layers1,
        initial_view_state=pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom_c),
        tooltip={"html": "<b>Board #:</b> {cb}<br/><b>Precip:</b> {value} mm/hr", "style": {"backgroundColor":"white","color":"black"}}
    )
    st.pydeck_chart(deck1, use_container_width=True)

    # Download of hourly, board-aggregated values
    dl_df = board_series.rename("precip_mm_hr").reset_index()
    dl_df.columns = ["unit_id", "precip_mm_hr"]
    dl_df["cb_number"] = dl_df["unit_id"].map(lambda uid: normalize_cb_id(uid))
    st.download_button(
        "Download this hour (board-aggregated) CSV",
        data=dl_df.to_csv(index=False).encode("utf-8"),
        file_name=f"nyc_boards_precip_{dt_selected:%Y%m%d_%H}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption(f"Hourly precip range: min={board_series.min():.2f} mm/hr, max={board_series.max():.2f} mm/hr.")

    # --- Per-map assistant: Single-hour precipitation ---
    top_points = hour_df.sort_values("precip_mm_hr", ascending=False).head(5)[["cell_id","lat","lon","precip_mm_hr"]]
    top_points = top_points.assign(cb=top_points["cell_id"].map(lambda x: normalize_cb_id(x)))
    context_single = {
        "type": "single_hour_precip",
        "timestamp_local": dt_selected.isoformat(),
        "stats": {
            "min_mm_hr": float(board_series.min()),
            "max_mm_hr": float(board_series.max()),
            "mean_mm_hr": float(board_series.mean())
        },
        "top5_points": top_points[["cb","lat","lon","precip_mm_hr"]].to_dict(orient="records")
    }
    render_map_assistant(
        block_key="single_hour",
        context=context_single,
        default_hint="Where are the hotspots and why there?",
        describe_by_cb=True
    )

    # ---- Step 3b: Community Board Attributes Visualization ----
    st.divider()
    st.subheader("Community Board Attributes Visualization")
    st.caption("Visualize key attributes from the merged data across community boards.")

    has_merged_data = any(board.get('attrs', {}).get('Buildings') is not None for board in boards)

    if not has_merged_data:
        st.warning("No merged CSV data available. Please ensure DataForBoxPlots.csv is loaded.")
    else:
        attribute_options = {
            "Buildings per Area (BLDperArea)": "BLDperArea",
            "Footprint per Area (FTPperArea)": "FTPperArea",
            "Percentage Impervious Cover": "Imperv",
            "Elevation": "Elevation",
            "Slope": "Slope",
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
        st.info(attribute_info(selected_attr_label))

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

        if attr_series.max() == 0 and attr_series.min() == 0:
            st.warning(f"No valid data found for {selected_attr_label}.")
        else:
            breaks_attr = compute_breaks(attr_series, k=len(BLUES))

            feats_attr = []
            for b in boards:
                v = float(attr_series.get(b["unit_id"], 0.0))
                feats_attr.append({
                    "type": "Feature",
                    "properties": {
                        "unit_id": b["unit_id"],
                        "cb": get_board_display_id(b),
                        "name": b.get("name", b["unit_id"]),
                        "value": v,
                        "attribute": selected_attr_label,
                        "fill_color": color_for_value_dynamic(v, breaks_attr),
                    },
                    "geometry": b["feature_geom"]
                })
            fc_attr = {"type": "FeatureCollection", "features": feats_attr}

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

            label_layer = cb_label_layer_from_boards(boards, text_size=12)
            layers_attr = [esri_light_gray_basemap(), cb_attr_layer] + ([label_layer] if label_layer else [])

            deck_attr = pdk.Deck(
                layers=layers_attr,
                initial_view_state=pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom_c),
                tooltip={
                    "html": "<b>Board #:</b> {cb}<br/><b>{attribute}:</b> {value}",
                    "style": {"backgroundColor": "white", "color": "black"}
                }
            )

            st.pydeck_chart(deck_attr, use_container_width=True)

            attr_top = pd.DataFrame({
                "cb": [get_board_display_id(b) for b in boards],
                "unit_id": [b["unit_id"] for b in boards],
                "value": [attr_series.get(b["unit_id"], 0.0) for b in boards]
            }).sort_values("value", ascending=False).head(7)

            context_attr = {
                "type": "attribute_map",
                "attribute": selected_attr_label,
                "stats": {
                    "min": float(attr_series.min()),
                    "max": float(attr_series.max()),
                    "mean": float(attr_series.mean()),
                    "median": float(attr_series.median())
                },
                "top_boards": attr_top[["cb","value"]].to_dict(orient="records")
            }
            render_map_assistant(
                block_key=f"attr_{attribute_options[selected_attr_label]}",
                context=context_attr,
                default_hint=f"What does high {selected_attr_label.lower()} imply for flooding here?",
                describe_by_cb=True
            )

            # Quick stats
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("Minimum", f"{attr_series.min():.2f}")
            with col_stat2:
                st.metric("Maximum", f"{attr_series.max():.2f}")
            with col_stat3:
                st.metric("Mean", f"{attr_series.mean():.2f}")
            with col_stat4:
                st.metric("Median", f"{attr_series.median():.2f}")

            dl_attr_df = attr_series.rename(selected_attr).reset_index()
            dl_attr_df.columns = ["unit_id", selected_attr]
            dl_attr_df["cb_number"] = dl_attr_df["unit_id"].map(lambda uid: normalize_cb_id(uid))
            st.download_button(
                f"Download {selected_attr_label} Data (CSV)",
                data=dl_attr_df.to_csv(index=False).encode("utf-8"),
                file_name=f"nyc_boards_{selected_attr}.csv",
                mime="text/csv",
                use_container_width=True,
            )

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
        baseline_method = st.radio(
            "Baseline precipitation aggregation over radar points:",
            options=["Mean over radars", "Max over radars"],
            index=0,
            help="This becomes the single citywide baseline time series shared with downstream agents."
        )
        st.session_state["baseline_method_choice"] = baseline_method

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

            # Shared baseline for downstream agents (mean or max over radars)
            if st.session_state.get("baseline_method_choice", "Mean over radars") == "Max over radars":
                precip_baseline = precip_matrix.max(axis=1)
                agg_label = "max"
            else:
                precip_baseline = precip_matrix.mean(axis=1)
                agg_label = "mean"

            precip_df = pd.DataFrame({
                "timestamp": hours_idx,
                "precip_mm_hr": np.round(precip_baseline, 3)
            })
            st.session_state[SESSION_KEYS["precip_df"]] = precip_df
            st.session_state[SESSION_KEYS["precip_meta"]] = {
                "source": "MappingAgent.mock_simulate_hourly_precip_range",
                "start": str(start_dt),
                "end_exclusive": str(end_dt),
                "radar_points": len(RADAR_DF),
                "aggregation": f"{agg_label} over radar points per hour",
                "baseline_method": st.session_state.get("baseline_method_choice", "Mean over radars")
            }
        st.success("Precipitation generated, daily metrics computed, and baseline forecast shared with Forecasting agent.")

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
        st.info(precip_metric_info(metric))
        opacity2 = st.slider("Polygon fill opacity (summary map)", 0.2, 1.0, 0.8, key="sum_opacity")

        if metric.startswith("Average of daily Max"):
            series = st.session_state["avg_of_daily_max"]; label = "avg_daily_max_mm_hr"; value_units = "mm/hr"
        elif metric.startswith("Average of daily Average"):
            series = st.session_state["avg_of_daily_mean"]; label = "avg_daily_avg_mm_hr"; value_units = "mm/hr"
        else:
            series = st.session_state["avg_of_daily_total"]; label = "avg_daily_total_mm"; value_units = "mm/day"

        board_series2, fc_sum = aggregate_series_to_boards(series, RADAR_DF, boards)
        cb_sum = pdk.Layer(
            "GeoJsonLayer",
            data=fc_sum,
            pickable=True, stroked=True, filled=True,
            get_fill_color="properties.fill_color",
            get_line_color=[40,40,40], lineWidthMinPixels=2,
            opacity=opacity2
        )
        label_layer = cb_label_layer_from_boards(boards, text_size=12)
        layers_sum = [esri_light_gray_basemap(), cb_sum] + ([label_layer] if label_layer else [])
        deck2 = pdk.Deck(
            layers=layers_sum,
            initial_view_state=pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom_c),
            tooltip={"html": "<b>Board #:</b> {cb}<br/><b>Value:</b> {value}",
                     "style": {"backgroundColor":"white","color":"black"}}
        )
        st.pydeck_chart(deck2, use_container_width=True)

        with st.expander("Color Legend (summary map)", expanded=False):
            breaks_attr = compute_breaks(board_series2, k=len(BLUES))
            legend_rows = []
            rngs = [board_series2.min()] + breaks_attr + [board_series2.max()]
            for i in range(len(BLUES)):
                legend_rows.append({
                    "Range": f"{rngs[i]:.2f} – {rngs[i+1]:.2f}",
                    "Color (RGB)": f"rgb({BLUES[i][0]}, {BLUES[i][1]}, {BLUES[i][2]})"
                })
            st.dataframe(pd.DataFrame(legend_rows), hide_index=True, use_container_width=True)

        st.caption(f"Summary range: min={board_series2.min():.2f}, max={board_series2.max():.2f} ({label}).")

        context_summary = {
            "type": "summary_metric_map",
            "metric_label": metric,
            "value_units": value_units,
            "stats": {
                "min": float(board_series2.min()),
                "max": float(board_series2.max()),
                "mean": float(board_series2.mean())
            },
            "note": "Values represent the average across days of the selected daily statistic.",
            "boards": [{"cb": get_board_display_id(b), "value": float(board_series2.get(b["unit_id"], 0.0))}
                       for b in boards]
        }
        render_map_assistant(
            block_key=f"summary_{label}",
            context=context_summary,
            default_hint="Which boards stand out in this summary and what actions follow?",
            describe_by_cb=True
        )

    st.info("✅ Baseline precipitation is now available to the 📈 Forecasting Agent (see sidebar).")


# -----------------------------
# Forecasting Agent (consumes shared precip)
# -----------------------------
def forecasting_agent():
    st.title("📈 Forecasting Agent (boards + static modifiers)")
    st.caption("Consumes the precipitation forecast saved by the Flood Mapping agent. Produces a board-level flood-index choropleth that accounts for static risk (buildings, impervious, slope, elevation, footprint).")

    boards = st.session_state.get(SESSION_KEYS["boards"])
    lat_c = st.session_state.get(SESSION_KEYS["lat"])
    lon_c = st.session_state.get(SESSION_KEYS["lon"])
    zoom_c = st.session_state.get(SESSION_KEYS["zoom"])
    precip_df = st.session_state.get(SESSION_KEYS["precip_df"])

    if any(v is None for v in [boards, lat_c, lon_c, zoom_c]) or precip_df is None:
        st.error("This agent needs the Flood Mapping agent to run first (to load boards and produce a baseline precipitation forecast). Go to 🌊 Flooding, generate a date-range forecast, then return.")
        return

    with st.expander("See agent handoff", expanded=True):
        st.graphviz_chart('''
            digraph G {
                rankdir=LR;
                node [shape=box, style=rounded];
                "User" -> "Flood Mapping Agent";
                "Flood Mapping Agent" -> "Forecasting Agent" [label="precip time-series (baseline)"];
                "Community Board Agent (beta)" -> "Forecasting Agent" [style=dashed, label="context (CB info)"];
                "Forecasting Agent" -> "User" [label="Explain + QA"];
            }
        ''')

    # --- Model descriptions + info toggles ---
    with st.expander("How the models work (tap to expand)", expanded=False):
        st.markdown("""
**Rule-based (multiplier × precip)**  
- Method: For each board and hour, `flood_index = precip_mm_hr × static_multiplier`.  
- Static multiplier: Derived from scaled **Buildings, Footprint, Impervious, Slope, Elevation, Commuting** via your chosen weight set (Model 1).

**Linear residual (p + α·p_prev)**  
- Method: `flood_index_t = m·p_t + α·(m·p_{t-1})`, where `m` is the static multiplier, `p_t` is baseline precip at time *t*, and `α` captures short-term persistence.

**Graph diffusion (spatial smoothing)**  
- Method: Start with `m·p_t` then diffuse across neighboring boards based on geographic proximity and static similarity.  
- Intuition: Adjacent boards influence each other during events, smoothing noisy spikes while preserving hotspots.
        """)

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        with st.expander("What are the **static values**?", expanded=False):
            st.markdown("""
The static values are board-level, time-invariant features (Buildings, Footprint, Impervious, Slope, Elevation, Commuting).
They are robustly scaled to ~[0,1] and combined with **Model 1** weights to form a **static multiplier** that up/down-weights the precipitation signal per board.
            """)
    with col_info2:
        with st.expander("What is the **precipitation baseline**?", expanded=False):
            st.markdown("""
The Flood Mapping agent produced a single, citywide hourly series by aggregating precipitation over radar points (you chose **Mean** or **Max**).
This exact series is used here; we **do not** regenerate precipitation.
            """)

    # 1) Static-variable weighting and model
    st.subheader("1) Static-variable weighting and model")
    col1, col2 = st.columns(2)
    with col1:
        weight_options = ["Random Forest (Model 1)"]
        default_key = weight_options[0] if weight_options[0] in WEIGHT_SETS else list(WEIGHT_SETS.keys())[0]
        weight_key = st.selectbox("Static importance set", [default_key], index=0)
    with col2:
        model = st.selectbox(
            "Forecasting model",
            ["Rule-based (multiplier × precip)",
             "Linear residual (p + α·p_prev)",
             "Graph diffusion (spatial smoothing)"],
            index=2
        )

    multiplier_by_board, static_scaled = build_board_multiplier(boards, weight_set_key=weight_key)
    st.session_state[SESSION_KEYS["static_scaled"]] = static_scaled
    st.session_state[SESSION_KEYS["weights_used"]] = weight_key

    with st.expander("Transparency: show static variables (scaled to ~[0,1])", expanded=False):
        st.dataframe(static_scaled.reset_index().rename(columns={"index": "unit_id"}), use_container_width=True, hide_index=True)

    with st.expander("Transparency: show precipitation baseline used here", expanded=False):
        st.dataframe(precip_df, use_container_width=True, hide_index=True)

    # 2) Run the chosen model
    st.subheader("2) Run forecast")
    if model.startswith("Rule-based"):
        fcst_long = forecast_rule_based(precip_df, multiplier_by_board)
        method_desc = "Rule-based: flood_index = precip_mm_hr × static_multiplier"
    elif model.startswith("Linear"):
        fcst_long = forecast_linear_residual(precip_df, multiplier_by_board, alpha=0.15)
        method_desc = "Linear residual: flood_index_t = m·p_t + α·(m·p_{t-1})"
    else:
        cent = pd.DataFrame({
            "unit_id": [b["unit_id"] for b in boards],
            "lat": [b["geom"].representative_point().y for b in boards],
            "lon": [b["geom"].representative_point().x for b in boards],
        }).set_index("unit_id")
        fcst_long = forecast_graph_diffusion(
            precip_df, multiplier_by_board, cent, static_scaled,
            k_geo=6, beta=0.5, diffusion_steps=2, gamma=0.3
        )
        method_desc = "Graph diffusion: smooth m·p_t across neighboring boards with static-guided edges"

    st.caption(f"**Method:** {method_desc}")

    # 3) Pick a time to visualize + QA
    st.subheader("3) Visualize a time step & ask questions")
    times = fcst_long["timestamp"].astype(str).unique().tolist()
    t_sel_str = st.selectbox("Timestamp", times, index=0)
    t_sel = pd.Timestamp(t_sel_str)
    snap = fcst_long[fcst_long["timestamp"] == t_sel].set_index("unit_id")["flood_index"]

    s = snap.fillna(0.0)

    def _compute_breaks(vals, k=7):
        v = vals.dropna().to_numpy()
        if len(v) == 0: return [0]*(k-1)
        qs = np.linspace(0,1,k); b = np.quantile(v, qs)
        return [float(b[i]) for i in range(1,k)]
    def _color_for_value_dynamic(val, breaks):
        idx=0
        for b in breaks:
            if val>b: idx+=1
            else: break
        return BLUES[min(idx,len(BLUES)-1)]

    br = _compute_breaks(s, k=7)
    feats = []
    for b in boards:
        uid = b["unit_id"]
        v = float(s.get(uid, 0.0))
        feats.append({
            "type":"Feature",
            "properties":{
                "unit_id": uid,
                "cb": get_board_display_id(b),
                "name": b.get("name", uid),
                "flood_index": v,
                "fill_color": _color_for_value_dynamic(v, br)
            },
            "geometry": b["feature_geom"]
        })
    fc = {"type":"FeatureCollection", "features": feats}
    layer = pdk.Layer(
        "GeoJsonLayer", data=fc,
        pickable=True, stroked=True, filled=True,
        get_fill_color="properties.fill_color",
        get_line_color=[40,40,40], lineWidthMinPixels=2, opacity=0.85
    )
    label_layer = cb_label_layer_from_boards(boards, text_size=12)
    layers_fc = [esri_light_gray_basemap(), layer] + ([label_layer] if label_layer else [])
    deck = pdk.Deck(
        layers=layers_fc,
        initial_view_state=pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom_c),
        tooltip={"html":"<b>Board #:</b> {cb}<br/><b>Flood index:</b> {flood_index:.2f}",
                 "style":{"backgroundColor":"white","color":"black"}}
    )
    st.pydeck_chart(deck, use_container_width=True)

    with st.expander("Color Legend (snapshot)", expanded=False):
        legend_rows = []
        rngs = [float(s.min())] + br + [float(s.max())]
        for i in range(len(BLUES)):
            legend_rows.append({
                "Range": f"{rngs[i]:.2f} – {rngs[i+1]:.2f}",
                "Color (RGB)": f"rgb({BLUES[i][0]}, {BLUES[i][1]}, {BLUES[i][2]})"
            })
        st.dataframe(pd.DataFrame(legend_rows), hide_index=True, use_container_width=True)

    st.download_button(
        "Download full forecast (long table, all times × boards)",
        data=fcst_long.to_csv(index=False).encode("utf-8"),
        file_name="forecast_long_boards.csv", mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        f"Static multiplier range across boards: "
        f"{multiplier_by_board.min():.2f} – {multiplier_by_board.max():.2f} (using **{weight_key}**)."
    )

    context_forecast = {
        "type": "forecast_snapshot",
        "timestamp": t_sel_str,
        "method": method_desc,
        "static_weights": weight_key,
        "by_board": [{"cb": get_board_display_id(b), "flood_index": float(s.get(b['unit_id'], 0.0))} for b in boards],
        "stats": {
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "median": float(s.median())
        }
    }
    render_map_assistant(
        block_key=f"forecast_{model.split()[0].lower()}",
        context=context_forecast,
        default_hint="Where should we prepare first and why?",
        describe_by_cb=True
    )

    # 4) Aggregate forecast over a selected time range
    st.divider()
    st.subheader("4) Aggregate forecast over a time range (mean flood index per board)")
    all_ts = sorted(fcst_long["timestamp"].astype(str).unique().tolist())
    start_ts = st.select_slider("Start time", options=all_ts, value=all_ts[0], key="fc_range_start")
    end_ts = st.select_slider("End time", options=all_ts, value=all_ts[-1], key="fc_range_end")

    start_idx = all_ts.index(start_ts)
    end_idx = all_ts.index(end_ts)
    if start_idx > end_idx:
        st.error("End time must be after start time.")
        return

    mask = (fcst_long["timestamp"].astype(str) >= start_ts) & (fcst_long["timestamp"].astype(str) <= end_ts)
    fc_window = fcst_long.loc[mask].copy()
    if fc_window.empty:
        st.warning("No forecast entries in the selected window.")
        return

    mean_by_board = fc_window.groupby("unit_id")["flood_index"].mean()

    br2 = _compute_breaks(mean_by_board, k=7)
    feats2 = []
    for b in boards:
        uid = b["unit_id"]
        v = float(mean_by_board.get(uid, 0.0))
        feats2.append({
            "type":"Feature",
            "properties":{
                "unit_id": uid,
                "cb": get_board_display_id(b),
                "name": b.get("name", uid),
                "flood_index_mean": v,
                "fill_color": _color_for_value_dynamic(v, br2)
            },
            "geometry": b["feature_geom"]
        })
    fc2 = {"type":"FeatureCollection", "features": feats2}
    layer2 = pdk.Layer(
        "GeoJsonLayer", data=fc2,
        pickable=True, stroked=True, filled=True,
        get_fill_color="properties.fill_color",
        get_line_color=[40,40,40], lineWidthMinPixels=2, opacity=0.85
    )
    label_layer = cb_label_layer_from_boards(boards, text_size=12)
    layers_fc2 = [esri_light_gray_basemap(), layer2] + ([label_layer] if label_layer else [])
    deck2 = pdk.Deck(
        layers=layers_fc2,
        initial_view_state=pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom_c),
        tooltip={"html":"<b>Board #:</b> {cb}<br/><b>Mean Flood index:</b> {flood_index_mean:.2f}",
                 "style":{"backgroundColor":"white","color":"black"}}
    )
    st.pydeck_chart(deck2, use_container_width=True)

    with st.expander("Color Legend (range mean)", expanded=False):
        rngs2 = [float(mean_by_board.min())] + br2 + [float(mean_by_board.max())]
        legend_rows2 = []
        for i in range(len(BLUES)):
            legend_rows2.append({
                "Range": f"{rngs2[i]:.2f} – {rngs2[i+1]:.2f}",
                "Color (RGB)": f"rgb({BLUES[i][0]}, {BLUES[i][1]}, {BLUES[i][2]})"
            })
        st.dataframe(pd.DataFrame(legend_rows2), hide_index=True, use_container_width=True)

    out_mean = mean_by_board.rename("mean_flood_index").reset_index()
    out_mean["cb_number"] = out_mean["unit_id"].map(lambda uid: normalize_cb_id(uid))
    st.download_button(
        "Download range-mean flood index by board (CSV)",
        data=out_mean.to_csv(index=False).encode("utf-8"),
        file_name=f"forecast_mean_by_board_{start_ts}_to_{end_ts}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    context_range = {
        "type": "forecast_range_mean",
        "start": start_ts,
        "end": end_ts,
        "by_board": [{"cb": get_board_display_id(b), "mean_flood_index": float(mean_by_board.get(b['unit_id'], 0.0))}
                     for b in boards],
        "stats": {
            "min": float(mean_by_board.min()),
            "max": float(mean_by_board.max()),
            "mean": float(mean_by_board.mean()),
            "median": float(mean_by_board.median())
        }
    }
    render_map_assistant(
        block_key=f"forecast_mean_{start_ts}_{end_ts}",
        context=context_range,
        default_hint="Which boards should prioritize maintenance and GI over this multi-hour window?",
        describe_by_cb=True
    )


# -----------------------------
# Router + sidebar
# -----------------------------
def sidebar_and_route():
    with st.sidebar:
        st.markdown("### Agents")
        st.button("🏠 Home", on_click=lambda: st.session_state.update(page="landing"))
        st.button("🌊 Flood Mapping", on_click=lambda: st.session_state.update(page="flooding"))
        st.button("📈 Forecasting", on_click=lambda: st.session_state.update(page="forecasting"))

    if st.session_state.page == "landing":
        landing_page()
    elif st.session_state.page == "flooding":
        flooding_mapping_agent()
    elif st.session_state.page == "forecasting":
        forecasting_agent()


# ===== Render =====
sidebar_agents()
sidebar_and_route()
