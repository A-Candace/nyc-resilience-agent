# app.py
from io import BytesIO
import os
import math
import json
from datetime import datetime, timedelta, timezone, date
from typing import Optional  # <-- added

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from dotenv import load_dotenv

# Optional / conditional imports
try:
    import shapely.geometry as sg
    from shapely.geometry import shape as shp_shape, mapping as shp_mapping
    from shapely.ops import transform as shp_transform
    SHAPELY_OK = True
except Exception:
    SHAPELY_OK = False

# ---- MLflow (works with 2.17.x, and conditionally with genai if present) ----
import mlflow

def _mlflow_genai_available():
    try:
        import mlflow.genai  # noqa: F401
        return True
    except Exception:
        return False

GENAI_OK = _mlflow_genai_available()

# ---- Environment ----
load_dotenv()

# Claude / Bedrock config
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
HAS_BEDROCK = all([
    os.getenv("AWS_ACCESS_KEY_ID"),
    os.getenv("AWS_SECRET_ACCESS_KEY"),
    os.getenv("AWS_REGION")
])

# MLflow config
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "nyc-resilience-agent")
os.environ.setdefault("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")

if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
try:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
except Exception as e:
    # Show friendly error but continue (user may fix allowed hosts)
    st.toast(f"MLflow experiment set error: {e}", icon="⚠️")

# ---- App-wide UI config ----
st.set_page_config(page_title="NYC Resilience AI Agent", page_icon="🌆", layout="wide")

# =====================================================================
# Utilities (Claude calls, MLflow GenAI logging, geometry, coloring)
# =====================================================================

def call_claude(prompt: str, system: str = None, temperature: float = 0.2, max_tokens: int = 600) -> str:
    """
    Bedrock Claude call with safe defaults. Falls back to a stub if Bedrock isn't configured.
    """
    if not HAS_BEDROCK:
        return ("(Claude disabled — add AWS credentials & region)\n\n"
                "Draft explanation: Concentrations align with higher imperviousness "
                "and lower elevation. Use curb-inlet cleaning and bioswales where feasible.")

    import boto3
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError, ClientError

    cfg = Config(
        region_name=AWS_REGION,
        read_timeout=20,
        connect_timeout=5,
        retries={"max_attempts": 2, "mode": "standard"},
    )
    client = boto3.client("bedrock-runtime", config=cfg)

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
            modelId=BEDROCK_MODEL_ID,
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
        return f"(Bedrock error: {e})"

def genai_log(prompt_text: str, response_text: str, meta: dict):
    """
    Log a GenAI prompt→response to MLflow.
    - If mlflow.genai is available, use trace/span.
    - Otherwise, log params + artifacts as a simple fallback.
    """
    if GENAI_OK:
        import mlflow.genai as mgen
        with mgen.start_trace() as trace:
            with mgen.start_span(name=meta.get("span_name", "map_explanation")) as span:
                span.set_inputs({"prompt": prompt_text, **{f"meta_{k}": v for k, v in meta.items()}})
                span.set_outputs({"response": response_text})
        # Also attach params to the active run if any
        try:
            mlflow.log_params({f"explain_{k}": str(v) for k, v in meta.items()})
        except Exception:
            pass
    else:
        # Fallback: write small text artifacts
        try:
            with mlflow.start_run(run_name=f"genai_{meta.get('span_name','explain')}"):
                mlflow.log_params({f"explain_{k}": str(v) for k, v in meta.items()})
                # Save prompt & response into tmp artifacts
                import tempfile
                with tempfile.TemporaryDirectory() as td:
                    pth_prompt = os.path.join(td, "prompt.txt")
                    pth_resp = os.path.join(td, "response.txt")
                    with open(pth_prompt, "w", encoding="utf-8") as f:
                        f.write(prompt_text)
                    with open(pth_resp, "w", encoding="utf-8") as f:
                        f.write(response_text)
                    mlflow.log_artifacts(td, artifact_path="genai_logs")
        except Exception:
            pass

# --- Color helpers ---
# Red (high risk) to Yellow (low risk) color scheme
RED_YELLOW = [
    [255, 255, 178],  # Light yellow (lowest)
    [254, 217, 118],
    [254, 178, 76],
    [253, 141, 60],
    [252, 78, 42],
    [227, 26, 28],
    [177, 0, 38],     # Dark red (highest)
]

# 4-class palette (yellow→red) used by legends / equal-interval bins
PALETTE4 = [
    [255, 255, 178],
    [254, 217, 118],
    [253, 141, 60],
    [227, 26, 28],
]

def compute_breaks(values: pd.Series, k: int = 7) -> list:
    v = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
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
    idx = min(idx, len(RED_YELLOW) - 1)
    return RED_YELLOW[idx]

def normalize_cb_id(val):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    try:
        return str(int(float(val)))
    except Exception:
        return str(val)

def get_board_display_id(board):
    attrs = board.get("attrs", {}) or {}
    # For census tracts, try TRACTCE or GEOID
    tract_id = attrs.get("TRACTCE") or attrs.get("GEOID") or attrs.get("tract_id")
    if tract_id is not None and str(tract_id) != "":
        return normalize_cb_id(tract_id)
    return normalize_cb_id(board.get("unit_id", ""))

# Helper accessors used elsewhere
def get_attr_ci(attrs: dict, key: str):
    if key in attrs:
        return attrs.get(key)
    # case-insensitive
    lower = key.lower()
    for k, v in attrs.items():
        if (k or "").lower() == lower:
            return v
    return None

def coerce_nonneg_numeric(val, default=0.0):
    try:
        x = float(pd.to_numeric(val, errors="coerce"))
        if np.isnan(x) or x < 0:
            return 0.0
        return x
    except Exception:
        return 0.0 if default is None else default

def normalize_0_1(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)
    minv = s.min()
    maxv = s.max()
    if maxv - minv <= 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - minv) / (maxv - minv)

def equal_interval_bins_4(series_0_1: pd.Series) -> pd.Series:
    # 0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1
    s = pd.to_numeric(series_0_1, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    bins = pd.cut(s, bins=[0, 0.25, 0.5, 0.75, 1.0], labels=[0, 1, 2, 3], include_lowest=True, right=True)
    return bins.astype(int)

def legend_html(title: str, labels: list, colors_rgb: list) -> str:
    # simple vertical legend
    rows = []
    for lab, col in zip(labels, colors_rgb):
        r, g, b = col
        rows.append(f'<div style="display:flex;align-items:center;margin:4px 0;">'
                    f'<span style="display:inline-block;width:16px;height:16px;background:rgb({r},{g},{b});margin-right:8px;border:1px solid #333;"></span>'
                    f'<span style="font-size:12px;">{lab}</span></div>')
    return (f'<div style="border:1px solid #ccc;border-radius:8px;padding:8px 10px;'
            f'background:#fff;max-width:220px;">'
            f'<div style="font-weight:600;margin-bottom:6px;">{title}</div>'
            f'{"".join(rows)}</div>')

# =====================================================================
# Geometry load & transforms
# =====================================================================

def _zipfile_from_any(obj):
    from zipfile import ZipFile
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

def load_community_boards_from_json(js: dict):
    if not SHAPELY_OK:
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

def load_census_tracts_from_geojson(js: dict):
    """
    Load census tracts from GeoJSON with NRI and static attributes.
    Expected columns:
    - TRACTCE or GEOID (tract identifier)
    - elevation_stats:MEAN or Elevation
    - slope_stats:MEAN or Slope
    - Summarized Area in SQUAREKILOMETERS (building footprint sum)
    - RFLD_RISKS (riverine flood risk)
    - CFLD_RISKS (coastal flood risk)
    """
    if not SHAPELY_OK:
        st.error("Census tract loading requires 'shapely'.")
        return None
    
    feats = js.get("features", [])
    tracts = []
    
    for idx, f in enumerate(feats):
        props = f.get("properties", {}) or {}
        
        # Get tract ID (try multiple possible field names)
        unit_id = (props.get("TRACTCE") or 
                   props.get("GEOID") or 
                   props.get("TRACT") or 
                   props.get("tract_id") or 
                   f"TRACT_{idx:05d}")
        
        name = props.get("NAME") or props.get("name") or str(unit_id)
        
        geom_geojson = f.get("geometry")
        if not geom_geojson:
            continue
        
        try:
            geom = shp_shape(geom_geojson)
        except Exception:
            continue
        
        if geom.is_empty:
            continue
        
        # Extract and clean attribute names (ArcGIS sometimes adds prefixes)
        clean_attrs = {}
        for k, v in props.items():
            # Handle ArcGIS field name patterns
            if "elevation" in k.lower() and "mean" in k.lower():
                clean_attrs["Elevation"] = v
            elif "slope" in k.lower() and "mean" in k.lower():
                clean_attrs["Slope"] = v
            elif "summarized" in k.lower() and "area" in k.lower():
                clean_attrs["Footprint"] = v
            elif k == "RFLD_RISKS":
                clean_attrs["RFLD_RISKS"] = v
            elif k == "CFLD_RISKS":
                clean_attrs["CFLD_RISKS"] = v
            else:
                clean_attrs[k] = v
        
        tracts.append({
            "unit_id": str(unit_id),
            "name": str(name),
            "geom": geom,
            "feature_geom": json.loads(json.dumps(geom_geojson)),
            "attrs": clean_attrs,
        })
    
    if not tracts:
        st.error("No valid census tract features found in the GeoJSON.")
    else:
        st.success(f"Loaded {len(tracts)} census tracts.")
    
    return tracts or None

def load_boards_from_shapefile_path(zip_path: str):
    if not os.path.exists(zip_path):
        return None
    return _load_boards_from_shapefile_zipfilelike(zip_path)

def load_boards_from_shapefile_zip_streamlit():
    zip_file = st.file_uploader("Upload Community Board/District **Shapefile (.zip)**", type=["zip"], key="shpzip_upl")
    if zip_file is None:
        return None
    return _load_boards_from_shapefile_zipfilelike(zip_file)

def _load_boards_from_shapefile_zipfilelike(zip_filelike):
    if not SHAPELY_OK:
        st.error("Shapefile support requires 'shapely>=2.0'.")
        return None
    try:
        import shapefile  # pyshp
    except ImportError:
        st.error("Missing dependency: 'pyshp' (pip install pyshp).")
        return None
    try:
        from pyproj import CRS, Transformer
    except Exception:
        st.error("Shapefile reprojection requires 'pyproj>=3.6'.")
        return None

    def to_pure_geojson(geom):
        m = shp_mapping(geom)
        return json.loads(json.dumps(m))

    import tempfile, os
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
        
        # For census tracts, try TRACTCE or GEOID
        unit_id = rec.get("TRACTCE") or rec.get("GEOID") or rec.get("BoroCD") or rec.get("boro_cd") or rec.get("cd") or f"TRACT_{idx:05d}"
        name = rec.get("NAME") or rec.get("cd_name") or rec.get("name") or unit_id
        
        geom_geojson = sr.shape.__geo_interface__
        try:
            geom = shp_shape(geom_geojson)
        except Exception as e:
            st.warning(f"Could not parse geometry for {unit_id}: {e}")
            continue
        if geom.is_empty:
            continue

        # Reproject guess if needed
        if transformer is None:
            try:
                if geom.geom_type == 'Polygon':
                    x0, y0 = list(geom.exterior.coords)[0]
                elif geom.geom_type == 'MultiPolygon':
                    x0, y0 = list(list(geom.geoms)[0].exterior.coords)[0]
                else:
                    x0, y0 = (0, 0)
                if not looks_like_lonlat(x0, y0):
                    from pyproj import Transformer
                    transformer = Transformer.from_crs(2263, 4326, always_xy=True)
            except Exception:
                pass
        if transformer is not None:
            try:
                geom = shp_transform(lambda x, y, z=None: transformer.transform(x, y), geom)
            except Exception as e:
                st.warning(f"Could not transform geometry for {unit_id}: {e}")
                continue

        if geom.geom_type not in ['Polygon', 'MultiPolygon']:
            st.warning(f"Skipping {unit_id}: geometry is {geom.geom_type}, not Polygon/MultiPolygon")
            continue

        feature_geom = to_pure_geojson(geom)
        
        # Clean attributes for census tracts
        clean_attrs = {}
        for k, v in rec.items():
            # Handle ArcGIS field name patterns
            if "elevation" in k.lower() and "mean" in k.lower():
                clean_attrs["Elevation"] = v
            elif "slope" in k.lower() and "mean" in k.lower():
                clean_attrs["Slope"] = v
            elif "summarized" in k.lower() and "area" in k.lower():
                clean_attrs["Footprint"] = v
            elif k == "RFLD_RISKS":
                clean_attrs["RFLD_RISKS"] = v
            elif k == "CFLD_RISKS":
                clean_attrs["CFLD_RISKS"] = v
            else:
                clean_attrs[k] = v

        polys.append({
            "unit_id": str(unit_id),
            "name": str(name),
            "geom": geom,
            "feature_geom": feature_geom,
            "attrs": clean_attrs,
        })

    if not polys:
        st.error("No valid polygon features found in the shapefile.")
    else:
        st.success(f"Loaded {len(polys)} census tracts from shapefile.")
    return polys or None

# =====================================================================
# Data merges and helpers
# =====================================================================

def load_and_merge_board_data(boards: list, csv_path: str = "DataForBoxPlots.csv") -> list:
    """
    Merge CSV static attributes onto boards by CB_id <-> boro_cd.
    NOTE: This function is kept for backward compatibility but may not be needed for census tracts.
    """
    if not os.path.exists(csv_path):
        return boards
    try:
        csv_df = pd.read_csv(csv_path)
        csv_cols = ['CB_id', 'Buildings', 'Elevation', 'Slope', 'Commuting',
                    'Imperv', 'Footprint', 'BLDperArea', 'FTPperArea']
        missing = [c for c in csv_cols if c not in csv_df.columns]
        if missing:
            st.warning(f"Static CSV missing columns: {missing}")
            return boards
        csv_df = csv_df[csv_cols].copy()
        csv_df['CB_id'] = csv_df['CB_id'].apply(normalize_cb_id)
        lookup = csv_df.set_index('CB_id').to_dict('index')

        out = []
        for b in boards:
            boro_cd_raw = b.get('attrs', {}).get('BoroCD') or b.get('attrs', {}).get('boro_cd') or b.get('attrs', {}).get('cd')
            boro_cd = normalize_cb_id(boro_cd_raw)
            filtered_attrs = {
                'boro_cd': boro_cd,
                'shape_area': b.get('attrs', {}).get('shape_area'),
                'shape_leng': b.get('attrs', {}).get('shape_leng'),
                'Area': b.get('attrs', {}).get('Area'),
            }
            if boro_cd in lookup:
                filtered_attrs.update(lookup[boro_cd])
            b['attrs'] = filtered_attrs
            out.append(b)
        return out
    except Exception as e:
        st.warning(f"Error merging CSV: {e}")
        return boards

def boards_bbox(fc):
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
        if not geom:
            continue
        t = geom.get("type")
        c = geom.get("coordinates", [])
        if t == "Polygon":
            update_bbox(c, False)
        elif t == "MultiPolygon":
            update_bbox(c, True)

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
    rows = []
    for b in boards:
        cb_num = get_board_display_id(b)
        if not cb_num or not SHAPELY_OK:
            continue
        p = b["geom"].representative_point()
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

# =====================================================================
# PNG exporter (matplotlib) for choropleths
# =====================================================================

def save_choropleth_png(boards: list, values_by_board: pd.Series, title: str, palette=None, breaks=None) -> bytes:
    """
    Render a simple static choropleth (no basemap) using matplotlib and return PNG bytes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    # Build color breaks
    s = pd.to_numeric(values_by_board, errors="coerce").fillna(0.0)
    if breaks is None:
        breaks = compute_breaks(s, k=(len(palette) if palette else len(RED_YELLOW)))
    colorset = palette if palette else RED_YELLOW

    def rgb_norm(c):
        return (c[0]/255.0, c[1]/255.0, c[2]/255.0)

    fig, ax = plt.subplots(figsize=(8, 8))
    patches = []
    colors = []

    for b in boards:
        uid = b["unit_id"]
        v = float(s.get(uid, 0.0))
        # For PNG with bins (when values are 0..3), handle gracefully
        if palette is not None and (v in [0.0, 1.0, 2.0, 3.0]):
            col = colorset[int(v)]
        else:
            col = color_for_value_dynamic(v, breaks)
        col = rgb_norm(col)

        geom = b["geom"]
        if geom.geom_type == "Polygon":
            rings = [np.asarray(geom.exterior.coords)]
            for r in rings:
                from matplotlib.patches import Polygon
                patches.append(Polygon(r, closed=True))
                colors.append(col)
        elif geom.geom_type == "MultiPolygon":
            for g in geom.geoms:
                r = np.asarray(g.exterior.coords)
                patches.append(Polygon(r, closed=True))
                colors.append(col)

    pc = PatchCollection(
        patches,
        facecolor=colors,
        edgecolor=(0.2, 0.2, 0.2),
        linewidths=0.5,
        alpha=0.9
    )
    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect('equal', adjustable='box')
    ax.axis("off")
    ax.set_title(title, fontsize=12)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# =====================================================================
# Methodology placeholders (edit later)
# =====================================================================

METH_RISK = {
    "NRI Coastal": """**NRI Coastal Flooding Risk (methodology)**
- Data source: FEMA National Risk Index (NRI) dataset
- Downscaling: Census tract level data aggregated from NRI
- Variables: CFLD_RISKS score (0-100 scale representing coastal flood risk)
- Combination: Direct use of NRI's composite risk score which includes expected annual loss, social vulnerability, and community resilience
- Caveats: Risk scores are relative and should be interpreted in context of local conditions
""",
    "NRI Riverine": """**NRI Riverine Flooding Risk (methodology)**
- Data source: FEMA National Risk Index (NRI) dataset
- Downscaling: Census tract level data aggregated from NRI
- Variables: RFLD_RISKS score (0-100 scale representing riverine flood risk)
- Combination: Direct use of NRI's composite risk score which includes expected annual loss, social vulnerability, and community resilience
- Caveats: Risk scores are relative and should be interpreted in context of local conditions
""",
    "My Risk Map": """**TODO – Custom Risk Map (methodology)**
- TODO: input features (impervious, footprint, elevation, slope, etc.)
- TODO: scaling/weighting scheme
- TODO: normalization/indexing
- TODO: validation notes
"""
}

METH_FORECAST = {
    "Water sensors": """**TODO – Next-Day Forecast (Water Sensors)**
- TODO: sensors & units
- TODO: aggregation to boards
- TODO: baseline/persistence definition
- TODO: known limitations
""",
    "Street flooding": """**TODO – Next-Day Forecast (Street Flooding)**
- TODO: 311 ingestion and cleaning
- TODO: temporal aggregation window
- TODO: baseline/persistence definition
- TODO: known limitations
""",
    "Catch basins": """**TODO – Next-Day Forecast (Catch Basins)**
- TODO: clog/cleaning data source
- TODO: aggregation to boards & cadence
- TODO: baseline/persistence definition
- TODO: known limitations
"""
}

METH_URBAN = """**Static Urban Features (context)**
- **Elevation**: Mean elevation per census tract from USGS DEM data, aggregated using Zonal Statistics
- **Slope**: Mean slope per census tract derived from elevation data, aggregated using Zonal Statistics
- **Building Footprint**: Total building footprint area (square kilometers) per census tract, summed from NYC building footprint dataset
- **Data Processing**: All spatial data aggregated to census tract level using ArcGIS Pro Zonal Statistics and Summarize Within tools
- **Interpretation**: Higher footprint and lower elevation typically correlate with increased flood vulnerability
"""

# =====================================================================
# Map builders
# =====================================================================

# --- REPLACED: safer color handling for Series-of-lists (no fillna(list)) ---
def fc_from_boards_and_values(boards: list, values: pd.Series, prop_name: str, color_series: Optional[pd.Series] = None) -> dict:
    """
    Build a FeatureCollection. If color_series is provided, it should map unit_id -> [r,g,b].
    We avoid .fillna(list) and instead default per-feature when missing.
    """
    feats = []
    breaks = None
    if color_series is None:
        breaks = compute_breaks(values, k=len(RED_YELLOW))

    for b in boards:
        uid = b["unit_id"]
        v = float(pd.to_numeric(values.get(uid, 0.0), errors="coerce") or 0.0)

        if color_series is not None:
            col = color_series.get(uid)
            if not isinstance(col, (list, tuple)) or len(col) != 3:
                col = [255, 255, 178]  # default light yellow
            fill = [int(col[0]), int(col[1]), int(col[2])]
        else:
            fill = color_for_value_dynamic(v, breaks)

        feats.append({
            "type": "Feature",
            "properties": {
                "unit_id": uid,
                "cb": get_board_display_id(b),
                prop_name: v,
                prop_name + "_formatted": f"{v:.4f}",
                "fill_color": fill
            },
            "geometry": b["feature_geom"]
        })
    return {"type": "FeatureCollection", "features": feats}

def pydeck_choropleth(fc: dict, lat_c: float, lon_c: float, zoom_c: float, prop_name: str, tooltip_label: str, opacity=0.85, label_layer=None):
    layer = pdk.Layer(
        "GeoJsonLayer",
        data=fc,
        pickable=True, stroked=True, filled=True,
        get_fill_color="properties.fill_color",
        get_line_color=[40, 40, 40], lineWidthMinPixels=2, opacity=opacity
    )
    layers = [
        pdk.Layer(
            "TileLayer",
            data="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
            minZoom=0, maxZoom=19, tileSize=256
        ),
        layer
    ]
    if label_layer:
        layers.append(label_layer)
    
    # Use the formatted property
    formatted_prop = prop_name + "_formatted"
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom_c),
        tooltip={
            "html": "<b>Tract #:</b> {cb}<br><b>" + tooltip_label + ":</b> {" + formatted_prop + "}",
            "style": {"backgroundColor": "white", "color": "black"}
        }
    )
    return deck

# =====================================================================
# Sidebar & routing
# =====================================================================

def sidebar():
    with st.sidebar:
        st.markdown("### NYC Resilience AI Agent")
        st.caption("Census Tract Analysis: Static Features, Risk Maps, and Forecasting\nwith MLflow + GenAI logging.")
        st.divider()
        st.button("🏠 Home", use_container_width=True, on_click=lambda: st.session_state.update(page="landing"), key="nav_home")
        st.button("🏙️ Urban Features", use_container_width=True, on_click=lambda: st.session_state.update(page="urban"), key="nav_urban")
        st.button("🗺️ Risk Mapping", use_container_width=True, on_click=lambda: st.session_state.update(page="risk"), key="nav_risk")
        st.button("📈 Forecasting", use_container_width=True, on_click=lambda: st.session_state.update(page="forecast"), key="nav_forecast")
        st.divider()
        if HAS_BEDROCK:
            st.success("Claude (Bedrock) connected.")
        else:
            missing = [k for k in ("AWS_ACCESS_KEY_ID","AWS_SECRET_ACCESS_KEY","AWS_REGION") if not os.getenv(k)]
            st.info("Claude: missing " + ", ".join(missing))
        try:
            mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
            st.success("MLflow tracking ready.")
        except Exception as e:
            st.warning(f"MLflow not ready: {e}")

# =====================================================================
# Pages
# =====================================================================

# MODIFIED: Changed to look for census tracts shapefile
LOCAL_CENSUS_TRACTS = os.getenv("LOCAL_CENSUS_TRACTS", "nyc_census_tracts.shp")

def ensure_boards() -> list:
    """
    Load census tracts (now called 'boards' for code compatibility).
    Tries: 1) local shapefile, 2) upload
    """
    boards = st.session_state.get("boards")
    if boards:
        return boards

    # 1) Try local shapefile (without .shp extension for the loader)
    base_path = LOCAL_CENSUS_TRACTS.replace('.shp', '')
    if os.path.exists(base_path + '.shp'):
        st.info(f"Loading census tracts from {LOCAL_CENSUS_TRACTS}...")
        boards = _load_census_tracts_from_shapefile_direct(base_path)
        if boards:
            st.session_state["boards"] = boards
            return boards

    # 2) Upload alternatives
    st.info("Upload Census Tracts GeoJSON or Shapefile ZIP.")
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        uploaded_geojson = st.file_uploader("GeoJSON", type=["geojson", "json"], key="geojson_upl")
        if uploaded_geojson is not None:
            try:
                b = uploaded_geojson.read()
                js = json.loads(b.decode("utf-8"))
            except UnicodeDecodeError:
                js = json.loads(b.decode("latin-1"))
            boards = load_census_tracts_from_geojson(js)
    with col_u2:
        boards = boards or load_boards_from_shapefile_zip_streamlit()

    if boards:
        st.session_state["boards"] = boards
    return boards

def _load_census_tracts_from_shapefile_direct(base_path: str):
    """
    Load census tracts directly from shapefile components (not zipped).
    """
    if not SHAPELY_OK:
        st.error("Shapefile support requires 'shapely>=2.0'.")
        return None
    try:
        import shapefile  # pyshp
    except ImportError:
        st.error("Missing dependency: 'pyshp' (pip install pyshp).")
        return None
    try:
        from pyproj import CRS, Transformer
    except Exception:
        st.error("Shapefile reprojection requires 'pyproj>=3.6'.")
        return None

    def to_pure_geojson(geom):
        m = shp_mapping(geom)
        return json.loads(json.dumps(m))

    shp_path = base_path + '.shp'
    prj_path = base_path + '.prj'

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
    
    # DEBUG: Show what fields are actually in the shapefile
    st.write("**DEBUG: Fields found in shapefile:**")
    st.write(fields)

    def looks_like_lonlat(x, y):
        return (-180.0 <= x <= 180.0) and (-90.0 <= y <= 90.0)

    tracts = []
    for idx, sr in enumerate(r.shapeRecords()):
        rec = {fields[i]: sr.record[i] for i in range(len(fields))}
        
        # For census tracts, try TRACTCE or GEOID
        unit_id = rec.get("TRACTCE") or rec.get("GEOID") or rec.get("tract_id") or f"TRACT_{idx:05d}"
        name = rec.get("NAME") or rec.get("name") or str(unit_id)
        
        geom_geojson = sr.shape.__geo_interface__
        try:
            geom = shp_shape(geom_geojson)
        except Exception as e:
            st.warning(f"Could not parse geometry for {unit_id}: {e}")
            continue
        if geom.is_empty:
            continue

        # Reproject if needed
        if transformer is None:
            try:
                if geom.geom_type == 'Polygon':
                    x0, y0 = list(geom.exterior.coords)[0]
                elif geom.geom_type == 'MultiPolygon':
                    x0, y0 = list(list(geom.geoms)[0].exterior.coords)[0]
                else:
                    x0, y0 = (0, 0)
                if not looks_like_lonlat(x0, y0):
                    from pyproj import Transformer
                    transformer = Transformer.from_crs(2263, 4326, always_xy=True)
            except Exception:
                pass
        if transformer is not None:
            try:
                geom = shp_transform(lambda x, y, z=None: transformer.transform(x, y), geom)
            except Exception as e:
                st.warning(f"Could not transform geometry for {unit_id}: {e}")
                continue

        if geom.geom_type not in ['Polygon', 'MultiPolygon']:
            st.warning(f"Skipping {unit_id}: geometry is {geom.geom_type}, not Polygon/MultiPolygon")
            continue

        feature_geom = to_pure_geojson(geom)
        
        # Clean attributes for census tracts - handle various field name patterns
        clean_attrs = {}
        for k, v in rec.items():
            # Always keep the original field
            clean_attrs[k] = v
            
            # Also create standardized names
            k_lower = k.lower()
            if "elevation" in k_lower and "mean" in k_lower:
                clean_attrs["Elevation"] = v
            elif "slope" in k_lower and "mean" in k_lower:
                clean_attrs["Slope"] = v
            elif "summarized" in k_lower and "area" in k_lower and "squarekilometer" in k_lower:
                clean_attrs["Footprint"] = v
            elif k.upper() == "RFLD_RISKS":
                clean_attrs["RFLD_RISKS"] = v
            elif k.upper() == "CFLD_RISKS":
                clean_attrs["CFLD_RISKS"] = v

        tracts.append({
            "unit_id": str(unit_id),
            "name": str(name),
            "geom": geom,
            "feature_geom": feature_geom,
            "attrs": clean_attrs,
        })

    if not tracts:
        st.error("No valid census tract features found in the shapefile.")
    else:
        st.success(f"✅ Loaded {len(tracts)} census tracts from shapefile.")
    return tracts or None

def home_page():
    st.title("🏙️ NYC Resilience AI Agent")
    st.subheader("Census Tract Analysis: Static Urban Features • Risk Maps • Forecasting")
    st.write("This application analyzes NYC census tracts with elevation, slope, building footprint data, and NRI flood risk scores.")
    st.divider()
    st.markdown("#### Workflow & Logging")
    st.markdown("- MLflow logs map generations, parameters, and GenAI (Claude) explanations.")
    st.markdown("- Each map is downloadable as **PNG** and **CSV**.")
    st.markdown("- Risk maps use FEMA NRI data at census tract level.")
    st.markdown("- Color scheme: **Red (high values/risk) → Yellow (low values/risk)**")

def urban_features_page():
    st.title("🏙️ Urban Features (Static) - Census Tracts")
    boards = ensure_boards()
    if not boards:
        st.stop()

    # Outline view
    fc_outlines = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": get_board_display_id(b)}, "geometry": b["feature_geom"]} for b in boards
    ]}
    lat_c, lon_c, zoom_c = boards_bbox(fc_outlines)

    label_layer = cb_label_layer_from_boards(boards, text_size=10)
    outline = pdk.Layer(
        "GeoJsonLayer",
        data=fc_outlines,
        pickable=True, stroked=True, filled=False,
        get_line_color=[255, 0, 0], line_width_min_pixels=2
    )
    bg = pdk.Layer(
        "TileLayer",
        data="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        minZoom=0, maxZoom=19, tileSize=256
    )
    st.pydeck_chart(
        pdk.Deck(layers=[bg, outline] + ([label_layer] if label_layer else []),
                 initial_view_state=pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom_c),
                 tooltip={"html": "<b>Tract:</b> {name}"})
    )
    st.caption("NYC Census Tracts (outline).")

    # Attribute choices - MODIFIED for census tract data
    st.subheader("Visualize a static attribute as a choropleth")
    attribute_options = {
        "Mean Elevation": "Elevation",
        "Mean Slope": "Slope",
        "Total Building Footprint (sq km)": "Footprint",
    }
    sel_label = st.selectbox("Attribute", list(attribute_options.keys()), index=0)
    sel_attr = attribute_options[sel_label]

    # Gather values
    vals = {}
    for b in boards:
        val = b.get('attrs', {}).get(sel_attr)
        try:
            vals[b["unit_id"]] = float(val) if val is not None and pd.notna(val) else 0.0
        except Exception:
            vals[b["unit_id"]] = 0.0
    series = pd.Series(vals).sort_index()

    # Map
    fc = fc_from_boards_and_values(boards, series, prop_name="value")
    label_layer = cb_label_layer_from_boards(boards, text_size=10)
    deck = pydeck_choropleth(fc, lat_c, lon_c, zoom_c, prop_name="value", tooltip_label=sel_label, opacity=0.85, label_layer=label_layer)
    st.pydeck_chart(deck, use_container_width=True)

    # Downloads: CSV + PNG
    c1, c2 = st.columns(2)
    with c1:
        dl_df = series.rename(sel_attr).reset_index()
        dl_df.columns = ["unit_id", sel_attr]
        dl_df["tract_id"] = dl_df["unit_id"].map(lambda x: normalize_cb_id(x))
        st.download_button(
            "Download CSV",
            data=dl_df.to_csv(index=False).encode("utf-8"),
            file_name=f"urban_{sel_attr}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c2:
        png_bytes = save_choropleth_png(boards, series, f"Urban Feature: {sel_label}")
        st.download_button(
            "Download PNG",
            data=png_bytes,
            file_name=f"urban_{sel_attr}.png",
            mime="image/png",
            use_container_width=True
        )

    st.divider()
    st.markdown("#### Methodology")
    st.info(METH_URBAN)

    # Claude explanation + MLflow logging
    st.subheader("Explain this map (Claude)")
    if st.button("Explain"):
        context = {
            "map_type": "urban_static",
            "attribute": sel_label,
            "stats": {
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "median": float(series.median()),
            }
        }
        prompt = (
            "You are a flood-planning assistant. Explain the static attribute map for NYC Census Tracts, "
            "focusing on patterns and practical interpretation for flood risk.\n\n"
            f"Context JSON:\n{json.dumps(context, indent=2)}\n\n"
            "Use census tract numbers when referencing locations."
        )
        system = "Be concise and practical. Mention hotspots and why they matter."
        answer = call_claude(prompt, system=system)

        st.markdown("**Claude:**")
        st.write(answer)

        # MLflow GenAI logging
        genai_log(prompt, answer, meta={"span_name": "explain_urban_feature", "attribute": sel_label})

# ------------------------------------------------------------
# NEW HELPERS (field detection & fallback selection for risk)
# ------------------------------------------------------------
def list_numeric_candidates(boards, max_show=60):
    """
    Scan attribute keys and return numeric-looking fields with example numeric values.
    Helps pick the correct field when names don't match exactly.
    """
    keys = set()
    samples = []
    for b in boards[:min(len(boards), max_show)]:
        attrs = b.get("attrs", {}) or {}
        for k, v in attrs.items():
            kl = (k or "").lower()
            if any(tok in kl for tok in ["shape", "geom", "globalid", "objectid"]):
                continue
            num = coerce_nonneg_numeric(v, None)
            if num is not None:
                keys.add(k)
                samples.append((k, num))
    # crude scoring: fields containing substrings of interest first
    scores = {}
    for k in keys:
        kc = k.lower()
        score = 0
        for token in ["risk", "cfl", "rfl", "hwav", "score", "eal", "alr"]:
            if token in kc:
                score += 1
        scores[k] = score
    ordered = sorted(list(keys), key=lambda x: (-scores.get(x,0), x))
    return ordered, dict(scores)

def pick_field_with_fallback(boards, requested: str):
    """
    Try requested field; if missing everywhere, surface a selectbox of numeric candidates.
    Returns (field_name_used, was_fallback_boolean).
    """
    # quick check if requested exists somewhere (case-insensitive)
    exists = False
    for b in boards:
        if get_attr_ci(b.get("attrs", {}) or {}, requested) is not None:
            exists = True
            break
    if exists:
        return requested, False

    st.warning(f"Requested field `{requested}` not found. Pick a field from your layer that holds numeric risk values.")
    candidates, _ = list_numeric_candidates(boards)
    if not candidates:
        st.error("No numeric-looking fields found in the layer’s attributes. Please export a layer with numeric risk columns.")
        return requested, False
    chosen = st.selectbox("Choose field", options=candidates, index=0, key=f"risk_field_fallback_{requested}")
    return chosen, True

# ------------------------------------------------------------
# REPLACED: Risk Mapping page with fallback picker & stronger debug
# ------------------------------------------------------------
def page_risk_mapping():
    st.title("🗺️ Risk Mapping (NRI + Custom)")
    boards = ensure_boards()
    if not boards:
        st.stop()

    risk_choice = st.radio(
        "Risk layer:",
        ["Coastal Flooding Risk (CFLD_RISKS)", "Riverine Flooding Risk (RFLD_RISKS)", "In-House Flood Risk (score_total)"],
        horizontal=False
    )
    if "Coastal" in risk_choice:
        requested_field = "CFLD_RISKS"
        pretty          = "NRI Coastal Flood Risk"
        meth_key        = "NRI Coastal"
    elif "Riverine" in risk_choice:
        requested_field = "RFLD_RISKS"
        pretty          = "NRI Riverine Flood Risk"
        meth_key        = "NRI Riverine"
    else:
        requested_field = "score_total"
        pretty          = "In-House Flood Risk (total)"
        meth_key        = "My Risk Map"

    field_name, used_fallback = pick_field_with_fallback(boards, requested_field)
    if used_fallback:
        st.info(f"Using `{field_name}` (fallback selection)")

    # Extract robust numeric values
    raw_vals = {}
    for b in boards:
        v_raw = get_attr_ci(b.get('attrs', {}) or {}, field_name)
        num = coerce_nonneg_numeric(v_raw, 0.0)
        raw_vals[b["unit_id"]] = num if num is not None else 0.0
    raw_series = pd.Series(raw_vals).sort_index()

    # Normalize and bin
    cleaned = normalize_0_1(raw_series)
    bins    = equal_interval_bins_4(cleaned)
    color_map = {0: PALETTE4[0], 1: PALETTE4[1], 2: PALETTE4[2], 3: PALETTE4[3]}
    color_series = pd.Series({uid: color_map[bins.loc[uid]] for uid in cleaned.index})

    # Debug panel
    with st.expander("🔎 Debug distribution & field check", expanded=True):
        st.write("**Raw value stats (pre-normalization):**")
        st.dataframe(raw_series.describe().to_frame(name="raw_stats"))
        st.write("**Normalized [0,1] stats:**")
        st.dataframe(cleaned.describe().to_frame(name="norm_stats"))

        counts = bins.value_counts().sort_index().reindex([0,1,2,3], fill_value=0)
        counts = counts.rename(index={0:"0–0.25", 1:"0.25–0.5", 2:"0.5–0.75", 3:"0.75–1.0"})
        st.write("**Bin counts (equal intervals on [0,1]):**")
        st.dataframe(counts.to_frame(name="count").T)

        # Show example keys & raw
        sample_attrs = boards[0].get("attrs", {}) or {}
        st.write("**Example feature attribute keys (first 60):**")
        st.write(list(sample_attrs.keys())[:60])
        st.write(f"**Field in use:** `{field_name}` → example raw:", get_attr_ci(sample_attrs, field_name))

        # If everything collapsed, highlight why
        unique_vals = int(cleaned.nunique())
        if unique_vals <= 1:
            st.error(
                "All normalized values are identical (likely 0). This usually means the chosen field is constant "
                "or non-numeric in the layer. Try picking a different field above."
            )
        elif counts.loc["0–0.25"] >= 0.95 * len(cleaned):
            st.info("Most tracts fell into 0–0.25. If this looks wrong, the field may be very low variance or scaled oddly.")
        preview_df = pd.DataFrame({
            "unit_id": cleaned.index,
            "raw": raw_series.values,
            "norm_0_1": cleaned.values,
            "bin(0..3)": bins.values
        }).head(12)
        st.write("**Sample rows:**")
        st.dataframe(preview_df)

    # Build FC with fixed colors
    fc = fc_from_boards_and_values(boards, cleaned, prop_name="risk_norm", color_series=color_series)

    lat_c, lon_c, zoom_c = boards_bbox({"type": "FeatureCollection",
                                        "features": [{"type": "Feature", "geometry": b["feature_geom"]} for b in boards]})
    label_layer = cb_label_layer_from_boards(boards, text_size=10)
    deck = pydeck_choropleth(fc, lat_c, lon_c, zoom_c, prop_name="risk_norm",
                             tooltip_label=f"{pretty} (0–1)", opacity=0.90, label_layer=label_layer)
    c_map, c_leg = st.columns([4,1])
    with c_map:
        st.pydeck_chart(deck, use_container_width=True)
    with c_leg:
        st.markdown(
            legend_html(f"{pretty} (equal intervals)", ["0–0.25","0.25–0.5","0.5–0.75","0.75–1.0"], PALETTE4),
            unsafe_allow_html=True
        )

    # Downloads
    c1, c2 = st.columns(2)
    with c1:
        dl_df = pd.DataFrame({"unit_id": cleaned.index, "risk_norm_0_1": cleaned.values, "bin_0..3": bins.values})
        dl_df["tract_id"] = dl_df["unit_id"].map(lambda x: normalize_cb_id(x))
        st.download_button(
            "Download CSV",
            data=dl_df.to_csv(index=False).encode("utf-8"),
            file_name=f"risk_map_{field_name.lower()}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c2:
        png_series = bins.astype(float)
        png_bytes = save_choropleth_png(boards, png_series, f"Risk Map: {pretty}", palette=PALETTE4, breaks=None)
        st.download_button(
            "Download PNG",
            data=png_bytes,
            file_name=f"risk_map_{field_name.lower()}.png",
            mime="image/png",
            use_container_width=True
        )

    st.divider()
    st.markdown("#### Methodology")
    st.info(METH_RISK[meth_key])

    src = st.session_state.get("boards_source", "Uploaded GeoJSON/Shapefile")
    st.caption(f"**Source**: {src} • **Field used**: `{field_name}`")

    st.subheader("Explain this risk map (Claude)")
    if st.button("Explain Risk Map", key="explain_risk_btn"):
        stats = {
            "min": float(cleaned.min()),
            "max": float(cleaned.max()),
            "mean": float(cleaned.mean()),
            "median": float(cleaned.median())
        }
        context = {"domain": "urban_flooding", "map_type": "risk", "variant": pretty, "field": field_name, "stats": stats}
        prompt = (
            "You are a flood-planning assistant. Explain the displayed risk map for NYC Census Tracts with actionable insights. "
            "Use census tract numbers when referencing locations.\n\n"
            f"Context JSON:\n{json.dumps(context, indent=2)}"
        )
        system = "Be concise and practical for city stakeholders."
        answer = call_claude(prompt, system=system)
        st.markdown("**Claude:**")
        st.write(answer)
        genai_log(prompt, answer, meta={"span_name": "explain_risk_map", "variant": pretty, "field": field_name})

def forecasting_page():
    st.title("📈 Forecasting (Stubs + Next-Day UI)")
    boards = ensure_boards()
    if not boards:
        st.stop()

    st.caption("Choose the signal you plan to forecast next-day (stubs—add methodology later).")
    choice = st.selectbox(
        "Signal:",
        ["Flooding (Water sensors)", "Flooding (Street flooding/311)", "Flooding (Catch basins)"],
        index=0
    )

    # Next-day date selector
    tomorrow = date.today() + timedelta(days=1)
    when = st.date_input("Forecast date (next-day)", value=tomorrow, min_value=date.today(), max_value=date.today()+timedelta(days=14))

    st.write("Upload an optional **census tract-level baseline** CSV to visualize (columns: unit_id, value). "
             "If omitted, a neutral placeholder map is shown.")
    upl = st.file_uploader("Optional: baseline tract scores for preview", type=["csv"], key="fcst_upload")

    # Determine values (placeholder)
    values = None
    if upl is not None:
        try:
            df = pd.read_csv(upl)
            if {"unit_id", "value"}.issubset(df.columns):
                s = df.set_index("unit_id")["value"]
                values = pd.Series({b["unit_id"]: float(pd.to_numeric(s.get(b["unit_id"]), errors="coerce") or 0.0) for b in boards})
            else:
                st.error("CSV must have: unit_id, value")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    if values is None:
        # Flat placeholder (zeros)
        values = pd.Series({b["unit_id"]: 0.0 for b in boards})

    # Map
    fc = fc_from_boards_and_values(boards, values, prop_name="forecast")
    lat_c, lon_c, zoom_c = boards_bbox({"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": b["feature_geom"]} for b in boards]})
    label_layer = cb_label_layer_from_boards(boards, text_size=10)
    deck = pydeck_choropleth(fc, lat_c, lon_c, zoom_c, prop_name="forecast", tooltip_label="Forecast (placeholder)", opacity=0.90, label_layer=label_layer)
    st.pydeck_chart(deck, use_container_width=True)

    # Downloads
    c1, c2 = st.columns(2)
    with c1:
        dl_df = values.rename("forecast_value").reset_index()
        dl_df.columns = ["unit_id", "forecast_value"]
        dl_df["tract_id"] = dl_df["unit_id"].map(lambda x: normalize_cb_id(x))
        st.download_button(
            "Download CSV",
            data=dl_df.to_csv(index=False).encode("utf-8"),
            file_name=f"forecast_preview_{when.isoformat()}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c2:
        png_bytes = save_choropleth_png(boards, values, f"Forecast Preview: {when.isoformat()}")
        st.download_button(
            "Download PNG",
            data=png_bytes,
            file_name=f"forecast_preview_{when.isoformat()}.png",
            mime="image/png",
            use_container_width=True
        )

    st.divider()
    st.markdown("#### Methodology (fill later)")
    meth_key = "Water sensors" if choice.startswith("Flooding (Water") else ("Street flooding" if "Street" in choice else "Catch basins")
    st.info(METH_FORECAST[meth_key])

    # Claude explanation + MLflow GenAI logging
    st.subheader("Explain this forecast preview (Claude)")
    if st.button("Explain Forecast"):
        stats = {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "median": float(values.median())
        }
        context = {
            "map_type": "forecast_stub",
            "signal": choice,
            "date": when.isoformat(),
            "stats": stats,
            "note": "Placeholder map (no model yet)."
        }
        prompt = (
            "You are a flood-forecast assistant. Explain this **next-day** forecast preview for NYC Census Tracts. "
            "Keep it practical and note any limitations of the placeholder.\n\n"
            f"Context JSON:\n{json.dumps(context, indent=2)}"
        )
        system = "Be concise and practical for city stakeholders."
        answer = call_claude(prompt, system=system)

        st.markdown("**Claude:**")
        st.write(answer)

        genai_log(prompt, answer, meta={"span_name": "explain_forecast_preview", "signal": choice, "date": when.isoformat()})

# =====================================================================
# Router
# =====================================================================

def page_landing():
    st.title("🧭 Choose a Domain")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💧 Urban Flooding", use_container_width=True, key="landing_flood"):
            st.session_state.page = "urban"
            st.rerun()
    with col2:
        if st.button("🌤️ Urban Heat Island (NRI HWAV only)", use_container_width=True, key="landing_heat"):
            st.session_state.page = "uhi"
            st.rerun()

def page_uhi():
    st.title("🌤️ Urban Heat Island (NRI Heat Wave Risk)")
    boards = ensure_boards()
    if not boards:
        st.stop()

    # Try preferred field; allow fallback if not present
    requested_field = "HWAV_RISKS"
    field_name, used_fallback = pick_field_with_fallback(boards, requested_field)
    if used_fallback:
        st.info(f"Using `{field_name}` (fallback selection)")

    raw_vals = {}
    for b in boards:
        v_raw = get_attr_ci(b.get('attrs', {}) or {}, field_name)
        num = coerce_nonneg_numeric(v_raw, 0.0)
        raw_vals[b["unit_id"]] = num if num is not None else 0.0
    raw_series = pd.Series(raw_vals).sort_index()

    cleaned = normalize_0_1(raw_series)
    bins    = equal_interval_bins_4(cleaned)
    color_map = {0: PALETTE4[0], 1: PALETTE4[1], 2: PALETTE4[2], 3: PALETTE4[3]}
    color_series = pd.Series({uid: color_map[bins.loc[uid]] for uid in cleaned.index})

    fc = fc_from_boards_and_values(boards, cleaned, prop_name="uhi_norm", color_series=color_series)

    lat_c, lon_c, zoom_c = boards_bbox({"type": "FeatureCollection",
                                        "features": [{"type": "Feature", "geometry": b["feature_geom"]} for b in boards]})
    label_layer = cb_label_layer_from_boards(boards, text_size=10)
    deck = pydeck_choropleth(fc, lat_c, lon_c, zoom_c, prop_name="uhi_norm",
                             tooltip_label="Heat Wave Risk (0–1)", opacity=0.90, label_layer=label_layer)
    c_map, c_leg = st.columns([4,1])
    with c_map:
        st.pydeck_chart(deck, use_container_width=True)
    with c_leg:
        st.markdown(
            legend_html("NRI Heat Wave Risk (equal intervals)", ["0–0.25","0.25–0.5","0.5–0.75","0.75–1.0"], PALETTE4),
            unsafe_allow_html=True
        )

    # Claude
    st.subheader("Explain this heat map (Claude)")
    if st.button("Explain Heat Map", key="explain_heat_btn"):
        stats = {
            "min": float(cleaned.min()),
            "max": float(cleaned.max()),
            "mean": float(cleaned.mean()),
            "median": float(cleaned.median())
        }
        context = {"domain": "urban_heat", "map_type": "risk", "variant": "HWAV_RISKS", "field": field_name, "stats": stats}
        prompt = (
            "You are a heat-risk planning assistant. Explain the displayed heat-wave risk map for NYC Census Tracts with actionable insights. "
            "Use census tract numbers when referencing locations.\n\n"
            f"Context JSON:\n{json.dumps(context, indent=2)}"
        )
        system = "Be concise and practical for city stakeholders."
        answer = call_claude(prompt, system=system)
        st.markdown("**Claude:**")
        st.write(answer)
        genai_log(prompt, answer, meta={"span_name": "explain_heat_map", "field": field_name})

if "page" not in st.session_state:
    st.session_state.page = "landing"

sidebar()
if st.session_state.page == "landing":
    page_landing()
elif st.session_state.page == "urban":
    urban_features_page()
elif st.session_state.page == "risk":
    page_risk_mapping()
elif st.session_state.page == "forecast":
    forecasting_page()
elif st.session_state.page == "uhi":
    page_uhi()
