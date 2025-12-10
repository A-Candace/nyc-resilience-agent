# app.py
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
    Bedrock Claude call with safe defaults. Falls back to a stub if Bedrock isn’t configured.
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
    idx = min(idx, len(BLUES) - 1)
    return BLUES[idx]

def normalize_cb_id(val):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    try:
        return str(int(float(val)))
    except Exception:
        return str(val)

def get_board_display_id(board):
    attrs = board.get("attrs", {}) or {}
    boro_cd = attrs.get("boro_cd")
    if boro_cd is not None and str(boro_cd) != "":
        return normalize_cb_id(boro_cd)
    return normalize_cb_id(board.get("unit_id", ""))

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

# =====================================================================
# Data merges and helpers
# =====================================================================

def load_and_merge_board_data(boards: list, csv_path: str = "DataForBoxPlots.csv") -> list:
    """
    Merge CSV static attributes onto boards by CB_id <-> boro_cd.
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

def save_choropleth_png(boards: list, values_by_board: pd.Series, title: str) -> bytes:
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
    breaks = compute_breaks(s, k=len(BLUES))

    def rgb_norm(c):
        return (c[0]/255.0, c[1]/255.0, c[2]/255.0)

    fig, ax = plt.subplots(figsize=(8, 8))
    patches = []
    colors = []

    for b in boards:
        uid = b["unit_id"]
        v = float(s.get(uid, 0.0))
        col = rgb_norm(color_for_value_dynamic(v, breaks))

        geom = b["geom"]
        if geom.geom_type == "Polygon":
            rings = [np.asarray(geom.exterior.coords)]
            for r in rings:
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
    "NRI Coastal": """**TODO – NRI Coastal Flooding Risk (methodology)**
- TODO: data source(s)
- TODO: downscaling to Community Boards
- TODO: variables/features & combination
- TODO: caveats/assumptions
""",
    "NRI Riverine": """**TODO – NRI Riverine Flooding Risk (methodology)**
- TODO: data source(s)
- TODO: downscaling to Community Boards
- TODO: variables/features & combination
- TODO: caveats/assumptions
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
- Based on published research about spatial drivers (imperviousness, building footprint, slope, elevation).
- **TODO:** cite the exact papers/sections you want here.
"""

# =====================================================================
# Map builders
# =====================================================================

def fc_from_boards_and_values(boards: list, values: pd.Series, prop_name: str) -> dict:
    breaks = compute_breaks(values, k=len(BLUES))
    feats = []
    for b in boards:
        uid = b["unit_id"]
        v = float(pd.to_numeric(values.get(uid, 0.0), errors="coerce") or 0.0)
        feats.append({
            "type": "Feature",
            "properties": {
                "unit_id": uid,
                "cb": get_board_display_id(b),
                prop_name: v,
                "fill_color": color_for_value_dynamic(v, breaks)
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
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom_c),
        tooltip={"html": f"<b>Board #:</b> {{cb}}<br/><b>{tooltip_label}:</b> {{{prop_name}:.2f}}",
                 "style": {"backgroundColor": "white", "color": "black"}}
    )
    return deck

# =====================================================================
# Sidebar & routing
# =====================================================================

def sidebar():
    with st.sidebar:
        st.markdown("### NYC Resilience AI Agent")
        st.caption("Rebuilt for Static Features, Risk Maps, and Forecasting stubs\nwith MLflow + GenAI logging.")
        st.divider()
        st.button("🏠 Home", use_container_width=True, on_click=lambda: st.session_state.update(page="landing"))
        st.button("🏙️ Urban Features", use_container_width=True, on_click=lambda: st.session_state.update(page="urban"))
        st.button("🗺️ Risk Mapping", use_container_width=True, on_click=lambda: st.session_state.update(page="risk"))
        st.button("📈 Forecasting", use_container_width=True, on_click=lambda: st.session_state.update(page="forecast"))
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

LOCAL_CB_ZIP = os.getenv("LOCAL_CB_ZIP", "NYC Community Boards_20251006.zip")

def ensure_boards() -> list:
    boards = st.session_state.get("boards")
    if boards:
        return boards

    # 1) Try local zip
    boards = load_boards_from_shapefile_path(LOCAL_CB_ZIP)
    # 2) Merge CSV if present
    if boards:
        boards = load_and_merge_board_data(boards, "DataForBoxPlots.csv")
        st.session_state["boards"] = boards
        return boards

    # 3) Upload alternatives
    st.info("Upload Community Board GeoJSON or Shapefile ZIP.")
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        uploaded_geojson = st.file_uploader("GeoJSON", type=["geojson", "json"], key="geojson_upl")
        if uploaded_geojson is not None:
            try:
                b = uploaded_geojson.read()
                js = json.loads(b.decode("utf-8"))
            except UnicodeDecodeError:
                js = json.loads(b.decode("latin-1"))
            boards = load_community_boards_from_json(js)
    with col_u2:
        boards = boards or load_boards_from_shapefile_zip_streamlit()

    if boards:
        boards = load_and_merge_board_data(boards, "DataForBoxPlots.csv")
        st.session_state["boards"] = boards
    return boards

def home_page():
    st.title("🏙️ NYC Resilience AI Agent")
    st.subheader("Static Urban Features • Risk Maps • Forecasting (stubs)")
    st.write("This build focuses on mapping Community Boards, static features, and placeholders for risk/forecasting.")
    st.divider()
    st.markdown("#### Workflow & Logging")
    st.markdown("- MLflow logs map generations, parameters, and GenAI (Claude) explanations.")
    st.markdown("- Each map is downloadable as **PNG** and **CSV**.")
    st.markdown("- Risk/Forecast pages include **TODO** methodology blocks for you to fill in later.")

def urban_features_page():
    st.title("🏙️ Urban Features (Static)")
    boards = ensure_boards()
    if not boards:
        st.stop()

    # Outline view
    fc_outlines = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": get_board_display_id(b)}, "geometry": b["feature_geom"]} for b in boards
    ]}
    lat_c, lon_c, zoom_c = boards_bbox(fc_outlines)

    label_layer = cb_label_layer_from_boards(boards, text_size=12)
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
                 tooltip={"html": "<b>Board:</b> {name}"})
    )
    st.caption("Community Boards/Districts (outline).")

    # Attribute choices
    st.subheader("Visualize a static attribute as a choropleth")
    attribute_options = {
        "Buildings per Area (BLDperArea)": "BLDperArea",
        "Footprint per Area (FTPperArea)": "FTPperArea",
        "Percentage Impervious Cover": "Imperv",
        "Elevation": "Elevation",
        "Slope": "Slope",
        "Commuting": "Commuting",
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
    label_layer = cb_label_layer_from_boards(boards, text_size=12)
    deck = pydeck_choropleth(fc, lat_c, lon_c, zoom_c, prop_name="value", tooltip_label=sel_label, opacity=0.85, label_layer=label_layer)
    st.pydeck_chart(deck, use_container_width=True)

    # Downloads: CSV + PNG
    c1, c2 = st.columns(2)
    with c1:
        dl_df = series.rename(sel_attr).reset_index()
        dl_df.columns = ["unit_id", sel_attr]
        dl_df["cb_number"] = dl_df["unit_id"].map(lambda x: normalize_cb_id(x))
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
            "You are a flood-planning assistant. Explain the static attribute map for NYC Community Boards, "
            "focusing on patterns and practical interpretation for flood risk.\n\n"
            f"Context JSON:\n{json.dumps(context, indent=2)}\n\n"
            "Use CB numbers when referencing locations."
        )
        system = "Be concise and practical. Mention hotspots and why they matter."
        answer = call_claude(prompt, system=system)

        st.markdown("**Claude:**")
        st.write(answer)

        # MLflow GenAI logging
        genai_log(prompt, answer, meta={"span_name": "explain_urban_feature", "attribute": sel_label})

def risk_mapping_page():
    st.title("🗺️ Risk Mapping")
    boards = ensure_boards()
    if not boards:
        st.stop()

    # Choose map
    risk_choice = st.radio(
        "Risk layer:",
        ["NRI Coastal Flooding Risk Map", "NRI Riverine Flooding Risk Map", "My Risk Map"],
        horizontal=False
    )

    st.caption("Upload a board-level CSV (optional) with columns: unit_id, value (0–1 or any numeric).")
    upl = st.file_uploader("Risk CSV (optional)", type=["csv"], key="risk_upload")

    # Build values_by_board
    values = None
    if upl is not None:
        try:
            df = pd.read_csv(upl)
            if not {"unit_id", "value"}.issubset(set(df.columns)):
                st.error("CSV must have columns: unit_id, value")
            else:
                s = df.set_index("unit_id")["value"]
                # ensure all boards get a value
                values = pd.Series({b["unit_id"]: float(pd.to_numeric(s.get(b["unit_id"]), errors="coerce") or 0.0) for b in boards})
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    if values is None:
        # Placeholder: derive a simple synthetic risk from available static fields if present
        # (equal-weight z-score over: Imperv, FTPperArea, BLDperArea; minus Elevation)
        vals = {}
        # Gather components
        comps = {
            "Imperv": [],
            "FTPperArea": [],
            "BLDperArea": [],
            "Elevation": []
        }
        for b in boards:
            a = b.get("attrs", {})
            for k in comps.keys():
                v = a.get(k)
                try:
                    comps[k].append(float(v) if v is not None and pd.notna(v) else np.nan)
                except Exception:
                    comps[k].append(np.nan)
        comps = {k: pd.Series(v, index=[brd["unit_id"] for brd in boards]) for k, v in comps.items()}

        def z(x):
            x = pd.to_numeric(x, errors="coerce")
            return (x - x.mean(skipna=True)) / (x.std(skipna=True) + 1e-9)

        score = 0
        cnt = 0
        if "Imperv" in comps:
            score = score + z(comps["Imperv"]); cnt += 1
        if "FTPperArea" in comps:
            score = score + z(comps["FTPperArea"]); cnt += 1
        if "BLDperArea" in comps:
            score = score + z(comps["BLDperArea"]); cnt += 1
        if "Elevation" in comps:
            score = score - z(comps["Elevation"]); cnt += 1
        if cnt == 0:
            score = pd.Series(0, index=[b["unit_id"] for b in boards])
        score = (score - score.min()) / (score.max() - score.min() + 1e-9)
        values = score

    # Map
    fc = fc_from_boards_and_values(boards, values, prop_name="risk")
    lat_c, lon_c, zoom_c = boards_bbox({"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": b["feature_geom"]} for b in boards]})
    label_layer = cb_label_layer_from_boards(boards, text_size=12)
    deck = pydeck_choropleth(fc, lat_c, lon_c, zoom_c, prop_name="risk", tooltip_label="Risk", opacity=0.90, label_layer=label_layer)
    st.pydeck_chart(deck, use_container_width=True)

    # Downloads
    c1, c2 = st.columns(2)
    with c1:
        dl_df = values.rename("risk").reset_index()
        dl_df.columns = ["unit_id", "risk"]
        dl_df["cb_number"] = dl_df["unit_id"].map(lambda x: normalize_cb_id(x))
        st.download_button(
            "Download CSV",
            data=dl_df.to_csv(index=False).encode("utf-8"),
            file_name=f"risk_map_{risk_choice.replace(' ','_').lower()}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c2:
        png_bytes = save_choropleth_png(boards, values, f"Risk Map: {risk_choice}")
        st.download_button(
            "Download PNG",
            data=png_bytes,
            file_name=f"risk_map_{risk_choice.replace(' ','_').lower()}.png",
            mime="image/png",
            use_container_width=True
        )

    st.divider()
    st.markdown("#### Methodology")
    meth_key = "My Risk Map"
    if risk_choice.startswith("NRI Coastal"):
        meth_key = "NRI Coastal"
    elif risk_choice.startswith("NRI Riverine"):
        meth_key = "NRI Riverine"
    st.info(METH_RISK[meth_key])

    # Claude explanation + MLflow GenAI logging
    st.subheader("Explain this risk map (Claude)")
    if st.button("Explain Risk Map"):
        # Stats
        stats = {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "median": float(values.median())
        }
        context = {"map_type": "risk", "variant": risk_choice, "stats": stats}
        prompt = (
            "You are a flood-planning assistant. Explain the displayed risk map for NYC Community Boards, "
            "with actionable insights. Use community board numbers when referencing locations.\n\n"
            f"Context JSON:\n{json.dumps(context, indent=2)}"
        )
        system = "Be concise and practical for city stakeholders."
        answer = call_claude(prompt, system=system)

        st.markdown("**Claude:**")
        st.write(answer)

        genai_log(prompt, answer, meta={"span_name": "explain_risk_map", "variant": risk_choice})

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

    st.write("Upload an optional **board-level baseline** CSV to visualize (columns: unit_id, value). "
             "If omitted, a neutral placeholder map is shown.")
    upl = st.file_uploader("Optional: baseline board scores for preview", type=["csv"], key="fcst_upload")

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
    label_layer = cb_label_layer_from_boards(boards, text_size=12)
    deck = pydeck_choropleth(fc, lat_c, lon_c, zoom_c, prop_name="forecast", tooltip_label="Forecast (placeholder)", opacity=0.90, label_layer=label_layer)
    st.pydeck_chart(deck, use_container_width=True)

    # Downloads
    c1, c2 = st.columns(2)
    with c1:
        dl_df = values.rename("forecast_value").reset_index()
        dl_df.columns = ["unit_id", "forecast_value"]
        dl_df["cb_number"] = dl_df["unit_id"].map(lambda x: normalize_cb_id(x))
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
            "You are a flood-forecast assistant. Explain this **next-day** forecast preview for NYC Community Boards. "
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

if "page" not in st.session_state:
    st.session_state.page = "landing"

sidebar()
if st.session_state.page == "landing":
    home_page()
elif st.session_state.page == "urban":
    urban_features_page()
elif st.session_state.page == "risk":
    risk_mapping_page()
elif st.session_state.page == "forecast":
    forecasting_page()
