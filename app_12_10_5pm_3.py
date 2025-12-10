# app.py
from io import BytesIO
import os
import math
import json
import re
from datetime import datetime, timedelta, timezone, date
from typing import Optional

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
        return ("(Claude disabled — add AWS credentials &amp; region)\n\n"
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
    """
    if GENAI_OK:
        import mlflow.genai as mgen
        with mgen.start_trace() as trace:
            with mgen.start_span(name=meta.get("span_name", "map_explanation")) as span:
                span.set_inputs({"prompt": prompt_text, **{f"meta_{k}": v for k, v in meta.items()}})
                span.set_outputs({"response": response_text})
        try:
            mlflow.log_params({f"explain_{k}": str(v) for k, v in meta.items()})
        except Exception:
            pass
    else:
        try:
            with mlflow.start_run(run_name=f"genai_{meta.get('span_name','explain')}"):
                mlflow.log_params({f"explain_{k}": str(v) for k, v in meta.items()})
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
RED_YELLOW = [
    [255, 255, 178],
    [254, 217, 118],
    [254, 178, 76],
    [253, 141, 60],
    [252, 78, 42],
    [227, 26, 28],
    [177, 0, 38],
]

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
    tract_id = attrs.get("TRACTCE") or attrs.get("GEOID") or attrs.get("tract_id")
    if tract_id is not None and str(tract_id) != "":
        return normalize_cb_id(tract_id)
    return normalize_cb_id(board.get("unit_id", ""))

def get_attr_ci(attrs: dict, key: str):
    """Case-insensitive attribute getter"""
    if key in attrs:
        return attrs.get(key)
    lower = key.lower()
    for k, v in attrs.items():
        if (k or "").lower() == lower:
            return v
    return None

def coerce_numeric(val, default=0.0):
    """
    Convert value to numeric, handling:
    - None, NaN, empty strings → default
    - Non-numeric strings → default
    - Valid numbers → float
    """
    if val is None:
        return default
    if isinstance(val, (int, float)):
        if np.isnan(val):
            return default
        return float(val)
    if isinstance(val, str):
        val = val.strip()
        if val == '' or val.lower() in ('nan', 'null', 'none', 'n/a', '#n/a'):
            return default
        # Remove any non-numeric characters except decimal point and minus
        cleaned = re.sub(r'[^\d.-]', '', val)
        if cleaned == '' or cleaned == '-' or cleaned == '.':
            return default
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return default
    return default

def normalize_0_1(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)
    minv = s.min()
    maxv = s.max()
    if maxv - minv <= 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - minv) / (maxv - minv)

def legend_html(title: str, labels: list, colors_rgb: list) -> str:
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
# Excel-backed attributes (NRI + statics)
# =====================================================================

RISK_XLSX = os.getenv("RISK_XLSX", "./Risk_Attributes_Table_v4.xlsx")

def _digits_only(x: object) -> str:
    s = "" if x is None else str(x)
    return re.sub(r"\D+", "", s)

def _pad_left(s: str, n: int) -> str:
    s = s or ""
    return s.zfill(n)

def _normalize_geoids_from_parts(state, county, tract) -> str:
    return _pad_left(_digits_only(state), 2) + _pad_left(_digits_only(county), 3) + _pad_left(_digits_only(tract), 6)

def _normalize_geoid_any(s: str) -> str:
    d = _digits_only(s)
    if len(d) >= 11:
        return d[-11:]
    return _pad_left(d, 11)

@st.cache_data
def load_risk_xlsx(path_or_bytes) -> pd.DataFrame:
    df = pd.read_excel(path_or_bytes, engine="openpyxl").copy()

    if all(c in df.columns for c in ["STATEFIPS", "COUNTYFIPS", "TRACT"]):
        df["__join_key__"] = df.apply(lambda r: _normalize_geoids_from_parts(r["STATEFIPS"], r["COUNTYFIPS"], r["TRACT"]), axis=1)
    else:
        candidates = [c for c in ["NRI_ID","GEOID","GEOID10","GEOID20","TRACTFIPS","TRACTCE","TRACT"] if c in df.columns]
        if candidates:
            col = candidates[0]
            df["__join_key__"] = df[col].map(_normalize_geoid_any)
        else:
            guess = None
            for c in df.columns:
                if re.search(r"(geoid|tract|fips)", str(c), flags=re.I):
                    guess = c; break
            if guess:
                df["__join_key__"] = df[guess].map(_normalize_geoid_any)
            else:
                df["__join_key__"] = ""

    return df

def merge_attrs_into_tracts(_tracts: list, df_excel: pd.DataFrame) -> tuple:
    """
    Merge Excel attributes onto tracts by NRI_ID.
    ONLY keeps Excel data - discards all shapefile attributes except geometry.
    Returns (tracts, coverage_ratio)
    """
    # Check if NRI_ID column exists in Excel
    if "NRI_ID" not in df_excel.columns:
        st.error("❌ Excel file must have an 'NRI_ID' column.")
        return _tracts, 0.0
    
    # Build lookup dictionary: NRI_ID -> all attributes
    df = df_excel.copy()
    df["NRI_ID"] = df["NRI_ID"].astype(str).str.strip()
    
    # Create dictionary keyed by NRI_ID
    excel_lookup = {}
    for _, row in df.iterrows():
        nri_id = str(row["NRI_ID"]).strip()
        if nri_id and nri_id != "" and nri_id.lower() != "nan":
            excel_lookup[nri_id] = row.to_dict()
    
    st.write(f"📊 Excel: {len(excel_lookup)} NRI_IDs loaded")
    
    # Match tracts to Excel using CensusTrac field
    merged = 0
    total = len(_tracts)
    sample_matches = []
    
    for idx, t in enumerate(_tracts):
        attrs = t.get("attrs", {}) or {}
        
        # Get NRI_ID from CensusTrac field
        tract_nri_id = None
        
        # Primary: Use CensusTrac field directly
        if "CensusTrac" in attrs:
            tract_nri_id = str(attrs["CensusTrac"]).strip()
        
        # Fallback: Build from FIPS components if CensusTrac missing
        if not tract_nri_id or tract_nri_id.lower() == "nan":
            state = str(attrs.get("CensusTr_3", "")).strip()
            county = str(attrs.get("CensusTr_6", "")).strip()
            tract = str(attrs.get("CensusTr_8", "")).strip()
            
            if state and county and tract:
                state = state.zfill(2)
                county = county.zfill(3)
                tract = tract.zfill(6)
                tract_nri_id = f"T{state}{county}{tract}"
        
        if not tract_nri_id:
            if idx < 3:
                sample_matches.append({
                    "tract_idx": idx,
                    "unit_id": t.get("unit_id"),
                    "status": "FAILED - no NRI_ID found"
                })
            continue
        
        # Look up in Excel
        if tract_nri_id in excel_lookup:
            excel_row = excel_lookup[tract_nri_id]
            
            # REPLACE attrs entirely with Excel data (keep only NRI_ID for reference)
            new_attrs = {
                "NRI_ID": tract_nri_id,
                # Keep only the columns we need from Excel
                "Mean_elevation": excel_row.get("Mean_elevation"),
                "Mean_slope": excel_row.get("Mean_slope"),
                "Total_ftp_area": excel_row.get("Total_ftp_area"),
                "AREA": excel_row.get("AREA"),
                "CFLD_RISKS": excel_row.get("CFLD_RISKS"),
                "RFLD_RISKS": excel_row.get("RFLD_RISKS"),
                "HWAV_RISKS": excel_row.get("HWAV_RISKS"),
                "score_precip": excel_row.get("score_precip"),
                "score_sf": excel_row.get("score_sf"),
                "score_cb": excel_row.get("score_cb"),
                "score_total": excel_row.get("score_total"),
            }
            
            # Replace attrs with ONLY Excel data
            t["attrs"] = new_attrs
            
            merged += 1
            
            if idx < 3:
                sample_matches.append({
                    "tract_idx": idx,
                    "unit_id": t.get("unit_id"),
                    "nri_id": tract_nri_id,
                    "has_elevation": new_attrs.get("Mean_elevation") is not None,
                    "has_slope": new_attrs.get("Mean_slope") is not None,
                    "has_footprint": new_attrs.get("Total_ftp_area") is not None,
                    "status": "✅ MATCHED"
                })
        else:
            if idx < 3:
                sample_matches.append({
                    "tract_idx": idx,
                    "unit_id": t.get("unit_id"),
                    "nri_id": tract_nri_id,
                    "status": "❌ Not in Excel"
                })
            # If not matched, clear attrs (keep only geometry)
            t["attrs"] = {"NRI_ID": tract_nri_id}
    
    coverage = merged / max(1, total)
    
    st.write(f"✅ Matched {merged}/{total} tracts ({coverage*100:.1f}%)")
    
    if sample_matches:
        st.write("**Sample merge attempts:**")
        import pandas as pd
        st.dataframe(pd.DataFrame(sample_matches))
    
    if coverage < 0.5:
        st.warning("⚠️ Low coverage - check if Excel NRI_IDs match shapefile CensusTrac values")
        sample_excel = list(excel_lookup.keys())[:5]
        st.write("**Sample Excel NRI_IDs:**", sample_excel)
    
    return _tracts, coverage

def get_attr_num(attrs: dict, col: str) -> float:
    """
    Safely fetch numeric from attrs[col] (case-insensitive).
    Handles all non-numeric values by converting to 0.0
    """
    raw = get_attr_ci(attrs, col)
    return coerce_numeric(raw, default=0.0)

def quantile_bins_4(series: pd.Series) -> tuple:
    """
    Returns (bin[0..3], edges[5]) where edges are raw-value edges used for the legend.
    Handles duplicate edges gracefully.
    """
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)

    if len(s) == 0:
        edges = np.array([0, 0.25, 0.5, 0.75, 1.0], dtype=float)
        bins = pd.Series(0, index=s.index, dtype=int)
        return bins, edges

    vmin, vmax = float(s.min()), float(s.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0

    # If all values are identical
    if np.isclose(vmin, vmax):
        edges = np.linspace(vmin, vmax + 1e-9, 5)
        bins = pd.Series(0, index=s.index, dtype=int)
        return bins, edges

    # Try quantile-based edges
    edges = np.quantile(s, [0, 0.25, 0.5, 0.75, 1.0]).astype(float)
    
    # Check for duplicate edges
    unique_edges = np.unique(edges)
    
    if len(unique_edges) < 5:
        # Fall back to equal-interval binning
        edges = np.linspace(vmin, vmax, 5)
        # Ensure strictly increasing
        for i in range(1, len(edges)):
            if edges[i] <= edges[i-1]:
                edges[i] = edges[i-1] + (vmax - vmin) * 0.01
    else:
        # Ensure strictly non-decreasing
        edges = np.maximum.accumulate(edges)
        # Add small epsilon to ensure strict inequality where needed
        for i in range(1, len(edges)):
            if np.isclose(edges[i], edges[i-1]):
                edges[i] = edges[i-1] + (vmax - vmin) * 0.001

    # Final safety check
    if np.isclose(edges[0], edges[-1]):
        edges[-1] = edges[0] + 1e-9

    labels = list(range(4))  # 0..3
    
    try:
        bins = pd.cut(s, bins=edges, labels=labels, include_lowest=True, right=True, duplicates='drop')
    except ValueError:
        # If still failing, use equal intervals
        edges = np.linspace(vmin, vmax + 1e-9, 5)
        bins = pd.cut(s, bins=edges, labels=labels, include_lowest=True, right=True, duplicates='drop')
    
    # Convert to int, handling any remaining NaN
    bins = bins.astype("Int64").fillna(0).astype(int).clip(0, 3)
    
    return bins, edges

def legend_with_ranges_html(title: str, edges: np.ndarray, colors=([255,255,178],[254,217,118],[253,141,60],[227,26,28])) -> str:
    labels = [
        f"{edges[0]:,.2f} – {edges[1]:,.2f}",
        f"{edges[1]:,.2f} – {edges[2]:,.2f}",
        f"{edges[2]:,.2f} – {edges[3]:,.2f}",
        f"{edges[3]:,.2f} – {edges[4]:,.2f}",
    ]
    rows=[]
    for lab,col in zip(labels, colors):
        r,g,b = col
        rows.append(
            f'<div style="display:flex;align-items:center;margin:4px 0;">'
            f'<span style="display:inline-block;width:16px;height:16px;background:rgb({r},{g},{b});margin-right:8px;border:1px solid #333;"></span>'
            f'<span style="font-size:12px;">{lab}</span></div>'
        )
    return (f'<div style="border:1px solid #ccc;border-radius:8px;padding:8px 10px;'
            f'background:#fff;max-width:280px;">'
            f'<div style="font-weight:600;margin-bottom:6px;">{title} (quartiles)</div>'
            f'{"".join(rows)}</div>')

# =====================================================================
# Geometry load &amp; transforms
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
    """Load census tracts from GeoJSON"""
    if not SHAPELY_OK:
        st.error("Census tract loading requires 'shapely'.")
        return None
    
    feats = js.get("features", [])
    tracts = []
    
    for idx, f in enumerate(feats):
        props = f.get("properties", {}) or {}
        
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
        
        clean_attrs = dict(props)
        
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
        import shapefile
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

    import tempfile
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
        clean_attrs = dict(rec)

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
        st.success(f"✅ Loaded {len(polys)} census tracts from shapefile.")
    return polys or None

# =====================================================================
# Data merges and helpers
# =====================================================================

def load_and_merge_board_data(boards: list, csv_path: str = "DataForBoxPlots.csv") -> list:
    """Legacy function kept for compatibility"""
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
    """Render a simple static choropleth using matplotlib and return PNG bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

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
# Methodology text
# =====================================================================

METH_RISK_URBAN = {
    "Urban Combined": """**Urban Flooding - Combined Risk Score (methodology)**

**Overview:**  
This in-house flood risk score captures **pluvial (rainfall-driven) urban flooding** by combining three community-based and sensor-based indicators. Unlike NRI coastal/riverine scores, this focuses on localized stormwater flooding across NYC census tracts.

**Data Sources:**
- **FloodNet water-depth sensors**: High-frequency depth measurements at 100+ locations across NYC
- **NOAA MRMS radar precipitation**: Hourly rainfall at ~1km resolution
- **NYC 311 service requests**: Community-reported street flooding and catch basin issues

**Three Component Scores:**

1. **score_precip** (Precipitation Sensitivity, range 0-5):
   - Derived from Negative Binomial GLM modeling sensor depth response to daily rainfall
   - Sensors with significant positive rain response (p<0.05, coef>0) score 4-5
   - Measures: *How much do water levels rise when it rains?*

2. **score_sf** (Street Flooding Reports, range 0-4):
   - Based on average daily 311 street flooding complaints within 0.5km of sensors on wet days
   - Normalized across all sensors with positive complaint rates
   - Measures: *How often do residents report street flooding during rain events?*

3. **score_cb** (Catch Basin Issues, range 0-4):
   - Based on average daily 311 catch basin complaints on wet days
   - Includes clogged, damaged, or malfunctioning drainage infrastructure reports
   - Measures: *How often do residents report drainage infrastructure problems?*

**score_total = score_precip + score_sf + score_cb** (range 0-13)

**Spatial Propagation:**
- Scores computed at 155 FloodNet sensor locations
- 500 synthetic "virtual sensors" fill spatial gaps using topographic similarity (elevation, slope, building proximity)
- Each census tract assigned scores from nearest sensor (real or synthetic) via spatial join

**Interpretation:**
- **High score_total (red)**: Areas with confirmed sensor-detected flooding AND high community reporting
- **High score_sf**: Community-identified street flooding hotspots (infrastructure may be overwhelmed)
- **High score_cb**: Areas with chronic drainage infrastructure issues (maintenance priority)
- **High score_precip**: Hydrologically sensitive locations (water accumulates during storms)

**Use Cases:**
- **score_total**: Overall urban flood vulnerability for capital planning
- **score_sf**: Prioritize emergency response and flood warning systems
- **score_cb**: Target catch basin cleaning, repair, and green infrastructure retrofits
- **score_precip**: Validate sensor network expansion locations

**Limitations:**
- Reflects 2020-2023 conditions (sensor deployment period)
- Underrepresents areas >0.5km from sensors (despite synthetic interpolation)
- 311 reports are biased toward higher-income, digitally-connected communities
- Does not include coastal/tidal flooding (see NRI CFLD_RISKS for that)

**Reference:** See Methods_Risk_Map.docx for full statistical workflow
""",
    
    "Urban Precip": """**Urban Flooding - Precipitation Sensitivity Score (methodology)**

**What it measures:**  
How strongly water depth at FloodNet sensors responds to rainfall, derived from daily-scale statistical modeling.

**Method:**
1. **Data**: 155 FloodNet sensors with hourly depth measurements merged with NOAA MRMS radar precipitation
2. **Aggregation**: Sum hourly depth and precipitation to daily totals per sensor
3. **Statistical Model**: Negative Binomial GLM per sensor:
   - Dependent variable: daily depth (mm) as count data
   - Predictors: daily precipitation (mm) + 311 complaint counts
   - Extract rain coefficient (`coef_precip`) and p-value
4. **Scoring**:
   - p-value ≥ 0.05 or coef ≤ 0 → **score = 0** (not rain-sensitive)
   - coef = 0 (exactly) → **score = 3** (neutral)
   - p < 0.05 and coef > 0 → normalize positive coefficients to **range 4-5** (most sensitive)

**Spatial Coverage:**  
Scores assigned to 500 synthetic sensors using topographic nearest-neighbor matching (elevation, slope, building distance), then propagated to census tracts via spatial join.

**Interpretation:**
- **High scores (red, 4-5)**: Locations where rain consistently causes measurable water accumulation
- **Zero scores (yellow)**: Areas with no detected rain-flood relationship (may be well-drained or sensors are dry)
- **~55 sensors** (out of 155) are statistically rain-sensitive

**Use for:** 
- Identifying chronic drainage problem areas
- Validating whether green infrastructure reduces rain response over time
- Sensor network optimization (place sensors where rain sensitivity is unknown)

**Limitation:** Does not capture human perception—use score_sf for community-reported impacts
""",
    
    "Urban SF": """**Urban Flooding - Street Flooding Reports Score (methodology)**

**What it measures:**  
Community-reported street flooding intensity during wet weather, based on NYC 311 service requests.

**Method:**
1. **311 Data Filtering**:
   - Select complaints with descriptor containing both "street" AND "flood" (case-insensitive)
   - Convert to hourly timestamps, filter to valid lat/lon coordinates
2. **Spatial Linking**:
   - Use BallTree (haversine distance) to assign each complaint to all FloodNet sensors within **0.5 km radius**
   - Aggregate to hourly complaint counts per sensor
3. **Wet-Day Focus**:
   - Merge with MRMS daily precipitation
   - Retain only days with precipitation > 0 mm
   - Compute **mean daily street flooding complaints per sensor on wet days**
4. **Scoring**:
   - Sensors with zero complaints → **score = 0**
   - Normalize positive values to **range 3-4** (linear scaling)

**Spatial Propagation:**  
Assigned to 500 synthetic sensors via topographic similarity, then to census tracts via nearest sensor.

**Interpretation:**
- **High scores (orange/red, 3.5-4)**: Areas where residents frequently report street flooding during rain
- **Zero scores**: Either well-drained OR underreporting (check against score_precip to distinguish)
- Captures **community perception** and **localized impacts** that sensors may miss

**Use for:**
- Emergency response prioritization during storms
- Identifying flood-prone intersections for signage/barricades
- Validating sensor placement (do sensors detect flooding where residents report it?)

**Limitations:**
- Reporting bias: wealthier, more engaged communities file more 311 requests
- Does not measure flood depth—only frequency of reports
- 0.5km radius may group unrelated flooding locations in dense areas

**Complementary to score_precip:** Sensors detect water, 311 detects *impacts on people*
""",
    
    "Urban CB": """**Urban Flooding - Catch Basin Issues Score (methodology)**

**What it measures:**  
Frequency of resident-reported catch basin problems (clogs, damage, capacity issues) during wet weather—an indicator of **drainage infrastructure failure**.

**Method:**
1. **311 Data Aggregation**:
   - Group all catch basin-related complaint descriptors into "Catch_Basin_Issues" category
   - Examples: "Clogged Catch Basin," "Broken Catch Basin," "Request Catch Basin Cleaning"
2. **Daily Complaint Counts**:
   - Merge 311 complaints with MRMS daily precipitation at sensor locations
   - Retain only **wet days** (daily precipitation > 0 mm)
3. **Scoring**:
   - Compute **mean daily catch basin complaints per sensor on wet days**
   - Sensors with zero complaints → **score = 0**
   - Normalize positive values to **range 3-4** (linear scaling)

**Spatial Propagation:**  
Assigned to 500 synthetic sensors via topographic nearest-neighbor, then to census tracts via spatial join.

**Interpretation:**
- **High scores (orange/red, 3.5-4)**: Chronic drainage infrastructure problems—catch basins are clogged, undersized, or broken
- **Zero scores**: Either well-maintained infrastructure OR underreporting
- Unlike score_sf (which reports *flooding*), this reports *the cause* (failed drainage)

**Use for:**
- **Infrastructure maintenance planning**: Prioritize catch basin cleaning and repair
- **Capital projects**: Identify areas needing upsized or additional catch basins
- **Green infrastructure targeting**: Where traditional gray infrastructure is failing, consider bioswales, permeable pavement
- **Real-time operations**: During storms, preemptively dispatch crews to high-score areas

**Limitations:**
- Reporting bias (same as score_sf)
- Does not measure actual catch basin condition—only community complaints
- Some complaints may be duplicate reports of the same basin
- Does not distinguish clog severity (partial vs. complete blockage)

**Complementary to score_sf:**  
- High score_cb + High score_sf = flooding caused by infrastructure failure → **fix catch basins**
- Low score_cb + High score_sf = flooding despite working drains → **inadequate capacity** → need upsizing or green infrastructure
""",
}

METH_RISK = {
    "NRI Coastal": """**NRI Coastal Flooding Risk (methodology)**
- **Source:** FEMA National Risk Index (NRI), tract-level download.
- **Metric used:** `CFLD_RISKS` from spreadsheet (raw NRI composite risk score; higher = higher risk).
- **Join:** Tract IDs normalized to digits-only and matched to feature attributes.
- **Mapping:** Colors are **quartiles of the raw NRI values**, not normalized; legend shows **actual value ranges**.
- **Interpretation:** The NRI "Risk" combines Expected Annual Loss, Social Vulnerability, and Community Resilience.
- **Caveats:** NRI scores are comparative; use alongside local knowledge, drainage and land-use data.
""",
    "NRI Riverine": """**NRI Riverine Flooding Risk (methodology)**
- **Source:** FEMA NRI (tract level).
- **Metric used:** `RFLD_RISKS` raw values from spreadsheet.
- **Join &amp; mapping:** Same as coastal — **quartiles on raw values** with range legend.
- **Interpretation/Caveats:** Same as coastal.
""",
    "My Risk Map": """**In-House Flood Risk (methodology)**
- **Source columns:** `score_total` in the spreadsheet (custom composite).
- **Intended inputs:** static features such as Elevation, Slope, Imperviousness, Footprint and any learned weights.
- **Mapping:** **Quartiles of raw `score_total`**, legend shows value ranges.
- **Next steps:** Document weighting scheme; add validation against observed flooding or 311 events.
"""
}

METH_FORECAST = {
    "Water sensors": """**TODO – Next-Day Forecast (Water Sensors)**
- TODO: sensors &amp; units
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
- TODO: aggregation to boards &amp; cadence
- TODO: baseline/persistence definition
- TODO: known limitations
"""
}

METH_URBAN = """**Static Urban Features (context)**
- **Elevation**: Mean elevation per census tract from USGS DEM data, aggregated using Zonal Statistics
- **Slope**: Mean slope per census tract derived from elevation data, aggregated using Zonal Statistics
- **Building Footprint**: Total building footprint area (square kilometers) per census tract, summed from NYC building footprint dataset
- **Building Footprint Density**: Ratio of building footprint area to total tract area (both in sq km). Higher values indicate more built-up areas.
- **Data Processing**: All spatial data aggregated to census tract level using ArcGIS Pro Zonal Statistics and Summarize Within tools
- **Interpretation**: Higher footprint density and lower elevation typically correlate with increased flood vulnerability
"""

METH_UHI = """**Urban Heat Island / Heat-Wave Risk (methodology)**
- **Source:** FEMA NRI heat-wave component.
- **Metric used:** `HWAV_RISKS` from the spreadsheet (raw values).
- **Join:** Same normalized tract-ID match as flood layers.
- **Mapping:** **Quartiles of raw values**; legend shows value ranges (not z-scores).
- **Use:** Prioritize cooling centers, tree planting, reflective surfaces in higher-risk tracts.
"""

# =====================================================================
# Map builders
# =====================================================================

def fc_from_boards_and_values(boards: list, values: pd.Series, prop_name: str, color_series: Optional[pd.Series] = None) -> dict:
    """Build a FeatureCollection with proper color handling"""
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
                col = [255, 255, 178]
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
    # Only add label layer if explicitly provided (which we won't do anymore)
    if label_layer:
        layers.append(label_layer)
    
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
# NEW: Query Engine for Multi-Criteria Selection
# =====================================================================

def parse_user_query_with_claude(user_query: str, boards: list) -> dict:
    """Use Claude to parse natural language into structured criteria"""
    
    available_fields = {
        "CFLD_RISKS": "Coastal flooding risk (NRI)",
        "RFLD_RISKS": "Riverine flooding risk (NRI)",
        "HWAV_RISKS": "Heat wave risk (NRI)",
        "score_total": "In-house flood risk score",
        "Mean_elevation": "Mean elevation (meters)",
        "Mean_slope": "Mean slope (degrees)",
        "Total_ftp_area": "Building footprint area (sq km)",
        "_COMPUTED_DENSITY_": "Building footprint density"
    }
    
    system = (
        "You are a query parser for NYC census tract data. "
        "Parse natural language queries into JSON criteria. "
        "Return ONLY valid JSON with no explanation.\n\n"
        "Available fields:\n" + "\n".join([f"- {k}: {v}" for k, v in available_fields.items()]) +
        "\n\nBin meanings (equal intervals):\n"
        "- bin 3 = highest 25% (red)\n"
        "- bin 2 = 50-75% (orange)\n"
        "- bin 1 = 25-50% (yellow)\n"
        "- bin 0 = lowest 25% (light yellow)"
    )
    
    prompt = f"""Parse this query into JSON:
"{user_query}"

Return format:
{{
  "criteria": [
    {{"field": "RFLD_RISKS", "bin": 3, "description": "highest riverine risk"}},
    {{"field": "Mean_elevation", "bin": 0, "description": "lowest elevation"}}
  ]
}}

Keywords:
- "highest", "high", "red" → bin 3
- "high-medium" → bin 2
- "low-medium" → bin 1  
- "lowest", "low" → bin 0

Return ONLY the JSON object."""
    
    response = call_claude(prompt, system=system, temperature=0.1, max_tokens=800)
    
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"criteria": []}
    except Exception as e:
        st.error(f"Parse error: {e}")
        return {"criteria": []}

def filter_tracts_by_criteria(boards: list, criteria: list) -> list:
    """Filter tracts meeting ALL criteria"""
    if not criteria:
        return []
    
    # Compute bins for all relevant fields
    field_bins = {}
    for crit in criteria:
        field = crit.get("field")
        if not field:
            continue
        
        # Handle computed density
        if field == "_COMPUTED_DENSITY_":
            vals = {}
            for b in boards:
                attrs = b.get("attrs", {})
                ftp = coerce_numeric(get_attr_ci(attrs, "Total_ftp_area"), default=0.0)
                area_sqmi = coerce_numeric(get_attr_ci(attrs, "AREA"), default=0.0)
                area_sqkm = area_sqmi * 2.58999
                if area_sqkm > 0:
                    vals[b["unit_id"]] = ftp / area_sqkm
                else:
                    vals[b["unit_id"]] = 0.0
        else:
            vals = {b["unit_id"]: get_attr_num(b.get("attrs", {}), field) for b in boards}
        
        series = pd.Series(vals)
        bins, _ = quantile_bins_4(series)
        field_bins[field] = bins
    
    # Filter tracts
    matching = []
    for b in boards:
        uid = b["unit_id"]
        meets_all = True
        
        for crit in criteria:
            field = crit.get("field")
            target_bin = crit.get("bin")
            
            if field not in field_bins:
                meets_all = False
                break
            
            tract_bin = field_bins[field].get(uid, 0)
            
            if tract_bin != target_bin:
                meets_all = False
                break
        
        if meets_all:
            matching.append(b)
    
    return matching

# =====================================================================
# Sidebar &amp; routing
# =====================================================================

def sidebar():
    with st.sidebar:
        st.markdown("### NYC Resilience AI Agent")
        st.caption("Census Tract Analysis: Static Features, Risk Maps, and Forecasting\nwith MLflow + GenAI logging.")
        st.divider()
        st.button("🏠 Home", use_container_width=True, on_click=lambda: st.session_state.update(page="landing"), key="nav_home")
        st.button("🏙️ Urban Features", use_container_width=True, on_click=lambda: st.session_state.update(page="urban"), key="nav_urban")
        st.button("🗺️ Risk Mapping", use_container_width=True, on_click=lambda: st.session_state.update(page="risk"), key="nav_risk")
        st.button("🤖 AI Query", use_container_width=True, on_click=lambda: st.session_state.update(page="query"), key="nav_query")
        st.button("💬 Chat", use_container_width=True, on_click=lambda: st.session_state.update(page="chat"), key="nav_chat")
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

LOCAL_CENSUS_TRACTS = os.getenv("LOCAL_CENSUS_TRACTS", "nyc_census_tracts.shp")

def ensure_boards() -> list:
    """Load census tracts with Excel merge"""
    boards = st.session_state.get("boards")
    if boards:
        return boards

    base_path = LOCAL_CENSUS_TRACTS.replace('.shp', '')
    if os.path.exists(base_path + '.shp'):
        st.info(f"Loading census tracts from {LOCAL_CENSUS_TRACTS}...")
        boards = _load_census_tracts_from_shapefile_direct(base_path)
        if boards:
            try:
                df_excel = load_risk_xlsx(RISK_XLSX)
                boards, cov = merge_attrs_into_tracts(boards, df_excel)
                if cov < 0.75:
                    st.warning(f"⚠️ Excel merge coverage: {cov*100:.1f}% — check tract IDs.")
                else:
                    st.success(f"✅ Excel attributes merged ({cov*100:.1f}% coverage).")
            except Exception as e:
                st.warning(f"Excel merge skipped: {e}")
            st.session_state["boards"] = boards
            return boards

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
        try:
            df_excel = load_risk_xlsx(RISK_XLSX)
            boards, cov = merge_attrs_into_tracts(boards, df_excel)
            if cov < 0.75:
                st.warning(f"⚠️ Excel merge coverage: {cov*100:.1f}% — check tract IDs.")
            else:
                st.success(f"✅ Excel attributes merged ({cov*100:.1f}% coverage).")
        except Exception as e:
            st.warning(f"Excel merge skipped: {e}")
        st.session_state["boards"] = boards
    return boards

def _load_census_tracts_from_shapefile_direct(base_path: str):
    """Load census tracts directly from shapefile components"""
    if not SHAPELY_OK:
        st.error("Shapefile support requires 'shapely>=2.0'.")
        return None
    try:
        import shapefile
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

    def looks_like_lonlat(x, y):
        return (-180.0 <= x <= 180.0) and (-90.0 <= y <= 90.0)

    tracts = []
    for idx, sr in enumerate(r.shapeRecords()):
        rec = {fields[i]: sr.record[i] for i in range(len(fields))}
        
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

        if transformer is None:
            try:
                if geom.geom_type == 'Polygon':
                    x0, y0 = list(geom.exterior.coords)[0]
                elif geom.geom_type == 'MultiPolygon':
                    x0, y0 = list(list(geom.geoms)[0].exterior.coords)[0]
                else:
                    x0, y0 = (0, 0)
                if not looks_like_lonlat(x0, y0):
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
            st.warning(f"Skipping {unit_id}: geometry is {geom.geom_type}")
            continue

        feature_geom = to_pure_geojson(geom)
        clean_attrs = dict(rec)

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
    st.markdown("#### Workflow &amp; Logging")
    st.markdown("- MLflow logs map generations, parameters, and GenAI (Claude) explanations.")
    st.markdown("- Each map is downloadable as **PNG** and **CSV**.")
    st.markdown("- Risk maps use FEMA NRI data at census tract level.")
    st.markdown("- Color scheme: **Red (high values/risk) → Yellow (low values/risk)**")

def urban_features_page():
    st.title("🏙️ Urban Features (Static) - Census Tracts")
    boards = ensure_boards()
    if not boards:
        st.stop()

    fc_outlines = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": get_board_display_id(b)}, "geometry": b["feature_geom"]} for b in boards
    ]}
    lat_c, lon_c, zoom_c = boards_bbox(fc_outlines)

    # Remove label_layer from outline view
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
        pdk.Deck(layers=[bg, outline],  # Removed label_layer here
                 initial_view_state=pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom_c),
                 tooltip={"html": "<b>Tract:</b> {name}"})
    )
    st.caption("NYC Census Tracts (outline).")

    st.subheader("Visualize a static attribute as a choropleth")
    
    # Define field mappings - now includes computed density
    field_mappings = {
        "Mean Elevation": ["Mean_elevation"],
        "Mean Slope": ["Mean_slope"],
        "Total Building Footprint (sq km)": ["Total_ftp_area"],
        "Building Footprint Density (ratio)": ["_COMPUTED_DENSITY_"],  # Special computed field
    }
    
    sel_label = st.selectbox("Attribute", list(field_mappings.keys()), index=0)
    possible_fields = field_mappings[sel_label]
    
    # Handle computed density field
    if possible_fields[0] == "_COMPUTED_DENSITY_":
        sel_attr = "_COMPUTED_DENSITY_"
        st.info(f"Using computed field: Total_ftp_area / AREA (both in sq km)")
        
        # Compute density for each tract
        vals = {}
        for b in boards:
            attrs = b.get("attrs", {})
            ftp = coerce_numeric(get_attr_ci(attrs, "Total_ftp_area"), default=0.0)  # Already in sq km
            area_sqmi = coerce_numeric(get_attr_ci(attrs, "AREA"), default=0.0)  # In square miles
            
            # Convert square miles to square kilometers (1 sq mi = 2.58999 sq km)
            area_sqkm = area_sqmi * 2.58999
            
            # Avoid division by zero
            if area_sqkm > 0:
                vals[b["unit_id"]] = ftp / area_sqkm
            else:
                vals[b["unit_id"]] = 0.0
        
        series = pd.Series(vals).sort_index()
    else:
        # Find which field actually exists in the data
        sel_attr = None
        for field in possible_fields:
            for b in boards[:5]:
                if get_attr_ci(b.get("attrs", {}), field) is not None:
                    sel_attr = field
                    break
            if sel_attr:
                break
        
        if not sel_attr:
            st.error(f"Could not find any of these fields: {possible_fields}")
            st.write("Available fields in first tract:", list(boards[0].get("attrs", {}).keys())[:20])
            st.stop()
        
        st.info(f"Using field: `{sel_attr}`")

        # Extract values using the robust numeric coercion
        vals = {}
        for b in boards:
            raw_val = get_attr_ci(b.get("attrs", {}), sel_attr)
            vals[b["unit_id"]] = coerce_numeric(raw_val, default=0.0)
        
        series = pd.Series(vals).sort_index()
    
    # Show stats
    with st.expander("📊 Data Statistics", expanded=False):
        st.write(f"**Field used:** `{sel_attr}`")
        st.write(f"**Non-zero values:** {(series > 0).sum()} / {len(series)}")
        st.dataframe(series.describe())

    bins, edges = quantile_bins_4(series)
    color_map = {0: PALETTE4[0], 1: PALETTE4[1], 2: PALETTE4[2], 3: PALETTE4[3]}
    color_series = pd.Series({uid: color_map[bins.loc[uid]] for uid in series.index})

    fc = fc_from_boards_and_values(boards, series, prop_name="value", color_series=color_series)
    
    # Remove label_layer from choropleth view too
    deck = pydeck_choropleth(fc, lat_c, lon_c, zoom_c, prop_name="value",
                             tooltip_label=sel_label, opacity=0.90, label_layer=None)  # Set to None
    c_map, c_leg = st.columns([4,1])
    with c_map:
        st.pydeck_chart(deck, use_container_width=True)
    with c_leg:
        st.markdown(legend_with_ranges_html(sel_label, edges, colors=PALETTE4), unsafe_allow_html=True)

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

    st.subheader("Explain this map (Claude)")
    if st.button("Explain"):
        context = {
            "map_type": "urban_static",
            "attribute": sel_label,
            "field_used": sel_attr,
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

        genai_log(prompt, answer, meta={"span_name": "explain_urban_feature", "attribute": sel_label, "field": sel_attr})

def list_numeric_candidates(boards, max_show=60):
    """Scan attribute keys and return numeric-looking fields"""
    keys = set()
    samples = []
    for b in boards[:min(len(boards), max_show)]:
        attrs = b.get("attrs", {}) or {}
        for k, v in attrs.items():
            kl = (k or "").lower()
            if any(tok in kl for tok in ["shape", "geom", "globalid", "objectid"]):
                continue
            num = coerce_numeric(v, None)
            if num is not None:
                keys.add(k)
                samples.append((k, num))
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
    """Try requested field; if missing, surface a selectbox"""
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
        st.error("No numeric-looking fields found in the layer's attributes.")
        return requested, False
    chosen = st.selectbox("Choose field", options=candidates, index=0, key=f"risk_field_fallback_{requested}")
    return chosen, True

def page_risk_mapping():
    st.title("🗺️ Risk Mapping (NRI + Custom)")
    boards = ensure_boards()
    if not boards:
        st.stop()

    # Choose which risk field to show
    risk_choice = st.radio(
        "Risk layer:",
        [
            "Coastal Flooding Risk (NRI - CFLD_RISKS)",
            "Riverine Flooding Risk (NRI - RFLD_RISKS)",
            "─────────────────────────────────",
            "Urban Flooding - Combined Score (score_total)",
            "Urban Flooding - Water Sensor Response (score_precip)",
            "Urban Flooding - Street Flooding Reports (score_sf)",
            "Urban Flooding - Catch Basin Issues (score_cb)"
        ],
        horizontal=False
    )
    
    # Skip divider selection
    if "─────" in risk_choice:
        st.info("👆 Please select a risk layer above")
        st.stop()
    
    if "Coastal" in risk_choice:
        requested_field = "CFLD_RISKS"
        pretty          = "NRI Coastal Flood Risk"
        meth_key        = "NRI Coastal"
        risk_type       = "nri"
        allow_fallback  = False
    elif "Riverine" in risk_choice:
        requested_field = "RFLD_RISKS"
        pretty          = "NRI Riverine Flood Risk"
        meth_key        = "NRI Riverine"
        risk_type       = "nri"
        allow_fallback  = False
    elif "Combined Score" in risk_choice:
        requested_field = "score_total"
        pretty          = "Urban Flooding - Combined Risk"
        meth_key        = "Urban Combined"
        risk_type       = "urban"
        allow_fallback  = True
    elif "Water Sensor" in risk_choice:
        requested_field = "score_precip"
        pretty          = "Urban Flooding - Precipitation Sensitivity"
        meth_key        = "Urban Precip"
        risk_type       = "urban"
        allow_fallback  = True
    elif "Street Flooding" in risk_choice:
        requested_field = "score_sf"
        pretty          = "Urban Flooding - Street Flooding Reports"
        meth_key        = "Urban SF"
        risk_type       = "urban"
        allow_fallback  = True
    elif "Catch Basin" in risk_choice:
        requested_field = "score_cb"
        pretty          = "Urban Flooding - Catch Basin Issues"
        meth_key        = "Urban CB"
        risk_type       = "urban"
        allow_fallback  = True

    # For NRI fields, use directly; for custom, allow fallback
    if allow_fallback:
        field_name, used_fallback = pick_field_with_fallback(boards, requested_field)
        if used_fallback:
            st.info(f"Using `{field_name}` (custom field selection)")
    else:
        field_name = requested_field
        # Check if field exists
        found = False
        for b in boards[:10]:
            if get_attr_ci(b.get("attrs", {}), field_name) is not None:
                found = True
                break
        if not found:
            st.error(f"❌ Required NRI field `{field_name}` not found in the merged Excel data. "
                    f"Please ensure Risk_Attributes_Table_v4.xlsx contains this column and the merge was successful.")
            st.stop()

    # Extract robust numeric RAW values from merged Excel attrs
    raw_vals = {}
    for b in boards:
        v_raw = get_attr_ci(b.get('attrs', {}) or {}, field_name)
        raw_vals[b["unit_id"]] = coerce_numeric(v_raw, default=0.0)
    raw_series = pd.Series(raw_vals).sort_index()

    # Bin by quartiles of RAW values
    bins, edges = quantile_bins_4(raw_series)

    # Map each quartile to a color
    color_map = {0: PALETTE4[0], 1: PALETTE4[1], 2: PALETTE4[2], 3: PALETTE4[3]}
    color_series = pd.Series({uid: color_map[int(bins.loc[uid])] for uid in bins.index})

    # Build GeoJSON FeatureCollection using RAW values for tooltips
    fc = fc_from_boards_and_values(
        boards,
        raw_series,
        prop_name="risk_value",
        color_series=color_series
    )

    # Render map + legend with numeric ranges per quartile
    lat_c, lon_c, zoom_c = boards_bbox({
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": b["feature_geom"]} for b in boards]
    })
    deck = pydeck_choropleth(
        fc, lat_c, lon_c, zoom_c,
        prop_name="risk_value",
        tooltip_label=f"{pretty} (raw)",
        opacity=0.90,
        label_layer=None
    )

    c_map, c_leg = st.columns([4, 1])
    with c_map:
        st.pydeck_chart(deck, use_container_width=True)
    with c_leg:
        range_labels = [f"{edges[i]:.2f} – {edges[i+1]:.2f}" for i in range(4)]
        st.markdown(
            legend_html(f"{pretty} (quartiles of raw values)", range_labels, PALETTE4),
            unsafe_allow_html=True
        )

    # Debug / QA panel
    with st.expander("🔎 Debug distribution &amp; field check", expanded=False):
        st.write("**Raw value stats:**")
        st.dataframe(raw_series.describe().to_frame(name="raw_stats"))

        counts = bins.value_counts().sort_index().reindex([0, 1, 2, 3], fill_value=0)
        counts.index = [f"Q{i+1}" for i in range(4)]
        st.write("**Quartile bin counts:**")
        st.dataframe(counts.to_frame(name="count").T)

        sample_attrs = boards[0].get("attrs", {}) or {}
        st.write("**Example feature attribute keys (first 60):**")
        st.write(list(sample_attrs.keys())[:60])
        st.write(f"**Field in use:** `{field_name}` → example raw:", get_attr_ci(sample_attrs, field_name))

        if raw_series.nunique() <= 1:
            st.info("All values are identical or nearly identical. Coloring still works via safe edges, but variance is minimal.")

    # Downloads
    c1, c2 = st.columns(2)
    with c1:
        dl_df = pd.DataFrame({
            "unit_id": raw_series.index,
            "risk_raw": raw_series.values,
            "bin_quartile_0..3": bins.values
        })
        dl_df["tract_id"] = dl_df["unit_id"].map(lambda x: normalize_cb_id(x))
        st.download_button(
            "Download CSV",
            data=dl_df.to_csv(index=False).encode("utf-8"),
            file_name=f"risk_map_{field_name.lower()}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c2:
        png_series = pd.Series({k: float(v) for k, v in bins.items()})
        png_bytes = save_choropleth_png(boards, png_series, f"Risk Map: {pretty}", palette=PALETTE4, breaks=None)
        st.download_button(
            "Download PNG",
            data=png_bytes,
            file_name=f"risk_map_{field_name.lower()}.png",
            mime="image/png",
            use_container_width=True
        )

    # Methodology + Claude explanation
    st.divider()
    st.markdown("#### Methodology")
    
    # Display appropriate methodology based on risk type
    if risk_type == "nri":
        st.info(METH_RISK[meth_key])
    else:  # urban flooding
        st.info(METH_RISK_URBAN.get(meth_key, METH_RISK["My Risk Map"]))

    src = st.session_state.get("boards_source", "Uploaded GeoJSON/Shapefile")
    st.caption(f"**Source**: {src} • **Field used**: `{field_name}`")

    st.subheader("Explain this risk map (Claude)")
    if st.button("Explain Risk Map", key="explain_risk_btn"):
        stats = {
            "min": float(raw_series.min()),
            "max": float(raw_series.max()),
            "mean": float(raw_series.mean()),
            "median": float(raw_series.median())
        }
        context = {
            "domain": "urban_flooding" if risk_type == "urban" else "nri_flooding",
            "map_type": "risk",
            "variant": pretty,
            "field": field_name,
            "stats": stats,
            "legend_ranges": [f"{edges[i]:.2f} – {edges[i+1]:.2f}" for i in range(4)]
        }
        
        # Customize prompt based on risk type
        if risk_type == "urban":
            prompt = (
                "You are a flood-planning assistant analyzing NYC census tracts with in-house urban flooding indicators. "
                "This analysis combines FloodNet sensor data, community-reported flooding, and infrastructure issues to assess pluvial (rainfall-driven) flood risk. "
                "Use census tract numbers when referencing locations.\n\n"
                f"Context JSON:\n{json.dumps(context, indent=2)}\n\n"
                "Explain patterns, hotspots, and actionable insights for city stakeholders."
            )
        else:
            prompt = (
                "You are a flood-planning assistant analyzing NYC census tracts with FEMA NRI flood risk data. "
                "This shows either coastal or riverine flooding risk from the National Risk Index. "
                "Use census tract numbers when referencing locations.\n\n"
                f"Context JSON:\n{json.dumps(context, indent=2)}\n\n"
                "Explain patterns, hotspots, and actionable insights for city stakeholders."
            )
        
        system = "Be concise and practical for city stakeholders. Focus on spatial patterns and recommended actions."
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

    tomorrow = date.today() + timedelta(days=1)
    when = st.date_input("Forecast date (next-day)", value=tomorrow, min_value=date.today(), max_value=date.today()+timedelta(days=14))

    st.write("Upload an optional **census tract-level baseline** CSV to visualize (columns: unit_id, value). "
             "If omitted, a neutral placeholder map is shown.")
    upl = st.file_uploader("Optional: baseline tract scores for preview", type=["csv"], key="fcst_upload")

    values = None
    if upl is not None:
        try:
            df = pd.read_csv(upl)
            if {"unit_id", "value"}.issubset(df.columns):
                s = df.set_index("unit_id")["value"]
                values = pd.Series({b["unit_id"]: coerce_numeric(s.get(b["unit_id"]), default=0.0) for b in boards})
            else:
                st.error("CSV must have: unit_id, value")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    if values is None:
        values = pd.Series({b["unit_id"]: 0.0 for b in boards})

    fc = fc_from_boards_and_values(boards, values, prop_name="forecast")
    lat_c, lon_c, zoom_c = boards_bbox({"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": b["feature_geom"]} for b in boards]})
    label_layer = cb_label_layer_from_boards(boards, text_size=10)
    deck = pydeck_choropleth(fc, lat_c, lon_c, zoom_c, prop_name="forecast", tooltip_label="Forecast (placeholder)", opacity=0.90, label_layer=None)
    st.pydeck_chart(deck, use_container_width=True)

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
    
    st.divider()
    st.subheader("🤖 AI-Powered Tools")
    
    col3, col4 = st.columns(2)
    with col3:
        if st.button("🔍 AI Query (Multi-Criteria Search)", use_container_width=True, key="landing_query"):
            st.session_state.page = "query"
            st.rerun()
        st.caption("Find tracts matching complex criteria using natural language")
    
    with col4:
        if st.button("💬 Chat with Claude", use_container_width=True, key="landing_chat"):
            st.session_state.page = "chat"
            st.rerun()
        st.caption("Ask questions about data, methodology, or get recommendations")

def page_uhi():
    st.title("🌤️ Urban Heat Island (NRI Heat Wave Risk)")
    boards = ensure_boards()
    if not boards:
        st.stop()

    # NRI field - should exist in Excel, no fallback needed
    requested_field = "HWAV_RISKS"
    field_name = requested_field
    
    # Check if field exists
    found = False
    for b in boards[:10]:
        if get_attr_ci(b.get("attrs", {}), field_name) is not None:
            found = True
            break
    if not found:
        st.error(f"❌ Required NRI field `{field_name}` not found in the merged Excel data. "
                f"Please ensure Risk_Attributes_Table_v4.xlsx contains this column and the merge was successful.")
        st.stop()

    # Pull RAW values
    raw_vals = {}
    for b in boards:
        v_raw = get_attr_ci(b.get('attrs', {}) or {}, field_name)
        raw_vals[b["unit_id"]] = coerce_numeric(v_raw, default=0.0)
    raw_series = pd.Series(raw_vals).sort_index()

    # Quartile bins + numeric range labels
    bins, edges = quantile_bins_4(raw_series)
    color_map = {0: PALETTE4[0], 1: PALETTE4[1], 2: PALETTE4[2], 3: PALETTE4[3]}
    color_series = pd.Series({uid: color_map[int(bins.loc[uid])] for uid in bins.index})

    # Build FC with colors driven by quartile, tooltip shows RAW values
    fc = fc_from_boards_and_values(
        boards,
        raw_series,
        prop_name="uhi_value",
        color_series=color_series
    )

    lat_c, lon_c, zoom_c = boards_bbox({
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": b["feature_geom"]} for b in boards]
    })
    label_layer = cb_label_layer_from_boards(boards, text_size=10)
    deck = pydeck_choropleth(
        fc, lat_c, lon_c, zoom_c,
        prop_name="uhi_value",
        tooltip_label="Heat Wave Risk (raw)",
        opacity=0.90,
        label_layer=None
    )
    c_map, c_leg = st.columns([4,1])
    with c_map:
        st.pydeck_chart(deck, use_container_width=True)
    with c_leg:
        range_labels = [f"{edges[i]:.2f} – {edges[i+1]:.2f}" for i in range(4)]
        st.markdown(
            legend_html("NRI Heat-Wave Risk (quartiles)", range_labels, PALETTE4),
            unsafe_allow_html=True
        )

    # Debug panel
    with st.expander("🔎 Debug distribution &amp; field check", expanded=False):
        st.write("**Raw value stats:**")
        st.dataframe(raw_series.describe().to_frame(name="raw_stats"))
        counts = bins.value_counts().sort_index().reindex([0,1,2,3], fill_value=0)
        counts.index = [f"Q{i+1}" for i in range(4)]
        st.write("**Quartile bin counts:**")
        st.dataframe(counts.to_frame(name="count").T)

        sample_attrs = boards[0].get("attrs", {}) or {}
        st.write("**Example feature attribute keys (first 60):**")
        st.write(list(sample_attrs.keys())[:60])
        st.write(f"**Field in use:** `{field_name}` → example raw:", get_attr_ci(sample_attrs, field_name))

        if raw_series.nunique() <= 1:
            st.info("All values are identical or nearly identical. Colors collapse to one bin.")

    # Downloads
    c1, c2 = st.columns(2)
    with c1:
        dl_df = pd.DataFrame({
            "unit_id": raw_series.index,
            "heat_risk_raw": raw_series.values,
            "bin_quartile_0..3": bins.values
        })
        dl_df["tract_id"] = dl_df["unit_id"].map(lambda x: normalize_cb_id(x))
        st.download_button(
            "Download CSV",
            data=dl_df.to_csv(index=False).encode("utf-8"),
            file_name=f"uhi_{field_name.lower()}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c2:
        png_series = pd.Series({k: float(v) for k, v in bins.items()})
        png_bytes = save_choropleth_png(boards, png_series, "UHI: Heat Wave Risk", palette=PALETTE4, breaks=None)
        st.download_button(
            "Download PNG",
            data=png_bytes,
            file_name=f"uhi_{field_name.lower()}.png",
            mime="image/png",
            use_container_width=True
        )

    # Claude explanation
    st.subheader("Explain this heat map (Claude)")
    if st.button("Explain Heat Map", key="explain_heat_btn"):
        stats = {
            "min": float(raw_series.min()),
            "max": float(raw_series.max()),
            "mean": float(raw_series.mean()),
            "median": float(raw_series.median())
        }
        context = {"domain": "urban_heat", "map_type": "risk", "variant": "HWAV_RISKS", "field": field_name,
                   "stats": stats, "legend_ranges": [f"{edges[i]:.2f} – {edges[i+1]:.2f}" for i in range(4)]}
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

# =====================================================================
# NEW: AI Query Page for Multi-Criteria Tract Selection
# =====================================================================

def page_ai_query():
    st.title("🤖 AI-Powered Tract Query")
    st.caption("Use natural language to find census tracts matching multiple criteria")
    
    boards = ensure_boards()
    if not boards:
        st.stop()
    
    st.markdown("### Example Queries:")
    st.markdown("- *Show tracts with highest riverine flooding risk AND highest heat wave risk*")
    st.markdown("- *Find tracts with lowest elevation AND highest coastal flooding risk*")
    st.markdown("- *Show tracts in the red portion (highest interval) for riverine, coastal, and heat wave*")
    st.markdown("- *Find tracts with highest building density and highest coastal flood risk*")
    
    user_query = st.text_area(
        "Enter your query:",
        height=100,
        placeholder="e.g., Show me tracts with highest coastal flood risk and lowest elevation",
        key="query_input"  # Add key to preserve state
    )
    
    if st.button("🔍 Search Tracts", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a query")
            st.stop()
        
        with st.spinner("🤖 Claude is parsing your query..."):
            parsed = parse_user_query_with_claude(user_query, boards)
        
        # Store results in session state
        st.session_state.query_results = {
            "parsed": parsed,
            "query": user_query
        }
    
    # Check if we have stored results
    if "query_results" not in st.session_state:
        st.info("👆 Enter a query above to get started")
        st.stop()
    
    # Use stored results
    parsed = st.session_state.query_results["parsed"]
    original_query = st.session_state.query_results["query"]
    
    st.write("**Original query:**", original_query)
    st.write("**Parsed criteria:**")
    st.json(parsed)
    
    criteria = parsed.get("criteria", [])
    if not criteria:
        st.error("Could not parse query. Try rephrasing with specific field names.")
        if st.button("🔄 Try Again"):
            del st.session_state.query_results
            st.rerun()
        st.stop()
    
    # Compute density if needed
    for crit in criteria:
        if crit.get("field") == "_COMPUTED_DENSITY_":
            for b in boards:
                attrs = b.get("attrs", {})
                ftp = coerce_numeric(get_attr_ci(attrs, "Total_ftp_area"), default=0.0)
                area_sqmi = coerce_numeric(get_attr_ci(attrs, "AREA"), default=0.0)
                area_sqkm = area_sqmi * 2.58999
                if area_sqkm > 0:
                    attrs["_COMPUTED_DENSITY_"] = ftp / area_sqkm
                else:
                    attrs["_COMPUTED_DENSITY_"] = 0.0
    
    matching_tracts = filter_tracts_by_criteria(boards, criteria)
    
    st.success(f"✅ Found **{len(matching_tracts)}** matching tracts (out of {len(boards)})")
    
    if not matching_tracts:
        st.info("No tracts match all criteria. Try relaxing your query.")
        if st.button("🔄 New Search"):
            del st.session_state.query_results
            st.rerun()
        st.stop()
    
    # Build map: grey for all, blue for matching
    fc_features = []
    for b in boards:
        uid = b["unit_id"]
        is_match = any(m["unit_id"] == uid for m in matching_tracts)
        fill_color = [0, 120, 255, 200] if is_match else [200, 200, 200, 100]
        
        fc_features.append({
            "type": "Feature",
            "properties": {
                "unit_id": uid,
                "cb": get_board_display_id(b),
                "matched": "Yes" if is_match else "No",
                "fill_color": fill_color
            },
            "geometry": b["feature_geom"]
        })
    
    fc = {"type": "FeatureCollection", "features": fc_features}
    
    layer = pdk.Layer(
        "GeoJsonLayer",
        data=fc,
        pickable=True,
        stroked=True,
        filled=True,
        get_fill_color="properties.fill_color",
        get_line_color=[40, 40, 40],
        lineWidthMinPixels=1,
        opacity=1.0
    )
    bg = pdk.Layer(
        "TileLayer",
        data="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        minZoom=0, maxZoom=19, tileSize=256
    )
    
    lat_c, lon_c, zoom_c = boards_bbox(fc)
    
    deck = pdk.Deck(
        layers=[bg, layer],
        initial_view_state=pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom_c),
        tooltip={"html": "<b>Tract:</b> {cb}<br><b>Matched:</b> {matched}<br><i>Click for details</i>"}
    )
    
    st.pydeck_chart(deck, use_container_width=True)
    
    # Legend
    st.markdown("""
    <div style="padding:10px; background:#f0f0f0; border-radius:5px; margin:10px 0;">
        <b>Legend:</b> 
        <span style="color:#0078ff;">● Blue</span> = Matching tracts | 
        <span style="color:#c8c8c8;">● Grey</span> = Non-matching tracts
    </div>
    """, unsafe_allow_html=True)
    
    # Show matching tracts with OpenStreetMap
    st.subheader(f"📍 {len(matching_tracts)} Matching Tracts")
    
    # Select a tract to view
    tract_options = {get_board_display_id(m): m for m in matching_tracts}
    
    # Use session state to preserve selection
    if "selected_tract_key" not in st.session_state:
        st.session_state.selected_tract_key = "-- Select --"
    
    selected_display = st.selectbox(
        "Select a tract to view in detail:",
        options=["-- Select --"] + list(tract_options.keys()),
        index=0 if st.session_state.selected_tract_key == "-- Select --" else 
              (["-- Select --"] + list(tract_options.keys())).index(st.session_state.selected_tract_key) 
              if st.session_state.selected_tract_key in tract_options else 0,
        key="tract_selector_dropdown"
    )
    
    # Update session state
    st.session_state.selected_tract_key = selected_display
    
    if selected_display != "-- Select --":
        selected_tract = tract_options[selected_display]
        attrs = selected_tract.get("attrs", {})
        nri_id = attrs.get("NRI_ID", "")
        
        # Get centroid
        centroid = selected_tract["geom"].centroid
        lat, lon = centroid.y, centroid.x
        
        st.markdown(f"### 🗺️ Tract {selected_display}")
        
        # Show attribute values
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**NRI ID:** {nri_id}")
            st.write(f"**Coordinates:** {lat:.6f}, {lon:.6f}")
            st.write("**Criteria values:**")
            for crit in criteria:
                field = crit.get("field")
                val = get_attr_num(attrs, field)
                desc = crit.get("description", field)
                st.write(f"- {desc}: **{val:.2f}**")
        
        with col2:
            st.write("**Quick Links:**")
            osm_url = f"https://www.openstreetmap.org/#map=18/{lat}/{lon}"
            osm_edit = f"https://www.openstreetmap.org/edit?editor=id#map=19/{lat}/{lon}"
            google_maps = f"https://www.google.com/maps/@{lat},{lon},18z"
            
            st.markdown(f"[🗺️ OpenStreetMap]({osm_url})")
            st.markdown(f"[📐 OSM Measurement Tool]({osm_edit})")
            st.markdown(f"[🛰️ Google Maps Satellite]({google_maps})")
        
        # Embed OpenStreetMap
        st.markdown("### 🗺️ Interactive Map View")
        
        try:
            import folium
            from streamlit_folium import st_folium
            
            # Create folium map centered on the tract
            m = folium.Map(
                location=[lat, lon],
                zoom_start=17,
                tiles='OpenStreetMap'
            )
            
            # Add satellite imagery tile layer
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Satellite',
                overlay=False,
                control=True
            ).add_to(m)
            
            # Add additional useful layers
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Satellite with Labels',
                overlay=False,
                control=True
            ).add_to(m)
            
            # Add OpenStreetMap as an alternative (already default, but making it explicit)
            folium.TileLayer(
                tiles='OpenStreetMap',
                name='Street Map',
                overlay=False,
                control=True
            ).add_to(m)
            
            # Add marker for the tract center
            folium.Marker(
                [lat, lon],
                popup=f"<b>Tract {selected_display}</b><br>NRI ID: {nri_id}<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}",
                tooltip=f"Tract {selected_display}",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
            
            # Add tract boundary polygon if available
            try:
                from shapely.geometry import mapping
                tract_geojson = mapping(selected_tract["geom"])
                folium.GeoJson(
                    tract_geojson,
                    style_function=lambda x: {
                        'fillColor': '#ff0000',
                        'color': '#ff0000',
                        'weight': 3,
                        'fillOpacity': 0.2,
                        'dashArray': '5, 5'
                    },
                    tooltip=f"Tract {selected_display} boundary"
                ).add_to(m)
            except Exception:
                pass  # Skip if polygon rendering fails
            
            # Add layer control to toggle between views
            folium.LayerControl(position='topright').add_to(m)
            
            # Add scale bar
            folium.plugins.MeasureControl(
                position='topleft',
                primary_length_unit='meters',
                secondary_length_unit='miles',
                primary_area_unit='sqmeters',
                secondary_area_unit='acres'
            ).add_to(m)
            
            # Add fullscreen button
            folium.plugins.Fullscreen(
                position='topleft',
                title='Fullscreen',
                title_cancel='Exit fullscreen',
                force_separate_button=True
            ).add_to(m)
            
            # Display the map
            st_folium(m, width=700, height=500)
            
            # Add link to full OSM and other mapping tools
            larger_map_url = f"https://www.openstreetmap.org/?mlat={lat}&amp;mlon={lon}#map=18/{lat}/{lon}"
            osm_id_editor_url = f"https://www.openstreetmap.org/edit?editor=id#map=19/{lat}/{lon}"
            google_maps_url = f"https://www.google.com/maps/@{lat},{lon},18z"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<span><span style="color: rgb(150, 34, 73); font-weight: bold;">&lt;small&gt;</span><span style="color: black; font-weight: normal;">🗺️ OpenStreetMap</span><span style="color: rgb(150, 34, 73); font-weight: bold;">&lt;/small&gt;</span><br><br></span>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<span><span style="color: rgb(150, 34, 73); font-weight: bold;">&lt;small&gt;</span><span style="color: black; font-weight: normal;">📐 OSM Measurement Tool</span><span style="color: rgb(150, 34, 73); font-weight: bold;">&lt;/small&gt;</span><br><br></span>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<span><span style="color: rgb(150, 34, 73); font-weight: bold;">&lt;small&gt;</span><span style="color: black; font-weight: normal;">🛰️ Google Maps</span><span style="color: rgb(150, 34, 73); font-weight: bold;">&lt;/small&gt;</span><br><br></span>', unsafe_allow_html=True)
            
        except ImportError:
            st.warning("⚠️ Folium not installed. Install with: `pip install folium streamlit-folium`")
            
            # Fallback: just show the link
            larger_map_url = f"https://www.openstreetmap.org/?mlat={lat}&amp;mlon={lon}#map=18/{lat}/{lon}"
            st.markdown(f'**[🗺️ View this location on OpenStreetMap]({larger_map_url})**')
        
        st.info("💡 **Tip:** Use the layer control (top-right) to toggle between Street Map, Satellite, and Satellite with Labels. "
                "Use the ruler icon to measure distances. Click 'OSM Measurement Tool' to access the iD Editor with street view (look for the 📷 person icon on the left sidebar).")    # Download matching tracts CSV
    st.divider()
    match_data = []
    for m in matching_tracts:
        attrs = m.get("attrs", {})
        row = {
            "NRI_ID": attrs.get("NRI_ID"),
            "tract_display": get_board_display_id(m),
            "latitude": m["geom"].centroid.y,
            "longitude": m["geom"].centroid.x,
        }
        # Add criteria fields
        for crit in criteria:
            field = crit.get("field")
            row[field] = get_attr_num(attrs, field)
        match_data.append(row)
    
    match_df = pd.DataFrame(match_data)
    
    col_dl, col_new = st.columns([3, 1])
    with col_dl:
        st.download_button(
            "📥 Download Matching Tracts CSV",
            data=match_df.to_csv(index=False).encode("utf-8"),
            file_name="matching_tracts.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_new:
        if st.button("🔄 New Search", use_container_width=True):
            del st.session_state.query_results
            if "selected_tract_key" in st.session_state:
                del st.session_state.selected_tract_key
            st.rerun()

# =====================================================================
# NEW: Interactive Chat Page
# =====================================================================

def page_chat():
    st.title("💬 Chat with Claude")
    st.caption("Ask questions about methodology, data, or get analysis recommendations")
    
    boards = ensure_boards()
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about NYC resilience data, methodologies, or analysis..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Build context from available data
        context_info = {
            "num_tracts": len(boards) if boards else 0,
            "available_fields": {
                "static": ["Mean_elevation", "Mean_slope", "Total_ftp_area", "AREA", "Building_density"],
                "nri_risks": ["CFLD_RISKS (coastal)", "RFLD_RISKS (riverine)", "HWAV_RISKS (heat wave)"],
                "custom_risks": ["score_precip", "score_sf", "score_cb", "score_total"]
            },
            "methodologies": {
                "urban_features": METH_URBAN,
                "nri_coastal": METH_RISK["NRI Coastal"],
                "nri_riverine": METH_RISK["NRI Riverine"],
                "custom_risk": METH_RISK["My Risk Map"],
                "heat": METH_UHI
            }
        }
        
        system_prompt = (
            "You are an expert assistant for NYC urban resilience and climate risk analysis. "
            "You help city planners understand flood risk, heat vulnerability, and infrastructure resilience. "
            "Provide accurate, actionable insights based on FEMA NRI data and local analysis. "
            "If you need to look up specific information not in the provided context, clearly state: "
            "'I would need to search for [specific information]' and suggest reliable sources like FEMA.gov, NOAA, or NYC Open Data. "
            "Be concise but thorough. Use technical terms when appropriate but explain them."
        )
        
        full_prompt = f"""User question: {prompt}

Context about available data:
{json.dumps(context_info, indent=2)}

Provide a helpful, accurate answer. If the question requires:
- Specific tract-level data → explain what fields to check
- External research → suggest searching official sources (FEMA, NOAA, NYC Open Data)
- Calculations → explain the methodology

Answer:"""
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Claude is thinking..."):
                response = call_claude(full_prompt, system=system_prompt, temperature=0.3, max_tokens=1500)
            st.markdown(response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Log to MLflow
        genai_log(full_prompt, response, meta={"span_name": "chat_interaction", "user_query": prompt})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# =====================================================================
# Router
# =====================================================================

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
elif st.session_state.page == "query":
    page_ai_query()
elif st.session_state.page == "chat":
    page_chat()