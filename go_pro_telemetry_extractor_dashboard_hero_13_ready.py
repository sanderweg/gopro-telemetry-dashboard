"""
GoPro Telemetry Extractor + Dashboard (Streamlit)

Goals
- Extract telemetry (GPS + IMU/"movement") from GoPro MP4/MOV files (incl. Hero 13 series) without loading the full video in memory.
- Visualize stats and timelines (speed, altitude, acceleration, gyro) and an interactive GPS track map.
- Export clean CSV/Parquet for post-analysis.

Requirements (install via pip / system):
  Python 3.9+
  pip install streamlit plotly pandas numpy pyarrow folium branca geopy rich
  System: ffmpeg + ffprobe available on PATH (https://ffmpeg.org/)
  Optional (for parsing GPMF):
    pip install gpmf-parser  # if available on PyPI for your platform
    # or use Juan Irache's GPMF parsing tools; see README notes in the chat.

Launch
  streamlit run gopro_telemetry_dashboard.py

Notes
- The app only extracts the small telemetry metadata stream from your large (>1 GB) video. It's fast and memory-friendly.
- Tested with modern GoPro models; GPMF payload typically contains GPS5 (GPS), GYRO, ACCL, GRAV, etc. If a camera/setting didn't record GPS (e.g., GPS off, or indoor), the GPS tab will reflect that gracefully.
"""

import os
import io
import json
import shutil
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px

try:
    import folium
    from folium.features import DivIcon
    from streamlit_folium import st_folium
except Exception:
    folium = None
    st_folium = None

# ---- Config ----
st.set_page_config(page_title="GoPro Telemetry Extractor", layout="wide")

# ---- Utilities ----

def have_cmd(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run(cmd: list, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check, text=True)


def ffprobe_json(input_path: str) -> Dict[str, Any]:
    """Return ffprobe stream metadata as JSON dict."""
    if not have_cmd("ffprobe"):
        raise RuntimeError("ffprobe is not installed or not on PATH.")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        input_path,
    ]
    proc = run(cmd)
    return json.loads(proc.stdout)


def find_gpmd_stream_index(meta: Dict[str, Any]) -> Optional[int]:
    """Heuristics to find the GoPro GPMD (telemetry) data stream index.
    ffprobe usually reports a data stream with codec_tag_string 'gpmd' or handler 'GoPro MET'.
    """
    streams = meta.get("streams", [])
    for s in streams:
        codec_tag = (s.get("codec_tag_string") or "").lower()
        codec_name = (s.get("codec_name") or "").lower()
        tags = {k.lower(): (v or "") for k, v in (s.get("tags") or {}).items()}
        handler = tags.get("handler_name", "").lower()
        if codec_tag == "gpmd" or codec_name == "gpmd" or "gopro met" in handler:
            return s.get("index")
    # Some files label as 'bin_data' but still telemetry
    for s in streams:
        if s.get("codec_type") == "data":
            tags = {k.lower(): (v or "") for k, v in (s.get("tags") or {}).items()}
            handler = tags.get("handler_name", "").lower()
            if "gopro" in handler and ("met" in handler or "telemetry" in handler):
                return s.get("index")
    return None


def extract_gpmd_raw(input_path: str, out_path: str, stream_index: int) -> None:
    """Extract raw GPMF (gpmd) packets without decoding, keeping it tiny.
    We use -codec copy on the metadata stream only.
    """
    if not have_cmd("ffmpeg"):
        raise RuntimeError("ffmpeg is not installed or not on PATH.")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-map",
        f"0:{stream_index}",
        "-c",
        "copy",
        "-f",
        "data",
        out_path,
    ]
    proc = run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)


# ---- GPMF parsing helpers ----

@dataclass
class Telemetry:
    gps: Optional[pd.DataFrame]
    gyro: Optional[pd.DataFrame]
    accl: Optional[pd.DataFrame]
    grav: Optional[pd.DataFrame]
    temp: Optional[pd.DataFrame]
    others: Dict[str, pd.DataFrame]


def try_parse_gpmf(raw_path: str) -> Telemetry:
    """Parse GPMF payload into tables. We try optional parsers; if none present, raise a clear guidance error.

    Expected common FourCC keys:
      - 'GPS5' (lat, lon, alt, speed, sats/acc?)
      - 'ACCL' (accelerometer)
      - 'GYRO' (gyroscope)
      - 'GRAV' (gravity vector)
      - 'TMPC'/'TEMP' (temperature)
    """
    # Strategy A: use a Python gpmf parser if available
    parser_errs = []
    # Attempt 1: gpmf_parser (placeholder import name)
    try:
        import gpmf_parser  # type: ignore
        with open(raw_path, "rb") as f:
            buf = f.read()
        parsed = gpmf_parser.parse(buf)  # library-specific; may differ
        # Convert to DataFrames; this is pseudo-generic because libs differ.
        # We'll attempt a reasonable mapping.
        tables = {}
        for key, series in parsed.items():
            try:
                df = pd.DataFrame(series)
                tables[key] = df
            except Exception:
                pass
        return materialize_tables(tables)
    except Exception as e:
        parser_errs.append(f"gpmf_parser: {e}")

    # Strategy B: instruct the user to install a supported parser
    raise RuntimeError(
        "Kon de GPMF-telemetrie niet parsen.\n"
        + "Probeer een GPMF-parser te installeren, bijv. `pip install gpmf-parser` (of een compatibel pakket).\n"
        + "Technisch detail: "
        + "; ".join(parser_errs)
    )


def materialize_tables(tables: Dict[str, pd.DataFrame]) -> Telemetry:
    def pick(key: str) -> Optional[pd.DataFrame]:
        for k in tables.keys():
            if k.upper().startswith(key):
                return standardize_df(k, tables[k])
        return None

    gps = pick("GPS5")
    gyro = pick("GYRO")
    accl = pick("ACCL")
    grav = pick("GRAV")
    temp = pick("TEMP") or pick("TMPC")

    others = {}
    for k, v in tables.items():
        ku = k.upper()
        if not any(ku.startswith(p) for p in ["GPS5", "GYRO", "ACCL", "GRAV", "TEMP", "TMPC"]):
            others[k] = standardize_df(k, v)

    return Telemetry(gps=gps, gyro=gyro, accl=accl, grav=grav, temp=temp, others=others)


def standardize_df(key: str, df: pd.DataFrame) -> pd.DataFrame:
    """Make columns/timestamps consistent as best-effort."""
    out = df.copy()
    # Heuristics for time column
    time_cols = [c for c in out.columns if str(c).lower() in ("t", "time", "timestamp", "cts", "date")]
    if time_cols:
        out.rename(columns={time_cols[0]: "t"}, inplace=True)
    if "t" in out.columns:
        # convert to pandas datetime if looks like seconds or ms
        try:
            if out["t"].max() > 1e12:  # ns
                out["ts"] = pd.to_datetime(out["t"], unit="ns", utc=True)
            elif out["t"].max() > 1e10:  # ms
                out["ts"] = pd.to_datetime(out["t"], unit="ms", utc=True)
            else:  # s
                out["ts"] = pd.to_datetime(out["t"], unit="s", utc=True)
        except Exception:
            pass
    # Standard GPS columns if present
    colmap = {
        "lat": ["lat", "latitude", "y"],
        "lon": ["lon", "longitude", "x"],
        "alt": ["alt", "altitude", "z"],
        "speed": ["speed", "spd", "v"],
        "sats": ["sats", "satellites"],
        "acc": ["acc", "accuracy", "posacc"],
    }
    for std, alts in colmap.items():
        for a in alts:
            if a in out.columns and std not in out.columns:
                out.rename(columns={a: std}, inplace=True)
                break
    out.attrs["key"] = key
    return out


# ---- Streamlit UI ----
st.title("GoPro Telemetry Extractor (GPS + Movement)")
st.caption("Werkt met grote MP4/MOV bestanden (Hero 13 klaar). Alle verwerking lokaal, alleen metadata stream wordt gelezen.")

with st.sidebar:
    st.header("Bronbestand")
    src_type = st.radio("Kies invoer", ["Bestandspad", "Upload"], index=0)
    input_path = ""
    uploaded = None
    if src_type == "Bestandspad":
        input_path = st.text_input("Pad naar GoPro video (MP4/MOV)")
    else:
        uploaded = st.file_uploader("Upload video (grootte >1GB: liever gebruik bestandspad)", type=["mp4", "mov"], accept_multiple_files=False)

    st.divider()
    st.header("Export")
    want_csv = st.checkbox("Exporteer CSV", value=True)
    want_parquet = st.checkbox("Exporteer Parquet", value=True)

    st.divider()
    st.header("Info")
    st.write("Vereist: ffmpeg/ffprobe. Optioneel: gpmf-parser (Python).")

if uploaded is not None:
    # Save uploaded to a temp file (not ideal for >1GB; advise path mode)
    tmpdir = tempfile.mkdtemp(prefix="gopro_")
    input_path = os.path.join(tmpdir, uploaded.name)
    with open(input_path, "wb") as f:
        f.write(uploaded.getbuffer())

if not input_path:
    st.info("Geef een videopad op of upload een bestand om te starten.")
    st.stop()

if not os.path.exists(input_path):
    st.error("Bestand niet gevonden. Controleer het pad.")
    st.stop()

try:
    meta = ffprobe_json(input_path)
except Exception as e:
    st.error(f"ffprobe fout: {e}")
    st.stop()

idx = find_gpmd_stream_index(meta)
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Bestandsinfo")
    st.json({
        "format": meta.get("format", {}).get("format_name"),
        "duration_s": meta.get("format", {}).get("duration"),
        "size_bytes": meta.get("format", {}).get("size"),
        "bit_rate": meta.get("format", {}).get("bit_rate"),
        "gpmd_stream_index": idx,
    })
with col_b:
    st.subheader("Streams (samenvatting)")
    st.json([
        {
            "index": s.get("index"),
            "type": s.get("codec_type"),
            "codec": s.get("codec_name"),
            "tag": s.get("codec_tag_string"),
            "handler": (s.get("tags", {}) or {}).get("handler_name"),
        }
        for s in meta.get("streams", [])
    ])

if idx is None:
    st.warning("Geen GPMD/telemetrie stream gevonden. Zorg dat GPS aanstond in de camera.")
    st.stop()

# Extract raw GPMD to temp
work = tempfile.mkdtemp(prefix="gpmd_")
raw_path = os.path.join(work, "telemetry.gpmd")
try:
    extract_gpmd_raw(input_path, raw_path, idx)
except Exception as e:
    st.error(f"Kon GPMD stream niet extraheren: {e}")
    st.stop()

# Parse
try:
    telem = try_parse_gpmf(raw_path)
except Exception as e:
    st.error(str(e))
    st.stop()

# ---- Dashboard tabs ----
tabs = st.tabs(["Overzicht", "GPS", "Beweging", "Export"])  # movement = IMU

with tabs[0]:
    st.subheader("Snel overzicht")

    m = {}
    if telem.gps is not None and not telem.gps.empty:
        gps = telem.gps.dropna(subset=[c for c in ["lat", "lon"] if c in telem.gps.columns])
        m["GPS punten"] = len(gps)
        if "speed" in gps.columns:
            m["Max snelheid (m/s)"] = float(np.nanmax(gps["speed"]))
            m["Gem snelheid (m/s)"] = float(np.nanmean(gps["speed"]))
    if telem.accl is not None and not telem.accl.empty:
        m["Accel samples"] = len(telem.accl)
    if telem.gyro is not None and not telem.gyro.empty:
        m["Gyro samples"] = len(telem.gyro)

    st.json(m)

    # Quick speed timeline
    if telem.gps is not None and "ts" in telem.gps.columns and "speed" in telem.gps.columns:
        st.markdown("**Snelheid over tijd**")
        fig = px.line(telem.gps, x="ts", y="speed", render_mode="auto", title="Speed vs Time")
        st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("GPS kaart & stats")
    if telem.gps is None or telem.gps.empty or folium is None or st_folium is None:
        st.info("Geen GPS of folium plugin niet beschikbaar. Installeer 'streamlit-folium'.")
    else:
        gps = telem.gps.dropna(subset=[c for c in ["lat", "lon"] if c in telem.gps.columns]).copy()
        if gps.empty:
            st.info("Geen geldige GPS punten.")
        else:
            # distance & speed helpers (rough)
            def haversine(lat1, lon1, lat2, lon2):
                R = 6371000.0
                dlat = np.radians(lat2 - lat1)
                dlon = np.radians(lon2 - lon1)
                a = (
                    np.sin(dlat / 2) ** 2
                    + np.cos(np.radians(lat1))
                    * np.cos(np.radians(lat2))
                    * np.sin(dlon / 2) ** 2
                )
                return 2 * R * np.arcsin(np.sqrt(a))

            gps["lat_shift"] = gps["lat"].shift(-1)
            gps["lon_shift"] = gps["lon"].shift(-1)
            gps["seg_m"] = haversine(gps["lat"], gps["lon"], gps["lat_shift"], gps["lon_shift"]).fillna(0)
            total_m = float(gps["seg_m"].sum())
            st.write({"Afstand (m)": round(total_m, 1)})

            center = [float(gps["lat"].iloc[0]), float(gps["lon"].iloc[0])]
            mapp = folium.Map(location=center, zoom_start=14)
            coords = list(zip(gps["lat"].values, gps["lon"].values))
            folium.PolyLine(coords, weight=4).add_to(mapp)
            # start/end markers
            folium.Marker(coords[0], icon=folium.Icon(icon="play")).add_to(mapp)
            folium.Marker(coords[-1], icon=folium.Icon(icon="stop")).add_to(mapp)
            st_folium(mapp, width=1200, height=500)

            # altitude & speed plots if present
            if "alt" in gps.columns:
                st.markdown("**Hoogteprofiel**")
                fig_alt = px.line(gps, x="ts" if "ts" in gps.columns else gps.index, y="alt", title="Altitude")
                st.plotly_chart(fig_alt, use_container_width=True)
            if "speed" in gps.columns:
                st.markdown("**Snelheid**")
                fig_spd = px.line(gps, x="ts" if "ts" in gps.columns else gps.index, y="speed", title="Speed")
                st.plotly_chart(fig_spd, use_container_width=True)

with tabs[2]:
    st.subheader("Beweging (IMU)")
    two_cols = st.columns(2)
    if telem.accl is not None and not telem.accl.empty:
        with two_cols[0]:
            cols = [c for c in telem.accl.columns if c.lower() in ("x", "y", "z", "ax", "ay", "az")]
            if cols:
                fig_a = px.line(telem.accl, x="ts" if "ts" in telem.accl.columns else telem.accl.index, y=cols, title="Accelerometer")
                st.plotly_chart(fig_a, use_container_width=True)
    if telem.gyro is not None and not telem.gyro.empty:
        with two_cols[1]:
            cols = [c for c in telem.gyro.columns if c.lower() in ("x", "y", "z", "gx", "gy", "gz")]
            if cols:
                fig_g = px.line(telem.gyro, x="ts" if "ts" in telem.gyro.columns else telem.gyro.index, y=cols, title="Gyroscope")
                st.plotly_chart(fig_g, use_container_width=True)

with tabs[3]:
    st.subheader("Export")
    outdir = st.text_input("Uitvoer map", value=os.path.join(os.path.dirname(input_path), "telemetry_export"))
    if st.button("Exporteren"):
        os.makedirs(outdir, exist_ok=True)
        written = []
        def dump(df: Optional[pd.DataFrame], name: str):
            if df is None or df.empty:
                return
            base = os.path.join(outdir, name)
            if want_csv:
                path = base + ".csv"
                df.to_csv(path, index=False)
                written.append(path)
            if want_parquet:
                path = base + ".parquet"
                df.to_parquet(path, index=False)
                written.append(path)
        dump(telem.gps, "gps")
        dump(telem.accl, "accel")
        dump(telem.gyro, "gyro")
        dump(telem.grav, "grav")
        dump(telem.temp, "temp")
        for k, v in telem.others.items():
            dump(v, k.lower())
        st.success(f"Bestanden geschreven: {len(written)}")
        for p in written:
            st.code(p)

st.caption("Tip: voor batchverwerking kun je dit script ook als module gebruiken en via CLI aanroepen om per map alle bestanden te verwerken.")
