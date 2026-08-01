"""
Data-thon Dashboard — Excel in, insights out.
Run:  streamlit run streamlit_app.py

Everything is computed — pandas/numpy for the dashboard, and deterministic
statistical/rule engines for the analysis layers (Insights, Action Plan,
Briefs, Competency Report, What-If, Archetypes & Risk).

No LLM or generative AI is used anywhere, as the Datathon rules require.
"""
import hashlib
import io
import json
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# analysis layers live one directory up
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import adapter
import insights as L_insights
import playbook as L_playbook
import brief as L_brief
import competency as L_competency
import models_ml as L_models
import charts as L_charts
import secondary as L_secondary          # <-- ADDED: cross-dataset layer
import insights_cross as L_cross         # <-- ADDED: cross-dataset insights
import gka as L_gka                      # <-- ADDED: GKA programme impact
from stats_tests import proportion_test
import verbalize as L_verbalize

st.set_page_config(page_title="Vishvena AI — Data Studio", layout="wide")

# ---------------- LIGHT THEME SKIN (matches pages/Missions design) ---------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stApp * { font-family: 'Inter', 'Segoe UI', sans-serif; }
span[data-testid="stIconMaterial"],
[class*="material-symbols"], [class*="material-icons"],
[data-testid="stExpanderToggleIcon"] {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined',
                 'Material Icons' !important; }
.stApp { background: #f2f4f8; }
h1, h2, h3 { color: #101828 !important; font-weight: 700 !important;
    letter-spacing: -0.3px; }
/* ---- soft dark sidebar (Linear/Vercel style) ---- */
[data-testid="stSidebar"] { background: #111827;
    border-right: 1px solid #1F2937; }
[data-testid="stSidebar"] *, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
    color: #E5E7EB !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] small, [data-testid="stSidebar"] .stCaption {
    color: #9CA3AF !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #F9FAFB !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] [data-testid="stNumberInputContainer"],
[data-testid="stSidebar"] [data-baseweb="input"] {
    background: #1F2937 !important; border-color: #374151 !important;
    color: #E5E7EB !important; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: #1F2937 !important;
    border: 1.5px dashed #4B5563 !important; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
    color: #9CA3AF !important; }
[data-testid="stSidebar"] button {
    background: #10B981 !important; color: #06281d !important;
    border: 0 !important; font-weight: 600 !important; }
[data-testid="stSidebar"] button:hover { background: #0ea371 !important; }
[data-testid="stSidebar"] [data-testid="stExpander"],
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: #1F2937 !important; border: 1px solid #374151 !important; }
[data-testid="stSidebar"] [data-testid="stAlert"] * {
    color: #E5E7EB !important; }
[data-testid="stSidebar"] hr { border-color: #374151; }
[data-testid="stSidebar"] a, [data-testid="stSidebarNav"] a span {
    color: #E5E7EB !important; }
[data-testid="stSidebar"] a:hover { color: #10B981 !important; }
[data-testid="stSidebarNav"] { background: #111827; padding-top: 8px;
    display: none !important; }  /* Missions page hidden from nav for now — remove this line to restore */
[data-testid="stSidebar"] [role="radiogroup"] label,
[data-testid="stSidebar"] [data-baseweb="checkbox"] {
    background: transparent !important; }
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff; border: 1px solid #e6e9ef !important;
    border-radius: 14px;
    box-shadow: 0 1px 3px rgba(16,24,40,.05); }
div[data-testid="stMetric"] { background: #ffffff;
    border: 1px solid #e6e9ef; border-top: 3px solid #0F6E56;
    border-radius: 12px; padding: 10px 14px; min-width: 0;
    box-shadow: 0 1px 3px rgba(16,24,40,.05); }
div[data-testid="stMetric"] label p { color: #67707f !important;
    font-size: 11px !important; text-transform: none;
    letter-spacing: .2px; font-weight: 600 !important;
    white-space: normal !important; line-height: 1.3; }
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] * {
    color: #0F6E56 !important; font-weight: 700 !important;
    font-size: 19px !important; line-height: 1.25 !important;
    white-space: normal !important; overflow: visible !important;
    text-overflow: clip !important; overflow-wrap: anywhere; }
div[data-testid="stMetricDelta"],
div[data-testid="stMetricDelta"] * { font-size: 13px !important; }
[data-baseweb="tab-list"] { background: #ffffff; border-radius: 12px;
    border: 1px solid #e6e9ef; padding: 4px 8px; }
[data-baseweb="tab-highlight"] { background: #0F6E56; }
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
    background: #ffffff; border: 1px solid #e6e9ef; border-radius: 12px;
    padding: 4px; }
[data-testid="stExpander"] { background: #ffffff;
    border: 1px solid #e6e9ef !important; border-radius: 12px; }
.stButton button, [data-testid="stFileUploader"] button {
    background: #0F6E56 !important; color: #ffffff !important;
    border: 0 !important; border-radius: 10px !important;
    font-weight: 600 !important; }
.stButton button:hover { background: #0a5340 !important; }
[data-testid="stFileUploaderDropzone"] { background: #ffffff;
    border: 1.5px dashed #c6cdd8 !important; border-radius: 12px; }
[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stHeader"] { background: #f2f4f8; }
[data-testid="stProgress"] > div > div {
    background: #dfe5ee !important; border-radius: 999px;
    height: 12px !important; }
[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, #0F6E56, #10B981) !important;
    border-radius: 999px; }
.stCode, pre, code { background: #f0f4f2 !important;
    color: #0a5340 !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ----------------------------- Data handling --------------------------------
HIERARCHY_CANDIDATES = ["Division", "District", "Block", "Cluster",
                        "GP Name", "GP"]

DATA_EXT = (".xlsx", ".xls", ".csv", ".parquet")
UPLOAD_EXT = [e.lstrip(".") for e in DATA_EXT] + ["zip"]


# Parsing .xlsx is by far the slowest thing this app does — openpyxl is pure
# Python and takes ~8 minutes on a 1.6M-row dataset, which alone would blow the
# competition's 3-minute reproducibility budget on a judge's fresh machine.
# calamine is a Rust reader exposed through pandas; measured on the real dataset
# it is 7.4x faster and returns byte-identical values AND identical dtypes.
# It is listed in requirements, but the chain degrades to openpyxl if a grader's
# environment lacks it, so the entry point can never fail for want of a wheel.
EXCEL_ENGINES = ("calamine", None)          # None = pandas' default (openpyxl)
ENGINE_USED = {}


def _read_excel_fast(name, src):
    last = None
    for eng in EXCEL_ENGINES:
        try:
            if hasattr(src, "seek"):
                src.seek(0)                  # a failed attempt consumes a buffer
            _kw = {"engine": eng} if eng else {}
            try:
                df = pd.read_excel(src, sheet_name="Assessment Data", **_kw)
            except Exception:
                if hasattr(src, "seek"):
                    src.seek(0)
                df = pd.read_excel(src, **_kw)
            ENGINE_USED[name] = eng or "openpyxl"
            return df
        except Exception as e:               # missing wheel, or an odd workbook
            last = e
    raise last


def _read_one(name, src):
    """Read a single table. `src` is a path or a file-like object."""
    low = name.lower()
    if low.endswith(".csv"):
        df = pd.read_csv(src)
    elif low.endswith(".parquet"):
        df = pd.read_parquet(src)
    else:
        df = _read_excel_fast(name, src)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _expand(name, src):
    """
    Yield (member_name, readable) pairs. A .zip expands to the data files
    inside it; anything else yields itself.
    """
    if not name.lower().endswith(".zip"):
        if hasattr(src, "seek"):
            src.seek(0)
        yield name, src
        return
    # An uploaded file is a ONE-SHOT stream. This function is called more than
    # once on the same handle — load_sources() reads the data, then
    # extract_embedded_qmaps() comes back for the Competency Mapping sheets —
    # and without rewinding, the second pass reads b"" and zipfile reports
    # "File is not a zip file" for a perfectly good archive.
    if hasattr(src, "seek"):
        src.seek(0)
    raw = open(src, "rb").read() if isinstance(src, str) else src.read()
    if not raw:
        raise ValueError(
            f"'{name}' read back empty. The upload buffer was already "
            f"consumed and could not be rewound.")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for info in sorted(z.infolist(), key=lambda i: i.filename):
            member = info.filename
            base = os.path.basename(member)
            # skip directories, macOS resource forks and Excel lock files —
            # and the app's OWN combined-parquet cache, which it writes next to
            # the source data. _scan_local already excludes those; without the
            # same rule here, zipping a folder the app has previously read
            # bundles the cache alongside the parts it was built from and the
            # whole dataset is silently loaded twice.
            if (info.is_dir() or not base or base.startswith((".", "~$"))
                    or "__MACOSX" in member
                    or base.lower().endswith(".cache.parquet")
                    or not base.lower().endswith(DATA_EXT)):
                continue
            yield f"{os.path.basename(name)}:{member}", io.BytesIO(z.read(info))


_YEAR_RE = re.compile(r"(20\d\d)\s*[-_]\s*(\d\d)")
_GRADE_RE = re.compile(r"[Gg]rade[\s_-]*(\d)")


def _stamp_year_grade(part, member):
    """Real GP-contest files carry Year in the folder/filename and Grade in
    the filename (e.g. 2023-24/GPContest_Grade_5_2023-24.xlsx). Stamp both
    as columns if the file doesn't already have them; fall back to the
    Unique Identifier column (Grade_5_23_24_...)."""
    cols = {str(c).strip().lower() for c in part.columns}
    need_year = "year" not in cols
    need_grade = "grade" not in cols
    if not (need_year or need_grade):
        return part
    ym = _YEAR_RE.search(member)
    gm = _GRADE_RE.search(os.path.basename(member)) or _GRADE_RE.search(member)
    year = f"{ym.group(1)}-{ym.group(2)}" if ym else None
    grade = int(gm.group(1)) if gm else None
    if (year is None or grade is None) and "Unique Identifier" in part.columns:
        u = str(part["Unique Identifier"].iloc[0])
        um = re.search(r"Grade[\s_-]*(\d)[\s_-]*(\d\d)[\s_-]*(\d\d)", u)
        if um:
            grade = grade if grade is not None else int(um.group(1))
            year = year or f"20{um.group(2)}-{um.group(3)}"
    if need_year and year:
        part["Year"] = year
    if need_grade and grade is not None:
        part["Grade"] = grade
    return part


# Karnataka canon: raw GP-contest spellings -> canonical district names, and
# district -> administrative division (the data has no Division column).
KA_DISTRICT_CANON = {
    "bagalkot": "Bagalkote", "ballari": "Ballari", "belagavi": "Belagavi",
    "belagavi chikkodi": "Belagavi", "bengaluru rural": "Bengaluru Rural",
    "bengaluru urban": "Bengaluru Urban", "bidar": "Bidar",
    "chamarajanagara": "Chamarajanagara", "chamarajanagar": "Chamarajanagara",
    "chikkaballapura": "Chikkaballapura", "chikkaballapur": "Chikkaballapura",
    "chikkamagaluru": "Chikkamagaluru", "chitradurga": "Chitradurga",
    "dakshina kannada": "Dakshina Kannada", "davanagere": "Davanagere",
    "dharwad": "Dharwad", "gadag": "Gadag", "hassan": "Hassan",
    "haveri": "Haveri", "kalaburgi": "Kalaburagi", "kalaburagi": "Kalaburagi",
    "kodagu": "Kodagu", "kolar": "Kolar", "koppal": "Koppal",
    "mandya": "Mandya", "mysuru": "Mysuru", "raichur": "Raichur",
    "ramanagara": "Ramanagara", "shivamogga": "Shivamogga",
    "tumakuru": "Tumakuru", "tumakuru madhugiri": "Tumakuru",
    "udupi": "Udupi", "uttara kannada": "Uttara Kannada",
    "uttara kannada sirsi": "Uttara Kannada", "vijayanagar": "Vijayanagara",
    "vijayanagara": "Vijayanagara", "vijayapura": "Vijayapura",
    "yadagiri": "Yadgir", "yadgir": "Yadgir",
}
KA_DIVISION = {
    "Belagavi": "Belagavi Division", "Bagalkote": "Belagavi Division",
    "Vijayapura": "Belagavi Division", "Dharwad": "Belagavi Division",
    "Gadag": "Belagavi Division", "Haveri": "Belagavi Division",
    "Uttara Kannada": "Belagavi Division",
    "Bengaluru Urban": "Bengaluru Division",
    "Bengaluru Rural": "Bengaluru Division",
    "Ramanagara": "Bengaluru Division", "Kolar": "Bengaluru Division",
    "Chikkaballapura": "Bengaluru Division", "Tumakuru": "Bengaluru Division",
    "Chitradurga": "Bengaluru Division", "Davanagere": "Bengaluru Division",
    "Shivamogga": "Bengaluru Division",
    "Kalaburagi": "Kalaburagi Division", "Bidar": "Kalaburagi Division",
    "Raichur": "Kalaburagi Division", "Yadgir": "Kalaburagi Division",
    "Koppal": "Kalaburagi Division", "Ballari": "Kalaburagi Division",
    "Vijayanagara": "Kalaburagi Division",
    "Mysuru": "Mysuru Division", "Mandya": "Mysuru Division",
    "Hassan": "Mysuru Division", "Chikkamagaluru": "Mysuru Division",
    "Kodagu": "Mysuru Division", "Chamarajanagara": "Mysuru Division",
    "Dakshina Kannada": "Mysuru Division", "Udupi": "Mysuru Division",
}


def competency_coverage_note(rqmaps):
    """One honest line about competencies NOT tested in every paper,
    generated from the embedded maps themselves."""
    if not rqmaps:
        return None
    papers = sorted(rqmaps.keys())
    universe = sorted({c for m in rqmaps.values() for c in m.values()})
    bits = []
    for comp in universe:
        have = {k for k, m in rqmaps.items() if comp in set(m.values())}
        if len(have) == len(papers):
            continue
        yrs_all = sorted({y for y, g in papers})
        yrs_missing = [y for y in yrs_all
                       if not any((y, g) in have for g in (4, 5, 6))]
        if yrs_missing:
            bits.append(f"**{comp}** — not tested in "
                        f"{', '.join(yrs_missing)}")
        else:
            bits.append(f"**{comp}** — in {len(have)} of "
                        f"{len(papers)} papers")
    if not bits:
        return None
    return ("📋 **Coverage:** the paper changes yearly, so some "
            "competencies have gaps — " + " · ".join(bits)
            + ". Averages use only the papers where a competency was "
              "actually tested; blank cells mean it never appeared there.")


def comp_score_frame(frame, item_cols, rqmaps, flat_qmap=None,
                     year_col="Year", grade_col="Grade"):
    """Per-student competency %-correct columns. With per-paper maps
    (rqmaps={(year,grade):{Q:comp}}) each (year,grade) chunk uses ITS OWN
    paper; with a flat map every row uses the same one. Returns a frame of
    competency columns aligned to `frame`."""
    comps = sorted({c for m in (rqmaps or {}).values() for c in m.values()}
                   | set((flat_qmap or {}).values()))
    out = pd.DataFrame(index=frame.index, columns=comps, dtype=float)
    if rqmaps and year_col in frame.columns and grade_col in frame.columns:
        _yg = frame.groupby([frame[year_col].astype(str),
                             pd.to_numeric(frame[grade_col],
                                           errors="coerce")]).groups
        for (y, g), idx in _yg.items():
            m = rqmaps.get((str(y), int(g)) if not pd.isna(g) else None)
            if not m:
                continue
            for comp in comps:
                cc = [q for q, c in m.items()
                      if c == comp and q in item_cols]
                if cc:
                    out.loc[idx, comp] = frame.loc[idx, cc].mean(axis=1) * 100
    elif flat_qmap:
        for comp in comps:
            cc = [q for q, c in flat_qmap.items()
                  if c == comp and q in item_cols]
            if cc:
                out[comp] = frame[cc].mean(axis=1) * 100
    return out.dropna(axis=1, how="all")


@st.cache_data(show_spinner=False)
def extract_embedded_qmaps(_sources, sig):
    """Real GP-contest workbooks carry a 'Competency Mapping' sheet — and the
    paper CHANGES every year and grade. Returns {(year, grade): {Q: comp}}
    plus {(year, grade): {Q: question_name}}. Empty dicts when absent."""
    qmaps, qnames = {}, {}
    for name, src2 in _sources:
        for member, handle in _expand(name, src2):
            if not member.lower().endswith((".xlsx", ".xls")):
                continue
            ym = _YEAR_RE.search(member)
            gm = _GRADE_RE.search(member)
            if not (ym and gm):
                continue
            key = (f"{ym.group(1)}-{ym.group(2)}", int(gm.group(1)))
            try:
                if hasattr(handle, "seek"):
                    handle.seek(0)
                cm = pd.read_excel(handle, sheet_name="Competency Mapping",
                                   header=None)
            except Exception:
                continue
            cm = cm.dropna(how="all").dropna(axis=1, how="all")
            _hdr = cm[cm.apply(lambda r: r.astype(str)
                               .str.contains("Competency", case=False)
                               .any(), axis=1)]
            if _hdr.empty:
                continue
            h = _hdr.index[0]
            cm.columns = cm.loc[h].astype(str).str.strip()
            cm = cm.loc[h + 1:]
            qc = next((c for c in cm.columns
                       if str(c).lower().startswith("question")
                       and "name" not in str(c).lower()), None)
            cc = next((c for c in cm.columns
                       if "compet" in str(c).lower()), None)
            nc = next((c for c in cm.columns
                       if "name" in str(c).lower()), None)
            if not (qc and cc):
                continue
            q = cm[qc].astype(str).str.strip()
            comp = (cm[cc].astype(str).str.strip().str.lower()
                    .replace({"mensuration": "measurement"}))
            qmaps[key] = dict(zip(q, comp))
            if nc:
                qnames[key] = dict(zip(q, cm[nc].astype(str).str.strip()))
    return qmaps, qnames


@st.cache_data(show_spinner=False)
def load_many(_sources, sig):
    """
    Read one or many files into a single frame.

    Excel cannot hold more than 1,048,576 rows, so a full state dataset arrives
    SPLIT — several .xlsx parts, or a .zip of them. Those parts are one logical
    dataset and every layer must see the whole thing, so they are read
    individually and concatenated before anything else happens.

    `_sources` is a list of (name, path-or-buffer); the leading underscore keeps
    Streamlit from trying to hash file handles. `sig` is the real cache key.

    Returns (combined_frame, manifest) where manifest has one row per part.
    """
    frames, manifest = [], []
    _n_src = max(len(_sources), 1)
    _pb = st.progress(0.0, text="📖 Reading your dataset…")

    for _si, (name, src) in enumerate(_sources):
        _pb.progress(min(max(_si / _n_src, 0.12), 0.95),
                     text=f"📖 Reading {name} — part {_si + 1} of {_n_src}. "
                          "Large Excel files take a while; Parquet/CSV are "
                          "much faster.")
        for member, handle in _expand(name, src):
            try:
                part = _read_one(member, handle)
            except Exception as e:                    # one bad part must not
                manifest.append({"file": member, "rows": 0,               # kill the rest
                                 "columns": 0, "status": f"❌ {type(e).__name__}: {e}"})
                continue
            part = _stamp_year_grade(part, member)
            frames.append(part)
            manifest.append({"file": member, "rows": len(part),
                             "columns": part.shape[1],
                             "read by": ENGINE_USED.get(member, "—"),
                             "status": "✅ loaded",
                             # Identical rows occur naturally without a student
                             # ID (same block, same answers). Counting them per
                             # part lets us tell that apart from real overlap.
                             "_self_dupes": int(part.duplicated().sum())})

    if not frames:
        _pb.empty()
        return None, manifest

    _pb.progress(0.97, text="🧩 Combining parts into one dataset…")
    if len(frames) == 1:
        df = frames[0]
    else:
        # Not every file in a folder or zip is a PART of the dataset. A zip of
        # assessment data often also carries a lookup table — a question map, a
        # district key — and concatenating a 20-row lookup onto a million rows
        # of results silently invents columns that are almost entirely null,
        # which then get auto-detected as real dimensions further down.
        # A genuine part shares most of its columns with the main table.
        ok_rows = [r for r in manifest if r["status"] == "✅ loaded"]
        main = max(frames, key=len)
        main_cols = set(main.columns)
        keep, kept_rows = [], []
        for m, f in zip(ok_rows, frames):
            share = len(set(f.columns) & main_cols) / max(len(f.columns), 1)
            if f is main or share >= 0.5:
                keep.append(f)
                kept_rows.append(m)
            else:
                m["status"] = (f"↩️ not part of the dataset — shares only "
                               f"{share:.0%} of its columns; treated as a "
                               f"lookup table and excluded")
        frames = keep

        # Align on the union of columns so a part missing an optional column
        # still contributes its rows (the gap becomes NaN, not a dropped file).
        all_cols = list(dict.fromkeys(c for f in frames for c in f.columns))
        for m, f in zip(kept_rows, frames):
            missing = [c for c in all_cols if c not in f.columns]
            if missing:
                m["status"] = f"⚠️ loaded, missing {len(missing)}: {', '.join(missing[:4])}"
        df = (frames[0] if len(frames) == 1 else
              pd.concat([f.reindex(columns=all_cols) for f in frames],
                        ignore_index=True, sort=False))

    _pb.progress(1.0, text="✅ Dataset ready")
    _pb.empty()
    return df, manifest


def combined_cache_path(paths):
    """Parquet sidecar for a specific SET of on-disk files, keyed by content."""
    key = hashlib.md5("|".join(
        f"{os.path.abspath(p)}:{os.path.getmtime(p)}:{os.path.getsize(p)}"
        for p in sorted(paths)).encode()).hexdigest()[:12]
    return os.path.join(os.path.dirname(sorted(paths)[0]), f"_combined_{key}.cache.parquet")


def load_sources(sources, disk_paths):
    """
    load_many + a Parquet sidecar when every source is a real file on disk.

    Re-parsing several 100 MB spreadsheets on every restart is the single
    slowest thing in the app; the sidecar turns that into about a second.
    """
    sig = tuple(n for n, _ in sources)
    if disk_paths:
        cache = combined_cache_path(disk_paths)
        if os.path.exists(cache):
            try:
                return pd.read_parquet(cache), [
                    {"file": os.path.basename(p), "rows": "—", "columns": "—",
                     "status": "⚡ from Parquet cache"} for p in sorted(disk_paths)]
            except Exception:
                pass                       # a corrupt cache just means re-read

    df, manifest = load_many(sources, sig)

    if df is not None and disk_paths:
        try:
            df.to_parquet(combined_cache_path(disk_paths), index=False)
        except Exception:
            pass          # caching is an optimisation, never a hard requirement
    return df, manifest

def classify_columns(df: pd.DataFrame) -> dict:
    """Objective (numeric) vs subjective (text) vs categorical vs id."""
    out = {"objective": [], "subjective": [], "categorical": [], "id": []}
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(df[col]):
            if nunique > 15:
                out["objective"].append(col)
            else:
                out["categorical"].append(col)
        else:
            avg_len = df[col].astype(str).str.len().mean()
            if nunique >= len(df) * 0.9:
                out["id"].append(col)
            elif avg_len > 25 and nunique > max(50, len(df) * 0.5):
                # long AND mostly-unique -> free text (subjective answers);
                # long but repeating -> names/labels (e.g. "Division East Halli 003")
                out["subjective"].append(col)
            else:
                out["categorical"].append(col)
    return out

def grade_on_curve(series: pd.Series) -> pd.Series:
    """Objective -> subjective: bucket by percentile (grading on a curve)."""
    labels = ["Poor", "Below Average", "Average", "Above Average", "Excellent"]
    pct = series.rank(pct=True)
    return pd.cut(pct, bins=[0, .2, .4, .6, .8, 1.0], labels=labels, include_lowest=True)


# ----------------------- Format adapter (schema may change!) -----------------
# The real data format is not final. This layer inspects whatever arrives and
# adapts it into the long shape the dashboard expects — nothing downstream
# needs to change when the schema shifts. Everything stays user-overridable
# in the Column mapping expander.

_WIDE_EXCLUDE = ("total", "overall", "percent", "max", "aggregate", "grand")

def _find_wide_competencies(df: pd.DataFrame) -> list:
    """Wide format: one numeric column per competency, e.g. Numeracy_score."""
    out = []
    for c in df.columns:
        cl = str(c).lower()
        if cl.endswith("score") and not any(k in cl for k in _WIDE_EXCLUDE):
            if pd.api.types.is_numeric_dtype(df[c]):
                out.append(c)
    return out

def _find_item_columns(df: pd.DataFrame) -> list:
    """Binary item responses: Q1..Qn with values in {0,1}."""
    items = []
    for c in df.columns:
        if re.fullmatch(r"[Qq]\s*_?\d+", str(c).strip()):
            v = pd.to_numeric(df[c], errors="coerce").dropna().unique()
            if len(v) and set(v).issubset({0, 1}):
                items.append(c)
    return sorted(items, key=lambda x: int(re.sub(r"\D", "", x)))

def normalize_dataset(df: pd.DataFrame):
    """Adapt the incoming file into the long shape the dashboard expects.

    Handles (auto-detected, in any combination):
    - wide competency columns (X_score, Y_score...) -> melted to Competency/Score,
      normalized to % of each competency's max so different maxima are comparable
    - binary item columns (Q1..Qn) -> kept aside for the Item Analysis tab
    - total_score/total_questions -> computed 'Overall (%)', sanity-checked
      against any existing percentage column (which may be miscomputed)

    Returns (long_df, info, items_df).
    """
    info = {"reshaped": False, "notes": []}
    comp_wide = _find_wide_competencies(df)
    # A genuine wide format has SEVERAL competency columns. A single "Score"
    # column is just the total — melting it invents a fake one-value competency.
    if len(comp_wide) < 2:
        comp_wide = []
    items = _find_item_columns(df)
    items_df = df.drop(columns=comp_wide, errors="ignore").copy() if items else None

    # --- Overall percent: prefer computing it ourselves --------------------
    tot = next((c for c in df.columns
                if str(c).lower() in ("total_score", "total", "overall_score")), None)
    totq = next((c for c in df.columns
                 if "question" in str(c).lower() and "total" in str(c).lower()), None)
    pct = next((c for c in df.columns if "percent" in str(c).lower()), None)
    if tot and totq:
        computed = (pd.to_numeric(df[tot], errors="coerce")
                    / pd.to_numeric(df[totq], errors="coerce").replace(0, np.nan)
                    * 100).round(2)
        if pct is not None:
            drift = (computed - pd.to_numeric(df[pct], errors="coerce")).abs().mean()
            if pd.notna(drift) and drift > 1:
                info["notes"].append(
                    f"⚠️ '{pct}' disagrees with {tot}/{totq} (avg gap {drift:.1f} pts) — "
                    "using a recomputed 'Overall (%)' instead.")
        df = df.assign(**{"Overall (%)": computed})
    elif pct is not None:
        df = df.assign(**{"Overall (%)": pd.to_numeric(df[pct], errors="coerce")})

    # --- Wide -> long competencies -----------------------------------------
    if comp_wide:
        already_long = any(any(k in str(c).lower()
                               for k in ("competen", "subject", "skill", "domain"))
                           for c in df.columns)
        if not already_long:
            id_vars = [c for c in df.columns if c not in comp_wide and c not in items]
            melted = df.melt(id_vars=id_vars, value_vars=comp_wide,
                             var_name="Competency", value_name="_raw")
            melted["Competency"] = (melted["Competency"].astype(str)
                                    .str.replace(r"(?i)_?score$", "", regex=True)
                                    .str.replace("_", " ").str.strip().str.title())
            maxes = (melted.groupby("Competency")["_raw"]
                     .transform("max").replace(0, np.nan))
            melted["Score"] = (pd.to_numeric(melted["_raw"], errors="coerce")
                               / maxes * 100).round(1)
            df = melted.drop(columns="_raw")
            info["reshaped"] = True
            info["notes"].append(
                f"🔁 Wide assessment format detected — reshaped into "
                f"{len(df):,} competency observations "
                f"({len(comp_wide)} competencies, scores normalized to % of max).")
    if items:
        info["notes"].append(
            f"🧩 {len(items)} binary question items detected (Q1–Q{len(items)}) — "
            "see the Item Analysis tab.")
        # No competency column, but 20 question items? Then the ITEMS are the
        # competency dimension (the datathon's "twenty competency-based
        # questions"). Melt them so every competency/gender tab has something
        # real to work with instead of nothing.
        has_comp = any(any(k in str(c).lower()
                           for k in ("competen", "subject", "skill", "domain"))
                       for c in df.columns)
        # Melting materialises rows x items. At 2M students x 20 questions that
        # is 40M rows and pandas raises MemoryError. Above the limit we keep the
        # wide frame: the dashboard tabs fall back to the total score, while the
        # analysis layers still get full per-question detail because the adapter
        # unpivots inside DuckDB instead.
        MELT_ROW_LIMIT = 5_000_000
        projected = len(df) * len(items)
        if not has_comp and projected > MELT_ROW_LIMIT:
            info["notes"].append(
                f"⚠️ {len(df):,} rows × {len(items)} questions = {projected:,} "
                f"question-level rows — too large to expand in memory. Dashboard "
                f"tabs will use the total score column; the analysis layers still "
                f"use every question (they aggregate in DuckDB).")
            has_comp = True          # skip the melt
        if not has_comp:
            id_vars = [c for c in df.columns if c not in items]
            df = df.melt(id_vars=id_vars, value_vars=items,
                         var_name="Competency", value_name="_correct")
            df["_correct"] = pd.to_numeric(df["_correct"], errors="coerce")
            df = df[df["_correct"].notna()]
            df["Item Score (%)"] = df["_correct"] * 100.0
            df = df.drop(columns="_correct")
            info["reshaped"] = True
            info["notes"].append(
                f"🔁 Each question treated as a competency → {len(df):,} "
                f"question-level observations. A correct answer scores 100, "
                f"a wrong answer 0.")
    return df, info, items_df


# ------------------------------- Sidebar ------------------------------------
_SHOW_MISSIONS_LINK = False   # hidden for now — flip to True to bring the link back
if _SHOW_MISSIONS_LINK:
    try:
        st.sidebar.page_link("pages/Missions.py",
                             label="**🎯 National Missions →**")
        st.sidebar.markdown("---")
    except Exception:
        pass
st.sidebar.title("📂 Data")
# ---- optional question map: maps Q items to named competencies -----------
@st.cache_data(show_spinner=False)
def _load_qmap(_file_bytes):
    import io as _io
    qm = pd.read_csv(_io.BytesIO(_file_bytes))
    qm.columns = [str(c).strip().lower() for c in qm.columns]
    qcol = next((c for c in qm.columns if c in ("question", "item", "q")), None)
    ccol = next((c for c in qm.columns if "compet" in c or c in ("skill", "domain")), None)
    if not (qcol and ccol):
        return None, None, None, "needs 'question' and 'competency' columns"
    qm[qcol] = qm[qcol].astype(str).str.strip()
    mapping = dict(zip(qm[qcol], qm[ccol].astype(str).str.strip()))
    dcol = next((c for c in qm.columns if "diff" in c), None)
    diff = dict(zip(qm[qcol], qm[dcol])) if dcol else None
    order = list(dict.fromkeys(qm[ccol].astype(str).str.strip()))
    return mapping, diff, order, None

QMAP, QDIFF, COMP_ORDER = None, None, None
# Real GP-contest workbooks embed their own per-paper competency maps and
# override any CSV — so the uploader is tucked away; it only matters for
# datasets WITHOUT an embedded "Competency Mapping" sheet.
with st.sidebar.expander("⚙️ Advanced: question map CSV", expanded=False):
    st.caption("Only needed for datasets without an embedded Competency "
               "Mapping sheet — the real GP-contest workbooks carry their "
               "own maps and ignore this.")
    _qm_up = st.file_uploader(
        "Question map (CSV)", type=["csv"],
        help="CSV with columns question, competency. Groups Q1..Qn into "
             "named skills. Ignored when the workbooks embed their own "
             "maps.")
_qm_bytes = None
if _qm_up is not None:
    _qm_bytes = _qm_up.getvalue()
    try:
        os.makedirs(os.path.join(_HERE, "_upload_cache"), exist_ok=True)
        with open(os.path.join(_HERE, "_upload_cache",
                               "qmap_override.csv"), "wb") as _qf:
            _qf.write(_qm_bytes)
    except Exception:
        pass
else:
    import pathlib as _qpl
    for _cand in (os.path.join("_upload_cache", "qmap_override.csv"),
                  "DATATHON_QUESTION_MAP.csv", "question_map.csv"):
        _p = _qpl.Path(__file__).parent / _cand
        if _p.exists():
            _qm_bytes = _p.read_bytes()
            break
if _qm_bytes:
    QMAP, QDIFF, COMP_ORDER, _qm_err = _load_qmap(_qm_bytes)
    if _qm_err:
        st.sidebar.warning(f"Question map ignored: {_qm_err}")
        QMAP = QDIFF = COMP_ORDER = None
    else:
        st.sidebar.success(f"🗺️ Question map: {len(QMAP)} items → "
                           f"{len(COMP_ORDER)} competencies")

MINN = int(st.sidebar.number_input(
    "🔬 Min students per group", min_value=5, max_value=200, value=15, step=5,
    help="Charts hide any group (block-year, gender-within-district, …) with "
         "fewer students than this — tiny groups produce lucky/unlucky "
         "averages, not evidence. Raise for stricter, lower for small samples."))
uploaded = st.sidebar.file_uploader(
    "Upload dataset (multiple files or a .zip)", type=UPLOAD_EXT,
    accept_multiple_files=True,
    help="Excel stops at 1,048,576 rows, so a full dataset usually arrives as "
         "several parts. Select them all — or one .zip containing them — and "
         "they are analysed as a single dataset.")

# Also allow loading straight from disk. The datathon requires a reproducible
# entry point that rebuilds outputs from a relative path with no manual upload,
# so the dashboard must be able to read files without the browser widget.
_DATA_DIR = os.path.dirname(_HERE)


def _scan_local(root, depth=2):
    """
    Data files in `root` and its subfolders. Split datasets normally arrive as
    a folder of parts, so a flat listing of the top directory would not show
    them at all.
    """
    found = []
    for cur, dirs, files in os.walk(root):
        rel = os.path.relpath(cur, root)
        if rel != "." and rel.count(os.sep) + 1 > depth:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs
                   if not d.startswith((".", "__")) and d != "streamlit_app"]
        for f in files:
            if (f.lower().endswith(DATA_EXT + (".zip",))
                    and not f.startswith("~$")
                    and not f.endswith(".cache.parquet")):
                found.append(os.path.normpath(os.path.join(rel, f))
                             if rel != "." else f)
    return sorted(found)


_local_files = _scan_local(_DATA_DIR)
_picked_local = []
if _local_files:
    _sel = st.sidebar.multiselect(
        "…or load files already on disk", _local_files, key="local_files",
        help="Subfolders are included — a split dataset is usually a folder of "
             "parts. Select every part; they load as one dataset.")
    _picked_local = [os.path.join(_DATA_DIR, f) for f in _sel]
    # Selecting a whole folder's worth of parts is the common case, so make it
    # one click rather than N.
    _folders = sorted({os.path.dirname(f) for f in _local_files if os.path.dirname(f)})
    if _folders and not _sel:
        _fp = st.sidebar.selectbox("…or a whole folder of parts",
                                   ["(none)"] + _folders, key="local_folder")
        if _fp != "(none)":
            _picked_local = [os.path.join(_DATA_DIR, f) for f in _local_files
                             if os.path.dirname(f) == _fp]

# ---- refresh-proof uploads: restore the last browser upload from cache ---
_UPLOAD_CACHE = os.path.join(_HERE, "_upload_cache")
_CACHE_FILE = os.path.join(_UPLOAD_CACHE, "last_upload.parquet")

# Remember whether an upload happened IN THIS SESSION. If it did and the
# uploader is now empty, the user clicked ✕ — that means "unload", so the
# cache must go too instead of resurrecting the file. A refresh starts a
# new session (flag resets) and restore works as intended.
if uploaded:
    st.session_state["_had_upload"] = True
elif st.session_state.get("_had_upload"):
    for _p in (_CACHE_FILE, _CACHE_FILE + ".meta.json"):
        try:
            os.remove(_p)
        except OSError:
            pass
    st.session_state.pop("_pcache_sig", None)

if (not uploaded and not _picked_local and os.path.exists(_CACHE_FILE)
        and not st.session_state.get("_had_upload")):
    _picked_local = [_CACHE_FILE]
    try:
        import json as _json
        with open(_CACHE_FILE + ".meta.json") as _mf:
            _meta = _json.load(_mf)
    except Exception:
        _meta = {}
    _mnames = "<br>".join(f"📄 {n}" for n in _meta.get("names", []))         or "📄 last uploaded dataset"
    _mrows = _meta.get("rows")
    st.sidebar.markdown(
        f"<div style='background:#1F2937; border:1px solid #374151; "
        f"border-left:3px solid #10B981; border-radius:10px; "
        f"padding:10px 12px; margin:4px 0 8px;'>"
        f"<div style='color:#10B981; font-size:11px; font-weight:700; "
        f"letter-spacing:.5px;'>♻️ RESTORED — REFRESH-PROOF</div>"
        f"<div style='color:#E5E7EB; font-size:13px; font-weight:600; "
        f"overflow-wrap:anywhere;'>{_mnames}</div>"
        + (f"<div style='color:#9CA3AF; font-size:12px;'>{_mrows:,} rows"
           f"</div>" if _mrows else "")
        + "<div style='color:#9CA3AF; font-size:11px;'>Upload again to "
          "replace.</div></div>", unsafe_allow_html=True)
    if st.sidebar.button("🗑️ Forget restored dataset"):
        try:
            os.remove(_CACHE_FILE)
            os.remove(_CACHE_FILE + ".meta.json")
        except OSError:
            pass
        st.session_state.pop("_pcache_sig", None)
        st.rerun()

if not uploaded and not _picked_local:
    st.title("📊 Education Insights Dashboard")
    st.info("⬅️ Upload your dataset to begin, or pick files already on disk. "
            "Expected: marks/scores + hierarchy columns "
            "(Division/District/Block/Cluster) + any text columns.")
    st.caption("Multiple parts and .zip archives are supported — Excel caps a "
               "single sheet at 1,048,576 rows, so large datasets arrive split. "
               "All parts are concatenated and analysed as one dataset.")
    st.stop()

# One list of (name, handle) whatever the origin, so everything below is
# identical for 1 uploaded file, 12 files, or a zip.
_sources = [(f.name, f) for f in (uploaded or [])] + \
           [(os.path.basename(p), p) for p in _picked_local]
_disk_paths = list(_picked_local) if not uploaded else []

# Widgets below default from a per-file GUESS. Streamlit only honors index=/
# default= the FIRST time a widget with a given key is created — on every
# later rerun it keeps whatever was already picked, ignoring a freshly
# computed guess. Without a file-specific key, switching to a new file (or
# even re-running after an earlier guess failed) leaves stale picks like
# "(none)" in place, which is exactly what "Gender/Competency not detected"
# looks like even when the new file has perfectly good columns. Suffixing
# every guess-driven widget's key with the file identity forces a fresh
# widget — and a fresh guess — whenever the uploaded file actually changes.
# Now keyed on the whole SET of parts, so adding or removing one part
# re-guesses the mapping instead of silently reusing the previous file's picks.
_fsig = hashlib.md5("|".join(
    f"{n}:{getattr(h, 'size', '') if not isinstance(h, str) else os.path.getsize(h)}"
    for n, h in _sources).encode()).hexdigest()[:10]

df, _manifest = load_sources(_sources, _disk_paths)
if df is None or df.empty:
    st.error("Could not read any data from the selected file(s).")
    if _manifest:
        st.dataframe(pd.DataFrame(_manifest), use_container_width=True)
    st.stop()

# ---- real-data normalization (GP-contest format) --------------------------
# The 2022-25 GP-contest files have: lowercase district spellings (incl.
# educational districts like "belagavi chikkodi"), no Division, no Score.
if "District" in df.columns:
    _dl = df["District"].astype(str).str.strip().str.lower()
    df["District"] = _dl.map(KA_DISTRICT_CANON).fillna(
        df["District"].astype(str).str.strip().str.title())
if "Division" not in df.columns and "District" in df.columns:
    df["Division"] = df["District"].map(KA_DIVISION).fillna("Karnataka")
_qitem_cols = [c for c in df.columns if re.fullmatch(r"Q\d+", str(c))]
if "Score" not in df.columns and _qitem_cols:
    df["Score"] = df[_qitem_cols].sum(axis=1)

# ---- per-paper competency maps embedded in the workbooks ------------------
RQMAPS, RQNAMES = extract_embedded_qmaps(_sources, _fsig)
if RQMAPS:
    _n_comps = len({c for m in RQMAPS.values() for c in m.values()})
    st.sidebar.success(f"🗺️ Competency maps read from the workbooks: "
                       f"{len(RQMAPS)} papers (per year & grade), "
                       f"{_n_comps} competencies")
    # the papers change every year+grade — override any flat CSV map
    QMAP, QDIFF = None, None
    COMP_ORDER = sorted({c for m in RQMAPS.values() for c in m.values()})

# Persist browser uploads to a local Parquet cache (once per file set) so a
# page refresh — which wipes Streamlit's upload widget — restores instantly.
if uploaded and st.session_state.get("_pcache_sig") != _fsig:
    try:
        with st.spinner("💾 Saving a refresh-proof copy…"):
            os.makedirs(_UPLOAD_CACHE, exist_ok=True)
            df.to_parquet(_CACHE_FILE, index=False)
            import json as _json
            with open(_CACHE_FILE + ".meta.json", "w") as _mf:
                _json.dump({"names": [n for n, _ in _sources],
                            "rows": int(len(df))}, _mf)
        st.session_state["_pcache_sig"] = _fsig
    except Exception:
        pass          # caching is best-effort; analysis continues either way

# Show what actually got combined — with several parts, a silently skipped or
# short file is the easiest way to analyse the wrong dataset without noticing.
_loaded = [m for m in _manifest if str(m["status"]).startswith(("✅", "⚡", "⚠️"))]
_failed = [m for m in _manifest if str(m["status"]).startswith("❌")]
if len(_manifest) > 1 or _failed:
    with st.sidebar.expander(f"📦 {len(_loaded)} parts → {len(df):,} rows",
                             expanded=bool(_failed)):
        st.dataframe(pd.DataFrame([{k: v for k, v in m.items()
                                    if not k.startswith("_")} for m in _manifest]),
                     use_container_width=True, hide_index=True)
        if _failed:
            st.error(f"{len(_failed)} file(s) could not be read — the analysis "
                     f"below EXCLUDES them.")
        if _disk_paths:
            st.caption("Combined result is cached as Parquet next to the files, "
                       "so later runs skip the Excel parsing entirely.")
        else:
            st.caption("Tip: .xlsx parses slowly (minutes per million rows). "
                       "Putting the parts on disk as .csv or .parquet and "
                       "loading them from the picker above makes restarts "
                       "near-instant.")

    # Selecting overlapping exports would double-count children. But identical
    # rows also arise by chance — with no student ID, two children in the same
    # block who answered the same way are indistinguishable. So compare the
    # combined duplicates against the duplicates each part already had on its
    # own; only the EXCESS indicates the parts actually overlap.
    # Reading from the Parquet cache skips the per-part pass, and without it
    # the comparison below is meaningless — so don't guess, just stay quiet.
    _can_compare = all("_self_dupes" in m for m in _loaded)
    _dups = int(df.duplicated().sum()) if _can_compare else 0
    _own = sum(m.get("_self_dupes", 0) for m in _loaded)
    _excess = _dups - _own
    if _can_compare and _excess > 0.02 * len(df):
        st.sidebar.error(
            f"🚫 The selected parts appear to OVERLAP: combining them created "
            f"{_excess:,} duplicate rows beyond what the parts contain "
            f"individually ({100*_excess/len(df):.1f}%). Every percentage below "
            f"double-counts those children. Check you have not selected both a "
            f"full export and its parts.")
    elif _dups:
        st.sidebar.caption(
            f"{_dups:,} identical rows ({100*_dups/len(df):.2f}%), of which "
            f"{_own:,} already occur within a single part. Expected without a "
            f"student ID — not treated as duplication.")

df, fmt_info, items_df = normalize_dataset(df)
for _note in fmt_info["notes"]:
    st.sidebar.info(_note)
cols = classify_columns(df)

def _guess_roles(df, cols):
    """Best-effort role guesses for any dataset; user can override in the UI."""
    cats = cols["categorical"]

    def name_match(keys, pool):
        for c in pool:
            if any(k in c.lower() for k in keys):
                return c
        return None

    # Year: name, or an integer-ish column with plausible year values
    year = name_match(["year", "session"], cats + cols["objective"])
    if year is None:
        for c in cats + cols["objective"]:
            v = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(v) and v.between(1990, 2100).all() and df[c].nunique() <= 25:
                year = c
                break

    # Gender: name, or a 2-3 value column with gender-like values
    gender = name_match(["gender", "sex"], cats)
    if gender is None:
        gset = {"male", "female", "m", "f", "boy", "girl", "boys", "girls",
                "other", "others"}
        for c in cats:
            vals = {str(x).strip().lower() for x in df[c].dropna().unique()}
            if 2 <= len(vals) <= 3 and vals <= gset:
                gender = c
                break

    comp = name_match(["competen", "subject", "skill", "domain"], cats)
    sid = (cols["id"][0] if cols["id"] else
           name_match(["student", "child", "roll", "_id", "id"], cats))

    # Hierarchy: known geo names first (ordered), else leftover categoricals
    # ordered by cardinality (fewest values = outermost ring)
    geo_keys = ["division", "state", "region", "zone", "district", "block",
                "taluk", "tehsil", "mandal", "cluster", "panchayat", "gram",
                "ward", "habitation", "village", "school"]
    used = {year, gender, comp, sid}
    non_geo = ("quality", "infra", "background", "income", "ratio",
               "attendance", "score", "percent")
    geo = [c for c in cats if c not in used
           and any(k in c.lower() for k in geo_keys)
           and not any(k in c.lower() for k in non_geo)]
    if geo:
        hier = sorted(geo, key=lambda c: df[c].nunique())
    else:
        blocked = ("score", "percent", "mark", "grade", "class", "band", "total")
        rest = [c for c in cats if c not in used
                and not pd.api.types.is_numeric_dtype(df[c])
                and not any(k in c.lower() for k in blocked)
                and 1 < df[c].nunique() <= max(200, len(df) // 10)]
        hier = sorted(rest, key=lambda c: df[c].nunique())[:4]
    return hier, year, gender, comp, sid

_g_hier, _g_year, _g_gender, _g_comp, _g_sid = _guess_roles(df, cols)
_NONE = "(none)"
with st.sidebar.expander("🧭 Column mapping", expanded=not _g_hier):
    st.caption("Auto-detected — fix anything that's wrong for this dataset.")
    hierarchy = st.multiselect("Hierarchy levels (outer → inner)",
                               cols["categorical"], default=_g_hier,
                               key=f"hier_{_fsig}")

    def role_select(label, guess, pool):
        opts = [_NONE] + pool
        pick = st.selectbox(label, opts,
                            index=opts.index(guess) if guess in opts else 0,
                            key=f"role_{label}_{_fsig}")
        return None if pick == _NONE else pick

    year_col = role_select("Year column", _g_year,
                           cols["categorical"] + cols["objective"])
    gender_col = role_select("Gender column", _g_gender, cols["categorical"])
    comp_col = role_select("Competency/Subject column", _g_comp,
                           cols["categorical"])
    sid_col = role_select("Student/record ID column", _g_sid,
                          cols["categorical"] + cols["id"])
    if st.button("🔄 Reset to auto-detected", key=f"reset_{_fsig}"):
        for k in list(st.session_state):
            if k.endswith(f"_{_fsig}"):
                del st.session_state[k]
        st.rerun()

# Academic years arrive as "2022-23". Several tabs do int(year) / polyfit on it,
# so convert to the numeric start year ONCE here rather than at each call site.
def _numeric_year(frame, col):
    """In-place: academic-year strings -> numeric start year. Returns (frame, ok)."""
    if frame is None or col not in frame.columns:
        return frame, True
    if pd.api.types.is_numeric_dtype(frame[col]):
        return frame, True
    parsed = adapter.parse_year(frame[col])
    if not parsed.notna().any():
        return frame, False
    frame = frame.assign(**{col: parsed})
    frame = frame[frame[col].notna()]
    frame[col] = frame[col].astype(int)
    return frame, True


if year_col and year_col in df.columns:
    _sample = df[year_col].dropna().iloc[0] if len(df[year_col].dropna()) else ""
    _was_text = not pd.api.types.is_numeric_dtype(df[year_col])
    df, _ok = _numeric_year(df, year_col)
    # items_df was captured before this point — convert it too, or the Item
    # Analysis tab compares its string years against the numeric slider range.
    items_df, _ = _numeric_year(items_df, year_col)
    if not _ok:
        st.sidebar.warning(f"Could not read years from '{year_col}' — "
                           f"trend and prediction tabs will be limited.")
        year_col = None
    elif _was_text:
        st.sidebar.caption(f"📅 '{year_col}' parsed to start year "
                           f"(e.g. {_sample} → {df[year_col].iloc[0]}).")

_score_opts = cols["objective"] or df.columns.tolist()
_score_opts = [c for c in _score_opts
               if not re.search(r"\b(id|identifier)\b", str(c), re.I)] \
    or _score_opts
_default_score = next((c for c in ("Score", "Overall (%)") if c in _score_opts),
                      _score_opts[0])
score_col = st.sidebar.selectbox("Score column", _score_opts,
                                 index=_score_opts.index(_default_score),
                                 key=f"score_col_{_fsig}")

# Curve-graded bucket for the chosen score
df["Performance_Band"] = grade_on_curve(df[score_col])

# --- The column to use whenever data is sliced BY COMPETENCY ----------------
# After the Q1..Qn melt, one student becomes 20 rows and `Score` is still their
# TOTAL — the same value repeated on all 20. So grouping `Score` by competency
# returns an IDENTICAL number for every competency: charts look populated but
# carry zero information (measured: 1 distinct value across 20 competencies).
# The real per-question result is "Item Score (%)" — 100 for correct, 0 for
# wrong. Anything broken down by competency must use this, not score_col.
ITEM_COL = "Item Score (%)"
COMP_VALUE_COL = (ITEM_COL if fmt_info.get("reshaped") and ITEM_COL in df.columns
                  else score_col)
COMP_VALUE_IS_PCT = COMP_VALUE_COL == ITEM_COL

# ---------------------------------------------------------------------------
#  Competency / question filtering — shared by Deep Dive and the Map
# ---------------------------------------------------------------------------
# The questions reach a tab in one of two shapes and a tab should not have to
# care which:
#   LONG  normalize_dataset melted Q1..Qn into `comp_col`, so one row is one
#         child's answer to one question and the value is in COMP_VALUE_COL.
#   WIDE  the melt was skipped (dataset too large to expand, or the file
#         already had its own competency column) and Q1..Qn are still columns.
# Both are reduced below to the same thing: "% correct on the selected
# questions", so a filtered chart means exactly what an unfiltered one means.
def _qnum(q):
    """Sort key so Q2 comes before Q10 and unnumbered labels sort last."""
    _m = re.search(r"\d+", str(q))
    return (0, int(_m.group()), "") if _m else (1, 0, str(q))


def question_universe(frame):
    """('long' | 'wide' | None, [question ids]) for `frame`."""
    if (comp_col and comp_col in frame.columns
            and COMP_VALUE_COL in frame.columns and COMP_VALUE_IS_PCT):
        qs = [str(q) for q in pd.unique(frame[comp_col].dropna())]
        if qs:
            return "long", sorted(qs, key=_qnum)
    wide = [c for c in frame.columns
            if re.fullmatch(r"[Qq]\s*_?\d+", str(c).strip())
            and pd.api.types.is_numeric_dtype(frame[c])
            and pd.notna(frame[c].max()) and float(frame[c].max()) <= 1]
    if wide:
        return "wide", sorted(wide, key=_qnum)
    return None, []


def competency_of(q):
    """Question id -> competency name from the uploaded map (identity if none)."""
    return QMAP.get(str(q), str(q)) if QMAP else str(q)


def competency_question_filter(frame, key, box=None, show=True):
    """Resolve the question selection for a tab, optionally with UI.

    With show=True this renders Competency + Question multiselects. With
    show=False it renders nothing and selects every question — a tab that
    only wants the shape-independent plumbing (which column holds the value,
    how to break skills down) can call it without adding widgets.

    An empty selection means "everything", so the default state leaves the
    tab's data exactly as it would have been without this filter.

    Returns a dict:
      frame        the filtered frame
      col          the column to average for the metric
      is_pct       whether that column is already a percentage
      label        what to call it on an axis
      narrowed     True when a strict subset of questions is active
      questions    the selected question ids
      competencies the selected competency names
      kind         'long' | 'wide' | None
    """
    box = box or st
    kind, qs = question_universe(frame)
    res = {"frame": frame, "col": score_col, "is_pct": False,
           "label": f"Avg {score_col}", "narrowed": False, "questions": qs,
           "competencies": [], "kind": kind, "n_all": len(qs)}
    if not qs:
        return res

    cmap = {q: competency_of(q) for q in qs}
    comps = list(dict.fromkeys(cmap[q] for q in qs))
    # Only offer the competency box when the map actually groups things. With
    # no map loaded every question is its own "competency" and the two boxes
    # would be duplicates of each other.
    mapped = bool(QMAP) and len(comps) < len(qs)

    csel, qsel = [], []
    if show:
        _fc = box.columns([1, 1.3])
        csel = (_fc[0].multiselect(
            f"🎯 Competency — {len(comps)} in the question map", comps,
            default=[], key=f"{key}_comp", placeholder="All competencies",
            help="Groups from the question-map CSV. Picking one narrows the "
                 "question list beside it.") if mapped else [])
        pool = [q for q in qs if not csel or cmap[q] in csel]
        qsel = _fc[1].multiselect(
            f"❓ Question — {len(pool)} available", pool, default=[],
            key=f"{key}_q", placeholder="All questions in that selection",
            format_func=(lambda q: f"{q} · {cmap[q]}") if mapped else str,
            help="Leave empty for every question above.")
    else:
        pool = list(qs)

    sel = list(qsel) if qsel else pool
    res["questions"] = sel
    res["competencies"] = csel or comps
    res["narrowed"] = len(sel) < len(qs)

    if kind == "long":
        res["col"] = COMP_VALUE_COL
        res["is_pct"] = COMP_VALUE_IS_PCT
        res["label"] = "% correct" if COMP_VALUE_IS_PCT else f"Avg {score_col}"
        if res["narrowed"]:
            res["frame"] = frame[frame[comp_col].astype(str)
                                 .isin(set(map(str, sel)))]
    else:                                   # wide: rebuild the % from columns
        _f = frame
        if res["narrowed"]:
            _f = frame.copy()
            _f["_qsel_pct"] = _f[sel].mean(axis=1) * 100.0
        elif (score_col in frame.columns
                and pd.notna(frame[score_col].max())
                and frame[score_col].max() <= len(qs)):
            # score_col is a raw total out of len(qs) — put it on the same
            # 0-100 scale as the filtered case so the axis never jumps
            _f = frame.copy()
            _f["_qsel_pct"] = _f[score_col] * 100.0 / len(qs)
        if _f is not frame:
            res.update(frame=_f, col="_qsel_pct", is_pct=True,
                       label="% correct")

    if res["narrowed"]:
        box.caption(
            f"🔎 **{len(sel)} of {len(qs)} questions** selected"
            + (f" · competencies: {', '.join(map(str, csel))}" if csel else "")
            + " — every number below is *% correct on these questions only*, "
              "not the child's overall score.")
    return res


def question_means(frame, res, by="Question"):
    """Mean % correct per question (or per competency) — works on both shapes."""
    if res["kind"] == "long":
        s = frame.groupby(comp_col, observed=True)[COMP_VALUE_COL].mean()
        s.index = s.index.astype(str)
        s = s.reindex([q for q in res["questions"] if q in set(s.index)])
    elif res["kind"] == "wide":
        _q = [q for q in res["questions"] if q in frame.columns]
        s = frame[_q].mean() * 100.0
        s.index = s.index.astype(str)
    else:
        return pd.Series(dtype=float)
    if by == "Competency":
        s = s.groupby(s.index.map(competency_of)).mean()
    return s.dropna()


def question_label(q):
    """'Q3 · Numeracy' when a map is loaded, plain 'Q3' when it isn't."""
    c = competency_of(q)
    return f"{q} · {c}" if str(c) != str(q) else str(q)


# Grade/class filter, when such a column exists
grade_col = next((c for c in cols["categorical"]
                  if str(c).lower() in ("grade", "class", "std", "standard")), None)

# Filters — auto-generated from hierarchy + categoricals
st.sidebar.title("🔍 Slice & Dice")
# ---- share with the 🎯 Missions page (pages/Missions.py) ----------------
st.session_state["_mx_primary_df"] = df
st.session_state["_mx_score_col"] = score_col
st.session_state["_mx_qmap"] = RQMAPS if RQMAPS else (QMAP or None)

fdf = df.copy()
_active_filters = {}          # reused by the analysis layers on the raw item frame
for col in hierarchy + ([comp_col] if comp_col else []) \
        + ([gender_col] if gender_col else []) \
        + ([grade_col] if grade_col else []):
    if col:
        options = ["All"] + sorted(fdf[col].dropna().unique().tolist())
        pick = st.sidebar.selectbox(col, options)
        if pick != "All":
            fdf = fdf[fdf[col] == pick]
            _active_filters[col] = pick

if year_col:
    years = sorted(df[year_col].unique())
    year_range = st.sidebar.select_slider("Years", options=years, value=(years[0], years[-1]))
    fdf = fdf[(fdf[year_col] >= year_range[0]) & (fdf[year_col] <= year_range[1])]

# ------------------------------- Header + KPIs -------------------------------
st.title("📊 Education Insights Dashboard")
# After the Q1..Qn melt one row is ONE ANSWER, not one student — label it
# honestly so "1,000 students" never silently reads as "20,000 records".
_row_unit = "question responses" if fmt_info.get("reshaped") else "records"
_n_students = len(items_df) if items_df is not None else len(fdf)
_src_note = (f"  ·  {len(_loaded)} files combined" if len(_loaded) > 1 else "")
st.caption(f"{len(fdf):,} {_row_unit} in current selection"
           + (f"  ·  {_n_students:,} student rows loaded"
              if fmt_info.get("reshaped") else "")
           + _src_note)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Average Score", f"{fdf[score_col].mean():.1f}")
k2.metric("Question responses" if fmt_info.get("reshaped") else "Records",
          f"{len(fdf):,}",
          help=(f"{_n_students:,} students × {len(_find_item_columns(items_df)) if items_df is not None else 0} "
                f"questions. Each row is one answer."
                if fmt_info.get("reshaped") else None))
if hierarchy:
    grp = fdf.groupby(hierarchy[-1])[score_col].mean()
    if len(grp):
        k3.metric(f"Best {hierarchy[-1]}", grp.idxmax(), f"{grp.max():.1f}")
        k4.metric(f"Weakest {hierarchy[-1]}", grp.idxmin(), f"{grp.min():.1f}",
                  delta_color="inverse")

tabs = st.tabs(["🌞 Hierarchy",
                # GKA's Impact sits second on purpose: "is the programme
                # helping?" is the question the whole submission answers, and
                # it must be reachable before any of the descriptive tabs.
                "🎓 GKA's Impact",
                "📈 Trends", "⚖️ Gender Gap", "🎯 Competencies",
                "🔬 Deep Dive", "🚨 Rankings & Alerts", "🔮 Prediction",
                "🗂️ Data", "🧩 Item Analysis", "🗺️ Map",
                # Cross-dataset sits BEFORE Insights on purpose: the district
                # -context join is an INPUT to the final insights, so the tab
                # that shows that join has to come first when reading L to R.
                "📄 Facts & Health", "🔗 Cross-dataset", "🧠 Insights",
                "📋 Action Plan", "📝 Briefs", "🎓 Competency Report",
                "🎛️ What-If", "🧬 Archetypes & Risk"])

# ---------------------------------------------------------------------------
#  Analysis layers — shared aggregate, built once from the current selection
# ---------------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.subheader("🧠 Analysis layers")

# When the file carries Q1..Qn item columns we analyse THOSE (each question is a
# competency). Otherwise we fall back to the score column and need thresholds.
_raw_for_layers = items_df if items_df is not None else fdf
for _c, _v in _active_filters.items():          # re-apply the sidebar slice
    if _c in _raw_for_layers.columns:
        _raw_for_layers = _raw_for_layers[_raw_for_layers[_c] == _v]

_has_items = bool(adapter.find_item_columns(_raw_for_layers))
if _has_items:
    st.sidebar.caption("🧩 Question items detected — each question is treated as a "
                       "competency; a wrong answer counts as below grade level.")
    _below_cut, _above_cut = 50.0, 75.0
else:
    _below_cut = st.sidebar.slider("‘Below grade level’ is a score under", 20, 70, 50, 5)
    _above_cut = st.sidebar.slider("‘Above grade level’ is a score at or over",
                                   _below_cut + 5, 95, max(_below_cut + 25, 75), 5)


_grain_opts = [c for c in hierarchy if c in _raw_for_layers.columns]
_finest = st.sidebar.selectbox(
    "Report findings down to", _grain_opts,
    index=min(2, len(_grain_opts) - 1) if _grain_opts else 0,
    help="Going finer than the data supports produces confident nonsense. "
         "If group sizes are tiny, move this up a level.")
_by_grade = st.sidebar.checkbox("Split by grade", value=True,
                                help="Uncheck to pool grades — quadruples group size.")


@st.cache_data(show_spinner="Aggregating…")
def _build_agg(_df, fsig, hierarchy, score_col, year_col, gender_col, comp_col,
               grade_col, below_cut, above_cut, finest, by_grade, qmap=None):
    return adapter.build_agg(_df, hierarchy=hierarchy, score_col=score_col,
                             year_col=year_col, gender_col=gender_col,
                             comp_col=comp_col, grade_col=grade_col,
                             below_cut=below_cut, above_cut=above_cut,
                             finest=finest, by_grade=by_grade,
                             qmap=dict(qmap) if qmap else None)


# ---- Performance: cache every heavy analysis-layer call -------------------
# Streamlit reruns this whole script on EVERY widget interaction. Without
# caching, that means re-training ridge/KMeans and re-generating insights on
# every slider drag anywhere in the app — the main source of lag.
#
# CAREFUL: a leading underscore tells Streamlit to LEAVE THAT ARGUMENT OUT of
# the cache key (it is the escape hatch for unhashable objects). So `_agg`
# alone would key these purely on district/limit — load a second file, change
# the reporting grain, or move the year filter, and every one of them would
# hand back the PREVIOUS dataset's answer, silently and with no error.
# `agg_sig` below is a real hash of the data, so it must be passed on every
# call. Verified: without it, two different aggregates return identical output.
def _agg_sig(a):
    """Content hash of the aggregate — the actual cache key for the wrappers."""
    if a is None or len(a) == 0:
        return "empty"
    return hashlib.md5(
        pd.util.hash_pandas_object(a, index=True).values).hexdigest()


@st.cache_data(show_spinner=False)
def _c_train_early_warning(_agg, agg_sig):
    return L_models.train_early_warning(_agg)

@st.cache_data(show_spinner=False)
def _c_cluster_blocks(_agg, agg_sig, k=3):
    return L_models.cluster_blocks(_agg, k=k)

# `context` is the cross-dataset bundle from insights_cross.prepare(). It is
# built from AGG plus the district file, so agg_sig + ctx_sig together key the
# cache — NOT the bundle itself, which holds DataFrames and is unhashable.
@st.cache_data(show_spinner=False)
def _c_insights_generate(_agg, agg_sig, district, limit=8, min_n=30,
                         _context=None, ctx_sig=None):
    return L_insights.generate(_agg, district, limit=limit, min_n=min_n,
                               context=_context)

@st.cache_data(show_spinner=False)
def _c_insights_describe(_agg, agg_sig, district, min_n=30):
    return L_insights.describe(_agg, district, min_n=min_n)

@st.cache_data(show_spinner=False)
def _c_cross_prepare(_agg, agg_sig, _sec, sec_sig, level):
    return L_cross.prepare(_agg, _sec, level=level)


def _gka_unit_col():
    """The district column as it is named in the raw item frame."""
    for c in ("District", "district", "DISTRICT"):
        if c in _raw_for_layers.columns:
            return c
    return (hierarchy[1] if len(hierarchy) > 1
            else (hierarchy[0] if hierarchy else None))


def _gka_gp_col():
    """
    The finest geography, if the file carries one.

    GP ID is preferred over GP Name: names repeat across blocks, so grouping
    by name silently merges different panchayats into one row.
    """
    for c in ("GP ID", "GP Id", "GP_ID", "GP Name", "GP", "Cluster"):
        if c in _raw_for_layers.columns:
            return c
    return None


def _gka_papers():
    """
    RQMAPS / RQNAMES re-keyed to the year representation the item frame uses.

    The workbooks key each paper by academic year ("2022-23"), while the
    dashboard converts Year to its numeric start (2022) so the range slider
    can work. Left unreconciled every anchor lookup misses, and the tab
    reports — quietly and wrongly — that no two papers share any skill.
    """
    if (year_col and year_col in _raw_for_layers.columns
            and pd.api.types.is_numeric_dtype(_raw_for_layers[year_col])):
        def _k(y):
            v = adapter.parse_year(pd.Series([y])).iloc[0]
            return int(v) if pd.notna(v) else y
        return ({(_k(y), g): m for (y, g), m in RQMAPS.items()},
                {(_k(y), g): m for (y, g), m in RQNAMES.items()})
    return RQMAPS, RQNAMES


# The GKA layer reads the RAW per-child item frame, not AGG: it needs GP, which
# the aggregate drops, and per-question responses, which it folds away.
@st.cache_data(show_spinner="Measuring programme impact…")
def _c_gka_analyse(_raw, raw_sig, year_col, grade_col, unit_col, gp_col, min_n):
    _qm, _qn = _gka_papers()
    return L_gka.analyse(_raw, _qm, _qn, year_col=year_col,
                         grade_col=grade_col, unit_col=unit_col,
                         gp_col=gp_col, min_n=min_n)


@st.cache_data(show_spinner=False)
def _c_read_context_file(path, mtime):
    return pd.read_excel(path)


def _find_context_file():
    """The district-context workbook shipped next to the app, if present."""
    for cand in ("secondary_dataset.xlsx", "secondary.xlsx"):
        p = os.path.join(_HERE, cand)
        if os.path.exists(p):
            return p, cand
    return None, None


def _insight_context(level=None):
    """
    The cross-dataset bundle, shared by the Insights and Action Plan tabs.

    One loader so the two tabs can never disagree about what the district
    context says. Returns None when there is no context file or the district
    names cannot be matched — callers then fall back to primary-only output.
    """
    path, name = _find_context_file()
    if path is None or "AGG" not in globals() or AGG is None:
        return None
    lvl = level or st.session_state.get("ins_level") or "District"
    try:
        sec = _c_read_context_file(path, os.path.getmtime(path))
        return _c_cross_prepare(AGG, AGG_SIG, sec, f"{name}:{sec.shape}", lvl)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def _c_playbook_recommend_v3(_agg, agg_sig, district, limit=12, min_n=30,
                             _context=None, ctx_sig=None):
    return L_playbook.recommend(_agg, district, limit=limit, min_n=min_n,
                                context=_context)

@st.cache_data(show_spinner=False)
def _c_playbook_coverage_v3(_agg, agg_sig, district, min_n=30,
                            _context=None, ctx_sig=None):
    return L_playbook.coverage_stats(_agg, district, min_n=min_n,
                                     context=_context)

@st.cache_data(show_spinner=False)
def _c_playbook_recommend(_agg, agg_sig, district, limit=12):
    return L_playbook.recommend(_agg, district, limit=limit)

@st.cache_data(show_spinner=False)
def _c_playbook_coverage(_agg, agg_sig, district):
    return L_playbook.coverage_stats(_agg, district)

@st.cache_data(show_spinner=False)
def _c_competency_report(_agg, agg_sig, district, comp):
    return L_competency.report(_agg, district, comp)

@st.cache_data(show_spinner=False)
def _c_competency_corr(_agg, agg_sig, district):
    return L_competency.correlation_matrix(_agg, district)

@st.cache_data(show_spinner=False)
def _c_what_if(_agg, agg_sig, district, comp, n_blocks, min_n=30):
    return L_models.what_if(_agg, district, comp, n_blocks=n_blocks,
                            min_n=min_n)

@st.cache_data(show_spinner=False)
def _c_benchmarks(_agg, agg_sig):
    return L_models.improvement_benchmarks(_agg)

@st.cache_data(show_spinner=False)
def _c_rebound(_agg, agg_sig):
    return L_models.natural_rebound(_agg)

@st.cache_data(show_spinner=False)
def _c_verbalize_district(_agg, agg_sig, district):
    return L_verbalize.verbalize_district(_agg, district)

@st.cache_data(show_spinner=False)
def _c_brief_build(_agg, agg_sig, district, role, block, min_n=30,
                   _context=None, ctx_sig=None):
    return L_brief.build(_agg, district, role=role, block=block, min_n=min_n,
                         context=_context)

@st.cache_data(show_spinner=False)
def _c_brief_build_all(_agg, agg_sig, district, block, min_n=30,
                       _context=None, ctx_sig=None):
    return L_brief.build_all(_agg, district, block=block, min_n=min_n,
                             context=_context)



# `_fsig` identifies the loaded file set. Without it in the key, loading a
# different file with the same column names would reuse the previous aggregate.
AGG, AGG_NOTE = _build_agg(_raw_for_layers, _fsig, tuple(hierarchy), score_col,
                           year_col, gender_col, comp_col, grade_col,
                           _below_cut, _above_cut, _finest, _by_grade,
                           qmap=(tuple(sorted(
                               ((int(y[:4]), g, q), c)
                               for (y, g), m in RQMAPS.items()
                               for q, c in m.items())) if RQMAPS
                               else (tuple(sorted(QMAP.items()))
                                     if QMAP else None)))

# Small groups produce unstable percentages (one child can swing a block by 20
# points). This drops them before any finding is generated.
if AGG is not None and not AGG.empty:
    _med = int(AGG["n"].median())
    _min_n = st.sidebar.slider(
        "Ignore groups smaller than (students)", 0, 100,
        0 if _med < 20 else 30, 5,
        help="Percentages computed on a handful of students are noise. Raise this "
             "on the full dataset; keep it at 0 only for small test files.")
    if _min_n:
        _kept = AGG[AGG["n"] >= _min_n]
        if _kept.empty:
            st.sidebar.error(f"No group has ≥{_min_n} students — showing all instead.")
        else:
            AGG = _kept.reset_index(drop=True)
    st.sidebar.caption(f"Median group size: **{_med}** students")

    # The default above is 0 on small files, which is exactly the case where
    # unfiltered output is least trustworthy. Say so plainly instead of letting
    # a 3-student "100% below grade level" read like a finding.
    if _min_n == 0 and _med < 20:
        st.sidebar.warning(
            f"⚠️ Median group is only **{_med}** students, and no minimum is set. "
            f"Individual findings here are illustrative, not evidence — one child "
            f"can move a block by {100/max(_med,1):.0f} points. Do not quote "
            f"block-level numbers from this file; use a coarser reporting unit "
            f"above, or run on the full dataset.")

AGG_WARNINGS = adapter.warnings(AGG)

# Computed AFTER the min_n filter, because that filter changes AGG — and every
# cached wrapper below must be keyed on the data it is actually given.
AGG_SIG = _agg_sig(AGG)


def _pick_district(key):
    """District selector shared by the analysis tabs."""
    opts = sorted(AGG["district"].unique())
    return st.selectbox("District", opts, key=key)


def _needs_agg(show_caveats=True):
    if AGG is None or AGG.empty:
        st.warning(AGG_NOTE or "Not enough data in the current selection.")
        return True
    if show_caveats:

        if AGG_WARNINGS:
            with st.expander(f"⚠️ {len(AGG_WARNINGS)} data caveat(s) — read before "
                             f"quoting these numbers"):
                for w in AGG_WARNINGS:
                    st.markdown(f"- {w}")
    return False

# ------------------------------- Sunburst ------------------------------------
def _score_to_hex(v, vmin, vmax):
    """Map a score to the RdYlGn palette (red→yellow→green), no dependencies."""
    anchors = [(0.0, (165, 0, 38)), (0.25, (244, 109, 67)), (0.5, (255, 255, 191)),
               (0.75, (102, 189, 99)), (1.0, (0, 104, 55))]
    t = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    t = min(max(t, 0.0), 1.0)
    for (t0, c0), (t1, c1) in zip(anchors, anchors[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0
            rgb = [round(a + (b - a) * f) for a, b in zip(c0, c1)]
            return "#{:02x}{:02x}{:02x}".format(*rgb)
    return "#006837"

@st.cache_data(show_spinner=False)
def _echarts_js():
    import pathlib as _pl
    return (_pl.Path(__file__).parent / "echarts.min.js").read_text(encoding="utf-8")

def _build_echarts_tree(agg, levels_all, vmin, vmax, topk):
    """Nested {name, value, children} data for ECharts sunburst/treemap.

    Each level shows at most `topk` children per parent (largest by record
    count); the rest is bundled into one '+N more' node with the weighted
    average, so node count stays bounded for any dataset size.
    """
    def build(level_df, levels, parent=""):
        col, rest = levels[0], levels[1:]
        # rank children by size, largest first
        groups = [(str(name), sub, int(sub["n"].sum()))
                  for name, sub in level_df.groupby(col, sort=False)]
        groups.sort(key=lambda g: -g[2])
        shown, extra = groups[:topk], groups[topk:]

        nodes = []
        for full, sub, ntot in shown:
            # Don't repeat the parent's name on inner text (keeps labels short)
            short = full
            for sep in ("-", " ", "_"):
                if parent and full.startswith(parent + sep):
                    short = full[len(parent) + 1:]
                    break
            # weighted average (correct even when children differ in size)
            avg = float((sub["avg"] * sub["n"]).sum() / max(ntot, 1))
            hexcol = _score_to_hex(avg, vmin, vmax)
            r, g, b = int(hexcol[1:3], 16), int(hexcol[3:5], 16), int(hexcol[5:7], 16)
            bright = (0.299 * r + 0.587 * g + 0.114 * b) / 255
            node = {"name": short, "full": full,
                    "itemStyle": {"color": hexcol},
                    "label": {"color": "#1a1a1a" if bright > 0.55 else "#ffffff"},
                    "avg": round(avg, 1)}
            if rest:
                node["children"] = build(sub, rest, full)
                node["value"] = ntot  # treemap needs parent totals too
            else:
                node["value"] = ntot
            nodes.append(node)

        if extra:
            n_sum = sum(g[2] for g in extra)
            wavg = sum(float((g[1]["avg"] * g[1]["n"]).sum()) for g in extra) / max(n_sum, 1)
            hexcol = _score_to_hex(wavg, vmin, vmax)
            nodes.append({
                "name": f"+{len(extra)} more",
                "full": f"{len(extra)} smaller units (bundled, weighted avg)",
                "itemStyle": {"color": hexcol, "opacity": 0.55,
                              "borderType": "dashed"},
                "label": {"color": "#1a1a1a"},
                "avg": round(wavg, 1),
                "value": n_sum,
            })
        return nodes

    return build(agg, levels_all)

def render_echarts_sunburst(agg, hierarchy, vmin, vmax, height=650, topk=12):
    """Smooth-animated sunburst via bundled ECharts (offline, no CDN)."""
    import json, pathlib
    import streamlit.components.v1 as components
    lib = pathlib.Path(__file__).parent / "echarts.min.js"
    if not lib.exists():
        st.warning("echarts.min.js not found next to the app — falling back to Plotly.")
        return False

    data = _build_echarts_tree(agg, hierarchy, vmin, vmax, topk)
    option = {
        "series": [{
            "type": "sunburst", "data": data, "radius": [0, "95%"],
            "sort": None,
            "emphasis": {"focus": "ancestor"},
            "label": {"rotate": "radial", "fontSize": 10, "minAngle": 7},
            "itemStyle": {"borderColor": "rgba(0,0,0,.35)", "borderWidth": 1.5},
            "animationDuration": 700,
            "animationDurationUpdate": 700,
            "animationEasing": "cubicOut",
            "animationEasingUpdate": "cubicInOut",
        }],
    }
    html = f"""
    <div id="sb" style="width:100%;height:{height}px;"></div>
    <script>{_echarts_js()}</script>
    <script>
      var chart = echarts.init(document.getElementById('sb'), null,
                               {{renderer: 'canvas'}});
      var option = {json.dumps(option)};
      option.tooltip = {{
        formatter: function(p) {{
          var s = '<b>' + ((p.data && p.data.full) || p.name) + '</b><br/>Avg score: ' +
                  (p.data && p.data.avg != null ? p.data.avg : '-');
          if (p.value != null) s += '<br/>Records: ' + p.value;
          return s;
        }}
      }};
      chart.setOption(option);
      window.addEventListener('resize', () => chart.resize());
    </script>"""
    components.html(html, height=height + 10)
    return True

def render_echarts_treemap(agg, hierarchy, vmin, vmax, height=650, topk=12):
    """Smooth-animated treemap via bundled ECharts (offline, no CDN).
    Click a tile to zoom in; breadcrumb at the bottom to zoom back out."""
    import json, pathlib
    import streamlit.components.v1 as components
    lib = pathlib.Path(__file__).parent / "echarts.min.js"
    if not lib.exists():
        st.warning("echarts.min.js not found next to the app — falling back to Plotly.")
        return False

    data = _build_echarts_tree(agg, hierarchy, vmin, vmax, topk)
    option = {
        "series": [{
            "type": "treemap", "data": data,
            "width": "100%", "height": "92%", "top": 0,
            "roam": False, "nodeClick": "zoomToNode",
            "breadcrumb": {"show": True, "bottom": 4, "height": 24,
                           "itemStyle": {"color": "#e8ecf2",
                                         "textStyle": {"color": "#26303e"}}},
            "leafDepth": 2,
            "upperLabel": {"show": True, "height": 22, "fontSize": 11,
                           "fontWeight": "bold"},
            "label": {"fontSize": 11},
            "itemStyle": {"borderColor": "rgba(0,0,0,.35)", "borderWidth": 1,
                          "gapWidth": 2},
            "levels": [
                {"itemStyle": {"borderWidth": 0, "gapWidth": 3}},
                {"itemStyle": {"gapWidth": 2, "borderColorSaturation": 0.35}},
                {"itemStyle": {"gapWidth": 1, "borderColorSaturation": 0.5}},
            ],
            "animationDurationUpdate": 600,
            "animationEasingUpdate": "cubicInOut",
        }],
    }
    html = f"""
    <div id="tm" style="width:100%;height:{height}px;"></div>
    <script>{_echarts_js()}</script>
    <script>
      var chart = echarts.init(document.getElementById('tm'), null,
                               {{renderer: 'canvas'}});
      var option = {json.dumps(option)};
      option.series[0].label.formatter = function(p) {{
        var lines = [p.name];
        if (p.data && p.data.avg != null) lines.push('avg ' + p.data.avg);
        if (p.value != null) lines.push(p.value.toLocaleString() + ' rec');
        return lines.join('\\n');
      }};
      option.tooltip = {{
        formatter: function(p) {{
          var s = '<b>' + ((p.data && p.data.full) || p.name) + '</b><br/>Avg score: ' +
                  (p.data && p.data.avg != null ? p.data.avg : '-');
          if (p.value != null) s += '<br/>Records: ' + p.value.toLocaleString();
          return s;
        }}
      }};
      chart.setOption(option);
      window.addEventListener('resize', () => chart.resize());
    </script>"""
    components.html(html, height=height + 10)
    return True

with tabs[0]:
    # fragment: widgets inside this tab rerun only this tab
    @st.fragment
    def _tab0_fragment():
        if hierarchy:
            # --- Chart type: treemap uses area (better for many small units /
            # comparing sizes); sunburst shows the ring hierarchy.
            chart_type = st.segmented_control(
                "Chart type", ["🟩 Treemap", "🌞 Sunburst"],
                default="🟩 Treemap", key="hier_chart_type")
            if chart_type is None:          # user tapped the active pill off
                chart_type = "🟩 Treemap"
            is_treemap = chart_type.endswith("Treemap")

            # --- Scale controls: bound what's DRAWN, not what's analyzed --------
            # Aggregation handles millions of rows fine; the browser dies on
            # thousands of arcs. Depth + top-K keep the arc count bounded no
            # matter how big the dataset gets.
            cc1, cc2, cc3 = st.columns([1.1, 1.1, 1])
            _lvl_word = "Levels shown" if is_treemap else "Rings shown"
            _slice_word = "Max tiles per level" if is_treemap else "Max slices per ring"
            if len(hierarchy) > 2:
                depth = cc1.slider(_lvl_word, 2, len(hierarchy),
                                   min(3, len(hierarchy)),
                                   help="Deeper levels load on click-zoom mentally; "
                                        "fewer levels = readable + fast.")
            else:
                depth = len(hierarchy)
            topk = cc2.slider(_slice_word, 5, 30, 12,
                              help="Largest units shown; the rest are bundled "
                                   "into a '+N more' slice (weighted average).")
            smooth = cc3.toggle("✨ Smooth renderer (ECharts)", value=False,
                                help="Silkier zoom animations. Turn off for the Plotly version.")

            levels = hierarchy[:depth]
            agg = fdf.groupby(levels, as_index=False).agg(
                avg=(score_col, "mean"), n=(score_col, "size"))
            rendered = False
            if smooth:
                _renderer = (render_echarts_treemap if is_treemap
                             else render_echarts_sunburst)
                rendered = _renderer(
                    agg, levels,
                    vmin=float(agg["avg"].min()), vmax=float(agg["avg"].max()),
                    topk=topk)
            if not rendered:
                if is_treemap:
                    fig = px.treemap(agg, path=levels, values="n", color="avg",
                                     color_continuous_scale="RdYlGn",
                                     color_continuous_midpoint=df[score_col].mean(),
                                     height=650)
                    # marker.colors holds the (weighted) avg for every node,
                    # including parents — reuse it as on-tile text
                    _tr = fig.data[0]
                    _tr.text = [f"Avg score: {c:.1f}" if c == c else ""
                                for c in _tr.marker.colors]
                    fig.update_traces(
                        maxdepth=3,
                        marker=dict(cornerradius=4),
                        texttemplate="<b>%{label}</b><br>%{text}"
                                     "<br>%{value:,} records"
                                     "<br>%{percentParent:.0%} of %{parent}",
                        hovertemplate="<b>%{label}</b><br>%{text}"
                                      "<br>Records: %{value:,}"
                                      "<br>%{percentParent:.0%} of %{parent}"
                                      "<extra></extra>")
                else:
                    fig = px.sunburst(agg, path=levels, values="n", color="avg",
                                      color_continuous_scale="RdYlGn",
                                      color_continuous_midpoint=df[score_col].mean(),
                                      height=650)
                    fig.update_traces(maxdepth=3)
                fig.update_layout(margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig, use_container_width=True)
            st.caption(("Nesting" if is_treemap else "Ring order") + ": " + " → ".join(levels) +
                       " · size = records · color = avg score (red weak → green strong) · click to zoom"
                       + (f" · deeper levels ({' → '.join(hierarchy[depth:])}) via the Rings slider"
                          if depth < len(hierarchy) else ""))

            # ---------- Compare mode: multiple units side by side ----------
            st.divider()
            if st.toggle("🆚 Compare units side by side"):
                # Any categorical column with 2-6 values can be compared,
                # not just hierarchy levels (Gender, Grade, Year, ...)
                extra_dims = [c for c in (cols["categorical"] + cols["objective"])
                              if c not in hierarchy
                              and 2 <= df[c].nunique(dropna=True) <= 6]
                cmp_options = hierarchy[:-1] or hierarchy
                clevel = st.selectbox("Compare at level", cmp_options,
                                      key="cmp_level")
                below = hierarchy[hierarchy.index(clevel) + 1:] or [clevel]
                opts = sorted(df[clevel].dropna().unique().tolist())
                picks = st.multiselect(f"Pick 2–6 {clevel}s to compare",
                                       opts, default=opts[:min(4, len(opts))],
                                       max_selections=6, key="cmp_picks")
                # Optional refinement: narrow the comparison to a specific slice
                refine_dims = extra_dims
                rc1, rc2 = st.columns(2)
                rdim = rc1.selectbox("Refine by (optional)", ["None"] + refine_dims,
                                     key="cmp_refine_dim")
                base = df
                refine_note = ""
                if rdim != "None":
                    rvals = sorted(df[rdim].dropna().unique().tolist())
                    rval = rc2.selectbox(f"{rdim} =", rvals, key="cmp_refine_val")
                    base = df[df[rdim] == rval]
                    refine_note = f" · showing only {rdim} = {rval}"

                # Competency filter. The generic "Refine by" box above only lists
                # columns with 2-6 distinct values, so a 20-question competency
                # column can never appear there — yet "compare these blocks on
                # the numeracy questions only" is the most common thing asked of
                # this view. Blank = all, so the default stays uncluttered.
                if comp_col and comp_col in base.columns:
                    all_comps = sorted(base[comp_col].dropna().unique().tolist())
                    csel = st.multiselect(
                        f"Competencies — leave blank for all {len(all_comps)}",
                        all_comps, default=[], key="cmp_comps",
                        help="Restrict the comparison to specific skills. The "
                             "sunbursts, the metrics and the bar chart below all "
                             "follow this selection.")
                    if csel:
                        base = base[base[comp_col].isin(csel)]
                        shown = ", ".join(map(str, csel[:6])) + ("…" if len(csel) > 6 else "")
                        refine_note += (f" · {len(csel)} of {len(all_comps)} "
                                        f"competencies ({shown})")

                if base.empty:
                    st.warning("Nothing left after these filters — widen the "
                               "competency or refine selection.")
                elif len(picks) < 2:
                    st.info("Pick at least two to compare.")
                else:
                    both = base[base[clevel].isin(picks)]
                    overall = both[score_col].mean()
                    # Shared color range so all charts are fairly comparable
                    leaf_avg = both.groupby(below[-1])[score_col].mean()
                    rng = [float(leaf_avg.min()) - 2, float(leaf_avg.max()) + 2]

                    def make_sun(dsub):
                        a = dsub.groupby(below, as_index=False).agg(
                            avg=(score_col, "mean"), n=(score_col, "size"))
                        f = px.sunburst(a, path=below, values="n", color="avg",
                                        color_continuous_scale="RdYlGn",
                                        range_color=rng, height=380)
                        f.update_layout(margin=dict(t=5, b=5, l=5, r=5),
                                        coloraxis_showscale=False)
                        return f

                    per_row = 2
                    for i in range(0, len(picks), per_row):
                        row_picks = picks[i:i + per_row]
                        row_cols = st.columns(per_row)
                        for col, name in zip(row_cols, row_picks):
                            dsub = base[base[clevel] == name]
                            with col:
                                st.metric(name, f"{dsub[score_col].mean():.1f}",
                                          f"{dsub[score_col].mean() - overall:+.1f} vs group avg")
                                st.plotly_chart(make_sun(dsub), use_container_width=True)

                    if comp_col:
                        comp_cmp = (both.groupby([clevel, comp_col], as_index=False)[score_col]
                                    .mean().round(1))
                        _n_c = comp_cmp[comp_col].nunique()
                        figc = px.bar(comp_cmp, x=comp_col, y=score_col, color=clevel,
                                      barmode="group", height=360,
                                      title=f"Competency comparison across selected units "
                                            f"({_n_c} competenc{'y' if _n_c == 1 else 'ies'})")
                        st.plotly_chart(figc, use_container_width=True)
                    st.caption("All sunbursts share one color scale, so colors are directly "
                               "comparable. Δ under each name is vs the average of the "
                               "selected group. Compare mode ignores sidebar hierarchy filters."
                               + refine_note)
        else:
            st.warning("No hierarchy columns (Division/District/Block/Cluster) detected.")
    _tab0_fragment()
# ============================================================================
#  Tab — GKA's Impact   (is the programme helping? cohort + competency + units)
# ============================================================================
with tabs[1]:
    @st.fragment
    def _tab_gka_fragment():
        st.subheader("🎓 GKA's Impact")
        if items_df is None or not RQMAPS:
            st.info("This tab needs the GP-contest workbooks: per-child question "
                    "columns plus each paper's own 'Competency Mapping' sheet. "
                    "Load the year folders to enable it.")
            return

        _gk = _c_gka_analyse(_raw_for_layers, _fsig + str(_active_filters),
                             year_col or "Year", grade_col or "Grade",
                             _gka_unit_col(), _gka_gp_col(), MINN)
        if not _gk:
            st.warning("Not enough of the dataset is loaded to measure impact — "
                       "at least two years and two grades are needed.")
            return
        if _gk.get("too_short"):
            st.warning(f"Impact needs at least two years and two grades. "
                       f"Loaded: {len(_gk['years'])} year(s), "
                       f"{len(_gk['grades'])} grade(s).")
            return

        # ---------------------------------------------------------------- 0
        _flip = _gk.get("flip") or {}
        if _flip.get("flipped"):
            st.error(
                "⚠️ **The nine papers are not equated to each other.** Scores "
                "rise with grade in " + ", ".join(_flip["rises_in"]) +
                " and fall with grade in " + ", ".join(_flip["falls_in"]) +
                ". Children cannot reverse that ordering in one year — the "
                "papers were rewritten. **Every figure on this tab is therefore "
                "measured on the skills two papers share, or on a unit's "
                "standing among units sitting the same paper. No raw "
                "cross-year score change is reported as if it meant learning.**")

        g1, g2 = st.columns([3, 2])
        with g1:
            _grid = _gk["grid"]
            if _grid is not None and len(_grid):
                _f = px.line(_grid, x=grade_col or "Grade", y="pct",
                             color=year_col or "Year", markers=True,
                             labels={"pct": "% correct"},
                             title="Mean % correct on each paper")
                _f.update_layout(height=330,
                                 margin=dict(t=48, b=10, l=10, r=10))
                st.plotly_chart(_f, use_container_width=True)
                st.caption("Each line is one year. If these were the same test "
                           "the lines could not cross — that they do is the "
                           "proof the papers changed.")
        with g2:
            st.markdown("**Papers and what they share**")
            _rows = []
            for _c in _gk.get("cohorts", []):
                for _s in _c["steps"]:
                    _rows.append({
                        "Step": f"G{_s['a'][1]} {_s['a'][0]} → G{_s['b'][1]} {_s['b'][0]}",
                        "Shared skills": _s["meta"]["n_anchors"],
                        "Comparable": "✅" if _s["meta"].get("ok") else "❌"})
            if _rows:
                st.dataframe(pd.DataFrame(_rows).drop_duplicates(),
                             use_container_width=True, hide_index=True)
            st.caption(f"A step needs at least {L_gka.MIN_ANCHORS} shared "
                       "skills before any score change is computed.")

        st.divider()

        # ---------------------------------------------------------------- 1
        st.markdown("### 1 · Overall performance — did cohorts improve?")
        st.caption("A cohort is the same children a year older: Grade 4 in one "
                   "year is Grade 5 the next. The bars compare what the raw "
                   "score suggests with what the shared skills actually show.")
        _bars = []
        for _c in _gk.get("cohorts", []):
            for _s in _c["steps"]:
                _lbl = (f"G{_s['a'][1]} {_s['a'][0][2:]} → "
                        f"G{_s['b'][1]} {_s['b'][0][2:]}")
                if not _s["meta"].get("ok"):
                    continue
                _r = _s["row"]
                _bars.append({"Step": _lbl, "Measure": "Raw (not comparable)",
                              "Points": float(_r["raw_drift"]), "err": 0.0})
                _bars.append({"Step": _lbl, "Measure": "Anchored on shared skills",
                              "Points": float(_r["drift"]),
                              "err": 1.96 * float(_r["se"])})
        if _bars:
            _bd = pd.DataFrame(_bars).drop_duplicates(subset=["Step", "Measure"])
            _fb = px.bar(_bd, x="Step", y="Points", color="Measure",
                         barmode="group", error_y="err",
                         color_discrete_map={
                             "Raw (not comparable)": "#b9c6c0",
                             "Anchored on shared skills": "#0a5340"},
                         labels={"Points": "change in % correct"})
            _fb.add_hline(y=0, line_width=1, line_color="#666")
            _fb.update_layout(height=340, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(_fb, use_container_width=True)
        _blocked = [s for c in _gk.get("cohorts", []) for s in c["steps"]
                    if not s["meta"].get("ok")]
        if _blocked:
            st.warning(
                f"**{len(_blocked)} cohort step(s) cannot be measured at all.** "
                + " ".join(sorted({s["meta"]["reason"] for s in _blocked}))[:400])

        _bu = _gk.get("by_unit")
        if _bu is not None and len(_bu):
            _up = int((_bu["drift"] > 0).sum())
            _solid = int(((_bu["drift"] > 0) & (_bu["evidence"] > 0)).sum())
            k = st.columns(4)
            k[0].metric("Districts improving", f"{_up} of {len(_bu)}")
            k[1].metric("Beyond their own error", _solid)
            k[2].metric("Best", f"{_bu.iloc[0]['unit']}",
                        f"{float(_bu.iloc[0]['drift']):+.1f} pts")
            k[3].metric("Weakest", f"{_bu.iloc[-1]['unit']}",
                        f"{float(_bu.iloc[-1]['drift']):+.1f} pts")
            with st.expander("Every district on this step"):
                st.dataframe(
                    _bu.rename(columns={
                        "unit": "District", "raw_drift": "Raw change",
                        "drift": "Anchored change", "evidence": "Evidence",
                        "n_a": "Children before", "n_b": "Children after"})
                    [["District", "Children before", "Children after",
                      "Raw change", "Anchored change", "Evidence"]],
                    use_container_width=True, hide_index=True)

        st.divider()

        # ---------------------------------------------------------------- 2
        st.markdown("### 2 · Competency level — what is not growing?")
        if _gk.get("coverage_note"):
            st.caption(_gk["coverage_note"])
        _cp = _gk.get("competency")
        if _cp is not None:
            _s = _cp["summary"].copy()
            _npap = int(_s["papers"].max())
            _s["Tested in"] = _s["papers"].astype(str) + f" of {_npap} papers"
            _fc = px.bar(_s.sort_values("mean_vs_paper"),
                         x="mean_vs_paper", y="competency", orientation="h",
                         color="mean_vs_paper", color_continuous_scale="RdYlGn",
                         color_continuous_midpoint=0,
                         labels={"mean_vs_paper": "points vs its own paper's average",
                                 "competency": ""},
                         hover_data=["papers", "below_in"])
            _fc.update_layout(height=360, showlegend=False,
                              coloraxis_showscale=False,
                              margin=dict(t=20, b=10, l=10, r=10))
            st.plotly_chart(_fc, use_container_width=True)
            st.caption("Measured against **each paper's own average**, so a "
                       "harder or easier paper moves every competency together "
                       "and cancels out. A competency below zero in every "
                       "paper is a real, stable weakness.")
            st.dataframe(
                _s.rename(columns={
                    "competency": "Competency", "mean_vs_paper": "Avg vs paper",
                    "below_in": "Papers below average", "worst": "Worst",
                    "best": "Best"})
                [["Competency", "Tested in", "Avg vs paper",
                  "Papers below average", "Worst", "Best"]],
                use_container_width=True, hide_index=True)

        st.divider()

        # ---------------------------------------------------------------- 3
        for _lvl, _title, _what in (
                ("gp", "3 · GPs in danger", "GP"),
                ("district", "4 · Districts in danger", "district")):
            _d = _gk.get("danger_" + _lvl)
            st.markdown(f"### {_title}")
            if _d is None or _d.empty:
                st.success(f"No {_what} shows a sustained decline in standing "
                           f"over the period.")
                continue
            _strict = int(_d["strict_decline"].sum())
            _sig = int(_d["beyond_normal_movement"].sum())
            st.caption(
                f"Ranked by how far each {_what}'s standing fell among the "
                f"{_what}s sitting the **same paper**, so paper difficulty "
                f"cancels. {_strict} of these fell in **both** steps; "
                f"**{_sig} fell by more than {_what}s normally move** "
                f"(Benjamini–Hochberg corrected across every {_what} tested). "
                f"Rows without that mark are the largest falls, not proven ones.")
            _show = _d.rename(columns={
                "unit": _what.title(), "start": f"{_gk['years'][0]} %ile",
                "mid": f"{_gk['years'][1]} %ile",
                "end": f"{_gk['years'][-1]} %ile",
                "change": "Change", "shape": "Pattern",
                "children": "Children",
                "strict_decline": "Fell both steps",
                "beyond_normal_movement": "Beyond normal movement"})
            _cols = [_what.title(), f"{_gk['years'][0]} %ile",
                     f"{_gk['years'][1]} %ile", f"{_gk['years'][-1]} %ile",
                     "Change", "Pattern", "Children", "Fell both steps",
                     "Beyond normal movement"]
            st.dataframe(_show[_cols], use_container_width=True,
                         hide_index=True)
            _recs = L_gka.recommendations(_gk, _lvl, limit=10)
            if _recs:
                with st.expander(f"What to do about these {_what}s "
                                 f"({len(_recs)} actions)"):
                    for _r in _recs:
                        with st.container(border=True):
                            st.markdown(
                                f"**{_r['priority']}** · {_r['unit']} — "
                                f"{_r['shape']}")
                            st.write(_r["recommendation"])
                            st.caption(f"`{_r['rule_fired']}` · "
                                       f"~{_r['children']:,} children")
            _st = _gk.get("stagnant_" + _lvl)
            if _st is not None and len(_st):
                with st.expander(f"…and {len(_st)} {_what}(s) that never moved "
                                 f"and sit low"):
                    st.dataframe(_st, use_container_width=True,
                                 hide_index=True)
            st.divider()

        # ---------------------------------------------------------------- 5
        st.markdown("### 5 · What this means")
        _fnd = L_gka.findings(_gk, limit=12)
        for _i, _f2 in enumerate(_fnd, 1):
            st.markdown(f"**{_i}. `{_f2['category']}`** — {_f2['text']}")
            st.caption(f"↳ {_f2['evidence']}  ·  `{_f2['source']}`")

        st.markdown("### 6 · Recommendations to Akshara")
        for _r in L_gka.programme_recommendations(_gk):
            with st.container(border=True):
                st.markdown(f"**{_r['priority']}** · {_r['area']}")
                st.write(_r["recommendation"])
                st.caption(f"↳ {_r['why']}")

        _space = L_gka.combination_space()
        with st.expander("Method"):
            st.dataframe(pd.DataFrame(L_gka.describe()),
                         use_container_width=True, hide_index=True)
            st.caption(
                f"Unit-level actions compose {_space['base_actions']} base "
                f"actions (9 trajectory shapes × 4 severity bands, every cell "
                f"filled) with up to {_space['max_clauses_shown']} of "
                f"{_space['modifiers']} qualifying clauses — "
                f"{_space['distinct_outputs']:,} distinct outputs, counted "
                f"from the rules rather than hardcoded.")
            if L_gka.ERRORS:
                st.error(f"⚠️ {len(L_gka.ERRORS)} stage(s) raised: "
                         + "; ".join(f"{e['stage']}: {e['error']}"
                                     for e in L_gka.ERRORS[:3]))
    _tab_gka_fragment()



# ------------------------------- Trends --------------------------------------
with tabs[2]:
    # fragment: widgets inside this tab rerun only this tab
    @st.fragment
    def _tab1_fragment():
        if year_col:
            # ---- setup: % correct + grade/cohort awareness -----------------
            _qcols = [c for c in fdf.columns if re.fullmatch(r"[Qq]\d+", str(c))]
            _out_of = len(_qcols) if _qcols and fdf[score_col].max() <= len(_qcols) else None
            _grade_col = next((c for c in fdf.columns if str(c).lower() == "grade"), None)
            _as_pct = _out_of is not None

            def _yval(s):
                """mean score -> displayed value (% correct when possible)"""
                return s * 100.0 / _out_of if _as_pct else s
            _ylab = "% correct" if _as_pct else "Average score"

            tc1, tc2, tc3 = st.columns([1.2, 1.3, 0.8])
            level = tc1.selectbox("Trend by", hierarchy or ["(overall)"],
                                  key="trend_level")
            _modes = ["📅 Year over year"]
            if _grade_col and pd.api.types.is_numeric_dtype(fdf[_grade_col]) \
                    and fdf[_grade_col].nunique() > 1 and fdf[year_col].nunique() > 1:
                _modes.append("🎓 Cohort journey")
            mode = tc2.radio("View", _modes, horizontal=True, key="trend_mode")
            neon = tc3.toggle("✨ Neon style", value=True, key="neon_trend")
            MIN_N = MINN
            _pctile = False
            if RQMAPS and fdf[year_col].nunique() > 1:
                _pctile = st.toggle(
                    "🏁 Percentile among peers (paper-proof)",
                    value=False, key="trend_pctile",
                    help="The paper changes every year, so raw % mixes "
                         "learning with paper difficulty. This view shows "
                         "each unit's standing among all units in the SAME "
                         "year (same paper) — 50 = middle of the pack. "
                         "Movement in percentile is real relative change.")

            cohort_mode = mode.startswith("🎓")
            if cohort_mode and RQMAPS:
                _pctile = True      # three grades = three different papers;
                                    # raw scores across them are not comparable
            if cohort_mode:
                # cohort = the year this child was in the LOWEST grade we track.
                _g0 = int(fdf[_grade_col].min())
                _co = fdf.copy()
                _co["_cohort"] = _co[year_col] - (_co[_grade_col] - _g0)
                _co["_cohort"] = _co["_cohort"].astype(int)
                xdim, sdim = _grade_col, "_cohort"
                t = _co.groupby([xdim, sdim], as_index=False).agg(
                    _m=(score_col, "mean"), _n=(score_col, "size"))
                _sname = lambda c: f"Grade {_g0} in {int(c)}-{(int(c) + 1) % 100:02d}"
                xs = sorted(t[xdim].unique().tolist())
                _xfmt = lambda x: f"Grade {int(x)}"
            else:
                xdim, sdim = year_col, (level if level in fdf.columns else None)
                keys = [xdim] + ([sdim] if sdim else [])
                t = fdf.groupby(keys, as_index=False).agg(
                    _m=(score_col, "mean"), _n=(score_col, "size"))
                _sname = str
                xs = sorted(t[xdim].unique().tolist())
                _xfmt = lambda x: f"{int(x)}-{(int(x) + 1) % 100:02d}"

            _small = int((t["_n"] < MIN_N).sum())
            t = t[t["_n"] >= MIN_N]
            t["_y"] = _yval(t["_m"]).round(1)
            if _pctile and (sdim or cohort_mode):
                # rank among same-year (same-paper) peers -> 0-100 percentile
                t["_y"] = (t.groupby(xdim)["_m"].rank(pct=True) * 100).round(1)
                _ylab = (f"percentile among "
                         f"{'cohorts' if cohort_mode else level + 's'} "
                         "(same-year peers)")
                if cohort_mode:
                    st.caption("🏁 Cohort lines shown as **percentile among "
                               "cohorts sitting the same paper** — the only "
                               "fair way to follow children across three "
                               "different papers.")

            # ---- choose which lines to draw: biggest movers, not alphabetical
            _shown, capped, _delta = ["Average"], False, None
            _too_thin = False
            if sdim and t[sdim].nunique() > 0:
                _w = t.pivot_table(index=sdim, columns=xdim, values="_y")
                _w = _w[_w.count(axis=1) >= 2]           # need 2+ points for a line
                if len(_w):
                    _delta = (_w.ffill(axis=1).iloc[:, -1] - _w.bfill(axis=1).iloc[:, 0])
                    _order = _delta.abs().sort_values(ascending=False).index.tolist()
                    _shown = _order[:8]
                    capped = len(_order) > 8
                else:
                    # every cell fell under MIN_N (tiny sample at this level) —
                    # fall back to the overall average instead of crashing/blank
                    _too_thin = True
                    sdim = None
                    t = fdf.groupby(xdim, as_index=False).agg(
                        _m=(score_col, "mean"), _n=(score_col, "size"))
                    t = t[t["_n"] >= MIN_N]
                    t["_y"] = _yval(t["_m"]).round(1)
            if _too_thin:
                st.info(f"Groups at **{level}** level have fewer than {MIN_N} "
                        "students per year in the current selection — too thin "
                        "for honest trend lines. Showing the overall average; "
                        "narrow with sidebar filters or pick a higher level.")

            # ---- KPI strip: the one-line story ------------------------------
            _ov = fdf.groupby(year_col)[score_col].mean()
            if len(_ov) >= 2:
                _first, _last = _yval(_ov.iloc[0]), _yval(_ov.iloc[-1])
                kk1, kk2, kk3 = st.columns(3)
                kk1.metric(f"State avg {_ylab} · all students · {_xfmt(_ov.index[0]) if not cohort_mode else int(_ov.index[0])}",
                           f"{_first:.1f}")
                kk2.metric(f"State avg {_ylab} · all students · {_xfmt(_ov.index[-1]) if not cohort_mode else int(_ov.index[-1])}",
                           f"{_last:.1f}")
                kk3.metric("Change over the period (pts)", f"{_last - _first:+.1f}",
                           delta=f"{_last - _first:+.1f}")
                st.caption("⚠️ **A different paper is set every year** — "
                           "year-to-year changes mix real learning with "
                           "paper difficulty. Within-year comparisons "
                           "(district vs district, gender, equity) are "
                           "exact; cross-year lines are best read as "
                           "relative movement.")

            # ---- line chart -------------------------------------------------
            def _series_pts(name):
                if sdim:
                    sub = t[t[sdim] == name]
                else:
                    sub = t
                return [round(float(sub[sub[xdim] == x]["_y"].mean()), 1)
                        if (sub[xdim] == x).any() else None for x in xs]

            if neon:
                import json as _j
                import pathlib as _p
                import streamlit.components.v1 as _c
                _lib = _p.Path(__file__).parent / "echarts.min.js"
                NEON = ["#00e5ff", "#ff2d78", "#ffd166", "#b388ff", "#2dff9e",
                        "#ff7a2d", "#40c4ff", "#ff8a80"]
                series = []
                for i, name in enumerate(_shown):
                    col = NEON[i % len(NEON)]
                    series.append({
                        "name": _sname(name), "type": "line",
                        "data": _series_pts(name),
                        "connectNulls": True,
                        "smooth": True, "symbol": "circle", "symbolSize": 9,
                        "lineStyle": {"width": 3, "color": col,
                                      "shadowBlur": 14, "shadowColor": col},
                        "itemStyle": {"color": col, "borderColor": "#ffffff",
                                      "borderWidth": 2},
                        "areaStyle": {"opacity": 0.12, "color": col},
                        "emphasis": {"focus": "series",
                                     "lineStyle": {"width": 4.5}},
                    })
                _opt = {
                    "backgroundColor": "#ffffff",
                    "grid": {"left": 48, "right": 24, "top": 36, "bottom": 56},
                    "tooltip": {"trigger": "axis"},
                    "legend": {"bottom": 4, "textStyle": {"color": "#67707f"},
                               "type": "scroll"},
                    "xAxis": {"type": "category",
                              "data": [_xfmt(x) for x in xs],
                              "axisLine": {"lineStyle": {"color": "#c9d0da"}},
                              "axisLabel": {"color": "#67707f"}},
                    "yAxis": {"type": "value", "name": _ylab, "scale": True,
                              "nameTextStyle": {"color": "#67707f"},
                              "splitLine": {"lineStyle": {"color": "rgba(60,70,90,.12)"}},
                              "axisLabel": {"color": "#67707f"}},
                    "series": series,
                    "animationDuration": 800, "animationEasing": "cubicOut",
                }
                _opt["tooltip"]["valueFormatter"] = None  # set in JS below
                _html = f"""
                <div id="ntrend" style="width:100%;height:430px;"></div>
                <script>{_echarts_js()}</script>
                <script>
                  var nt = echarts.init(document.getElementById('ntrend'));
                  var _o = {_j.dumps(_opt)};
              _o.tooltip.valueFormatter = (v) => v == null ? '-' : Number(v).toFixed(1);
              nt.setOption(_o);
                  window.addEventListener('resize', () => nt.resize());
                </script>"""
                _c.html(_html, height=440)
            else:
                _tt = t[t[sdim].isin(_shown)] if sdim else t
                if sdim:
                    _tt = _tt.assign(**{sdim: _tt[sdim].map(_sname)})
                    fig = px.line(_tt, x=xdim, y="_y", color=sdim, markers=True)
                else:
                    fig = px.line(_tt, x=xdim, y="_y", markers=True)
                fig.update_traces(hovertemplate="%{x} · %{y:.1f}"
                                  "<extra>%{fullData.name}</extra>")
                fig.update_layout(yaxis_title=_ylab, legend_title="")
                fig.update_xaxes(tickvals=xs, ticktext=[_xfmt(x) for x in xs])
                st.plotly_chart(fig, use_container_width=True)

            _bits = []
            if cohort_mode:
                _bits.append("each line follows one cohort of children as they "
                             "move up grades — the honest way to read change in "
                             "longitudinal data")
            elif sdim:
                _bits.append(f"lines = the 8 biggest movers among "
                             f"{level}s (by change, not alphabetical)"
                             if capped else f"lines = {level}s")
            if _as_pct:
                _bits.append(f"y = % of {_out_of} items correct")
            if _small:
                _bits.append(f"{_small} tiny groups (<{MIN_N} students) hidden — "
                             "they make lines jump meaninglessly")
            if _bits:
                st.caption(" · ".join(_bits))

            # ---- improvers vs decliners ------------------------------------
            if (not cohort_mode) and sdim and _delta is not None and len(_delta):
                st.markdown("##### 📈 Improving vs declining "
                            f"({_xfmt(xs[0])} → {_xfmt(xs[-1])})")
                _d = _delta.dropna().sort_values()
                _show_d = pd.concat([_d.head(10), _d.tail(10)])
                _show_d = _show_d[~_show_d.index.duplicated()]
                _bar = _show_d.reset_index()
                _bar.columns = [level, "delta"]
                figd = px.bar(_bar, x="delta", y=level, orientation="h",
                              color="delta", color_continuous_scale="RdYlGn",
                              color_continuous_midpoint=0,
                              height=max(300, 26 * len(_bar) + 80),
                              labels={"delta": f"Change in {_ylab} "
                                               f"({_xfmt(xs[0])} → {_xfmt(xs[-1])})"})
                figd.update_traces(hovertemplate="<b>%{y}</b><br>%{x:+.1f} pts "
                                   "change in " + _ylab + "<extra></extra>")
                _avg_d = float(_d.mean())
                figd.add_vline(x=_avg_d, line_dash="dot", line_color="#67707f",
                               annotation_text=f"typical change {_avg_d:+.1f}",
                               annotation_font_color="#67707f")
                figd.update_layout(coloraxis_showscale=False,
                                   yaxis_title="", margin=dict(t=10, b=10))
                st.plotly_chart(figd, use_container_width=True)
                st.caption(f"**How to read:** each bar = one {level}; length = "
                           f"points its {_ylab} changed between {_xfmt(xs[0])} "
                           f"and {_xfmt(xs[-1])}. Top = fastest improvement, "
                           "bottom = slowest (declining would show red, leftward). "
                           f"Showing top 10 + bottom 10 of {len(_d)} eligible "
                           f"{level}s; groups under {MIN_N} students excluded. "
                           "⚠️ **The test paper changes every year** — "
                           "cross-year movement partly reflects paper "
                           "difficulty; read bars as relative (who moved "
                           "more than peers), not absolute learning gains.")
                with st.expander(f"Full table — every eligible {level}, every year"):
                    st.caption(f"Each row = one {level}. Yearly columns = its "
                               f"average {_ylab}. 'Change' = last year minus "
                               "first year — the number the bars above are drawn "
                               "from. Sorted smallest change first.")
                    pivot = t.pivot_table(index=sdim, columns=xdim, values="_y")
                    pivot.columns = [_xfmt(c) for c in pivot.columns]
                    _chg = f"Change ({_xfmt(xs[0])} → {_xfmt(xs[-1])})"
                    pivot[_chg] = (pivot.ffill(axis=1).iloc[:, -1]
                                   - pivot.bfill(axis=1).iloc[:, 0]).round(1)
                    st.dataframe(pivot.round(1).sort_values(_chg))
        else:
            st.info("No Year column detected — trend view unavailable.")
    _tab1_fragment()

# ------------------------------- Gender gap ----------------------------------
with tabs[3]:
    # fragment: widgets inside this tab rerun only this tab
    @st.fragment
    def _tab2_fragment():
        if gender_col and gender_col in fdf.columns:
            # normalize labels: F/f/Female -> Female, M/m/Male -> Male
            _gmap = {"f": "Female", "female": "Female", "girl": "Female",
                     "m": "Male", "male": "Male", "boy": "Male"}
            _gg = fdf.copy()
            _gg["_g"] = (_gg[gender_col].astype(str).str.strip().str.lower()
                         .map(_gmap))
            _gg = _gg[_gg["_g"].notna()]
            _qg = [c for c in _gg.columns if re.fullmatch(r"[Qq]\d+", str(c))
                   and pd.api.types.is_numeric_dtype(_gg[c])]
            _oog = len(_qg) if _qg and _gg[score_col].max() <= len(_qg) else None
            _pctg = (lambda s: s * 100.0 / _oog) if _oog else (lambda s: s)
            _ylabg = "% correct" if _oog else "Avg score"
            GMIN_N = MINN

            if not len(_gg):
                st.info("Could not read genders from "
                        f"'{gender_col}' (expected F/M or Female/Male values).")
            else:
                # ---- KPI strip ---------------------------------------------
                _f = _gg[_gg["_g"] == "Female"][score_col].mean()
                _m = _gg[_gg["_g"] == "Male"][score_col].mean()
                _fn = int((_gg["_g"] == "Female").sum())
                _mn = int((_gg["_g"] == "Male").sum())
                _gap = _pctg(_f) - _pctg(_m)
                if abs(_gap) < 0.05:
                    _gap = 0.0          # avoid a misleading "-0.0" display
                gk1, gk2, gk3, gk4 = st.columns(4)
                gk1.metric(f"👧 Girls · {_ylabg}", f"{_pctg(_f):.1f}",
                           help=f"{_fn:,} records")
                gk2.metric(f"👦 Boys · {_ylabg}", f"{_pctg(_m):.1f}",
                           help=f"{_mn:,} records")
                gk3.metric("Gap (Girls − Boys)", f"{_gap:+.1f}",
                           delta=f"{_gap:+.1f}",
                           help="Positive = girls ahead, negative = boys ahead")
                _lead = "Girls lead" if _gap > 0.05 else \
                        ("Boys lead" if _gap < -0.05 else "Even")
                gk4.metric("Overall", _lead)

                st.caption("**This tab tests for a gender gap three ways** — "
                           "by skill, over time, and by geography. Values are in "
                           "percentage points of % correct; positive = girls "
                           "ahead, negative = boys ahead.")
                _t0, _t1, _t2, _t3 = st.tabs(
                    ["👀 Simple view",
                     "🧩 Gap by skill" + ("" if comp_col or _qg else " (n/a)"),
                     "📅 Gap over time", "🗺️ Where is the gap?"])

                # ---- 0: the plain-language view -----------------------------
                # A diverging bar of "gap in percentage points" asks the reader
                # to reason about a signed difference, and hides the levels
                # entirely — a +2 gap looks the same whether both groups are at
                # 90% or at 30%. This shows BOTH actual values as two dots with
                # the gap as the line between them, and greys out differences
                # that are within chance so nobody reads noise as a finding.
                with _t0:
                    # Skills live in ONE OF TWO PLACES. Melted datasets put
                    # them in comp_col; datasets too large to melt keep them
                    # as Q1..Qn COLUMNS. Only the first was handled, so this
                    # panel said "needs a competency column" on every dataset
                    # over the melt limit — including the full 1M-row file —
                    # even though the questions were sitting right there.
                    dd, _sk_err = None, None
                    if comp_col and comp_col in _gg.columns:
                        # Two plain unstacks rather than one named aggregation:
                        # .agg(mean=..., n=...).unstack() builds a MultiIndex
                        # whose level order depends on the data, and blew up
                        # with KeyError('mean') when a stray lookup table left
                        # the competency column almost entirely null.
                        _gb = _gg.groupby([comp_col, "_g"])[COMP_VALUE_COL]
                        _mean = _gb.mean().unstack()
                        _cnt = _gb.size().unstack()
                        _have = set(map(str, _mean.columns))
                        if not {"Female", "Male"}.issubset(_have) or _mean.empty:
                            _sk_err = "Need both girls and boys present to compare."
                        else:
                            dd = pd.DataFrame({
                                "skill": _mean.index.astype(str),
                                "girls": _mean["Female"].to_numpy(float),
                                "boys":  _mean["Male"].to_numpy(float),
                                "ng":    _cnt["Female"].to_numpy(float),
                                "nb":    _cnt["Male"].to_numpy(float)})
                            dd = dd.dropna(subset=["girls", "boys"])
                            if not COMP_VALUE_IS_PCT:
                                dd["girls"] = _pctg(dd["girls"])
                                dd["boys"] = _pctg(dd["boys"])
                    elif _qg:
                        _gfw = _gg[_gg["_g"] == "Female"]
                        _gmw = _gg[_gg["_g"] == "Male"]
                        if not len(_gfw) or not len(_gmw):
                            _sk_err = "Need both girls and boys present to compare."
                        else:
                            # One row per CHILD here, so ng/nb are the group
                            # sizes and the significance test below is valid.
                            dd = pd.DataFrame({
                                "skill": [str(q) for q in _qg],
                                "girls": [float(_gfw[q].mean()) * 100 for q in _qg],
                                "boys":  [float(_gmw[q].mean()) * 100 for q in _qg],
                                "ng":    float(len(_gfw)),
                                "nb":    float(len(_gmw))})
                            dd = dd.dropna(subset=["girls", "boys"])
                    else:
                        _sk_err = ("This dataset has no per-question or "
                                   "competency columns, so the gap cannot be "
                                   "split by skill. The other three sub-tabs "
                                   "still work.")

                    # roll up to named competencies, or at least label with them
                    if dd is not None and len(dd) and QMAP:
                        _gsk = {competency_of(s) for s in dd["skill"]}
                        if len(_gsk) < len(dd):
                            _gby = st.radio(
                                "Break the gap down by",
                                ["Competency", "Question"], horizontal=True,
                                key="gg_by")
                            if _gby == "Competency":
                                dd["skill"] = dd["skill"].map(competency_of)
                                dd = (dd.groupby("skill", as_index=False)
                                        .agg(girls=("girls", "mean"),
                                             boys=("boys", "mean"),
                                             # children, not child-questions:
                                             # averaging over 4 questions does
                                             # not give 4x the sample
                                             ng=("ng", "min"), nb=("nb", "min")))
                            else:
                                dd["skill"] = [question_label(s)
                                               for s in dd["skill"]]

                    if dd is None or not len(dd):
                        st.info(_sk_err or "No skill-level data in this selection.")
                    else:
                        dd["gap"] = dd["girls"] - dd["boys"]

                        # Within ONE competency each row is one child, so
                        # these counts are genuinely independent and the
                        # test is valid (see units.py for why pooling
                        # across competencies would not be).
                        _flags, _ps = [], []
                        for _r in dd.itertuples():
                            _, _p, _ = proportion_test(
                                _r.girls, _r.boys,
                                max(int(_r.ng), 1), max(int(_r.nb), 1))
                            _ps.append(_p)
                            _flags.append(_p < 0.05)
                        dd["p"] = _ps
                        dd["real"] = _flags
                        dd = dd.sort_values("gap").reset_index(drop=True)

                        _gall = float(np.average(dd["girls"], weights=dd["ng"]))
                        _ball = float(np.average(dd["boys"], weights=dd["nb"]))
                        _nreal = int(dd["real"].sum())
                        _lead_w = ("girls" if _gall > _ball else "boys")
                        st.markdown(
                            f"#### Out of every 100 questions answered, "
                            f"girls get **{_gall:.0f}** right and boys get "
                            f"**{_ball:.0f}** right.")
                        st.caption(
                            f"That is a {abs(_gall-_ball):.1f}-point overall "
                            f"lead for {_lead_w}. Below, each row is one "
                            f"skill: the two dots are the two groups, and "
                            f"the bar between them is the gap. "
                            f"**{_nreal} of {len(dd)} skills** show a gap "
                            f"bigger than chance would explain — the rest "
                            f"are drawn in grey and should not be acted on.")

                        PINK, BLUE, GREY = "#ff2d78", "#00b4d8", "#67707f"
                        figd = go.Figure()
                        for _real, _clr, _nm in ((True, "#ffb703", "Real difference"),
                                                 (False, GREY, "Within chance")):
                            _sub = dd[dd["real"] == _real]
                            if _sub.empty:
                                continue
                            _xs, _ys = [], []
                            for _r in _sub.itertuples():
                                _xs += [_r.boys, _r.girls, None]
                                _ys += [_r.skill, _r.skill, None]
                            figd.add_trace(go.Scatter(
                                x=_xs, y=_ys, mode="lines", name=_nm,
                                line=dict(color=_clr, width=7),
                                opacity=0.55, hoverinfo="skip"))
                        for _col, _clr, _nm, _cnt in (
                                ("boys", BLUE, "👦 Boys", "nb"),
                                ("girls", PINK, "👧 Girls", "ng")):
                            figd.add_trace(go.Scatter(
                                x=dd[_col], y=dd["skill"], mode="markers",
                                name=_nm,
                                marker=dict(color=_clr, size=15,
                                            line=dict(color="white", width=1.5)),
                                customdata=np.stack([dd[_cnt], dd["p"]], axis=-1),
                                hovertemplate=("<b>%{y}</b><br>" + _nm +
                                               ": %{x:.1f}% correct<br>"
                                               "%{customdata[0]:,.0f} children"
                                               "<extra></extra>")))
                        figd.update_layout(
                            height=max(340, 26 * len(dd) + 130),
                            xaxis_title="% of questions answered correctly",
                            yaxis_title="", margin=dict(t=10, b=10, l=10, r=10),
                            legend=dict(orientation="h", yanchor="bottom",
                                        y=1.02, x=0),
                            hovermode="closest")
                        figd.update_xaxes(ticksuffix="%")
                        st.plotly_chart(figd, use_container_width=True)
                        st.caption(
                            "**How to read:** pink dot = girls, blue dot = "
                            "boys, on the same 0–100% scale. The further "
                            "right a dot, the better that group did. A long "
                            "bar between them means a big gap; a short bar "
                            "means they are close. **Yellow = the gap is "
                            "real; grey = too small to distinguish from "
                            "chance.** Unlike a gap-only chart, this also "
                            "shows whether both groups are doing well or "
                            "both are struggling.")

                        _wr = dd[dd["real"]]
                        if not _wr.empty:
                            _b = _wr.iloc[0]
                            _g = _wr.iloc[-1]
                            st.markdown(
                                f"- Biggest **real** boys' lead: **{_b.skill}** "
                                f"— boys {_b.boys:.0f}% vs girls {_b.girls:.0f}% "
                                f"({abs(_b.gap):.1f} pts)\n"
                                f"- Biggest **real** girls' lead: **{_g.skill}** "
                                f"— girls {_g.girls:.0f}% vs boys {_g.boys:.0f}% "
                                f"({abs(_g.gap):.1f} pts)")
                        else:
                            st.success(
                                "No skill shows a girls-vs-boys difference "
                                "beyond chance in this selection — on this "
                                "evidence, gender is not where the problem is.")

                        with st.expander("Show the numbers"):
                            _tbl = dd[["skill", "girls", "boys", "gap",
                                       "ng", "nb", "p", "real"]].copy()
                            _tbl.columns = ["Skill", "Girls %", "Boys %",
                                            "Gap (G−B)", "Girls (n)",
                                            "Boys (n)", "p-value",
                                            "Real difference?"]
                            # use_container_width, NOT width='stretch':
                            # st.dataframe takes width as an int until
                            # Streamlit 1.49, so a string raises TypeError.
                            st.dataframe(
                                _tbl.round({"Girls %": 1, "Boys %": 1,
                                            "Gap (G−B)": 1, "p-value": 4}),
                                use_container_width=True, hide_index=True)

                # ---- 1: per competency / per question item ------------------
                with _t1:
                    if comp_col and comp_col in _gg.columns:
                        # COMP_VALUE_COL, not score_col — see its definition.
                        # score_col here is the student's total, identical on
                        # every one of their competency rows, so this chart used
                        # to draw 20 bars all showing the same number.
                        _by = (_gg.groupby([comp_col, "_g"])[COMP_VALUE_COL]
                               .mean().unstack())
                        _dim = comp_col
                    elif _qg:
                        _gb_mode = "Item"
                        if QMAP or RQMAPS:
                            _gb_mode = st.radio(
                                "Break the gap down by", ["Competency", "Item"],
                                horizontal=True, key="gap_skill_mode")
                            if RQMAPS and _gb_mode == "Item":
                                st.caption("⚠️ The paper changes every year "
                                           "and grade — the same Q number is "
                                           "a DIFFERENT question in each. "
                                           "Competency view is the fair "
                                           "comparison.")
                        _rows = []
                        if _gb_mode == "Competency" and (QMAP or RQMAPS):
                            _covg = competency_coverage_note(RQMAPS)
                            if _covg:
                                st.caption(_covg)
                            _csf = comp_score_frame(_gg, _qg, RQMAPS, QMAP,
                                                    year_col or "Year",
                                                    grade_col or "Grade")
                            for comp in _csf.columns:
                                _rows.append({
                                    "Competency": comp,
                                    "Female": _csf.loc[_gg["_g"] == "Female",
                                                       comp].mean(),
                                    "Male": _csf.loc[_gg["_g"] == "Male",
                                                     comp].mean()})
                            _by = pd.DataFrame(_rows).set_index("Competency")
                            _dim = "Competency"
                        else:
                            for q in _qg:
                                _rows.append({
                                    "Item": q,
                                    "Female": _gg.loc[_gg["_g"] == "Female", q].mean() * 100,
                                    "Male": _gg.loc[_gg["_g"] == "Male", q].mean() * 100})
                            _by = pd.DataFrame(_rows).set_index("Item")
                            _dim = "Item"
                    else:
                        _by = None
                    if _by is not None and {"Female", "Male"}.issubset(_by.columns):
                        if _dim in ("Item", "Competency") or COMP_VALUE_IS_PCT:
                            # already 0-100, no rescaling needed
                            _by["gap"] = _by["Female"] - _by["Male"]
                        else:
                            _by["Female"] = _pctg(_by["Female"])
                            _by["Male"] = _pctg(_by["Male"])
                            _by["gap"] = _by["Female"] - _by["Male"]
                        _byp = _by.sort_values("gap").reset_index()
                        _byp["_who"] = np.where(
                            _byp["gap"] > 0.05, "Girls ahead by",
                            np.where(_byp["gap"] < -0.05, "Boys ahead by",
                                     "Even — gap only"))
                        figb = px.bar(_byp, x="gap", y=_dim, orientation="h",
                                      color="gap", color_continuous_scale="RdBu",
                                      color_continuous_midpoint=0,
                                      height=max(320, 22 * len(_byp) + 90),
                                      labels={"gap": "Gap in %-points (Girls − Boys)"})
                        figb.update_traces(customdata=np.stack(
                            [_byp["_who"], _byp["gap"].abs(),
                             _byp["Female"], _byp["Male"]], axis=-1),
                            hovertemplate="<b>%{y}</b><br>%{customdata[0]} "
                            "%{customdata[1]:.1f} pts<br>Girls %{customdata[2]:.1f}%"
                            " correct · Boys %{customdata[3]:.1f}%<extra></extra>")
                        figb.update_layout(coloraxis_showscale=False,
                                           yaxis_title="",
                                           margin=dict(t=10, b=10))
                        st.plotly_chart(figb, use_container_width=True)
                        st.caption("**How to read:** each bar = one test "
                                   "question. Blue right = girls answer it "
                                   "correctly more often; red left = boys; "
                                   "no bar = dead even.")
                        _wq = _byp.iloc[0]; _bq = _byp.iloc[-1]
                        st.caption(
                            f"Blue right = girls stronger, red left = boys stronger. "
                            f"Biggest girls' lead: **{_bq[_dim]}** ({_bq['gap']:+.1f}); "
                            f"biggest boys' lead: **{_wq[_dim]}** ({_wq['gap']:+.1f}). "
                            + ("Each bar = one assessment item (% answering "
                               "correctly)." if _dim == "Item" else ""))
                    else:
                        st.info("No competency column or Q-item columns to break "
                                "the gap down by skill.")

                # ---- 2: gap over years --------------------------------------
                with _t2:
                    if year_col and _gg[year_col].nunique() > 1:
                        _yy = (_gg.groupby([year_col, "_g"])[score_col].mean()
                               .unstack())
                        if {"Female", "Male"}.issubset(_yy.columns):
                            _yy = _pctg(_yy)
                            _yy["gap"] = _yy["Female"] - _yy["Male"]
                            _yl = _yy.reset_index()
                            _yl["Year"] = _yl[year_col].map(
                                lambda x: f"{int(x)}-{(int(x) + 1) % 100:02d}")
                            _mlt = _yl.melt(id_vars=["Year"],
                                            value_vars=["Female", "Male"],
                                            var_name="Gender", value_name=_ylabg)
                            figt = px.line(_mlt, x="Year", y=_ylabg,
                                           color="Gender", markers=True,
                                           color_discrete_map={
                                               "Female": "#ff2d78",
                                               "Male": "#00e5ff"})
                            figt.update_traces(
                                line_width=3, marker_size=10,
                                hovertemplate="%{x} · %{y:.1f}% correct"
                                              "<extra>%{fullData.name}</extra>")
                            st.plotly_chart(figt, use_container_width=True)
                            st.caption("**How to read:** one line per gender, "
                                       "average % correct each year. A real gap "
                                       "would show as the lines visibly "
                                       "separating.")
                            _g0v, _g1v = _yl["gap"].iloc[0], _yl["gap"].iloc[-1]
                            _dirw = ("narrowed" if abs(_g1v) < abs(_g0v)
                                     else "widened" if abs(_g1v) > abs(_g0v)
                                     else "held steady")
                            st.caption(f"The gap {_dirw}: {_g0v:+.1f} pts in "
                                       f"{_yl['Year'].iloc[0]} → {_g1v:+.1f} pts "
                                       f"in {_yl['Year'].iloc[-1]}.")
                    else:
                        st.info("Need at least two years of data for a gap trend.")

                # ---- 3: gap by geography ------------------------------------
                with _t3:
                    if hierarchy:
                        _lvl = st.selectbox("Level", hierarchy, key="gap_level")
                        _geo = (_gg.groupby([_lvl, "_g"])[score_col]
                                .agg(["mean", "size"]).unstack())
                        _mean = _geo["mean"]
                        _size = _geo["size"]
                        if {"Female", "Male"}.issubset(_mean.columns):
                            _ok = ((_size["Female"] >= GMIN_N)
                                   & (_size["Male"] >= GMIN_N))
                            _hid = int((~_ok).sum())
                            _gd = (_pctg(_mean.loc[_ok, "Female"])
                                   - _pctg(_mean.loc[_ok, "Male"]))
                            if len(_gd):
                                _gd = _gd.sort_values()
                                _show = pd.concat([_gd.head(10), _gd.tail(10)])
                                _show = _show[~_show.index.duplicated()]
                                _gp = _show.reset_index()
                                _gp.columns = [_lvl, "gap"]
                                figg = px.bar(
                                    _gp, x="gap", y=_lvl, orientation="h",
                                    color="gap", color_continuous_scale="RdBu",
                                    color_continuous_midpoint=0,
                                    height=max(300, 26 * len(_gp) + 80),
                                    labels={"gap":
                                            "Gap in %-points (Girls − Boys)"})
                                figg.update_traces(hovertemplate=
                                    "<b>%{y}</b><br>%{x:+.1f} pts "
                                    "(positive = girls ahead)<extra></extra>")
                                figg.update_layout(coloraxis_showscale=False,
                                                   yaxis_title="",
                                                   margin=dict(t=10, b=10))
                                st.plotly_chart(figg, use_container_width=True)
                                st.caption("**How to read:** most girl-leaning "
                                           "units at top (blue), most boy-leaning "
                                           "at bottom (red); balanced units "
                                           "omitted — every unit is analyzed, "
                                           "the chart shows only the extremes.")
                                st.caption(
                                    "Most boy-leaning and most girl-leaning "
                                    f"{_lvl}s. " +
                                    (f"{_hid} unit(s) hidden (<{GMIN_N} records "
                                     "per gender)." if _hid else ""))
                            else:
                                st.info(f"All {_lvl}s have fewer than {GMIN_N} "
                                        "records per gender — pick a higher "
                                        "level or widen filters.")
                    else:
                        st.info("No hierarchy columns detected.")

                with st.expander("Full gender table"):
                    _keys = ([comp_col] if comp_col else []) or None
                    _tbl = (_gg.groupby(([comp_col] if comp_col else
                                         ([year_col] if year_col else ["_g"]))
                                        + (["_g"] if comp_col or year_col else []))
                            [score_col].mean().unstack() if (comp_col or year_col)
                            else _gg.groupby("_g")[score_col].mean().to_frame().T)
                    st.dataframe(_pctg(_tbl).round(1))
        else:
            st.info("Gender column not detected — pick it in the sidebar under "
                    "'Column roles'.")
    _tab2_fragment()

# ------------------------------- Competencies --------------------------------
with tabs[4]:
    if comp_col:
        c1, c2 = st.columns(2)
        with c1:
            # COMP_VALUE_COL: score_col is the student total, identical on
            # every competency row, so this bar chart used to show one flat
            # value repeated across all competencies.
            comp = fdf.groupby(comp_col, as_index=False)[COMP_VALUE_COL].mean()
            comp = comp.rename(columns={COMP_VALUE_COL: score_col})
            import json as _jb
            import pathlib as _pb
            import streamlit.components.v1 as _cb
            _libb = _pb.Path(__file__).parent / "echarts.min.js"
            PASTEL = ["#ff6b9d", "#7bf1c0", "#b3b8f7", "#6bd5f5",
                      "#ffd166", "#c792ea", "#8affc1", "#ffa07a"]
            bar_data = [{"value": round(float(r[score_col]), 1),
                         "itemStyle": {"color": PASTEL[i % len(PASTEL)],
                                       "borderRadius": [6, 6, 0, 0]}}
                        for i, (_, r) in enumerate(comp.iterrows())]
            _optb = {
                "backgroundColor": "#ffffff",
                "grid": {"left": 16, "right": 16, "top": 30, "bottom": 40},
                "tooltip": {"trigger": "item",
                            "formatter": "{b}: {c}"},
                "xAxis": {"type": "category",
                          "data": comp[comp_col].astype(str).str.upper().tolist(),
                          "axisLine": {"show": False}, "axisTick": {"show": False},
                          "axisLabel": {"color": "#26303e", "fontWeight": "bold",
                                        "fontSize": 11}},
                "yAxis": {"show": False},
                "series": [{
                    "type": "bar", "data": bar_data, "barWidth": "62%",
                    "label": {"show": True, "position": "inside",
                              "color": "#26303e", "fontWeight": "bold",
                              "fontSize": 14, "formatter": "{c}"},
                    "emphasis": {"itemStyle": {"shadowBlur": 16,
                                               "shadowColor": "rgba(255,255,255,.35)"}},
                    "animationDuration": 700, "animationEasing": "cubicOut",
                }],
            }
            _htmlb = f"""
            <div id="cbar" style="width:100%;height:360px;"></div>
            <script>{_echarts_js()}</script>
            <script>
              var cb = echarts.init(document.getElementById('cbar'));
              cb.setOption({_jb.dumps(_optb)});
              window.addEventListener('resize', () => cb.resize());
            </script>"""
            _cb.html(_htmlb, height=372)
        with c2:
            band = fdf.groupby(["Performance_Band"], observed=True, as_index=False).size()
            BAND_ORDER = ["Poor", "Below Average", "Average", "Above Average", "Excellent"]
            BAND_COLORS = {"Poor": "#d73027", "Below Average": "#fc8d59",
                           "Average": "#fee08b", "Above Average": "#91cf60",
                           "Excellent": "#1a9850"}
            band["Performance_Band"] = band["Performance_Band"].astype(str)
            band = (band.set_index("Performance_Band")
                    .reindex(BAND_ORDER).dropna().reset_index())
            if st.toggle("🧊 3D pie", value=True, key="pie3d"):
                import json
                import streamlit.components.v1 as components
                slices = [{"label": r["Performance_Band"], "value": int(r["size"]),
                           "color": BAND_COLORS[r["Performance_Band"]]}
                          for _, r in band.iterrows()]
                legend = " &nbsp; ".join(
                    f"<span style='color:{s['color']}'>&#9632;</span> {s['label']}"
                    for s in slices)
                html = f"""
                <canvas id="p3d" width="420" height="330"
                        style="display:block;margin:0 auto;"></canvas>
                <div style="text-align:center;font-family:sans-serif;font-size:12.5px;
                            color:#ccc;padding-top:2px;">{legend}</div>
                <script>
                const data = {json.dumps(slices)};
                const cv = document.getElementById('p3d'), ctx = cv.getContext('2d');
                const cx = 210, cy = 150, R = 150, squash = 0.55, depth = 26, POP = 14;
                const total = data.reduce((a, d) => a + d.value, 0);
                function shade(hex, f) {{
                  const n = parseInt(hex.slice(1), 16);
                  const r = Math.round(((n >> 16) & 255) * f),
                        g = Math.round(((n >> 8) & 255) * f),
                        b = Math.round((n & 255) * f);
                  return `rgb(${{r}},${{g}},${{b}})`;
                }}
                let angles = [], a = -Math.PI / 2;
                for (const d of data) {{
                  const sweep = d.value / total * 2 * Math.PI;
                  angles.push([a, a + sweep, d]); a += sweep;
                }}
                function draw(hover) {{
                  ctx.clearRect(0, 0, cv.width, cv.height);
                  const off = i => {{
                    if (i !== hover) return [0, 0];
                    const m = (angles[i][0] + angles[i][1]) / 2;
                    return [POP * Math.cos(m), POP * squash * Math.sin(m)];
                  }};
                  // side walls (front half), darker
                  angles.forEach(([a0, a1, d], i) => {{
                    const [ox, oy] = off(i);
                    ctx.beginPath(); let drew = false;
                    for (let t = a0; t <= a1 + 1e-9; t += 0.02) {{
                      if (Math.sin(t) < 0) continue;
                      const x = cx + ox + R * Math.cos(t),
                            y = cy + oy + R * squash * Math.sin(t);
                      if (!drew) {{ ctx.moveTo(x, y + depth); drew = true; }}
                      else ctx.lineTo(x, y + depth);
                    }}
                    if (!drew) return;
                    for (let t = a1; t >= a0 - 1e-9; t -= 0.02) {{
                      if (Math.sin(t) < 0) continue;
                      ctx.lineTo(cx + ox + R * Math.cos(t),
                                 cy + oy + R * squash * Math.sin(t));
                    }}
                    ctx.closePath();
                    ctx.fillStyle = shade(d.color, i === hover ? 0.72 : 0.62);
                    ctx.fill();
                  }});
                  // top faces
                  angles.forEach(([a0, a1, d], i) => {{
                    const [ox, oy] = off(i);
                    ctx.beginPath(); ctx.moveTo(cx + ox, cy + oy);
                    for (let t = a0; t <= a1 + 1e-9; t += 0.02)
                      ctx.lineTo(cx + ox + R * Math.cos(t),
                                 cy + oy + R * squash * Math.sin(t));
                    ctx.closePath();
                    ctx.fillStyle = i === hover ? shade(d.color, 1.12) : d.color;
                    ctx.fill();
                    ctx.strokeStyle = 'rgba(0,0,0,.35)'; ctx.lineWidth = 1; ctx.stroke();
                  }});
                  // labels
                  ctx.font = 'bold 13px sans-serif'; ctx.textAlign = 'center';
                  angles.forEach(([a0, a1, d], i) => {{
                    const [ox, oy] = off(i);
                    const m = (a0 + a1) / 2,
                          pct = (d.value / total * 100).toFixed(1) + '%';
                    ctx.fillStyle = '#111';
                    ctx.fillText(pct, cx + ox + R * 0.62 * Math.cos(m),
                                 cy + oy + R * squash * 0.62 * Math.sin(m) + 4);
                  }});
                  // hover tooltip
                  if (hover != null) {{
                    const d = angles[hover][2],
                          pct = (d.value / total * 100).toFixed(1);
                    const msg = d.label + ': ' + d.value.toLocaleString() +
                                ' (' + pct + '%)';
                    ctx.font = '12.5px sans-serif';
                    const w = ctx.measureText(msg).width + 16;
                    ctx.fillStyle = 'rgba(20,20,30,.92)';
                    ctx.beginPath();
                    ctx.roundRect((cv.width - w) / 2, 6, w, 24, 6); ctx.fill();
                    ctx.fillStyle = '#fff'; ctx.textAlign = 'center';
                    ctx.fillText(msg, cv.width / 2, 22);
                  }}
                }}
                function pick(mx, my) {{
                  const dx = mx - cx, dy = (my - cy) / squash;
                  if (Math.sqrt(dx * dx + dy * dy) > R + POP) return null;
                  let t = Math.atan2(dy, dx);
                  // normalize into [-PI/2, 3PI/2) to match slice angles
                  if (t < -Math.PI / 2) t += 2 * Math.PI;
                  for (let i = 0; i < angles.length; i++)
                    if (t >= angles[i][0] && t < angles[i][1]) return i;
                  return null;
                }}
                let cur = null;
                cv.addEventListener('mousemove', e => {{
                  const r = cv.getBoundingClientRect();
                  const h = pick((e.clientX - r.left) * cv.width / r.width,
                                 (e.clientY - r.top) * cv.height / r.height);
                  if (h !== cur) {{ cur = h; draw(cur); }}
                  cv.style.cursor = h == null ? 'default' : 'pointer';
                }});
                cv.addEventListener('mouseleave', () => {{ cur = null; draw(null); }});
                draw(null);
                </script>"""
                components.html(html, height=372)
            else:
                import json as _json
                import pathlib as _pl
                import streamlit.components.v1 as _components
                _lib = _pl.Path(__file__).parent / "echarts.min.js"
                pie_data = [{"name": r["Performance_Band"], "value": int(r["size"]),
                             "itemStyle": {"color": BAND_COLORS[r["Performance_Band"]]}}
                            for _, r in band.iterrows()]
                _opt = {
                    "tooltip": {"trigger": "item",
                                "formatter": "{b}: {c} ({d}%)"},
                    "legend": {"bottom": 0, "textStyle": {"color": "#67707f"}},
                    "series": [{
                        "type": "pie", "radius": ["42%", "72%"],
                        "center": ["50%", "46%"],
                        "data": pie_data,
                        "label": {"formatter": "{d}%", "color": "#ddd"},
                        "itemStyle": {"borderColor": "rgba(0,0,0,.4)",
                                      "borderWidth": 1},
                        "emphasis": {
                            "scale": True, "scaleSize": 14,
                            "itemStyle": {"shadowBlur": 18,
                                          "shadowColor": "rgba(0,0,0,0.45)"}},
                        "animationType": "scale",
                        "animationEasing": "elasticOut",
                        "animationDuration": 700,
                    }],
                }
                _html = f"""
                <div id="pie2d" style="width:100%;height:360px;"></div>
                <script>{_echarts_js()}</script>
                <script>
                  var p2 = echarts.init(document.getElementById('pie2d'));
                  p2.setOption({_json.dumps(_opt)});
                  window.addEventListener('resize', () => p2.resize());
                </script>"""
                _components.html(_html, height=372)
        st.divider()
    # (original competency charts above run only when a Competency column
    #  exists; the mastery view below works for every data shape)
    # Mastery view: how many children are at each proficiency band, by grade,
    # over time, and which skills lag at which grade. (Per-item difficulty
    # lives in the Item Analysis tab; this tab is about STUDENTS.)
    _qm = [c for c in fdf.columns if re.fullmatch(r"[Qq]\d+", str(c))
           and pd.api.types.is_numeric_dtype(fdf[c])]
    _oom = len(_qm) if _qm and fdf[score_col].max() <= len(_qm) else None
    _grade_c = next((c for c in fdf.columns if str(c).lower() == "grade"), None)

    if _oom or fdf[score_col].max() <= 100:
        _md = fdf.copy()
        _md["_pct"] = (_md[score_col] * 100.0 / _oom) if _oom else _md[score_col]
        BANDS = [("🔴 Struggling", 0, 40, "#e05252"),
                 ("🟠 Developing", 40, 60, "#f2a44a"),
                 ("🟡 Proficient", 60, 80, "#e8d44d"),
                 ("🟢 Advanced", 80, 101, "#57c26b")]
        def _band(p):
            for name, lo, hi, _ in BANDS:
                if lo <= p < hi:
                    return name
            return BANDS[-1][0]
        _md["_band"] = _md["_pct"].map(_band)
        _order = [b[0] for b in BANDS]
        _cmap = {b[0]: b[3] for b in BANDS}

        # ---- KPI strip ---------------------------------------------------
        _tot = len(_md)
        mk = st.columns(4)
        for i, (name, lo, hi, _c) in enumerate(BANDS):
            _n = int((_md["_band"] == name).sum())
            mk[i].metric(f"{name} ({lo}–{hi if hi <= 100 else 100}%)",
                         f"{_n * 100.0 / _tot:.0f}%", help=f"{_n:,} students")

        cm1, cm2 = st.columns([1, 1.3])

        # ---- donut: overall mastery mix ---------------------------------
        with cm1:
            _mix = (_md["_band"].value_counts().reindex(_order)
                    .fillna(0).reset_index())
            _mix.columns = ["Band", "Students"]
            figp = px.pie(_mix, names="Band", values="Students", hole=0.55,
                          color="Band", color_discrete_map=_cmap, height=360)
            figp.update_traces(textinfo="percent",
                               texttemplate="%{percent:.1%}",
                               hovertemplate="%{label}: %{percent:.1%} of all "
                               "students (%{value:,})<extra></extra>",
                               marker=dict(line=dict(color="#ffffff", width=2)))
            figp.update_layout(margin=dict(t=24, b=8, l=8, r=8),
                               legend=dict(orientation="h", y=-0.08),
                               paper_bgcolor="rgba(0,0,0,0)",
                               font=dict(color="#26303e"))
            st.plotly_chart(figp, use_container_width=True)

        # ---- stacked bar: mastery mix per grade -------------------------
        with cm2:
            if _grade_c:
                _bg = (_md.groupby([_grade_c, "_band"]).size()
                       .rename("n").reset_index())
                _bg["pct"] = (_bg["n"] * 100.0
                              / _bg.groupby(_grade_c)["n"].transform("sum"))
                _bg["Grade"] = "Grade " + _bg[_grade_c].astype(int).astype(str)
                figs = px.bar(_bg, x="Grade", y="pct", color="_band",
                              color_discrete_map=_cmap,
                              category_orders={"_band": _order},
                              height=360,
                              labels={"pct": "% of students", "_band": ""})
                figs.update_traces(hovertemplate="%{x} · %{y:.1f}% of "
                                   "students<extra>%{fullData.name}</extra>")
                figs.update_layout(barmode="stack",
                                   margin=dict(t=24, b=8, l=8, r=8),
                                   legend=dict(orientation="h", y=-0.14),
                                   paper_bgcolor="rgba(0,0,0,0)",
                                   font=dict(color="#26303e"))
                st.plotly_chart(figs, use_container_width=True)
                st.caption("**How to read:** each column = 100% of that "
                           "grade's students, sliced by their own score. "
                           "Healthy progress = orange shrinking and green "
                           "growing as grades rise.")
            else:
                st.info("No Grade column — per-grade mastery unavailable.")

        # ---- mastery over time ------------------------------------------
        if year_col and _md[year_col].nunique() > 1:
            st.markdown("##### 📅 Mastery over time")
            _bt = (_md.groupby([year_col, "_band"]).size()
                   .rename("n").reset_index())
            _bt["pct"] = (_bt["n"] * 100.0
                          / _bt.groupby(year_col)["n"].transform("sum"))
            _bt["Year"] = _bt[year_col].map(
                lambda x: f"{int(x)}-{(int(x) + 1) % 100:02d}")
            figt = px.bar(_bt, x="Year", y="pct", color="_band",
                          color_discrete_map=_cmap,
                          category_orders={"_band": _order}, height=320,
                          labels={"pct": "% of students", "_band": ""})
            figt.update_traces(hovertemplate="%{x} · %{y:.1f}% of "
                               "students<extra>%{fullData.name}</extra>")
            figt.update_layout(barmode="stack",
                               margin=dict(t=8, b=8, l=8, r=8),
                               legend=dict(orientation="h", y=-0.2),
                               paper_bgcolor="rgba(0,0,0,0)",
                               font=dict(color="#26303e"))
            st.plotly_chart(figt, use_container_width=True)
            st.caption("**How to read:** each column = one year, all "
                       "students sliced by score band. Orange shrinking + "
                       "green growing = children moving up bands year on "
                       "year.")
            _adv = _bt[_bt["_band"].isin(_order[2:])].groupby("Year")["pct"].sum()
            if len(_adv) >= 2:
                st.caption(f"Proficient-or-better: {_adv.iloc[0]:.0f}% in "
                           f"{_adv.index[0]} → {_adv.iloc[-1]:.0f}% in "
                           f"{_adv.index[-1]} "
                           f"({_adv.iloc[-1] - _adv.iloc[0]:+.0f} pts).")

        # ---- skill × grade heatmap --------------------------------------
        if _qm and _grade_c:
            st.markdown("##### 🔥 Skill × grade — where learning lags")
            _hm_mode = "Item"
            if QMAP or RQMAPS:
                _hm_mode = st.radio("Columns", ["Competency", "Item"],
                                    horizontal=True, key="hm_mode")
                if RQMAPS and _hm_mode == "Item":
                    st.caption("⚠️ Q numbers are different questions in "
                               "each year & grade — compare by Competency.")
            if _hm_mode == "Competency" and RQMAPS:
                _csf_hm = comp_score_frame(_md, _qm, RQMAPS, None,
                                           year_col or "Year",
                                           grade_col or "Grade")
                _hm = (_csf_hm.groupby(
                    pd.to_numeric(_md[_grade_c], errors="coerce"))
                    .mean().round(0).dropna(axis=1, how="all"))
                _cov = competency_coverage_note(RQMAPS)
                if _cov:
                    st.caption(_cov)
            else:
                _hm = (_md.groupby(_grade_c)[_qm].mean().mul(100).round(0))
                if _hm_mode == "Competency" and QMAP:
                    _hm = _hm.T.groupby(
                        lambda q: QMAP.get(str(q), str(q))).mean().T.round(0)
                    if COMP_ORDER:
                        _hm = _hm[[c for c in COMP_ORDER
                                   if c in _hm.columns]]
            _hm.index = ["Grade " + str(int(i)) for i in _hm.index]
            fighm = px.imshow(_hm, color_continuous_scale="RdYlGn",
                              range_color=[0, 100], aspect="auto",
                              height=90 + 60 * len(_hm),
                              labels={"x": "Item", "color": "% correct"},
                              text_auto=True)
            fighm.update_traces(hovertemplate="%{y} · %{x}: %{z:.0f}% "
                                "answer correctly<extra></extra>")
            fighm.update_layout(margin=dict(t=8, b=8, l=8, r=8),
                                paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#26303e"),
                                coloraxis_showscale=False)
            st.plotly_chart(fighm, use_container_width=True)
            st.caption("**How to read:** columns = test questions, rows = "
                       "grades, each cell = % of that grade answering "
                       "correctly. Orange columns = hard skills for "
                       "everyone; a flat column down the rows = a skill "
                       "children aren't gaining as they move up grades.")
            _lag = _hm.min().sort_values()
            st.caption(f"Red cells = skills weak at that grade. Weakest "
                       f"overall: **{_lag.index[0]}** "
                       f"({_lag.iloc[0]:.0f}% at its worst grade). "
                       "Per-item deep dive: Item Analysis tab.")
    else:
        st.info("Could not derive a % score (no Q-item columns and score "
                "isn't on a 0–100 scale) — mastery view unavailable.")

# ------------------------------- Deep dive -----------------------------------
with tabs[5]:
    # fragment: widgets inside this tab rerun only this tab
    @st.fragment
    def _tab4_fragment():
        # show=False: no filter widgets here, just the shape-independent
        # plumbing. This tab previously looked for Q1..Qn COLUMNS, which only
        # exist when the melt was skipped — on the melted dataset the
        # per-question chart below never rendered at all.
        QF = competency_question_filter(fdf, "dd", show=False)
        ddf, _dcol, _dlab = QF["frame"], QF["col"], QF["label"]
        _dpct = lambda s: s            # _dcol is already on the right scale
        DMIN_N = MINN
        # group the skill breakdown by named competency when a map is loaded
        _dd_by = "Question"
        if QF["kind"] and QMAP and len(QF["competencies"]) < len(QF["questions"]):
            _dd_by = st.radio("Break skills down by", ["Competency", "Question"],
                              horizontal=True, key="dd_by")

        # ================= A) Unit report card (drill-down) ===================
        st.markdown("#### 🔬 Unit report card")
        # one row is one ANSWER after the melt, so count children by id
        _dsid = sid_col if (sid_col and sid_col in ddf.columns) else None
        _cnt = ((_dsid, "nunique") if _dsid else (_dcol, "size"))
        _cntlab = "Students" if _dsid else ("Responses" if QF["kind"] == "long"
                                            else "Students")

        da, db = st.columns([1, 1.6])
        dlevel = da.selectbox("Level", hierarchy, key="dd_level") if hierarchy else None
        if dlevel:
            _units = (ddf.groupby(dlevel).agg(mean=(_dcol, "mean"), size=_cnt)
                      .query("size >= @DMIN_N").sort_values("mean"))
            if len(_units):
                dunit = db.selectbox(
                    f"{dlevel} (sorted weakest → strongest, "
                    f"min {DMIN_N} students)", _units.index.tolist(), key="dd_unit")
                _sub = ddf[ddf[dlevel] == dunit]
                _rank = int((_units["mean"] < _units.loc[dunit, "mean"]).sum()) + 1
                _gapv = _dpct(_sub[_dcol].mean()) - _dpct(ddf[_dcol].mean())
                dk = st.columns(4)
                dk[0].metric(_dlab, f"{_dpct(_sub[_dcol].mean()):.1f}")
                dk[1].metric(_cntlab, f"{int(_units.loc[dunit, 'size']):,}")
                dk[2].metric(f"Rank among {len(_units)} {dlevel}s",
                             f"#{len(_units) - _rank + 1}",
                             help="1 = strongest")
                dk[3].metric("vs overall", f"{_gapv:+.1f}",
                             delta=f"{_gapv:+.1f}")

                dc1, dc2 = st.columns(2)
                with dc1:
                    # a 0/100 item column has only two values — a histogram of
                    # it is two bars and says nothing. Show the distribution of
                    # each CHILD's percentage instead.
                    if QF["is_pct"] and _dsid:
                        _hsrc = _sub.groupby(_dsid)[_dcol].mean()
                    else:
                        _hsrc = _sub[_dcol]
                    _hall = (ddf.groupby(_dsid)[_dcol].mean().mean()
                             if (QF["is_pct"] and _dsid) else ddf[_dcol].mean())
                    _hd = pd.DataFrame({
                        "pct": _dpct(_hsrc.astype(float))})
                    fighd = px.histogram(_hd, x="pct", nbins=20, height=330,
                                         labels={"pct": _dlab})
                    fighd.add_vline(x=float(_dpct(_hall)),
                                    line_dash="dash", line_color="#e8d44d",
                                    annotation_text="overall avg",
                                    annotation_font_color="#e8d44d")
                    fighd.update_traces(hovertemplate="%{y:,} students "
                                        "scored around %{x}%<extra></extra>")
                    fighd.update_layout(margin=dict(t=28, b=8),
                                        yaxis_title="Students",
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        font=dict(color="#26303e"),
                                        title=f"Score distribution — {dunit}")
                    st.plotly_chart(fighd, use_container_width=True)
                with dc2:
                    # unit vs everyone, per skill — works on the melted shape
                    # too, and rolls up to named competencies when a map exists
                    _ov = question_means(ddf, QF, _dd_by)
                    _un = question_means(_sub, QF, _dd_by)
                    _dlt = (_un - _ov).dropna()
                    if len(_dlt):
                        _worst_gap = float(_dlt.min())
                        _dlt = _dlt.sort_values().head(8).iloc[::-1]
                        _ylab = ([question_label(i) for i in _dlt.index]
                                 if _dd_by == "Question" else list(_dlt.index))
                        figdq = px.bar(x=_dlt.values, y=_ylab,
                                       orientation="h", height=330,
                                       color=_dlt.values,
                                       color_continuous_scale="RdYlGn",
                                       color_continuous_midpoint=0,
                                       labels={"x": f"pts vs overall "
                                                    f"(per {_dd_by.lower()})",
                                               "y": ""})
                        figdq.update_traces(hovertemplate="<b>%{y}</b>: "
                                            "%{x:+.1f} pts vs the overall average"
                                            "<extra></extra>")
                        figdq.update_layout(coloraxis_showscale=False,
                                            margin=dict(t=28, b=8),
                                            paper_bgcolor="rgba(0,0,0,0)",
                                            font=dict(color="#26303e"),
                                            title=f"Weakest {_dd_by.lower()}s "
                                                  f"vs overall")
                        if _worst_gap > -1.0:
                            st.success(f"✅ No real skill gaps: {dunit} is within "
                                       f"{abs(_worst_gap):.1f} pts of the overall "
                                       f"average on every "
                                       f"{_dd_by.lower()}. The chart "
                                       "below zooms into those tiny differences.")
                        st.plotly_chart(figdq, use_container_width=True)
                        st.caption(f"**How to read:** each bar = one "
                                   f"{_dd_by.lower()}; length = how far {dunit} "
                                   "scores below everyone else on it. The "
                                   "most-negative bars are its remedial "
                                   "shortlist.")
                if year_col and _sub[year_col].nunique() > 1:
                    _tr = _sub.groupby(year_col).agg(mean=(_dcol, "mean"),
                                                     size=_cnt)
                    _tr = _tr[_tr["size"] >= DMIN_N]
                    if len(_tr) > 1:
                        _trd = pd.DataFrame({
                            "Year": [f"{int(x)}-{(int(x)+1) % 100:02d}"
                                     for x in _tr.index],
                            _dlab: _dpct(_tr["mean"]).round(1)})
                        figtr = px.line(_trd, x="Year", y=_dlab, markers=True,
                                        height=280,
                                        title=f"Trend — {dunit}")
                        figtr.update_traces(line_width=3, marker_size=10,
                                            line_color="#00e5ff",
                                            hovertemplate="%{x} · %{y:.1f}"
                                            "<extra></extra>")
                        figtr.update_layout(margin=dict(t=40, b=8),
                                            paper_bgcolor="rgba(0,0,0,0)",
                                            font=dict(color="#26303e"))
                        st.plotly_chart(figtr, use_container_width=True)
                        _t0, _t1v = _trd[_dlab].iloc[0], _trd[_dlab].iloc[-1]
                        st.caption(f"**{dunit} over time:** {_t0:.1f} → "
                                   f"{_t1v:.1f} {_dlab} "
                                   f"({_t1v - _t0:+.1f} pts across the period).")
            else:
                st.info(f"No {dlevel} has {DMIN_N}+ students in the current "
                        "selection — widen filters or pick a higher level.")

        st.divider()
        # ================= B) Equity lens =====================================
        st.markdown("#### ⚖️ Equity lens — same average, different fairness")
        if hierarchy:
            elevel = st.selectbox("Compare units at", hierarchy, key="eq_level")
            # Quantiles must be taken over CHILDREN. On the melted shape _dcol
            # is 0 or 100 per answer, so p10=0 and p90=100 for every unit and
            # the whole chart collapses to one horizontal line at 100.
            _eqsrc = (ddf.groupby([elevel, _dsid], observed=True)[_dcol]
                      .mean().reset_index() if _dsid else ddf)
            _eq = (_eqsrc.groupby(elevel)[_dcol]
                   .agg(avg="mean", p10=lambda s: s.quantile(.10),
                        p90=lambda s: s.quantile(.90), n="size")
                   .query("n >= @DMIN_N"))
            if len(_eq) >= 3:
                _eq["avg"] = _dpct(_eq["avg"])
                _eq["spread"] = _dpct(_eq["p90"]) - _dpct(_eq["p10"])
                _eqr = _eq.reset_index()
                figeq = px.scatter(_eqr, x="avg", y="spread", size="n",
                                   hover_name=elevel, height=430,
                                   color="spread",
                                   color_continuous_scale="RdYlGn_r",
                                   labels={"avg": f"Average {_dlab}",
                                           "spread": "Gap inside the unit "
                                                     "(top 10% − bottom 10%)"})
                figeq.add_hline(y=float(_eq["spread"].median()),
                                line_dash="dot", line_color="gray")
                figeq.add_vline(x=float(_eq["avg"].median()),
                                line_dash="dot", line_color="gray")
                figeq.update_traces(hovertemplate="<b>%{hovertext}</b><br>"
                                    "Average: %{x:.1f}% correct<br>"
                                    "Gap inside (top 10% − bottom 10% of its own "
                                    "students): %{y:.1f} pts<br>"
                                    "Students: %{marker.size:,}<extra></extra>")
                figeq.update_layout(coloraxis_showscale=False,
                                    margin=dict(t=10, b=8),
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    font=dict(color="#26303e"))
                st.plotly_chart(figeq, use_container_width=True)
                st.caption("**How to read:** each dot = one unit; dot size = "
                           "students. Right = better average. Higher = bigger "
                           "gap between that unit's own best and weakest "
                           "students. **Top-right = good average hiding "
                           "left-behind children.** Dotted lines = the middle "
                           "of the pack.")
                _worst = _eqr.sort_values("spread").iloc[-1]
                st.caption(
                    "Top-right = decent average hiding huge internal gaps — "
                    "children left behind behind a healthy-looking number. "
                    f"Widest gap: **{_worst[elevel]}** "
                    f"({_worst['spread']:.0f}-pt spread inside one "
                    f"{elevel.lower()}). Units under {DMIN_N} students excluded.")
            else:
                st.info(f"Fewer than 3 {elevel}s pass the {DMIN_N}-student "
                        "minimum in this selection.")

        st.divider()
        st.markdown("#### 📦 Classic views")
        # 1) Heatmap: geography x competency (named competencies when mapped)
        if hierarchy and comp_col:
            level = st.selectbox("Heatmap level", hierarchy, index=min(1, len(hierarchy)-1))
            # COMP_VALUE_COL, not score_col: with the questions melted into a
            # competency dimension, score_col is the student's TOTAL and is the
            # same on all of their rows — so every column of a given row held an
            # identical value and the grid was solid horizontal bands
            # (measured: 1 distinct value per row across all 20 competencies).
            _hsub = ddf if QF["kind"] == "long" else fdf
            _hkey = comp_col
            if _dd_by == "Competency" and QMAP and comp_col in _hsub.columns:
                _hsub = _hsub.assign(
                    _hcomp=_hsub[comp_col].astype(str).map(competency_of))
                _hkey = "_hcomp"
            hm = _hsub.pivot_table(index=level, columns=_hkey,
                                   values=COMP_VALUE_COL, aggfunc="mean")
            if _hkey == comp_col and QMAP:      # keep Q ids, add the skill name
                hm.columns = [question_label(c) for c in hm.columns]
            _hlab = "% correct" if COMP_VALUE_IS_PCT else "Avg score"
            fig = px.imshow(hm.round(1), text_auto=True, aspect="auto",
                            color_continuous_scale="RdYlGn",
                            labels=dict(color=_hlab))
            fig.update_layout(title=f"{_hlab} — {level} × "
                                    f"{_dd_by if QMAP else 'Competency'}")
            st.plotly_chart(fig, use_container_width=True)
            if QMAP:
                st.caption(f"Columns are **{_dd_by.lower()}s** from the question "
                           f"map. Red cells are the {level.lower()}-by-skill "
                           "combinations to target — a whole red column is a "
                           "curriculum problem, a whole red row is a unit "
                           "problem.")

        c1, c2 = st.columns(2)
        # 2) Box plot: spread/inequality within each unit
        with c1:
            if hierarchy:
                blevel = st.selectbox("Spread by", hierarchy, index=0, key="box_level")
                # per child, so the box describes children rather than answers
                _bsrc = (ddf.groupby([blevel, _dsid], observed=True)[_dcol]
                         .mean().reset_index() if _dsid else ddf)
                fig = px.box(_bsrc, x=blevel, y=_dcol, points=False)
                fig.update_layout(title=f"{_dlab} spread within each {blevel}",
                                  yaxis_title=_dlab)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Wide boxes = high inequality inside that unit, even if the average looks fine.")

                # --- Simpler alternate for the same question -----------------
                # A box plot asks the reader to know what a quartile and a
                # whisker are. This answers the same question — "how mixed are
                # the children inside this unit?" — as the share of children in
                # each performance band. A big red block means many struggling
                # children, and no statistics are needed to see it.
                _BANDS = ["Poor", "Below Average", "Average",
                          "Above Average", "Excellent"]
                _BANDC = ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"]
                _bd = (fdf.groupby([blevel, "Performance_Band"], observed=True)
                          .size().rename("n").reset_index())
                _bd = _bd[_bd["n"] > 0]
                if not _bd.empty:
                    _bd["Performance_Band"] = _bd["Performance_Band"].astype(str)
                    _bd["pct"] = (100 * _bd["n"]
                                  / _bd.groupby(blevel)["n"].transform("sum"))
                    # worst-off units first: biggest share in the bottom two bands
                    _rank = (_bd[_bd["Performance_Band"].isin(_BANDS[:2])]
                             .groupby(blevel)["pct"].sum()
                             .reindex(_bd[blevel].unique()).fillna(0)
                             .sort_values(ascending=False))
                    figb2 = px.bar(
                        _bd, x="pct", y=blevel, color="Performance_Band",
                        orientation="h",
                        category_orders={blevel: list(_rank.index),
                                         "Performance_Band": _BANDS},
                        color_discrete_sequence=_BANDC,
                        height=max(300, 30 * _bd[blevel].nunique() + 120),
                        labels={"pct": "% of children", "Performance_Band": ""})
                    figb2.update_traces(
                        hovertemplate="<b>%{y}</b><br>%{x:.0f}% of children"
                                      "<extra>%{fullData.name}</extra>")
                    figb2.update_layout(
                        barmode="stack", title=f"Same data — who is in each band",
                        yaxis_title="", margin=dict(t=40, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                    x=0, font=dict(size=11)))
                    figb2.update_xaxes(ticksuffix="%", range=[0, 100])
                    st.plotly_chart(figb2, use_container_width=True)
                    _wq = _rank.index[0]
                    st.caption(
                        f"**Easier read of the same thing.** Each bar is one "
                        f"{blevel.lower()}, split into 100%. Red = struggling "
                        f"children, green = doing well. Sorted worst-first, so "
                        f"**{_wq}** needs attention most "
                        f"({_rank.iloc[0]:.0f}% of its children in the bottom "
                        f"two bands). A unit with red *and* green is unequal "
                        f"inside — the same thing a wide box means.")
        # 3) Scatter: subjective factor vs outcome
        with c2:
            if cols["subjective"] and hierarchy:
                factor = st.selectbox("Factor", cols["subjective"], key="scatter_factor")
                unit = hierarchy[-1]
                sc = (ddf.groupby([unit, factor], as_index=False)
                         .agg(avg=(_dcol, "mean"), n=_cnt))
                fig = px.strip(sc, x=factor, y="avg", hover_name=unit)
                fig.update_layout(title=f"{factor} vs {_dlab.lower()} (per {unit})",
                                  yaxis_title=_dlab, xaxis_title="")
                fig.update_xaxes(tickangle=20)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Each dot = one " + unit.lower() + " — shows whether the factor tracks outcomes.")
    _tab4_fragment()

# --------------------------- Rankings & Alerts -------------------------------
with tabs[6]:
    # fragment: widgets inside this tab rerun only this tab
    @st.fragment
    def _tab5_fragment():
        unit = (st.selectbox("Rank by", hierarchy, index=len(hierarchy) - 1,
                             key="rank_level") if hierarchy else None)
        if unit:
            RMIN_N = MINN
            _qr = [c for c in fdf.columns if re.fullmatch(r"[Qq]\d+", str(c))
                   and pd.api.types.is_numeric_dtype(fdf[c])]
            _oor = len(_qr) if _qr and fdf[score_col].max() <= len(_qr) else None
            _rpct = (lambda s: s * 100.0 / _oor) if _oor else (lambda s: s)
            _rmax = 100 if _oor or fdf[score_col].max() <= 100 \
                else float(fdf[score_col].max())

            _rk = (fdf.groupby(unit)[score_col].agg(["mean", "size"])
                   .rename(columns={"mean": "Avg", "size": "Students"}))
            _hidden = int((_rk["Students"] < RMIN_N).sum())
            _rk = _rk[_rk["Students"] >= RMIN_N]
            eligible = set(_rk.index)          # reused by alerts + quadrant below
            _rk["Avg score"] = _rpct(_rk["Avg"]).round(1)
            ranked = (_rk.drop(columns="Avg").sort_values("Avg score",
                                                          ascending=False)
                      .reset_index())
            bar_cfg = {"Avg score": st.column_config.ProgressColumn(
                "Avg score" + (" (%)" if _oor else ""),
                min_value=0, max_value=_rmax, format="%.1f")}
            r1, r2 = st.columns(2)
            with r1:
                st.subheader(f"🏆 Top 10 {unit}s")
                st.dataframe(ranked.head(10), column_config=bar_cfg, hide_index=True)
            with r2:
                st.subheader(f"⚠️ Bottom 10 {unit}s")
                st.dataframe(ranked.tail(10).iloc[::-1], column_config=bar_cfg, hide_index=True)
            if _hidden:
                st.caption(f"{_hidden} {unit}(s) with fewer than {RMIN_N} students "
                           "excluded — tiny groups top rankings by pure luck.")

        # Early-warning alerts: units declining across years
        if unit and year_col and fdf[year_col].nunique() > 1:
            st.subheader("🚨 Early-Warning: consistently declining")
            pv = fdf.pivot_table(index=unit, columns=year_col, values=score_col, aggfunc="mean")
            pv = pv.dropna()
            pv = pv[pv.index.isin(eligible)]   # same min-N guard as the rankings
            declining = pv[pv.diff(axis=1).iloc[:, 1:].lt(0).all(axis=1)]
            if len(declining):
                for name, row in declining.iterrows():
                    trail = " → ".join(f"{v:.1f}" for v in row.values)
                    st.error(f"**{name}** declining every year: {trail}")
            else:
                st.success("No unit is declining across every year in the current selection.")

        # Quadrant scatter: current level vs 3-year change (proto-prediction view)
        if unit and year_col and fdf[year_col].nunique() > 1:
            st.subheader("🎯 Level vs Trend — who is low AND falling?")
            y0, yN = fdf[year_col].min(), fdf[year_col].max()
            lv = fdf.pivot_table(index=unit, columns=year_col, values=score_col, aggfunc="mean").dropna()
            lv = lv[lv.index.isin(eligible)]   # same min-N guard as the rankings
            quad = pd.DataFrame({
                "current": lv[yN].round(1),
                "change": (lv[yN] - lv[y0]).round(1),
                "n": fdf.groupby(unit)[score_col].size(),
            }).reset_index()
            fig = px.scatter(quad, x="current", y="change", size="n", hover_name=unit,
                             color="change", color_continuous_scale="RdYlGn",
                             labels={"current": f"Average score ({yN})",
                                     "change": f"Change {y0}→{yN}"})
            fig.add_hline(y=0, line_dash="dot", line_color="gray")
            fig.add_vline(x=float(fdf[score_col].mean()), line_dash="dot", line_color="gray")
            fig.update_layout(height=480)
            st.plotly_chart(fig, use_container_width=True)

            st.caption("Bottom-left quadrant = below-average AND declining → the at-risk list. "
                       "Dot size = number of records.")

        # Sankey: movement between performance bands (first year -> last year)
        sid = sid_col if (sid_col and sid_col in fdf.columns) else None
        if sid and year_col and fdf[year_col].nunique() > 1:
            st.subheader("🔀 Learning progression: band movement (first → last year)")
            import plotly.graph_objects as go
            y0, y1 = fdf[year_col].min(), fdf[year_col].max()
            per = (fdf[fdf[year_col].isin([y0, y1])]
                   .groupby([sid, year_col])[score_col].mean().reset_index())
            per["band"] = per.groupby(year_col)[score_col].transform(grade_on_curve)
            wide = per.pivot(index=sid, columns=year_col, values="band").dropna()
            order = ["Poor", "Below Average", "Average", "Above Average", "Excellent"]
            flows = wide.groupby([y0, y1], observed=True).size().reset_index(name="n")
            labels = [f"{b} ({y0})" for b in order] + [f"{b} ({y1})" for b in order]
            colors = ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"] * 2
            src = flows[y0].map({b: i for i, b in enumerate(order)})
            tgt = flows[y1].map({b: i + len(order) for i, b in enumerate(order)})
            fig = go.Figure(go.Sankey(
                node=dict(label=labels, color=colors, pad=12),
                link=dict(source=src, target=tgt, value=flows["n"])))
            fig.update_layout(height=480, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Each band shows how many students moved between performance "
                       f"bands from {y0} to {y1} — thick red→red flows mean students stuck at the bottom.")

            # --- Simpler alternate for the same question --------------------
            # A Sankey shows every band-to-band flow at once, which is a lot to
            # decode when the question a reader actually has is simply: did
            # children get better, stay the same, or fall back? Same data,
            # collapsed to that one answer.
            _oi = {b: i for i, b in enumerate(order)}
            _mv = wide.copy()
            # grade_on_curve returns a Categorical, and .map() on a Categorical
            # gives back a Categorical — which cannot be subtracted. Go through
            # str -> numeric so the band positions are plain integers.
            _r0 = pd.to_numeric(_mv[y0].astype(str).map(_oi), errors="coerce")
            _r1 = pd.to_numeric(_mv[y1].astype(str).map(_oi), errors="coerce")
            _mv["_d"] = _r1 - _r0
            _mv = _mv[_mv["_d"].notna()]
            _up = int((_mv["_d"] > 0).sum())
            _sm = int((_mv["_d"] == 0).sum())
            _dn = int((_mv["_d"] < 0).sum())
            _tt = _up + _sm + _dn
            if _tt:
                m1, m2, m3 = st.columns(3)
                m1.metric("⬆️ Moved up a band", f"{_up:,}",
                          f"{100*_up/_tt:.0f}% of children")
                m2.metric("➡️ Stayed the same", f"{_sm:,}",
                          f"{100*_sm/_tt:.0f}% of children", delta_color="off")
                m3.metric("⬇️ Fell back a band", f"{_dn:,}",
                          f"-{100*_dn/_tt:.0f}% of children", delta_color="inverse")
                _mvdf = pd.DataFrame({
                    "outcome": ["Fell back", "Stayed the same", "Moved up"],
                    "children": [_dn, _sm, _up],
                    "color": ["#d73027", "#67707f", "#1a9850"]})
                figs2 = px.bar(_mvdf, x="children", y="outcome", orientation="h",
                               height=260, text="children")
                figs2.update_traces(
                    marker_color=_mvdf["color"], texttemplate="%{text:,}",
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>%{x:,} children<extra></extra>")
                figs2.update_layout(
                    title=f"Same data — what happened to each child, {y0} → {y1}",
                    yaxis_title="", xaxis_title="children",
                    margin=dict(t=40, b=10),
                    xaxis_range=[0, max(_mvdf["children"]) * 1.18])
                st.plotly_chart(figs2, use_container_width=True)
                _verdict = ("more children moved up than fell back"
                            if _up > _dn else
                            "more children fell back than moved up"
                            if _dn > _up else
                            "exactly as many moved up as fell back")
                st.caption(
                    f"**Easier read of the same thing.** Of the {_tt:,} children "
                    f"tested in both {y0} and {y1}, {_verdict}. The Sankey above "
                    f"shows *which* bands they moved between; this shows the "
                    f"bottom line.")
    _tab5_fragment()

# ------------------------------- Prediction ----------------------------------
with tabs[7]:
    # fragment: widgets inside this tab rerun only this tab
    @st.fragment
    def _tab6_fragment():
        unit_p = (st.selectbox(
            "Forecast level", hierarchy,
            index=min(1, len(hierarchy) - 1),
            key="pred_level",
            help="Finer levels have fewer students per unit-year, so forecasts "
                 "get noisier — the backtest below shows the honest error at "
                 "the level you pick.") if hierarchy else None)
        if unit_p and year_col and fdf[year_col].nunique() >= 2:
            st.subheader("🔮 Next-year forecast (trend projection per unit)")
            PMIN_N = MINN
            _qp = [c for c in fdf.columns if re.fullmatch(r"[Qq]\d+", str(c))
                   and pd.api.types.is_numeric_dtype(fdf[c])]
            _oop = len(_qp) if _qp and fdf[score_col].max() <= len(_qp) else None
            # Range must follow the actual score scale. Hard-coding 20–60 assumed a
            # 0–100 percentage and raised StreamlitAPIException on raw marks (0–20),
            # because the computed default fell below the minimum.
            if _oop:
                _smin = float(df[score_col].min()) * 100.0 / _oop
                _smax = float(df[score_col].max()) * 100.0 / _oop
            else:
                _smin, _smax = float(df[score_col].min()), float(df[score_col].max())
            _lo, _hi = float(np.floor(_smin)), float(np.ceil(_smax))
            if _hi <= _lo:
                _hi = _lo + 1.0
            _dmean = df[score_col].mean() * (100.0 / _oop if _oop else 1)
            _dstd = df[score_col].std() * (100.0 / _oop if _oop else 1)
            _default = float(np.clip(round(_dmean - 0.5 * _dstd, 1), _lo, _hi))
            threshold = st.slider("At-risk threshold (projected score below…)",
                                  _lo, _hi, _default,
                                  help=f"'{score_col}' runs {_smin:g}–{_smax:g} in this "
                                       f"dataset; the default is half a standard "
                                       f"deviation below the mean.")
            years_sorted = sorted(fdf[year_col].unique())
            next_year = int(years_sorted[-1]) + 1
            pv = fdf.pivot_table(index=unit_p, columns=year_col, values=score_col,
                                 aggfunc="mean").dropna()
            _pn = fdf.pivot_table(index=unit_p, columns=year_col, values=score_col,
                                  aggfunc="size").reindex(pv.index)
            _pre = len(pv)
            pv = pv[(_pn >= PMIN_N).all(axis=1)]
            if _oop:
                pv = pv * 100.0 / _oop
            if _pre - len(pv):
                st.caption(f"{_pre - len(pv)} {unit_p}(s) excluded: fewer than "
                           f"{PMIN_N} students in some year — a trend line "
                           "through a handful of children is noise, not signal.")
            if not len(pv):
                # never st.stop() inside a tab — it would kill every later tab
                st.warning(f"⚠️ No {unit_p} has {PMIN_N}+ students in every year "
                           "in this selection — showing ALL units, but these "
                           "forecasts are statistically unreliable. Pick a "
                           "higher forecast level.")
                pv = fdf.pivot_table(index=unit_p, columns=year_col,
                                     values=score_col, aggfunc="mean").dropna()
                if _oop:
                    pv = pv * 100.0 / _oop
            xs = np.array(years_sorted, dtype=float)

            def project(row):
                slope, intercept = np.polyfit(xs, row.values.astype(float), 1)
                return slope * next_year + intercept, slope

            proj = pv.apply(lambda r: pd.Series(project(r),
                            index=["Projected", "Trend/yr"]), axis=1)
            proj["Projected"] = proj["Projected"].clip(0, 100).round(1)
            proj["Trend/yr"] = proj["Trend/yr"].round(2)
            proj[f"Current ({years_sorted[-1]})"] = pv[years_sorted[-1]].round(1)
            proj["At Risk"] = proj["Projected"] < threshold
            proj = proj.sort_values("Projected")

            n_risk = int(proj["At Risk"].sum())
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Units projected below {threshold} in {next_year}", n_risk)
            c2.metric("Steepest decline", f"{proj['Trend/yr'].min():+.1f}/yr")
            c3.metric("Fastest improvement", f"{proj['Trend/yr'].max():+.1f}/yr")

            # Chart: current vs projected, at-risk highlighted
            plot_df = proj.reset_index().rename(columns={proj.index.name or "index": unit_p})
            fig = px.scatter(plot_df, x=f"Current ({years_sorted[-1]})", y="Projected",
                             color="At Risk", hover_name=unit_p,
                             color_discrete_map={True: "#d73027", False: "#1a9850"})
            fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                          annotation_text=f"threshold {threshold}")
            lo = min(plot_df["Projected"].min(), plot_df[f"Current ({years_sorted[-1]})"].min()) - 3
            hi = max(plot_df["Projected"].max(), plot_df[f"Current ({years_sorted[-1]})"].max()) + 3
            fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                          line=dict(color="gray", dash="dot"))
            fig.update_layout(height=480)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Below the gray diagonal = projected to fall vs today. "
                       "Red dots = early-warning list for " + str(next_year) + ".")

            st.subheader(f"⚠️ Projected at-risk {unit_p}s for {next_year}")
            risk_table = proj[proj["At Risk"]].drop(columns="At Risk")
            if len(risk_table):
                st.dataframe(risk_table)
            else:
                st.success("No unit projected below the threshold at current trends.")
            # Backtest: predict the last year from earlier years, compare vs reality
            if len(years_sorted) >= 3:
                st.subheader("✅ Backtest — how accurate is this method?")
                hold = years_sorted[-1]
                train_years = np.array(years_sorted[:-1], dtype=float)

                def bt_project(row):
                    slope, intercept = np.polyfit(train_years,
                                                  row[years_sorted[:-1]].values.astype(float), 1)
                    return slope * hold + intercept

                bt_pred = pv.apply(bt_project, axis=1).clip(0, 100)
                bt_real = pv[hold]
                err = (bt_pred - bt_real).abs()
                real_risk = set(bt_real[bt_real < threshold].index)
                pred_risk = set(bt_pred[bt_pred < threshold].index)
                caught = len(real_risk & pred_risk)

                b1, b2, b3 = st.columns(3)
                b1.metric("Avg error (points)", f"±{err.mean():.1f}")
                b2.metric("Worst error", f"±{err.max():.1f}")
                b3.metric("At-risk caught", f"{caught} of {len(real_risk)}" if real_risk else "n/a")
                st.caption(f"Method: trained only on {', '.join(map(str, years_sorted[:-1]))}, "
                           f"predicted {hold} blind, compared against the real {hold} values. "
                           "The model never saw the year it predicted.")
                if len(train_years) < 3:
                    st.warning(
                        f"⚠️ **Interpret this backtest with care.** It trains on only "
                        f"{len(train_years)} year(s). A straight line through two points "
                        f"fits them exactly, so the fit carries no error estimate and the "
                        f"extrapolation is highly sensitive to noise in either year. "
                        f"These accuracy figures are indicative only — a trustworthy "
                        f"backtest needs at least 3 training years (4+ years of data).")

            with st.expander("Method (for the jury)"):
                st.markdown(
                    "Per unit: ordinary least-squares line fit on the yearly averages "
                    f"({', '.join(map(str, years_sorted))}) → extrapolated to {next_year}. "
                    "Simple, transparent, explainable — upgradeable to feature-based ML "
                    "(teacher quality, income, gender mix) as a next step.")
        else:
            st.info("Prediction needs a hierarchy column and at least 2 years of data.")
    _tab6_fragment()

# ------------------------------- Raw data ------------------------------------
with tabs[8]:
    st.write("**Detected column types** (auto-classified):")
    st.json({k: v for k, v in cols.items() if v})
    st.dataframe(fdf.head(500))


# ------------------------------- Item analysis -------------------------------
with tabs[9]:
    # fragment: widgets inside this tab rerun only this tab
    @st.fragment
    def _tab8_fragment():
        if items_df is not None:
            # ALL_ITEMS is the whole paper and never shrinks. The competency /
            # question picker below narrows `item_cols`, which controls what is
            # DISPLAYED — but the discrimination index further down must still
            # rank students on their full-paper total, or filtering to four
            # binary items would split the cohort on a 0-4 score and produce
            # meaningless quartiles.
            ALL_ITEMS = _find_item_columns(items_df)
            item_cols = list(ALL_ITEMS)
            idf = items_df.copy()

            # Apply the same sidebar slice where the columns exist in the item data
            for col in hierarchy + ([gender_col] if gender_col else []) \
                    + ([grade_col] if grade_col else []):
                if col and col in idf.columns and col in fdf.columns:
                    keep = fdf[col].unique()
                    idf = idf[idf[col].isin(keep)]
            if year_col and year_col in idf.columns:
                idf = idf[(idf[year_col] >= year_range[0]) & (idf[year_col] <= year_range[1])]

            _lmap, _lnames = dict(QMAP or {}), {}
            if RQMAPS:
                st.info("📄 The paper changes every year AND grade — item "
                        "statistics only make sense within one paper. Pick "
                        "which paper to analyse:")
                _pc1, _pc2 = st.columns(2)
                _yrs = sorted({y for (y, g) in RQMAPS})
                _sel_y = _pc1.selectbox("Paper year", _yrs,
                                        index=len(_yrs) - 1, key="ia_year")
                _grs = sorted({g for (y, g) in RQMAPS if y == _sel_y})
                _sel_g = _pc2.selectbox("Paper grade", _grs, key="ia_grade")
                _lmap = RQMAPS.get((_sel_y, _sel_g), {})
                _lnames = RQNAMES.get((_sel_y, _sel_g), {})
                if year_col and year_col in idf.columns:
                    idf = idf[idf[year_col].astype(str).str.contains(
                        _sel_y[:4])]
                if grade_col and grade_col in idf.columns:
                    idf = idf[pd.to_numeric(idf[grade_col],
                                            errors="coerce") == _sel_g]

            def _cof(q):
                return _lmap.get(str(q), str(q)) if _lmap else str(q)

            def _qlab(q):
                nm = _lnames.get(str(q))
                c = _cof(q)
                if nm:
                    return f"{q} · {nm}"
                return f"{q} · {c}" if c != str(q) else str(q)

            IQF = competency_question_filter(idf, "item")
            _pick = [q for q in IQF["questions"] if q in ALL_ITEMS]
            if _pick:
                item_cols = _pick
            _iby = "Question"
            if _lmap and len({_cof(q) for q in item_cols}) < len(item_cols):
                _iby = st.radio("Show items as", ["Question", "Competency"],
                                horizontal=True, key="item_by")

            def _roll(s):
                """Per-item series -> averaged into competencies, or relabelled.

                Rolling up must AGGREGATE, not just rename: four columns all
                called 'Algebra' would collide in the bar chart and heatmap.
                """
                if _iby == "Competency":
                    return s.groupby(s.index.map(_cof)).mean()
                return s.set_axis([_qlab(i) for i in s.index])

            st.caption(f"{len(idf):,} student responses × {len(item_cols)} of "
                       f"{len(ALL_ITEMS)} items in current selection")

            rates = (idf[item_cols].apply(pd.to_numeric, errors="coerce")
                     .mean().mul(100))
            order = _roll(rates).round(1).sort_values()

            st.subheader(f"📉 Hardest {_iby.lower()}s "
                         "(lowest % answered correctly)")
            fig = px.bar(x=order.index, y=order.values,
                         labels={"x": _iby, "y": "% correct"},
                         color=order.values, color_continuous_scale="RdYlGn",
                         range_color=[0, 100], height=380)
            fig.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            worst = order.index[0] if len(order) else None
            best = order.index[-1] if len(order) else None
            if worst is not None:
                c1, c2, c3 = st.columns(3)
                c1.metric(f"Hardest {_iby.lower()}", str(worst),
                          f"{order.iloc[0]:.1f}% correct",
                          delta_color="inverse")
                c2.metric(f"Easiest {_iby.lower()}", str(best),
                          f"{order.iloc[-1]:.1f}% correct")
                c3.metric("Spread", f"{order.iloc[-1] - order.iloc[0]:.1f} pts")

            st.divider()
            st.subheader("🔥 Item difficulty by group")
            dim_pool = [c for c in idf.columns
                        if c not in ALL_ITEMS     # every Q column, not just the
                        and c != score_col        # selected ones — breaking
                        and 1 < idf[c].nunique() <= 40]  # items down by the
            if dim_pool:                          # total score is circular
                dim = st.selectbox("Break down items by", dim_pool)
                heat = idf.groupby(dim)[item_cols].mean().mul(100)
                # same aggregate-don't-rename rule as the bar chart above
                heat = (heat.T.groupby(heat.columns.map(_cof)).mean().T
                        if _iby == "Competency"
                        else heat.set_axis([_qlab(c)
                                            for c in heat.columns], axis=1))
                heat = heat.round(1)
                figh = px.imshow(heat, color_continuous_scale="RdYlGn",
                                 zmin=0, zmax=100, aspect="auto",
                                 labels=dict(color="% correct"),
                                 height=min(220 + 26 * len(heat), 700))
                st.plotly_chart(figh, use_container_width=True)
                st.caption("Red cells = groups struggling on that item — "
                           "a direct target list for remedial content.")

            st.divider()
            st.subheader("🎯 Item quality — difficulty × discrimination")
            # Discrimination: top 27% of students minus bottom 27% (by total
            # score) on each item. Near zero = the question does not separate
            # strong from weak learners — likely ambiguous, miskeyed, or guessed.
            # ALL_ITEMS, not item_cols: the high/low groups must be split on the
            # student's WHOLE-PAPER score. Ranking them on a filtered subset of
            # four binary items gives a 0-4 total whose 27th and 73rd
            # percentiles land on the same value, and the index collapses.
            _im = idf[ALL_ITEMS].apply(pd.to_numeric, errors="coerce")
            _tot = _im.sum(axis=1)
            _hi = _im[_tot >= _tot.quantile(0.73)]
            _lo = _im[_tot <= _tot.quantile(0.27)]
            if len(_hi) >= 15 and len(_lo) >= 15:
                _qual = pd.DataFrame({
                    "Item": item_cols,
                    "difficulty": _im[item_cols].mean().mul(100).round(1).values,
                    "discrimination": (_hi[item_cols].mean()
                                       - _lo[item_cols].mean()).round(2).values})
                _qual["flag"] = np.where(_qual["discrimination"] < 0.2,
                                         "⚠️ review", "ok")
                figq = px.scatter(_qual, x="difficulty", y="discrimination",
                                  text="Item", color="flag",
                                  color_discrete_map={"ok": "#57c26b",
                                                      "⚠️ review": "#e05252"},
                                  height=440,
                                  labels={"difficulty": "% correct (easy →)",
                                          "discrimination":
                                          "Discrimination (top 27% − bottom 27%)"})
                if _lmap:
                    _qual["Competency"] = _qual["Item"].map(
                        lambda q: _lmap.get(str(q), "—"))
                    _qual["QName"] = _qual["Item"].map(
                        lambda q: _lnames.get(str(q), ""))
                    figq.update_traces(
                        customdata=_qual[["Competency", "QName"]],
                        hovertemplate="<b>%{text}</b> · %{customdata[0]}"
                        "<br>%{customdata[1]}<br>%{x:.0f}% correct · "
                        "discrimination %{y:.2f}<extra></extra>")
                figq.update_traces(textposition="top center", marker_size=11)
                figq.add_hline(y=0.2, line_dash="dash", line_color="gray",
                               annotation_text="0.2 = review threshold",
                               annotation_font_color="#67707f")
                figq.update_layout(margin=dict(t=10, b=10), legend_title="",
                                   paper_bgcolor="rgba(0,0,0,0)",
                                   font=dict(color="#26303e"))
                st.plotly_chart(figq, use_container_width=True)

                _bad = _qual[_qual["flag"] != "ok"]["Item"].tolist()
                st.caption(
                    ("Items below the line barely separate strong from weak "
                     "students — candidates for rewriting before the next "
                     "assessment: **" + ", ".join(_bad) + "**. "
                     if _bad else
                     "All items clear the 0.2 discrimination bar — the paper "
                     "separates strong from weak students consistently. ")
                    + "A very easy item (right side) is *expected* to have "
                      "modest discrimination; a mid-difficulty item with low "
                      "discrimination is the red flag.")
                if QMAP and QDIFF:
                    st.divider()
                    st.subheader("📐 Did the test behave as designed?")
                    _dd2 = pd.DataFrame({
                        "Item": item_cols,
                        "intended": [float(QDIFF.get(str(q), float("nan"))) * 100
                                     for q in item_cols],
                        "observed": [(1 - _im[q].mean()) * 100
                                     for q in item_cols]})
                    _dd2["Competency"] = _dd2["Item"].map(
                        lambda q: QMAP.get(str(q), "—"))
                    _dd2 = _dd2.dropna()
                    figdo = px.scatter(_dd2, x="intended", y="observed",
                                       text="Item", color="Competency",
                                       height=430,
                                       labels={"intended":
                                               "Intended difficulty (% expected "
                                               "to get it wrong, from the map)",
                                               "observed":
                                               "Observed (% actually wrong)"})
                    _mx = float(max(_dd2["intended"].max(),
                                    _dd2["observed"].max())) + 5
                    figdo.add_shape(type="line", x0=0, y0=0, x1=_mx, y1=_mx,
                                    line=dict(dash="dash", color="gray"))
                    figdo.update_traces(textposition="top center",
                                        marker_size=10,
                                        hovertemplate="<b>%{text}</b><br>"
                                        "designed: %{x:.0f}% wrong · actual: "
                                        "%{y:.0f}% wrong<extra></extra>")
                    figdo.update_layout(margin=dict(t=10, b=10),
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        font=dict(color="#26303e"))
                    st.plotly_chart(figdo, use_container_width=True)
                    st.caption("**How to read:** each dot = one question; "
                               "across = how hard the designers *intended* it "
                               "to be, up = how hard it *actually* was. Dots on "
                               "the dashed line behaved exactly as designed; "
                               "far above = harder than intended, far below = "
                               "easier. Colors = competency groups from the "
                               "question map.")
            else:
                st.info("Too few students in the current selection for a stable "
                        "discrimination estimate (need 15+ in the top and bottom "
                        "score groups).")
        else:
            st.info("No binary item columns (Q1, Q2, …) detected in this dataset — "
                    "this tab activates automatically when item-level data arrives.")
    _tab8_fragment()
# ------------------------------- Choropleth Map -------------------------------
# Karnataka district map. Needs karnataka_districts.geojson next to this file
# (feature property: properties.district, 31 districts incl. Vijayanagara).
DISTRICT_FIX = {
    # old spelling / common variant  ->  name used in the GeoJSON
    "Bangalore": "Bengaluru Urban", "Bangalore Urban": "Bengaluru Urban",
    "Bengaluru": "Bengaluru Urban", "Bangalore Rural": "Bengaluru Rural",
    "Mysore": "Mysuru", "Belgaum": "Belagavi", "Gulbarga": "Kalaburagi",
    "Bijapur": "Vijayapura", "Bellary": "Ballari", "Shimoga": "Shivamogga",
    "Tumkur": "Tumakuru", "Chikmagalur": "Chikkamagaluru",
    "Chickmagalur": "Chikkamagaluru", "Chamarajanagar": "Chamarajanagara",
    "Chikballapur": "Chikkaballapura", "Chikkaballapur": "Chikkaballapura",
    "Bagalkot": "Bagalkote", "Davangere": "Davanagere",
    "Ramanagar": "Ramanagara", "Ramnagar": "Ramanagara",
    "Mangalore": "Dakshina Kannada", "Karwar": "Uttara Kannada",
}

with tabs[10]:
    import os
    geo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "karnataka_districts.geojson")
    dist_col = next((c for c in df.columns if c.lower() == "district"), None)

    if not os.path.exists(geo_path):
        st.warning("`karnataka_districts.geojson` not found next to the app. "
                   "Download a Karnataka districts GeoJSON and save it as that "
                   "filename in the same folder as streamlit_app.py.")
    elif dist_col is None:
        st.warning("No `district` column detected in this dataset.")
    else:
        with open(geo_path) as _f:
            _geo = json.load(_f)
        geo_names = {f["properties"]["district"] for f in _geo["features"]}

        # Fragment: the skill filter, the demo toggle and a map click all rerun
        # ONLY this section, not the whole 18-tab script — that full rerun was
        # the long load. Everything the map needs is therefore built in here.
        @st.fragment
        def _map_drill_fragment():
            # ---- competency + question filter -----------------------------
            st.markdown("##### 🔎 Map a single skill")
            MQF = competency_question_filter(fdf, "map")
            mdf, _mcol, _mlab = MQF["frame"], MQF["col"], MQF["label"]
            _msid = sid_col if (sid_col and sid_col in mdf.columns) else None
            _mcnt = (_msid, "nunique") if _msid else (_mcol, "size")
            _mby = "Question"
            if MQF["kind"] and QMAP and len(MQF["competencies"]) < len(MQF["questions"]):
                _mby = st.radio("Break skills down by",
                                ["Competency", "Question"], horizontal=True,
                                key="map_by")

            # 1 row per district: mean + child count (respects sidebar filters)
            # `n` counts CHILDREN, not answers — after the melt one child is 20
            # rows, and labelling that "Students" overstates it 20x.
            m = mdf.groupby(dist_col, as_index=False).agg(
                avg=(_mcol, "mean"), n=_mcnt)
            # NOTE: do NOT round before coloring — with a tight spread, rounding
            # quantizes every district to the same value and the map goes one
            # flat color. Round only in the hover.
            # Fix spellings, then check the match rate
            m["geo_name"] = m[dist_col].astype(str).str.strip().replace(DISTRICT_FIX)
            matched = m[m["geo_name"].isin(geo_names)]
            unmatched = sorted(m.loc[~m["geo_name"].isin(geo_names), dist_col])

            # --- Demo mode: fake/unmatched district names get assigned to real
            # Karnataka districts purely for visualization. Deterministic
            # (sorted + round-robin over free slots), so no reshuffle on rerun.
            demo_assigned = False
            if unmatched:
                # Default OFF. This fabricates geographic positions, and a
                # reviewer who misses the caption would read a real score
                # against the wrong district. Opt in deliberately, never
                # by default.
                demo = st.toggle(
                    "🎲 Demo mode: place unmatched districts on the map anyway",
                    value=False, key="map_demo",
                    help="Assigns each unmatched district a real Karnataka "
                         "district slot so the choropleth renders. Positions "
                         "are FAKE — colors/values are your real data. Use "
                         "only for previews, never in a submission.")
                if demo:
                    free = sorted(geo_names - set(matched["geo_name"]))
                    assign = {name: free[i % len(free)]
                              for i, name in enumerate(unmatched)}
                    m.loc[~m["geo_name"].isin(geo_names), "geo_name"] = (
                        m.loc[~m["geo_name"].isin(geo_names), dist_col].map(assign))
                    # If several data-districts landed on one map slot, keep
                    # the first (a slot can only be painted one color anyway)
                    overflow = len(m) - m["geo_name"].nunique()
                    m = m.drop_duplicates(subset="geo_name", keep="first")
                    matched = m[m["geo_name"].isin(geo_names)]
                    demo_assigned = True
                    if overflow > 0:
                        st.warning(f"⚠️ Your data has more districts than the "
                                   f"map has regions (30) — {overflow} "
                                   f"district(s) couldn't be placed and are "
                                   f"hidden in demo mode. All districts still "
                                   f"appear in every other tab.")

            mc1, mc2 = st.columns(2)
            mc1.metric("Districts matched to map", f"{len(matched)} / {len(m)}")
            mc2.metric("Map regions with data",
                       f"{len(matched)} / {len(geo_names)}")
            if unmatched and not demo_assigned:
                st.warning("Not on the map (fix spellings in DISTRICT_FIX): "
                           + ", ".join(unmatched[:15])
                           + (" …" if len(unmatched) > 15 else ""))
            if demo_assigned:
                st.info("🎲 Demo mode ON — district *positions* are randomly "
                        "assigned for visualization only; scores/colors are "
                        "real. Turn this off for actual Karnataka data.")

            if len(matched):
                _spread = float(matched["avg"].max() - matched["avg"].min())
                stretch = st.toggle(
                    "🎨 Stretch colors to data range", value=True,
                    help="ON: red = weakest district, green = strongest, however "
                         "small the true gap — good for spotting relative "
                         "differences. OFF: fixed 0–100 scale, honest absolute "
                         "view (a tight dataset will look uniformly one color, "
                         "because it genuinely is).")
                _rng = ([float(matched["avg"].min()), float(matched["avg"].max())]
                        if stretch and _spread > 0
                        else ([0, 100] if MQF["is_pct"] else None))
                _mttl = ("Avg " + _mlab.lower() if not MQF["narrowed"] else
                         f"{_mlab} on {len(MQF['questions'])} selected "
                         f"question(s)")
                figm = px.choropleth(
                    matched, geojson=_geo, locations="geo_name",
                    featureidkey="properties.district",
                    color="avg", color_continuous_scale="RdYlGn",
                    range_color=_rng,
                    hover_name=dist_col,
                    hover_data={"geo_name": demo_assigned,
                                "avg": ":.1f", "n": True},
                    labels={"avg": _mttl,
                            "n": "Students" if _msid else "Records",
                            "geo_name": "Shown at"},
                    height=650)
                if stretch and _spread < (2.0 if MQF["is_pct"] else 0.5):
                    st.caption(f"⚠️ True spread between districts is only "
                               f"{_spread:.2f} {'pts' if MQF['is_pct'] else ''} — "
                               "colors are stretched to show *relative* rank; "
                               "the differences are tiny in absolute terms.")
                figm.update_geos(fitbounds="locations", visible=False,
                                 bgcolor="rgba(0,0,0,0)")
                figm.update_layout(margin=dict(t=10, b=10, l=10, r=10),
                                   paper_bgcolor="rgba(0,0,0,0)",
                                   font=dict(color="#26303e"))
                _ev = st.plotly_chart(figm, use_container_width=True,
                                      on_select="rerun",
                                      selection_mode="points",
                                      key="map_click")
                st.caption(f"Color = **{_mttl.lower()}** per district (red weak → "
                           "green strong) · respects the sidebar filters and "
                           "the skill filter above · grey districts have no "
                           "data in the current selection · **click a district "
                           "to drill in**.")
                if MQF["narrowed"]:
                    _wd = matched.sort_values("avg").iloc[0]
                    _bd = matched.sort_values("avg").iloc[-1]
                    st.info(
                        f"🎯 On the selected "
                        + (", ".join(map(str, MQF["competencies"]))
                           if len(MQF["competencies"]) <= 3
                           else f"{len(MQF['questions'])} questions")
                        + f": weakest is **{_wd[dist_col]}** at "
                          f"{_wd['avg']:.1f}%, strongest is **{_bd[dist_col]}** "
                          f"at {_bd['avg']:.1f}% — a "
                          f"{_bd['avg'] - _wd['avg']:.1f}-point gap.")

                # ---------- click-to-drill: district detail panel ----------
                _sel_geo = None
                try:
                    _sel = getattr(_ev, "selection", None) or {}
                    _pts = (_sel.get("points", []) if isinstance(_sel, dict)
                            else getattr(_sel, "points", []) or [])
                    if _pts:
                        _p0 = dict(_pts[0])
                        _sel_geo = (_p0.get("location")
                                    or (_p0.get("properties") or {}).get("district")
                                    or _p0.get("hovertext"))
                        if _sel_geo is None:
                            _pi = _p0.get("point_index", _p0.get("point_number"))
                            if _pi is not None:
                                _sel_geo = matched.iloc[int(_pi)]["geo_name"]
                except Exception:
                    _sel_geo = None
                # data-name click payloads: map back to geo_name if needed
                if _sel_geo is not None and _sel_geo not in set(matched["geo_name"]):
                    _bk = matched[matched[dist_col].astype(str) == str(_sel_geo)]
                    _sel_geo = _bk["geo_name"].iloc[0] if len(_bk) else None
                # fallback / sync: dropdown always works even if clicks don't
                _opts = ["— none —"] + matched.sort_values("avg")[dist_col].tolist()
                _pick = st.selectbox("…or pick a district to inspect", _opts,
                                     key="map_pick")
                if _sel_geo is None and _pick != "— none —":
                    _sel_geo = matched.loc[matched[dist_col] == _pick,
                                           "geo_name"].iloc[0]

                if _sel_geo is not None and _sel_geo in set(matched["geo_name"]):
                    _row = matched[matched["geo_name"] == _sel_geo].iloc[0]
                    _dname = _row[dist_col]          # name as it exists in the DATA
                    _dd = mdf[mdf[dist_col] == _dname]
                    st.divider()
                    st.markdown(f"### 📍 {_dname}"
                                + (f"  ·  shown at *{_sel_geo}*"
                                   if demo_assigned else ""))

                    _zc1, _zc2 = st.columns([1, 1.4])
                    with _zc1:
                        # zoomed map: just this district's polygon
                        _one = {"type": "FeatureCollection",
                                "features": [f for f in _geo["features"]
                                             if f["properties"]["district"]
                                             == _sel_geo]}
                        _fz = px.choropleth(
                            matched[matched["geo_name"] == _sel_geo],
                            geojson=_one, locations="geo_name",
                            featureidkey="properties.district",
                            color_discrete_sequence=["#2f9e5f"],
                            hover_name=dist_col, height=300)
                        _fz.update_geos(fitbounds="locations", visible=False,
                                        bgcolor="rgba(0,0,0,0)")
                        _fz.update_layout(margin=dict(t=4, b=4, l=4, r=4),
                                          paper_bgcolor="rgba(0,0,0,0)",
                                          showlegend=False,
                                          coloraxis_showscale=False)
                        st.plotly_chart(_fz, use_container_width=True)

                    with _zc2:
                        _rank_all = (matched.sort_values("avg", ascending=False)
                                     .reset_index(drop=True))
                        _rk_pos = int(_rank_all[_rank_all["geo_name"]
                                                == _sel_geo].index[0]) + 1
                        _gapd = float(_row["avg"]) - float(matched["avg"].mean())
                        zk = st.columns(2)
                        zk[0].metric(_mttl, f"{_row['avg']:.1f}")
                        zk[1].metric("Students" if _msid else "Records",
                                     f"{int(_row['n']):,}")
                        zk2 = st.columns(2)
                        zk2[0].metric(f"Rank of {len(matched)}", f"#{_rk_pos}")
                        zk2[1].metric("vs district average", f"{_gapd:+.1f}",
                                      delta=f"{_gapd:+.1f}")

                    # per-skill profile for this district, vs everyone else.
                    # Uses the shared helper so it works on the melted shape
                    # too, and rolls up to named competencies when a map is
                    # loaded — previously this needed Q1..Qn COLUMNS, which
                    # the melted dataset does not have, so it never appeared.
                    _mov = question_means(mdf, MQF, _mby)
                    _mun = question_means(_dd, MQF, _mby)
                    _mdlt = (_mun - _mov).dropna().sort_values()
                    if len(_mdlt):
                        _mlabels = ([question_label(i) for i in _mdlt.index]
                                    if _mby == "Question" else list(_mdlt.index))
                        _fs = px.bar(x=_mdlt.values, y=_mlabels,
                                     orientation="h", color=_mdlt.values,
                                     color_continuous_scale="RdYlGn",
                                     color_continuous_midpoint=0,
                                     height=max(240, 24 * len(_mdlt) + 80),
                                     labels={"x": f"pts vs the state average",
                                             "y": ""})
                        _fs.update_traces(
                            hovertemplate="<b>%{y}</b>: %{x:+.1f} pts vs "
                                          "the state average<extra></extra>")
                        _fs.update_layout(coloraxis_showscale=False,
                                          margin=dict(t=30, b=8),
                                          paper_bgcolor="rgba(0,0,0,0)",
                                          font=dict(color="#26303e"),
                                          title=f"{_dname} by "
                                                f"{_mby.lower()} — where it "
                                                f"is ahead and behind")
                        st.plotly_chart(_fs, use_container_width=True)
                        _w3 = ", ".join(f"**{_mlabels[i]}** ({_mdlt.iloc[i]:+.1f})"
                                        for i in range(min(3, len(_mdlt))))
                        st.caption(f"Red = {_dname} scores below the state on "
                                   f"that {_mby.lower()}, green = above. "
                                   f"Its weakest: {_w3}.")

                    # blocks inside the district
                    _blk_col = next((c for c in mdf.columns
                                     if str(c).lower() == "block"), None)
                    if _blk_col:
                        _bl = _dd.groupby(_blk_col).agg(avg=(_mcol, "mean"),
                                                        size=_mcnt)
                        _bl = _bl[_bl["size"] >= 5]
                        if len(_bl):
                            _blp = (_bl.sort_values("avg").reset_index()
                                    .rename(columns={_blk_col: "Block"}))
                            _fb = px.bar(_blp, x="avg", y="Block",
                                         orientation="h",
                                         color="avg",
                                         color_continuous_scale="RdYlGn",
                                         height=max(220, 30 * len(_blp) + 70),
                                         labels={"avg": _mttl})
                            _fb.update_layout(coloraxis_showscale=False,
                                              yaxis_title="",
                                              margin=dict(t=26, b=8),
                                              paper_bgcolor="rgba(0,0,0,0)",
                                              font=dict(color="#26303e"),
                                              title="Blocks inside "
                                                    f"{_dname}")
                            st.plotly_chart(_fb, use_container_width=True)
                    # its year trend
                    if year_col and _dd[year_col].nunique() > 1:
                        _dt = _dd.groupby(year_col)[_mcol].mean()
                        _dtp = pd.DataFrame({
                            "Year": [f"{int(x)}-{(int(x)+1) % 100:02d}"
                                     for x in _dt.index],
                            "avg": _dt.round(1).values})
                        _ft = px.line(_dtp, x="Year", y="avg", markers=True,
                                      height=240, labels={"avg": _mttl})
                        _ft.update_traces(line_width=3, marker_size=10,
                                          line_color="#00e5ff")
                        _ft.update_layout(margin=dict(t=8, b=8),
                                          paper_bgcolor="rgba(0,0,0,0)",
                                          font=dict(color="#26303e"))
                        st.plotly_chart(_ft, use_container_width=True)
                    st.caption("Click a different district on the map above to "
                               "switch, or double-click empty space to clear.")
            else:
                st.error("0 districts matched the GeoJSON. Your district names "
                         "don't look like Karnataka districts — if this is dummy "
                         "data, regenerate it with real district names "
                         "(see generate_sample_data.py).")
        _map_drill_fragment()

# ============================================================================
#  Tab 10 — Facts & Health   (Layer 1 verbalize + Layer 3 status rules)
# ============================================================================
with tabs[11]:
    # fragment: widgets inside this tab rerun only this tab
    @st.fragment
    def _tab10_fragment():
        st.subheader("📄 Facts & Learning Health")
        if not _needs_agg():
            d_f = _pick_district("facts_dist")

            # ---- Layer 3: status rules -----------------------------------------
            st.markdown("##### Learning-health status per competency")
            ctable = L_verbalize.competency_table(AGG, d_f)
            ctable["status"] = ctable["below_pct"].apply(L_models.health_status)
            st.dataframe(ctable, use_container_width=True, hide_index=True)
            st.caption("Threshold rules, no model: below ≥ 60 → **Critical**, "
                       "≥ 45 → **At-risk**, otherwise **Strong**. Fully explainable — "
                       "'why Critical?' is answered by the number itself.")

            st.divider()

            # ---- Layer 1: verbalized facts -------------------------------------
            st.markdown("##### Verbalized facts")
            sents, joined = _c_verbalize_district(AGG, AGG_SIG, d_f)
            st.caption(f"**{len(sents)} sentences** generated by fixed Python templates "
                       f"from the aggregated numbers — every figure traces to an exact "
                       f"computation, and the wording is identical on every run.")
            show = st.slider("Sentences to show", 5, max(10, min(len(sents), 200)),
                             min(15, len(sents)), key="facts_n") if len(sents) > 5 else len(sents)
            st.code("\n".join(sents[:show]), language=None)
            st.download_button("⬇️ Download all facts (.txt)", joined,
                               file_name=f"facts_{d_f}.txt", key="facts_dl")

            # ---- brief clause traceability -------------------------------------
            with st.expander("How the district brief's clauses are computed"):
                st.dataframe(pd.DataFrame(L_brief.build_breakdown(AGG, d_f)),
                             use_container_width=True, hide_index=True)
    _tab10_fragment()


# ============================================================================
#  Tab 11 — Insights   (ranked findings from the generator suite)
# ============================================================================
with tabs[13]:
    # fragment: widgets inside this tab rerun only this tab
    @st.fragment
    def _tab11_fragment():
        st.subheader("🧠 Ranked Insights")
        if not _needs_agg():

            # ---- what the assessment file says, plus what district context
            # adds. Both families score in the same unit (points of the
            # outcome), so they rank against each other in one list.
            ic1, ic2 = st.columns([1, 1])
            d_ins = _pick_district("ins_dist")
            _lvl = ic1.radio(
                "Analyse district context at", list(L_cross.LEVELS),
                horizontal=True, key="ins_level",
                help="Districts rolled up into the group they belong to in "
                     "your data. Counts are summed, rates size-weighted. "
                     "Fewer units measure each one more precisely but leave "
                     "too few points to correlate — the tab says so when that "
                     "happens rather than reporting a number nobody should "
                     "read.")

            # the district-context file, via the shared loader
            _sec_path, _sec_name = _find_context_file()
            _ctx = _insight_context(_lvl)
            if _ctx:
                _u, _lv = _ctx["n_units"], _ctx["level"].lower()
                ic2.metric(f"{_ctx['level']}s in the analysis", _u,
                           help=f"smallest detectable |r| here: "
                                f"{_ctx['min_detectable_r']:.2f}")
                if _ctx.get("descriptive_only"):
                    ic2.caption(f"⚠️ {_u} {_lv}s — too few to correlate, so "
                                f"comparisons only.")
            elif _sec_path is None:
                ic2.caption("No district-context file found next to the app — "
                            "primary insights only.")
            else:
                ic2.caption("District names could not be matched to the "
                            "context file — primary insights only.")

            items = _c_insights_generate(
                AGG, AGG_SIG, d_ins, limit=8, min_n=MINN, _context=_ctx,
                ctx_sig=(f"{_lvl}:{_ctx['n_units']}" if _ctx else "none"))
            if not items:
                st.info(f"No findings passed the {MINN}-student minimum and "
                        f"the evidence threshold for this selection. Lower "
                        f"'Min students per group' in the sidebar to see "
                        f"weaker signals — they will be less reliable.")
            for i, it in enumerate(items, 1):
                _tag = ("🔗" if str(it.get("source", "")).startswith("x_")
                        else "📊")
                st.markdown(f"**{i}. {_tag} `{it['category']}`** — {it['text']}")
                st.caption(f"↳ {it['evidence']}"
                           + (f"  ·  `{it['source']}`" if it.get("source")
                              else ""))
            if items:
                st.caption("📊 = from the assessment file · 🔗 = from the "
                           "district-context join. Ranked by evidence — the "
                           "lower bound of the effect, so a big number from a "
                           "tiny group cannot outrank a solid one.")

            with st.expander("Generator coverage"):
                reg = pd.DataFrame(_c_insights_describe(AGG, AGG_SIG, d_ins,
                                                        min_n=MINN))
                if _ctx:
                    reg = pd.concat(
                        [reg, pd.DataFrame(L_cross.describe(_ctx, d_ins))],
                        ignore_index=True)
                st.dataframe(reg, use_container_width=True, hide_index=True)
                _errs = L_insights.ERRORS + L_cross.ERRORS
                if _errs:
                    st.error(f"⚠️ {len(_errs)} generator(s) raised: "
                             + "; ".join(f"{e['generator']}: {e['error']}"
                                         for e in _errs[:3]))
                if _ctx and _ctx.get("rollup_rules"):
                    st.caption("How the context file was rolled up: "
                               + " · ".join(f"**{k}** {v}" for k, v in
                                            list(_ctx["rollup_rules"].items())[:6])
                               + " …")

            f = L_charts.insight_scores(items)
            if f:
                st.plotly_chart(f, use_container_width=True)
    _tab11_fragment()


# ============================================================================
#  Tab 12 — Action Plan   (composed intervention recommendations)
# ============================================================================
with tabs[14]:
    # fragment: widgets inside this tab rerun only this tab
    @st.fragment
    def _tab12_fragment():
        st.subheader("📋 Action Plan")
        if not _needs_agg():
            d_act = _pick_district("act_dist")
            # the same district-context bundle the Insights tab uses, so an
            # action can say "…but this district already beats its
            # circumstances, so the fix is local"
            _actx = _insight_context()
            _asig = (f"{_actx['level']}:{_actx['n_units']}" if _actx else "none")
            cov = _c_playbook_coverage_v3(AGG, AGG_SIG, d_act, min_n=MINN,
                                          _context=_actx, ctx_sig=_asig)
            if cov:
                k = st.columns(5)
                k[0].metric("Recommendations", cov["recommendations_generated"])
                k[1].metric("Unique rule combos", cov["unique_rule_combinations"])
                k[2].metric("Possible combinations",
                            f"{cov['distinct_outputs']:,}",
                            help=f"{cov['base_actions']} base actions × "
                                 f"{cov['clause_subsets']} clause combinations "
                                 f"(up to {cov['max_clauses_shown']} shown at "
                                 f"once). Computed from the rules, not "
                                 f"hardcoded.")
                k[3].metric("With a peer model", cov["with_peer_model"])
                k[4].metric("Using district context",
                            cov.get("using_district_context", 0),
                            help="Actions whose advice changed because of the "
                                 "cross-dataset join.")

            recs = _c_playbook_recommend_v3(AGG, AGG_SIG, d_act, limit=12,
                                            min_n=MINN, _context=_actx,
                                            ctx_sig=_asig)
            if not recs:
                st.success(f"No block-competency pairing met the criteria for "
                           f"intervention in this selection — every group is "
                           f"either performing adequately or below the "
                           f"{MINN}-student minimum needed to act on.")
            for r in recs:
                with st.container(border=True):
                    head = f"**{r['priority']}** · {r['block']} — {r['competency']}"
                    if r["peer_model"]:
                        head += f"  ·  📎 model block: **{r['peer_model']}**"
                    st.markdown(head)
                    st.write(r["recommendation"])
                    st.caption(f"`{r['rule_fired']}`  ·  ~{r['children']:,} children"
                               + (f"  ·  also applies: {r['also_applies']}"
                                  if r["also_applies"] else ""))

            st.plotly_chart(L_charts.playbook_grid(AGG, d_act, min_n=MINN),
                            use_container_width=True)
    _tab12_fragment()


# ============================================================================
#  Tab 13 — Briefs   (role-based narrative)
# ============================================================================
with tabs[15]:
    # fragment: widgets inside this tab rerun only this tab
    @st.fragment
    def _tab13_fragment():
        st.subheader("📝 Role-Based Briefs")
        if not _needs_agg():
            d_br = _pick_district("br_dist")
            blocks = sorted(AGG[AGG["district"] == d_br]["block"].unique())
            c1, c2 = st.columns([2, 1])
            with c1:
                role = st.radio("Audience", list(L_brief.ROLES.keys()), horizontal=True,
                                format_func=lambda r: f"{L_brief.ROLES[r]['icon']} "
                                                      f"{L_brief.ROLES[r]['label']}",
                                key="br_role")
            with c2:
                blk = st.selectbox("Block (for the block brief)", blocks, key="br_blk")

            # briefs read the same cross-dataset bundle, so the policy brief
            # REPORTS the socio-economic result instead of recommending that
            # somebody go and test it
            _bctx = _insight_context()
            _bsig = (f"{_bctx['level']}:{_bctx['n_units']}" if _bctx else "none")
            b = _c_brief_build(AGG, AGG_SIG, d_br, role, blk, min_n=MINN,
                               _context=_bctx, ctx_sig=_bsig)
            if b:
                meta = L_brief.ROLES[role]
                st.caption(f"**Scope:** {b['scope']}  ·  **Sees:** {meta['scope']}  ·  "
                           f"**Acts over:** {meta['horizon']}")
                st.info(b["text"])
                st.json(b["metrics"], expanded=False)

            with st.expander("All three audiences side by side"):
                for rk, bb in _c_brief_build_all(AGG, AGG_SIG, d_br, blk).items():
                    if not bb:
                        continue
                    m = L_brief.ROLES[rk]
                    st.markdown(f"**{m['icon']} {m['label']}** — *{bb['scope']}*")
                    st.write(bb["text"])
                    st.divider()
    _tab13_fragment()


# ============================================================================
#  Tab 14 — Competency Report   (per-competency deep dive)
# ============================================================================
with tabs[16]:
    # fragment: widgets inside this tab rerun only this tab
    @st.fragment
    def _tab14_fragment():
        st.subheader("🎓 Competency Intelligence Report")
        if not _needs_agg():
            c1, c2 = st.columns(2)
            with c1:
                d_cp = _pick_district("cp_dist")
            with c2:
                comps = sorted(AGG[AGG["district"] == d_cp]["competency"].unique())
                comp_pick = st.selectbox("Competency", comps, key="cp_comp")

            rep = _c_competency_report(AGG, AGG_SIG, d_cp, comp_pick)
            if rep is None:
                st.info("Not enough data for this combination.")
            else:
                o = rep["overview"]
                k = st.columns(5)
                k[0].metric("Below grade", f"{o['below_pct']}%")
                k[1].metric("At grade", f"{o['at_pct']}%")
                k[2].metric("Above grade", f"{o['above_pct']}%")
                k[3].metric("Children below", f"{o['children_below']:,}")
                k[4].metric("Rank", o["rank"])

                st.info(rep["summary"])

                # Plain-language view of the same three numbers above. Added
                # alongside the metrics, not instead of them.
                with st.expander("👶 What this looks like in a class of 100",
                                 expanded=True):
                    _fw = L_charts.hundred_children(
                        o["below_pct"], o["at_pct"], o["above_pct"],
                        n_children=o.get("students_assessed"))
                    st.plotly_chart(_fw, use_container_width=True)
                    st.caption(
                        f"**Each dot is one child.** If {comp_pick} were tested "
                        f"on a class of 100 children in {d_cp}: "
                        f"**{round(o['below_pct'])} are below grade level** (red), "
                        f"{round(o['at_pct'])} are at grade level (yellow), and "
                        f"{round(o['above_pct'])} are above it (green). "
                        f"Across the whole district that is "
                        f"**{o['children_below']:,} children** who need support "
                        f"in this one skill.")

                a, b_ = st.columns(2)
                with a:
                    st.markdown("**By grade**")
                    st.dataframe(rep["grade"]["table"], use_container_width=True,
                                 hide_index=True)
                    st.caption(rep["grade"]["summary"])
                    st.markdown("**Gender**")
                    st.caption(rep["gender"]["summary"])
                with b_:
                    st.markdown("**By block**")
                    st.dataframe(rep["geography"]["table"], use_container_width=True,
                                 hide_index=True)
                    st.caption(rep["geography"]["summary"])
                    st.markdown("**Trend**")
                    st.dataframe(rep["trend"]["table"], use_container_width=True,
                                 hide_index=True)
                    st.caption(rep["trend"]["summary"])

                st.markdown("**Risk**")
                st.caption(rep["risk"]["summary"])

                f = L_charts.competency_block_bars(rep)
                if f:
                    st.plotly_chart(f, use_container_width=True)

                corr = _c_competency_corr(AGG, AGG_SIG, d_cp)
                if corr is not None:
                    with st.expander("Which competencies fail together?"):
                        fc = L_charts.competency_correlation(corr)
                        if fc:
                            st.plotly_chart(fc, use_container_width=True)
                        for p in L_competency.strongest_pairs(corr):
                            st.caption(f"{p['pair']} — r = {p['r']}")
                        st.caption("High correlation = the same units are weak in both, "
                                   "pointing at a shared root cause, so one intervention "
                                   "can address the pair.")
    _tab14_fragment()


# ============================================================================
#  Tab 14 — What-If scenario planner
# ============================================================================
with tabs[17]:
    # fragment: widgets inside this tab rerun only this tab
    @st.fragment
    def _tab15_fragment():
        st.subheader("🎛️ What-If Scenario Planner")
        if not _needs_agg():
            c1, c2, c3 = st.columns(3)
            with c1:
                d_wi = _pick_district("wi_dist")
            with c2:
                comps_wi = sorted(AGG[AGG["district"] == d_wi]["competency"].unique())
                comp_wi = st.selectbox("Competency to target", comps_wi, key="wi_comp")
            with c3:
                n_blocks = st.slider("How many weakest units can you reach?",
                                     1, 20, 10, key="wi_n")

            bench = _c_benchmarks(AGG, AGG_SIG)
            rebound = _c_rebound(AGG, AGG_SIG)
            w = _c_what_if(AGG, AGG_SIG, d_wi, comp_wi, n_blocks, min_n=MINN)

            if w is None or not w["scenarios"]:
                st.info("Not enough history in this selection to model a scenario. "
                        "What-If needs at least two years so improvement rates can be "
                        "measured from the data rather than assumed.")
            else:
                st.caption(f"Targeting {w['blocks_targeted']} units covering "
                           f"{w['students_covered']:,} students · currently "
                           f"**{w['before_below_pct']}%** below grade level")

                sc = pd.DataFrame(w["scenarios"])
                k = st.columns(3)
                for i, row in sc.iterrows():
                    k[i].metric(row["scenario"],
                                f"{row['after_below_pct']}%",
                                delta=f"-{row['net_of_rebound_pts']} pts",
                                delta_color="inverse")

                # The same scenarios counted in children instead of percentage
                # points — added above the table, which stays as it was.
                _fwi = L_charts.whatif_children(
                    w["before_below_pct"], w["students_covered"], w["scenarios"])
                if _fwi is not None:
                    st.plotly_chart(_fwi, use_container_width=True)
                    # Derive the headline the SAME way the bars do, so the
                    # sentence and the picture can never disagree by a rounding
                    # step (the model's own children_moved rounds net% instead).
                    _now = round(w["students_covered"]
                                 * w["before_below_pct"] / 100.0)
                    _best = max(_now - round(w["students_covered"]
                                             * s["after_below_pct"] / 100.0)
                                for s in w["scenarios"])
                    st.caption(
                        f"**How to read:** the top bar is today — "
                        f"**{_now:,} children** below grade level in "
                        f"{comp_wi} across the {w['blocks_targeted']} units you "
                        f"can reach. Each bar below is one scenario: the green "
                        f"part is how many of those children move above the "
                        f"line, the red part is how many are still behind. "
                        f"Best case here is **{_best:,} children**."
                        + ("" if bench["reliable"] else
                           " ⚠️ Group sizes are too small for these numbers to "
                           "be quoted — see the warning below."))

                st.dataframe(sc, use_container_width=True, hide_index=True)

                st.info(
                    f"⚠️ **These are scenarios, not forecasts.** Improvement rates come "
                    f"from gains actually observed in this dataset "
                    f"({bench['n_observed']} year-over-year improvements in groups of "
                    f"≥{bench['min_n_used']:.0f} students: median "
                    f"{bench['typical']:.1f} pts, 90th percentile {bench['best']:.1f} pts). "
                    f"A **{w['natural_rebound_pts']}-point natural rebound is subtracted** "
                    f"from every scenario, because the weakest units tend to improve "
                    f"somewhat on their own (regression to the mean) — we do not claim "
                    f"that as programme impact.")
                if not bench["reliable"]:
                    st.error(
                        "🚫 **Do not quote these numbers.** There are too few adequately "
                        "sized groups in this selection to measure real improvement "
                        "rates, so the benchmarks are dominated by sampling noise — in a "
                        "group of 2 students, one child changing answer reads as a "
                        "50-point swing. Raise **‘Report findings down to’** to District "
                        "or Division, and untick **‘Split by grade’**, until the median "
                        "group size is at least 30.")

                with st.expander("Units this scenario targets"):
                    st.write(", ".join(w["block_names"]))
    _tab15_fragment()


# ============================================================================
#  Tab 15 — Learning archetypes (KMeans) + risk model
# ============================================================================
with tabs[18]:
    # fragment: widgets inside this tab rerun only this tab
    @st.fragment
    def _tab16_fragment():
        st.subheader("🧬 Learning Archetypes & Risk")
        if not _needs_agg():
            st.markdown("##### Archetypes — units grouped by *pattern* of weakness")
            try:
                clusters, names = _c_cluster_blocks(AGG, AGG_SIG, k=3)
                st.dataframe(clusters, use_container_width=True, hide_index=True)
                st.caption("KMeans on the unit × competency matrix. Units in the same "
                           "archetype share a weakness profile, so one intervention "
                           "design can serve the whole group. Unsupervised — no labels, "
                           "no pre-trained model, fitted live on this data.")
            except Exception as e:
                st.info(f"Clustering needs at least 3 units and 2 competencies "
                        f"in the current selection. ({type(e).__name__})")

            st.divider()
            st.markdown("##### Early-warning model")
            ew = _c_train_early_warning(AGG, AGG_SIG)

            if not ew["ok"]:
                st.info(f"Not available for this selection — {ew['reason']}")
            else:
                st.caption(
                    f"Trained on {'/'.join(map(str, ew['train_years']))} → next year "
                    f"({ew['n_train']:,} rows). **Validated on {ew['test_year']} → "
                    f"{ew['test_year']+1}, a transition the model never saw** "
                    f"({ew['n_test']:,} rows).")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Error (MAE)", f"{ew['mae']} pts")
                m2.metric("Naive baseline", f"{ew['naive_mae']} pts",
                          help="'Assume nothing changes' — the bar any forecast must clear.")
                m3.metric("Better than naive", f"{ew['improvement_pct']}%",
                          delta="beats baseline" if ew["beats_naive"] else "FAILS baseline",
                          delta_color="normal" if ew["beats_naive"] else "inverse")
                m4.metric("R²", ew["r2"])

                if not ew["beats_naive"]:
                    st.error(
                        "🚫 **This model does not beat simply assuming nothing changes.** "
                        f"Its error is {ew['mae']} points against the naive baseline's "
                        f"{ew['naive_mae']} — so year-to-year movement at this "
                        "granularity is mostly noise, and no forecast is offered below. "
                        "The watchlist instead uses the naive rule, which is the better "
                        "predictor here.")
                    st.caption(
                        f"R² is {ew['r2']}, which looks strong and is misleading: a unit "
                        f"at 60% this year is near 60% next year, so simply copying last "
                        f"year's number already earns a high R². Error against the naive "
                        f"baseline is the test that matters, and this model fails it.")
                else:
                    st.success(
                        f"✅ On a held-out year the model is **{ew['improvement_pct']}% more "
                        f"accurate than assuming no change**, catching "
                        f"{ew['recall']*100:.0f}% of the units that actually ended up "
                        f"below the risk line (precision {ew['precision']*100:.0f}%).")

                # Never print a forecast under a banner telling the reader not to
                # trust one. When the model loses to persistence, show what the
                # naive rule flags — same rows, ranked by something defensible.
                if ew["beats_naive"]:
                    st.markdown(f"**Forecast for {ew['forecast_year']} — "
                                f"highest predicted risk**")
                    st.dataframe(ew["forecast"].head(15),
                                 use_container_width=True, hide_index=True)
                else:
                    _cut = ew.get("risk_cut", 50.0)
                    st.markdown(f"**Watchlist for {ew['forecast_year']} — "
                                f"units already above {_cut:.0f}% below grade level "
                                f"in {ew.get('last_observed_year', '')}**")
                    st.dataframe(ew["naive_watchlist"].head(15),
                                 use_container_width=True, hide_index=True)
                    st.caption(
                        f"Not a prediction — these are the units that are already "
                        f"struggling, on the assumption that nothing changes. On the "
                        f"held-out year that rule caught "
                        f"{ew.get('naive_recall', 0)*100:.0f}% of the units that did end "
                        f"up at risk (precision "
                        f"{ew.get('naive_precision', 0)*100:.0f}%), against the model's "
                        f"{ew['recall']*100:.0f}% / {ew['precision']*100:.0f}%. Acting on "
                        f"today's worst units is the defensible move when next year "
                        f"cannot be predicted.")
                    with st.expander("Model output anyway (not usable as a forecast)"):
                        st.dataframe(ew["forecast"].head(15),
                                     use_container_width=True, hide_index=True)

                with st.expander("What the model learned"):
                    st.dataframe(pd.DataFrame(
                        [{"feature": k, "coefficient": v}
                         for k, v in ew["coefficients"].items()]),
                        use_container_width=True, hide_index=True)
                    st.caption(
                        "Ridge regression on features known at the base year only — "
                        "current below%, above%, gender gap, group size, the unit's "
                        "overall level and the competency's overall difficulty. "
                        "Nothing from the future enters the features, so the held-out "
                        "score is honest.")
                    st.caption(
                        "This replaces an earlier model that predicted `below_pct ≥ 50` "
                        "while using `below_pct` as an input — circular, and it only "
                        "restated the present. This one predicts a different year.")
    _tab16_fragment()

# ============================================================================
#  Tab 17 — Cross-dataset: assessment outcomes x district context
#  ADDED by the analysis-layer work. Self-contained: reads AGG plus one
#  uploaded district-level file, calls secondary.py, renders. It does not
#  modify anything above this line.
# ============================================================================
with tabs[12]:
    @st.fragment
    def _tab17_fragment():
        st.subheader("🔗 Cross-dataset — does district context explain results?")

        if _needs_agg():
            return

        # ---- pick the context file --------------------------------------
        _sec_dir = os.path.dirname(_HERE)
        _sec_local = [f for f in _scan_local(_sec_dir)
                      if any(k in f.lower() for k in
                             ("secondary", "context", "district", "census",
                              "merged"))]
        c1, c2 = st.columns([1.2, 1])
        with c1:
            _sec_up = st.file_uploader(
                "District context file (one row per district)",
                type=["xlsx", "xls", "csv"], key="xds_up",
                help="Needs a District column plus numeric indicators — "
                     "income, literacy, teacher counts, libraries…")
        with c2:
            _sec_pick = st.selectbox("…or one already on disk",
                                     ["(none)"] + _sec_local, key="xds_local")

        _sec_df = None
        if _sec_up is not None:
            _sec_df = (pd.read_csv(_sec_up)
                       if _sec_up.name.lower().endswith(".csv")
                       else pd.read_excel(_sec_up))
        elif _sec_pick != "(none)":
            _p = os.path.join(_sec_dir, _sec_pick)
            _sec_df = (pd.read_csv(_p) if _p.lower().endswith(".csv")
                       else pd.read_excel(_p))
        else:
            # PRE-LOADED default: the fixed secondary dataset ships with the
            # app and never changes — no upload needed. The uploader above
            # remains as an override only.
            _default_sec = os.path.join(_HERE, "secondary_dataset.xlsx")
            if os.path.exists(_default_sec):
                _sec_df = pd.read_excel(_default_sec)
                st.success("📎 Using the built-in secondary dataset "
                           "(secondary_dataset.xlsx) — upload a file above "
                           "only to override it.")

        if _sec_df is None:
            st.info("⬆️ Upload a district-level context file to run the analysis.")
            return

        # ---- build the district-level outcome ---------------------------
        _outcome = "Below grade level (%)"
        _grp = AGG.groupby("district")
        _prim = (_grp.apply(lambda g: (g["below_pct"] * g["n"]).sum() / g["n"].sum(),
                            include_groups=False)
                 .rename(_outcome).reset_index()
                 .rename(columns={"district": "District"}))

        _key = next((c for c in _sec_df.columns
                     if str(c).strip().lower() in
                     ("district", "district name", "dist", "districts")), None)
        if _key is None:
            st.error(f"No District column in the context file. Columns found: "
                     f"{list(_sec_df.columns)[:8]}")
            return
        _sec_df = _sec_df.rename(columns={_key: "District"})

        _map, _rep = L_secondary.align_districts(
            _prim["District"].tolist(),
            _sec_df["District"].astype(str).tolist())
        _prim["District"] = _prim["District"].map(_map).fillna(_prim["District"])

        merged, _jr = L_secondary.join(_prim, _sec_df)
        if merged is None or merged.empty:
            st.error("Nothing joined — the two files name their districts "
                     "differently in a way the resolver could not bridge.")
            return

        res = L_secondary.analyse(merged, _outcome)
        tab, v, fit = res["table"], res["variance"], res["fit"]
        tested = tab[~tab["derived"]]
        rd = res.get("redundancy")
        _crit = tab.attrs.get("min_detectable_r", float("nan"))
        _nsig = int((tested["verdict"] ==
                     "significant after FDR correction").sum())
        _nred = int(rd["redundant"].sum()) if rd is not None and not rd.empty else 0

        st.markdown("### What we found")

        _lines = []
        if not v["ok"]:
            _lines.append(
                f"**Nothing to explain.** Every district scored almost the "
                f"same, so there is no difference for anything to cause.")
        elif _nsig:
            for r in tested[tested["verdict"] ==
                            "significant after FDR correction"].head(4).itertuples():
                _lines.append(
                    f"**{r.variable} matters.** Districts with more of it have "
                    f"{'fewer' if r.r < 0 else 'more'} children falling behind, "
                    f"and this is strong enough that we can rule out "
                    f"coincidence.")
        else:
            _t = tested.iloc[0] if len(tested) else None
            _good = tested[tested["r"] < 0].head(4)
            _lines.append(
                "**Nothing about a district reliably explains how its children "
                "do.** Not money, not libraries, not teacher numbers, not adult "
                "literacy. So you cannot tell which districts are struggling by "
                "looking at their circumstances — you have to look at the "
                "schools.")
            if len(_good):
                _names = ", ".join(_good["variable"].head(3))
                _lines.append(
                    f"**But a few point the right way.** {_names} all lean "
                    f"towards *more of it, fewer children behind* — around "
                    f"{_good['r'].abs().mean():.2f} on a scale where 1 would be "
                    f"a perfect match and 0 is nothing at all. That is too weak "
                    f"to prove with only {int(tested['n'].max())} districts, but "
                    f"it is not nothing either.")
            _lines.append(
                f"**Why we cannot prove it:** with {int(tested['n'].max())} "
                f"districts, pure chance alone throws up patterns this strong "
                f"about one time in twenty. Our strongest result is inside that "
                f"range. **With 220 blocks instead of 31 districts, the same "
                f"numbers would be strong enough to confirm.**")

        if _nred:
            _p0 = rd[rd["redundant"]].iloc[0]
            _lines.append(
                f"**Much of the district data repeats itself.** {_nred} pairs "
                f"move almost identically — {_p0['a']} and {_p0['b']}, for "
                f"instance. They are mostly measuring how *big* a district is, "
                f"not how *good* it is.")

        if int(tab["derived"].sum()):
            _lines.append(
                f"**{int(tab['derived'].sum())} columns were thrown out** "
                f"because they were the test results in disguise, not facts "
                f"about the district. Comparing the results to themselves would "
                f"always look like a perfect match.")

        for _l in _lines:
            st.markdown("- " + _l)

        # ---- the statistics, made visible ------------------------------
        _ff = L_charts.effect_forest(tested, crit=_crit)
        if _ff is not None:
            st.plotly_chart(_ff, use_container_width=True)
            st.markdown(
                f"""**Graph 1 — every indicator, with its uncertainty.**

| On the chart | Means |
|---|---|
| **White dot** | how strong the link looks (the correlation) |
| **Grey bar** | the range the true value could be in |
| **Red dashed line at 0** | no relationship at all |
| **Grey band** | too small to detect with {int(tested['n'].max()) if len(tested) else 0} districts |

**A bar touching the red line = we cannot even be sure of the direction.**
A dot inside the grey band = invisible at this sample size, however it looks.""")

        if len(tested):
            _t0 = tested.iloc[0]
            _sc = L_charts.relationship_scatter(
                merged, _t0["variable"], _outcome, label_col="District",
                r=float(_t0["r"]), p=float(_t0["p_raw"]))
            if _sc is not None:
                st.plotly_chart(_sc, use_container_width=True)
                _dirw = ("fewer" if _t0["r"] < 0 else "more")
                st.markdown(
                    f"""**Graph 2 — what that strongest link actually looks like.**

Each dot is one district: **{_t0['variable']}** across, **% of children below
grade level** up. Green = doing well, red = struggling. The dotted line is the
trend through them.

It slopes {'**down**' if _t0['r'] < 0 else '**up**'} — districts with more
{_t0['variable']} have {_dirw} children behind. But the dots sit a long way
from the line, and that scatter is exactly what r = {_t0['r']:+.2f} measures.
Graph 1 turns that scatter into the grey bar.""")

        # =================================================================
        #  the detail, folded away
        # =================================================================
    _tab17_fragment()