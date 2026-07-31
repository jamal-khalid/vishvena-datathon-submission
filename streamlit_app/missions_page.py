"""
🎯 National Missions page — fully self-contained, safe to drop into any
future version of the app without touching the dev's streamlit_app.py.

FILES it expects next to itself (all already in the repo):
    secondary_dataset.xlsx        district socio-economic context (never changes)
    DATATHON_QUESTION_MAP.csv     question -> competency map
    karnataka_districts.geojson   31-district boundaries

HOW TO INTEGRATE (2 lines for the dev, whenever ready):
    import missions_page
    ...
    tabs = st.tabs([...existing..., "🎯 Missions"])          # add one tab
    with tabs[-1]:
        missions_page.render(df=df, score_col=score_col)     # df optional!

`df`/`score_col` are OPTIONAL: without them (or with district names that
don't match real Karnataka districts) Sections 2-3 politely explain what
they need and Section 1 still works fully from the secondary dataset.

STANDALONE PREVIEW (no integration needed):
    streamlit run missions_page.py
"""

from __future__ import annotations

import json
import pathlib
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

_HERE = pathlib.Path(__file__).parent
MIN_MATCH = 5          # least matched districts for Sections 2-3 to render
_MAP_NAME_KEY = "properties.district"


_STANDALONE = False   # set True by the __main__ block below


def _theme():
    """Chart colors matching the page theme (light when standalone)."""
    if not _STANDALONE:
        try:
            if (st.get_option("theme.base") or "light") == "dark":
                return {"text": "#cdd3f7", "muted": "#9aa3c7",
                        "accent": "#5eead4"}
        except Exception:
            pass
    return {"text": "#1f2533", "muted": "#67707f", "accent": "#0F6E56"}


_LIGHT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stApp * { font-family: 'Inter', 'Segoe UI', sans-serif; }
span[data-testid="stIconMaterial"],
[class*="material-symbols"], [class*="material-icons"],
[data-testid="stExpanderToggleIcon"] {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined',
                 'Material Icons' !important; }
.stApp { background: #eef1f6; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp span { color: #26303e; }
.block-container { padding-top: 3.2rem; max-width: 1180px; }

h1, h2, h3 { color: #101828 !important; font-weight: 700 !important;
    letter-spacing: -0.3px; }
h2 { margin-top: .2rem !important; }

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
[data-testid="stSidebarNav"] { background: #111827; padding-top: 8px; }
[data-testid="stSidebar"] [role="radiogroup"] label,
[data-testid="stSidebar"] [data-baseweb="checkbox"] {
    background: transparent !important; }

/* section + component cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff; border: 1px solid #e6e9ef !important;
    border-radius: 16px; padding: 6px 10px;
    box-shadow: 0 1px 3px rgba(16,24,40,.06), 0 8px 24px rgba(16,24,40,.04);
    margin-bottom: 1.1rem; }

/* metric cards */
div[data-testid="stMetric"] { background: #ffffff;
    border: 1px solid #e6e9ef; border-top: 3px solid #0F6E56;
    border-radius: 14px; padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(16,24,40,.06); }
div[data-testid="stMetric"] label p { color: #67707f !important;
    font-size: 12px !important; text-transform: uppercase;
    letter-spacing: .6px; font-weight: 600 !important; }
div[data-testid="stMetricValue"] { color: #0F6E56 !important;
    font-weight: 800 !important; }

/* hero */
.mx-hero { background: linear-gradient(120deg, #07332a 0%, #0F6E56 55%, #14876a 100%);
    border-radius: 18px; padding: 26px 30px; margin: 0 0 1.2rem;
    box-shadow: 0 10px 30px rgba(15,110,86,.25); }
.mx-hero h1 { color: #ffffff !important; margin: 0 0 6px; font-size: 30px; }
.mx-hero p { color: #d7efe7 !important; margin: 0; font-size: 15px;
    max-width: 900px; }
.mx-kicker { display: inline-block; background: rgba(255,255,255,.14);
    color: #d7efe7; font-size: 12px; font-weight: 600; letter-spacing: 1px;
    text-transform: uppercase; padding: 4px 12px; border-radius: 999px;
    margin-bottom: 10px; }

/* section headers with icon badges */
.mx-sec { display: flex; align-items: center; gap: 12px;
    margin: .2rem 0 .6rem; }
.mx-sec .mx-ic { width: 40px; height: 40px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; background: #e5f3ee; flex: 0 0 40px; }
.mx-sec h2 { margin: 0 !important; font-size: 21px; }
.mx-sec .mx-sub { color: #67707f; font-size: 13px; margin: 2px 0 0; }

/* colored divider */
.mx-div { height: 3px; border: 0; border-radius: 3px; margin: 1.4rem 0;
    background: linear-gradient(90deg, #0F6E56, #5DCAA5 45%, transparent); }

/* top/bottom performer rows */
.mx-row { display: flex; align-items: center; gap: 10px;
    background: #f8faf9; border: 1px solid #e6e9ef; border-radius: 10px;
    padding: 8px 12px; margin-bottom: 6px; }
.mx-rank { width: 26px; height: 26px; border-radius: 8px; flex: 0 0 26px;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 700; color: #ffffff; }
.mx-rank.g { background: #0F6E56; } .mx-rank.r { background: #C0392B; }
.mx-name { flex: 1; font-weight: 600; color: #26303e; font-size: 14px; }
.mx-val { font-weight: 800; font-size: 14px; padding: 3px 10px;
    border-radius: 999px; }
.mx-val.g { background: #e5f3ee; color: #0a5340; }
.mx-val.r { background: #fbeae7; color: #8c2318; }
.mx-grp { font-size: 12px; font-weight: 700; letter-spacing: .6px;
    text-transform: uppercase; color: #67707f; margin: 4px 0 8px; }

/* code chips = formulas + items */
.stCode, pre, code { background: #f0f4f2 !important;
    color: #0a5340 !important; border-radius: 8px !important;
    border: 1px solid #dfe8e4 !important; font-size: 12.5px !important; }

/* controls */
[data-testid="stSelectbox"] > div > div { background: #ffffff;
    border: 1px solid #d8dde6 !important; border-radius: 10px; }
.stButton button, [data-testid="stFileUploader"] button {
    background: #0F6E56 !important; color: #ffffff !important;
    border: 0 !important; border-radius: 10px !important;
    font-weight: 600 !important; }
.stButton button:hover { background: #0a5340 !important; }
[data-testid="stFileUploaderDropzone"] { background: #ffffff;
    border: 1.5px dashed #c6cdd8 !important; border-radius: 12px; }

/* charts + tables inside white containers */
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
    background: #ffffff; border: 1px solid #e6e9ef; border-radius: 14px;
    padding: 2px; }
[data-testid="stExpander"] { background: #ffffff;
    border: 1px solid #e6e9ef !important; border-radius: 12px; }
[data-testid="stAlert"] { border-radius: 12px; border: 1px solid #e6e9ef; }
[data-testid="stHeader"] { background: #eef1f6; }
</style>
"""


def _hero(kicker: str, title: str, sub: str):
    if _STANDALONE:
        st.markdown(
            f"<div class='mx-hero'>"
            f"<span style='display:inline-block; background:rgba(255,255,255,.16); "
            f"color:#eafff7; font-size:12px; font-weight:600; letter-spacing:1px; "
            f"text-transform:uppercase; padding:4px 12px; border-radius:999px; "
            f"margin-bottom:10px;'>{kicker}</span>"
            f"<div style='color:#ffffff; font-size:30px; font-weight:800; "
            f"line-height:1.2; margin:0 0 8px; letter-spacing:-0.3px;'>{title}</div>"
            f"<div style='color:#d7efe7; font-size:15px; line-height:1.55; "
            f"max-width:900px;'>{sub}</div></div>",
            unsafe_allow_html=True)
    else:
        st.title(title)
        st.caption(sub)


def _sec_header(icon: str, title: str, sub: str = ""):
    if _STANDALONE:
        st.markdown(
            f"<div class='mx-sec'><div class='mx-ic'>{icon}</div><div>"
            f"<h2>{title}</h2>"
            + (f"<p class='mx-sub'>{sub}</p>" if sub else "")
            + "</div></div>", unsafe_allow_html=True)
    else:
        st.header(title)
        if sub:
            st.caption(sub)


def _cdiv():
    if _STANDALONE:
        st.markdown("<hr class='mx-div'>", unsafe_allow_html=True)
    else:
        st.divider()


def _apply_light_skin():
    st.markdown(_LIGHT_CSS, unsafe_allow_html=True)

# spellings seen in secondary data / likely real data -> geojson canonical
CANON = {
    "Chamarajanagar": "Chamarajanagara", "Chikkaballapur": "Chikkaballapura",
    "Chikballapur": "Chikkaballapura", "Vijayanagar": "Vijayanagara",
    "Bangalore": "Bengaluru Urban", "Bangalore Urban": "Bengaluru Urban",
    "Bangalore Rural": "Bengaluru Rural", "Bengaluru": "Bengaluru Urban",
    "Mysore": "Mysuru", "Belgaum": "Belagavi", "Gulbarga": "Kalaburagi",
    "Bijapur": "Vijayapura", "Bellary": "Ballari", "Shimoga": "Shivamogga",
    "Tumkur": "Tumakuru", "Chikmagalur": "Chikkamagaluru",
    "Bagalkot": "Bagalkote", "Davangere": "Davanagere",
    "Ramanagar": "Ramanagara", "Chamrajnagar": "Chamarajanagara",
}


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_secondary() -> pd.DataFrame:
    d = pd.read_excel(_HERE / "secondary_dataset.xlsx")
    d.columns = [str(c).strip() for c in d.columns]
    d["District"] = (d["District"].astype(str).str.strip()
                     .replace(CANON))
    d = d.set_index("District")
    # known defect: Total Household holds a copy-paste error for Bagalkote —
    # always derive the total ourselves
    d["HH"] = d["Rural Household"] + d["Urban Household"]
    d["lit_total"] = (d["Rural Total Literacy"] + d["Urban Total Literacy"])
    d["lit_f"] = d["Rural Female Literacy"] + d["Urban Female Literacy"]
    d["lit_m"] = d["Rural Male Literacy"] + d["Urban Male Literacy"]
    return d


@st.cache_data(show_spinner=False)
def _load_qmap() -> dict:
    p = _HERE / "DATATHON_QUESTION_MAP.csv"
    if not p.exists():
        return {}
    qm = pd.read_csv(p)
    qm.columns = [str(c).strip().lower() for c in qm.columns]
    return dict(zip(qm["question"].astype(str).str.strip(),
                    qm["competency"].astype(str).str.strip()))


@st.cache_data(show_spinner=False)
def _load_geo() -> dict:
    return json.loads((_HERE / "karnataka_districts.geojson").read_text())


def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


# --------------------------------------------------------------------------
# mission definitions — the four national missions
# each component: (label, formula shown to the user, series builder)
# --------------------------------------------------------------------------
def _mission_specs(d: pd.DataFrame) -> dict:
    return {
        "🤖 IndiaAI Mission": {
            "index_name": "CII — Contribution to IndiaAI Index",
            "about": (
                "The **IndiaAI Mission** is the Government of India's "
                "national programme (approved March 2024, ≈₹10,300 crore "
                "over five years) to build the country's AI ecosystem: "
                "compute infrastructure, foundation models, quality "
                "datasets, AI skilling, and support for AI startups — "
                "with an explicit goal of taking AI capability beyond the "
                "big metros."),
            "point": (
                "**Why look at districts?** AI capability does not appear "
                "from nowhere — it grows where there is economic capacity, "
                "a literate population, knowledge infrastructure and urban "
                "density. This page asks: *which Karnataka districts are "
                "ready to contribute to IndiaAI today, which will need "
                "groundwork first — and are the children in those "
                "districts building the skills the mission will need?*"),
            "intro": "",
            "components": [
                ("Economic base", "log(per-capita income)",
                 np.log(d["Per Capita Income"]),
                 "AI activity needs purchasing power and investment "
                 "capacity; income is its broadest proxy (log-scaled so "
                 "Bengaluru's outlier income doesn't drown everyone else)."),
                ("Knowledge access", "libraries ÷ households × 100000",
                 d["Total Libraries"] / d["HH"] * 100000,
                 "Libraries are the district's public knowledge "
                 "infrastructure — the same access layer digital skilling "
                 "programmes build on."),
                ("Literate base", "literates ÷ households",
                 d["lit_total"] / d["HH"],
                 "A digital-and-AI workforce is drawn from the literate "
                 "population; this measures how much of it surrounds each "
                 "household."),
                ("Urbanisation", "urban households ÷ total households",
                 d["Urban Household"] / d["HH"],
                 "Connectivity, institutions and tech employers cluster in "
                 "urban areas — urban share captures that density."),
            ],
            "competencies": ["Logical Reasoning", "Problem Solving",
                             "Numeracy"],
            "comp_whys": {
                "Logical Reasoning": "The core cognitive skill of AI work "
                    "— pattern recognition, if-then thinking, deduction.",
                "Problem Solving": "Turning a messy situation into "
                    "solvable steps: what engineers and data scientists "
                    "do all day.",
                "Numeracy": "All of AI stands on mathematics; comfort "
                    "with numbers is the entry ticket."},
            "comp_reason": ("No single skill makes an AI-ready child — but "
                            "a district whose children can reason, "
                            "decompose problems AND handle numbers is "
                            "growing exactly the talent pipeline the "
                            "IndiaAI Mission will recruit from in 10 "
                            "years. That is why we read the three "
                            "cumulatively: their combined average is the "
                            "district's contribution signal."),
        },
        "🧮 NIPUN Bharat": {
            "index_name": "FLN Capacity Index",
            "about": (
                "**NIPUN Bharat** (launched 2021 under Samagra Shiksha) is "
                "the national mission to guarantee every child foundational "
                "literacy and numeracy by the end of Grade 3/5 — the exact "
                "grades and skills our assessment measures."),
            "point": (
                "**Why look at districts?** Foundational learning is "
                "delivered by teachers in classrooms. This page asks: "
                "*which districts have the capacity to teach the "
                "fundamentals — enough teachers, small enough classes — "
                "and does that capacity show up in children's actual "
                "foundational scores?*"),
            "intro": "",
            "components": [
                ("Teacher availability", "primary teachers ÷ 1000 enrolled",
                 d["Primary Teachers Total"] / d["Student Enrolment"] * 1000,
                 "More teachers per 1,000 enrolled children = more "
                 "instruction per child."),
                ("Class size", "− student-teacher ratio (lower is better)",
                 -d["Student Teacher Ratio"],
                 "Crowded classes are the enemy of foundational teaching; "
                 "the ratio is inverted so smaller classes score higher."),
                ("Female teachers", "female ÷ total primary teachers",
                 d["Primary Teachers Female"] / d["Primary Teachers Total"],
                 "Female teachers are consistently linked to better "
                 "early-grade outcomes and girls' attendance."),
                ("Knowledge access", "libraries ÷ households × 100000",
                 d["Total Libraries"] / d["HH"] * 100000,
                 "Reading material beyond the textbook is what FLN "
                 "practice runs on."),
            ],
            "competencies": ["Numeracy", "Arithmetic", "Problem Solving"],
            "comp_whys": {
                "Numeracy": "Number sense — the first pillar NIPUN "
                    "Bharat names.",
                "Arithmetic": "Operations fluency — the second pillar, "
                    "and the gateway to all later maths.",
                "Problem Solving": "Applying both to a real question — "
                    "the proof that the foundation actually holds."},
            "comp_reason": ("Together these three ARE foundational "
                            "numeracy — the exact outcome NIPUN Bharat "
                            "exists to secure by Grade 5. Their combined "
                            "average is each district's FLN achievement."),
        },
        "👧 Beti Bachao Beti Padhao": {
            "index_name": "Girls' Education Environment Index",
            "about": (
                "**Beti Bachao Beti Padhao** (launched 2015) is the "
                "national campaign for the girl child; its *Beti Padhao* "
                "pillar targets equal education for girls."),
            "point": (
                "**Why look at districts?** A girl's chance at education "
                "is set by the environment around her. This page asks: "
                "*in which districts is female literacy closest to male "
                "literacy, where do girls see women teaching — and do "
                "girls actually score better where the environment "
                "supports them?*"),
            "intro": "",
            "components": [
                ("Literacy parity", "female literates ÷ male literates",
                 d["lit_f"] / d["lit_m"],
                 "The core equality measure: 1.0 means as many literate "
                 "women as men."),
                ("Rural parity", "rural female ÷ rural male literates",
                 d["Rural Female Literacy"] / d["Rural Male Literacy"],
                 "The gender gap concentrates in villages — rural parity "
                 "is the stricter test."),
                ("Role models", "female ÷ total primary teachers",
                 d["Primary Teachers Female"] / d["Primary Teachers Total"],
                 "Girls who see women teaching stay in school longer."),
            ],
            "competencies": ["Numeracy", "Arithmetic", "Logical Reasoning"],
            "comp_whys": {
                "Numeracy": "The subject with the most persistent "
                    "gender stereotype — where girls' progress is the "
                    "sharpest signal.",
                "Arithmetic": "Fluency that translates directly into "
                    "staying with maths in higher grades.",
                "Logical Reasoning": "Higher-order thinking — beyond "
                    "rote, the skill that predicts girls continuing "
                    "into STEM."},
            "comp_reason": ("Scored on GIRLS only. The mission question "
                            "is not whether the district scores well — "
                            "it is how girls themselves perform where "
                            "the environment supports (or fails) them, "
                            "read across the three skills together."),
            "girls_only": True,
        },
        "📖 ULLAS (Adult Literacy)": {
            "index_name": "Community Literacy Index",
            "about": (
                "**ULLAS — the New India Literacy Programme** (2022-2027) "
                "targets adult literacy: crores of adult non-literates, "
                "because the home is a child's first classroom."),
            "point": (
                "**Why look at districts?** Literate homes produce "
                "school-ready children — one of the most robust findings "
                "in education research. This page asks: *how much adult "
                "literacy surrounds each district's children, especially "
                "in villages — and do children learn better where the "
                "community around them reads?*"),
            "intro": "",
            "components": [
                ("Adult literacy stock", "literates ÷ households",
                 d["lit_total"] / d["HH"],
                 "The amount of adult literacy surrounding an average "
                 "household — the child's home learning environment."),
                ("Rural literacy", "rural literates ÷ rural households",
                 d["Rural Total Literacy"] / d["Rural Household"],
                 "Adult illiteracy concentrates in villages; ULLAS effort "
                 "is needed most where this is low."),
                ("Even reach", "− (urban rate − rural rate)",
                 -(d["Urban Total Literacy"] / d["Urban Household"]
                   - d["Rural Total Literacy"] / d["Rural Household"]),
                 "A small town-village gap means literacy reaches "
                 "everywhere, not just district headquarters."),
            ],
            "competencies": ["Logical Reasoning", "Problem Solving",
                             "Geometry"],
            "comp_whys": {
                "Logical Reasoning": "Grows from conversation and "
                    "reading at home more than from drilling.",
                "Problem Solving": "Word-problem comprehension — a "
                    "child must READ the problem before solving it.",
                "Geometry": "Spatial language and reasoning, strongly "
                    "shaped by the vocabulary of the home."},
            "comp_reason": ("The three most comprehension-heavy skills — "
                            "the ones a literate home environment shapes "
                            "most and rote drilling shapes least. Their "
                            "combined average is where community "
                            "literacy should leave its fingerprint."),
        },
    }


def _index_series(spec: dict) -> pd.Series:
    parts = [_z(c[2]) for c in spec["components"]]
    return pd.concat(parts, axis=1).mean(axis=1)


# --------------------------------------------------------------------------
# primary-data helpers
# --------------------------------------------------------------------------
def _primary_district_scores(df, spec, qmap):
    """Per-district avg % on the mission's 3 competencies. None if impossible."""
    if df is None or not qmap:
        return None, "no data"
    dist_col = next((c for c in df.columns
                     if str(c).strip().lower() == "district"), None)
    if dist_col is None:
        return None, "no district column"
    comp_items = {}
    for q, comp in qmap.items():
        comp_items.setdefault(comp, []).append(q)
    items = [q for comp in spec["competencies"]
             for q in comp_items.get(comp, []) if q in df.columns]
    if not items:
        return None, "no matching item columns"
    sub = df
    if spec.get("girls_only"):
        gcol = next((c for c in df.columns
                     if str(c).strip().lower() == "gender"), None)
        if gcol is not None:
            g = df[gcol].astype(str).str.strip().str.lower()
            sub = df[g.isin(["f", "female", "girl"])]
    s = (sub.groupby(sub[dist_col].astype(str).str.strip()
                     .replace(CANON))[items].mean().mean(axis=1) * 100)
    s.index.name = "District"
    return s, None


# --------------------------------------------------------------------------
# small UI helpers
# --------------------------------------------------------------------------
def _centroids(geo) -> dict:
    out = {}
    for ft in geo["features"]:
        gm = ft["geometry"]
        polys = (gm["coordinates"] if gm["type"] == "MultiPolygon"
                 else [gm["coordinates"]])
        pts = [p for poly in polys for p in poly[0]]
        out[ft["properties"]["district"]] = (
            sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts))
    return out


def _choropleth(values: pd.Series, title: str, key: str,
                colorbar: str, fmt: str = ":.2f",
                mark_top=None, mark_bottom=None):
    geo = _load_geo()
    m = values.dropna().rename("v").reset_index()
    fig = px.choropleth(
        m, geojson=geo, locations="District",
        featureidkey=_MAP_NAME_KEY, color="v",
        color_continuous_scale="RdYlGn",
        range_color=[float(m["v"].min()), float(m["v"].max())],
        hover_name="District", height=620)
    fig.update_traces(marker_line_color="#ffffff",
                      marker_line_width=1.3)
    fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>"
                      + colorbar + ": %{z" + fmt + "}<extra></extra>")
    if mark_top or mark_bottom:
        import plotly.graph_objects as go
        cent = _centroids(geo)
        for names, symbol in [(mark_top or [], "🏆"),
                              (mark_bottom or [], "⚠")]:
            pts = [(cent[n], n) for n in names if n in cent]
            if pts:
                fig.add_trace(go.Scattergeo(
                    lon=[p[0][0] for p in pts], lat=[p[0][1] for p in pts],
                    text=[symbol] * len(pts), mode="text",
                    textfont=dict(size=16), hoverinfo="skip",
                    showlegend=False))
    fig.update_geos(fitbounds="locations", visible=False,
                    bgcolor="rgba(0,0,0,0)",
                    domain=dict(x=[0, 1], y=[0.08, 1]))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=0, b=0, l=0, r=0),
                      font=dict(color=_theme()["text"]),
                      coloraxis_colorbar=dict(
                          title=dict(text=colorbar, side="top"),
                          orientation="h", y=-0.02, x=0.5, xanchor="center",
                          thickness=10, len=0.55, outlinewidth=0,
                          tickfont=dict(color=_theme()["text"], size=11)))
    st.plotly_chart(fig, width="stretch", key=key,
                    config={"scrollZoom": True, "displaylogo": False})


def _rows_html(items, cls):
    return "".join(
        f"<div class='mx-row'><div class='mx-rank {cls}'>{r}</div>"
        f"<div class='mx-name'>{k}</div>"
        f"<div class='mx-val {cls}'>{v}</div></div>"
        for r, k, v in items)


def _top_bottom(values: pd.Series, unit_label: str, fmt: str = "{:+.2f}"):
    s = values.dropna().sort_values(ascending=False)
    n = len(s)
    top = [(i, k, fmt.format(v))
           for i, (k, v) in enumerate(s.head(3).items(), 1)]
    bot = [(n - i, k, fmt.format(v))
           for i, (k, v) in enumerate(s.tail(3).iloc[::-1].items())]
    if _STANDALONE:
        st.markdown(f"<p class='mx-grp'>🏆 Top 3 · {unit_label}</p>"
                    + _rows_html(top, "g"), unsafe_allow_html=True)
        st.markdown(f"<p class='mx-grp'>⚠️ Bottom 3 · {unit_label}</p>"
                    + _rows_html(bot, "r"), unsafe_allow_html=True)
    else:
        st.markdown(f"**🏆 Top 3 — {unit_label}**")
        for r, k, v in top:
            st.markdown(f"{r}. {k} — `{v}`")
        st.markdown(f"**⚠️ Bottom 3 — {unit_label}**")
        for r, k, v in bot:
            st.markdown(f"{r}. {k} — `{v}`")
    missing = 31 - len(s)
    if missing > 0:
        st.caption(f"{missing} district(s) not scored — missing values in "
                   "the secondary dataset (e.g. Vijayanagara has only "
                   "income data).")


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------
def render(df: pd.DataFrame | None = None, score_col: str | None = None,
           qmap: dict | None = None):
    """Render the missions page. Every argument is optional."""
    if _STANDALONE:
        _apply_light_skin()
    sec = _load_secondary()
    qmap = qmap or _load_qmap()
    specs = _mission_specs(sec)

    _pick_host = st.sidebar if _STANDALONE else st
    mission = _pick_host.selectbox("Mission", list(specs.keys()),
                                   key="mission_pick")
    spec = specs[mission]
    idx = _index_series(spec)

    # ---------------- Section 0 — the mission -------------------------
    _hero("National mission · Section 0",
          mission,
          spec["about"].replace("**", ""))
    with st.container(border=True):
        st.write(spec["point"])
    z0a, z0b, z0c = st.columns(3)
    z0a.metric("Districts analysed", int(idx.notna().sum()))
    z0b.metric("Data sources", "2",
               help="Secondary: district socio-economic context. "
                    "Primary: student assessment (Q1-Q20).")
    z0c.metric("Index", spec["index_name"].split("—")[0].strip())
    _cdiv()

    # ---------------- Section 1 — secondary dataset only ---------------
    _sec_header("🧭", f"Section 1 · {spec['index_name']}",
                "Secondary dataset only — no assessment data enters this "
                "section")
    st.markdown("**How we calculate it** — each component is a simple, "
                "checkable ratio from the district statistics:")
    cols = st.columns(len(spec["components"]))
    for col, comp in zip(cols, spec["components"]):
        label, formula, series, why = comp
        with col.container(border=True):
            st.markdown(f"**{label}**")
            st.code(formula, language=None)
            st.caption(why)
    st.markdown(
        f"Each component is standardised (z-score: how many standard "
        f"deviations a district sits from the state average), then the "
        f"**{spec['index_name'].split('—')[0].strip()} = the average of "
        f"the {len(spec['components'])} standardised components**. "
        "0 = a typical district; +1 well above the state norm; −1 well "
        "below. No single component can dominate, and every number can "
        "be traced back to the source table.")
    c_map, c_tbl = st.columns([1.6, 1])
    with c_map:
        _choropleth(idx, spec["index_name"], f"map1_{mission}", "Index")
    with c_tbl:
        _top_bottom(idx, spec["index_name"])
        with st.expander("All districts"):
            st.dataframe(idx.dropna().sort_values(ascending=False)
                         .round(2).rename("Index"))

    # ---------------- Section 2 — primary dataset only ------------------
    _cdiv()
    _sec_header("🎯", "Section 2 · The 3 competencies we chose",
                "Primary dataset only — children's actual answers on the "
                "assessment, nothing else")
    _ci = {q: c for q, c in (qmap or {}).items()}
    _by_comp = {}
    for q, c in _ci.items():
        _by_comp.setdefault(c, []).append(q)
    ccols = st.columns(3)
    for col, comp in zip(ccols, spec["competencies"]):
        with col.container(border=True):
            st.markdown(f"**{comp}**")
            qs = sorted(_by_comp.get(comp, []),
                        key=lambda x: int(re.sub(r"\D", "", x) or 0))
            st.code(", ".join(qs) if qs else "items n/a", language=None)
            st.caption(spec.get("comp_whys", {}).get(comp, ""))
    st.markdown("**Why these three, read together:** " + spec["comp_reason"])
    st.caption("District score = average % correct across all items of the "
               "three competencies"
               + (" · girls only" if spec.get("girls_only") else "")
               + " · every child counts equally.")

    prim, why = _primary_district_scores(df, spec, qmap)
    matched = None
    if prim is not None:
        matched = prim[prim.index.isin(idx.index)]
    if prim is None:
        st.info("Needs the primary dataset (upload it in the sidebar) and "
                f"the question map. ({why})")
    elif matched is None or len(matched) < MIN_MATCH:
        st.info(f"Only {0 if matched is None else len(matched)} of the "
                "primary dataset's districts match real Karnataka district "
                "names — the map and rankings activate when real (or "
                "realistically named) data is loaded.")
    else:
        c_map2, c_tbl2 = st.columns([1.6, 1])
        with c_map2:
            _choropleth(matched, "Avg % correct", f"map2_{mission}",
                        "% correct", fmt=":.1f",
                        mark_top=matched.nlargest(3).index.tolist(),
                        mark_bottom=matched.nsmallest(3).index.tolist())
            st.caption("Color = district score on the 3 competencies · "
                       "🏆 = top 3 · ⚠ = bottom 3")
        with c_tbl2:
            _top_bottom(matched, "% correct on these competencies",
                        fmt="{:.1f}%")
            with st.expander("All districts"):
                st.dataframe(matched.sort_values(ascending=False)
                             .round(1).rename("% correct"))

    # ---------------- Section 3 ----------------------------------------
    _cdiv()
    _sec_header("🔗", "Section 3 · Merged — environment × achievement",
                "Both datasets joined per district")
    if prim is None or matched is None or len(matched) < MIN_MATCH:
        st.info("The merged view joins the index above with district "
                "performance — it activates together with Section 2.")
        return
    both = pd.DataFrame({"index": idx, "score": matched}).dropna()
    r = float(np.corrcoef(both["index"], both["score"])[0, 1])
    merged_rank = (_z(both["index"]) + _z(both["score"])) / 2

    k1, k2, k3 = st.columns(3)
    k1.metric("Districts merged", len(both))
    k2.metric("Correlation (index ↔ score)", f"r = {r:+.2f}",
              help="+1 = environment fully explains performance, 0 = no "
                   "relationship. Judge with the scatter, not the number "
                   "alone.")
    k3.metric("Strongest merged district",
              merged_rank.idxmax())

    xm, ym = float(both["index"].median()), float(both["score"].median())
    figq = px.scatter(both.reset_index(), x="index", y="score",
                      text="District", height=460,
                      labels={"index": spec["index_name"],
                              "score": "Avg % correct (3 competencies)"})
    figq.update_traces(marker=dict(size=11, color=_theme()["accent"]),
                       textposition="top center",
                       textfont=dict(size=10),
                       hovertemplate="<b>%{text}</b><br>index %{x:.2f} · "
                                     "score %{y:.1f}%<extra></extra>")
    figq.add_vline(x=xm, line_dash="dot", line_color="gray")
    figq.add_hline(y=ym, line_dash="dot", line_color="gray")
    for tx, ty, lab in [(0.02, 0.98, "learn from them"),
                        (0.98, 0.98, "showcase"),
                        (0.02, 0.02, "invest"),
                        (0.98, 0.02, "investigate")]:
        figq.add_annotation(xref="paper", yref="paper", x=tx, y=ty,
                            text=lab, showarrow=False,
                            font=dict(size=11, color=_theme()["muted"]))
    figq.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)",
                       margin=dict(t=10, b=10),
                       font=dict(color=_theme()["text"]))
    st.plotly_chart(figq, width="stretch", key=f"quad_{mission}")
    st.caption("**How to read:** each dot = one district. Right = better "
               "environment, up = better learning. Bottom-left = invest "
               "(they lack the means) · bottom-right = investigate "
               "(means exist, results don't) · top-left = learn from them "
               "(overachievers) · top-right = showcase.")

    _top_bottom(merged_rank, "merged rank (environment + achievement)")


# --------------------------------------------------------------------------
# standalone preview:  streamlit run missions_page.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    _STANDALONE = True
    st.set_page_config(page_title="National Missions", page_icon="🎯",
                       layout="wide")
    _up = st.sidebar.file_uploader(
        "Primary dataset (optional — activates Sections 2-3)",
        type=["xlsx", "csv", "parquet"])
    _pdf = None
    if _up is not None:
        _n = _up.name.lower()
        _pdf = (pd.read_csv(_up) if _n.endswith(".csv")
                else pd.read_parquet(_up) if _n.endswith(".parquet")
                else pd.read_excel(_up))
    render(df=_pdf)