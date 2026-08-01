"""
Mission Intelligence — NIPUN Bharat + PARAKH pages.

Deterministic analytics only (datathon rule: no LLM). Every number on these
pages is computed from the loaded GP-contest data; nothing is fabricated.

Data notes this module is built around:
- the paper CHANGES every year and grade (9 papers), so competency accuracy
  is always computed per (year, grade) against that paper's own mapping;
- cross-year movement uses percentile standing among districts covered in
  every year (papers and coverage both changed);
- secondary data is DISTRICT level only — it is context, never used to
  explain a Block/GP, and association is never phrased as causation.
"""
import json
import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))

# competency families used by NIPUN framing (foundational = FLN core)
FOUNDATIONAL = ["number sense", "place value", "addition", "subtraction"]
OPERATIONS = ["multiplication", "division", "fraction"]
APPLIED = ["measurement", "shapes", "data handling"]

ACCENT = "#0F6E56"
CLASS_COLORS = {"Critical": "#d73027", "Needs Attention": "#fc8d59",
                "Moderate": "#fee08b", "Healthy": "#1a9850"}


# --------------------------------------------------------------------------
#  data access — session first, refresh-proof cache as fallback
# --------------------------------------------------------------------------
def _year_key(y):
    m = re.search(r"(20\d\d)", str(y))
    return int(m.group(1)) if m else None


def _year_label(y):
    yk = _year_key(y)
    return f"{yk}-{(yk + 1) % 100:02d}" if yk is not None else str(y)


def load_context():
    """(df, rqmaps, err) from the running app or the upload cache."""
    df = st.session_state.get("_mx_primary_df")
    rq = st.session_state.get("_mx_qmap")
    if df is None:
        cache = os.path.join(_HERE, "_upload_cache", "last_upload.parquet")
        if os.path.exists(cache):
            df = pd.read_parquet(cache)
            try:
                with open(cache + ".qmaps.json") as f:
                    j = json.load(f)
                rq = {(k.split("|")[0], int(k.split("|")[1])): v
                      for k, v in j.get("maps", {}).items()}
            except Exception:
                rq = None
    if df is None:
        return None, None, ("No dataset loaded. Open the main DataTalk page, "
                            "upload the GP-contest data, then return here.")
    if not isinstance(rq, dict) or not any(isinstance(k, tuple) for k in rq):
        return None, None, ("This page needs the per-paper competency maps "
                            "from the GP-contest workbooks. Upload the real "
                            "year folders on the main page first.")
    dcol = next((c for c in df.columns if str(c).lower() == "district"), None)
    ycol = next((c for c in df.columns if str(c).lower() == "year"), None)
    gcol = next((c for c in df.columns if str(c).lower() == "grade"), None)
    if not (dcol and ycol and gcol):
        return None, None, "District/Year/Grade columns missing."
    return df, {(str(y), int(g)): m for (y, g), m in rq.items()}, None


def _sig(df):
    return f"{len(df)}:{df.columns.size}:{float(pd.to_numeric(df.iloc[:200].select_dtypes('number').sum().sum(), errors='coerce') or 0):.0f}"


def _qcols(df):
    return [c for c in df.columns if re.fullmatch(r"Q\d+", str(c))]


# --------------------------------------------------------------------------
#  §37 shared analytics — pure, per-paper correct, cached
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def competency_accuracy(_df, sig, by=(), girls_only=False):
    """Accuracy % per competency, per optional group keys, honoring each
    (year, grade) paper's own mapping. by ⊆ {'Year','Grade','District',...}"""
    df, rq, _ = load_context()
    qs = _qcols(df)
    ycol = next(c for c in df.columns if str(c).lower() == "year")
    gcol = next(c for c in df.columns if str(c).lower() == "grade")
    sub = df
    if girls_only:
        gc = next((c for c in df.columns if str(c).lower() == "gender"), None)
        if gc is not None:
            sub = df[df[gc].astype(str).str.strip().str.lower()
                     .isin(["f", "female", "girl"])]
    rqn = {(_year_key(y), int(g)): m for (y, g), m in rq.items()}
    out = []
    for (y, g), idx in sub.groupby([sub[ycol].map(_year_key),
                                    pd.to_numeric(sub[gcol],
                                                  errors="coerce")]).groups.items():
        m = rqn.get((y, int(g))) if pd.notna(g) else None
        if not m:
            continue
        chunk = sub.loc[idx]
        comp_items = {}
        for q, c in m.items():
            if q in qs:
                comp_items.setdefault(c, []).append(q)
        for comp, items in comp_items.items():
            per = chunk[items].mean(axis=1)          # per child, own paper
            row = {"Year": f"{y}-{(y + 1) % 100:02d}", "Grade": int(g),
                   "competency": comp}
            if by:
                for keys, gidx in per.groupby(
                        [chunk[b] for b in by if b in chunk.columns]).groups.items():
                    keys = keys if isinstance(keys, tuple) else (keys,)
                    r = dict(row)
                    r.update({b: k for b, k in zip(by, keys)})
                    r["acc"] = float(per.loc[gidx].mean() * 100)
                    r["n"] = int(len(gidx))
                    out.append(r)
            else:
                row["acc"] = float(per.mean() * 100)
                row["n"] = int(len(per))
                out.append(row)
    return pd.DataFrame(out)


@st.cache_data(show_spinner=False)
def district_year_standing(_df, sig):
    """District × year: overall accuracy + percentile among districts covered
    in EVERY year (balanced panel — coverage changed across years)."""
    df, rq, _ = load_context()
    qs = _qcols(df)
    ycol = next(c for c in df.columns if str(c).lower() == "year")
    dcol = next(c for c in df.columns if str(c).lower() == "district")
    acc = (df[qs].mean(axis=1) * 100).rename("acc")
    t = pd.DataFrame({"y": df[ycol].map(_year_key), "d": df[dcol], "acc": acc})
    g = t.groupby(["d", "y"])["acc"].agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= 30]
    piv = g.pivot(index="d", columns="y", values="mean")
    full = piv.dropna()
    pct = full.rank(pct=True) * 100
    return full.round(1), pct.round(1)


@st.cache_data(show_spinner=False)
def persistent_gaps(_df, sig, threshold=50.0, level="District"):
    """Unit × competency below `threshold` in EVERY year it was tested
    (≥2 years) + accuracy trail. The §18 persistence rule."""
    ca = competency_accuracy(_df, sig, by=(level,))
    ca = ca[ca["n"] >= 30]
    g = (ca.groupby([level, "competency", "Year"])
         .apply(lambda x: np.average(x["acc"], weights=x["n"]),
                include_groups=False).rename("acc").reset_index())
    rows = []
    for (u, comp), sub in g.groupby([level, "competency"]):
        sub = sub.sort_values("Year")
        if len(sub) >= 2 and (sub["acc"] < threshold).all():
            rows.append({level: u, "competency": comp,
                         "years": len(sub),
                         "trail": " → ".join(f"{a:.0f}%" for a in sub["acc"]),
                         "latest": float(sub["acc"].iloc[-1]),
                         "change": float(sub["acc"].iloc[-1]
                                         - sub["acc"].iloc[0])})
    return (pd.DataFrame(rows).sort_values("latest")
            if rows else pd.DataFrame())


@st.cache_data(show_spinner=False)
def gender_gap_by(_df, sig, key="competency"):
    ca_g = competency_accuracy(_df, sig, girls_only=True)
    ca_a = competency_accuracy(_df, sig)
    # boys = all minus girls, weighted
    g = ca_g.groupby(key).apply(lambda x: np.average(x["acc"], weights=x["n"]),
                                include_groups=False).rename("Girls")
    a = ca_a.groupby(key).apply(lambda x: np.average(x["acc"], weights=x["n"]),
                                include_groups=False)
    ng = ca_g.groupby(key)["n"].sum()
    na = ca_a.groupby(key)["n"].sum()
    boys = ((a * na - g * ng) / (na - ng)).rename("Boys")
    out = pd.concat([g, boys], axis=1).dropna()
    out["Gap (G−B)"] = out["Girls"] - out["Boys"]
    return out.round(1)


def movement_classes(pct, up=5.0, down=-5.0):
    """Improving / Stable / Declining by Δ percentile first→last year."""
    d = (pct.iloc[:, -1] - pct.iloc[:, 0]).round(1)
    cls = pd.Series(np.where(d >= up, "Improving ↑",
                    np.where(d <= down, "Declining ↓", "Stable →")),
                    index=d.index)
    return d, cls


@st.cache_data(show_spinner=False)
def intervention_signals(_df, sig, threshold=50.0, min_years=2):
    """§18 early-intervention engine: persistent low competency + no real
    improvement + standing context → prioritized, evidence-backed signals."""
    pg = persistent_gaps(_df, sig, threshold, "District")
    if pg.empty:
        return []
    _, pct = district_year_standing(_df, sig)
    dpct, _cls = movement_classes(pct)
    out = []
    for r in pg.itertuples():
        d = getattr(r, "District")
        falling = float(dpct.get(d, 0)) < 0
        hi = (r.latest < threshold - 5) and (r.change <= 2 or falling)
        out.append({
            "priority": "HIGH" if hi else "MEDIUM",
            "district": d, "competency": r.competency,
            "signal": f"Persistent {r.competency} weakness",
            "evidence": r.trail + f" (below {threshold:.0f}% in all "
                        f"{r.years} tested years)",
            "trend": (f"district standing {dpct.get(d, 0):+.0f} percentile "
                      f"pts over the period"),
            "suggestion": _suggest(r.competency),
            "latest": r.latest,
        })
    out.sort(key=lambda x: (x["priority"] != "HIGH", x["latest"]))
    return out


def _suggest(comp):
    if comp in FOUNDATIONAL:
        return ("Foundational numeracy remediation — structured practice on "
                f"{comp} with grade-appropriate materials")
    if comp in OPERATIONS:
        return (f"Targeted instructional support on {comp} — procedure "
                "fluency built on verified foundational skills")
    return (f"Applied-skills practice for {comp} — contextual tasks "
            "(calendars, measurement, data reading) in daily teaching")


@st.cache_data(show_spinner=False)
def secondary_context(_df, sig):
    """District performance vs district context; Pearson r. Association only."""
    try:
        import missions_page as _mp
        sec = _mp._load_secondary()
    except Exception:
        p = os.path.join(_HERE, "secondary_dataset.xlsx")
        if not os.path.exists(p):
            return None
        sec = pd.read_excel(p)
        sec.columns = [str(c).strip() for c in sec.columns]
        sec = sec.set_index("District")
    full, _ = district_year_standing(_df, sig)
    perf = full.mean(axis=1).rename("Assessment accuracy (%)")
    m = sec.join(perf, how="inner").dropna(subset=["Assessment accuracy (%)"])
    return m if len(m) >= 8 else None


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 8 or x.std() == 0 or y.std() == 0:
        return None, len(x)
    return float(np.corrcoef(x, y)[0, 1]), len(x)


# --------------------------------------------------------------------------
#  shared UI pieces (Fabric-ish: dense, bordered, restrained)
# --------------------------------------------------------------------------
def _css():
    try:
        import missions_page as _mp
        st.markdown(_mp._LIGHT_CSS, unsafe_allow_html=True)
    except Exception:
        pass
    st.markdown("""<style>
    .mi-flow { display:flex; align-items:center; flex-wrap:wrap; gap:6px; }
    .mi-step { background:#ffffff; border:1px solid #e6e9ef;
        border-radius:8px; padding:8px 14px; font-size:13px;
        font-weight:600; color:#26303e; }
    .mi-arrow { color:#98a2b3; font-weight:700; }
    .mi-tag { display:inline-block; padding:2px 10px; border-radius:999px;
        font-size:11px; font-weight:700; letter-spacing:.4px; }
    </style>""", unsafe_allow_html=True)


def _header(title, subtitle, note):
    lp = os.path.join(_HERE, "images", "logo.png")
    img = ""
    if os.path.exists(lp):
        import base64
        with open(lp, "rb") as f:
            img = ("<img src='data:image/png;base64,"
                   + base64.b64encode(f.read()).decode()
                   + "' style='height:54px;width:54px;border-radius:12px;'/>")
    st.markdown(
        f"<div style='display:flex;gap:14px;align-items:center;'>{img}"
        f"<div><div style='font-size:34px;font-weight:800;color:#101828;"
        f"letter-spacing:-.8px;line-height:1.05;'>{title}</div>"
        f"<div style='font-size:14px;color:{ACCENT};font-weight:700;'>"
        f"{subtitle}</div></div></div>", unsafe_allow_html=True)
    st.caption(note)


def _flow(steps):
    html = "<div class='mi-flow'>"
    html += "<span class='mi-arrow'>→</span>".join(
        f"<span class='mi-step'>{s}</span>" for s in steps)
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _load_geo():
    p = os.path.join(_HERE, "karnataka_districts.geojson")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _choropleth(frame, value_col, color_args, hover_extra=None):
    geo = _load_geo()
    if geo is None:
        st.info("karnataka_districts.geojson not found — map unavailable.")
        return None
    fig = px.choropleth(frame, geojson=geo, locations="District",
                        featureidkey="properties.district",
                        color=value_col, hover_name="District",
                        hover_data=hover_extra, height=560, **color_args)
    fig.update_geos(fitbounds="locations", visible=False,
                    bgcolor="rgba(0,0,0,0)")
    fig.update_layout(margin=dict(t=6, b=6, l=6, r=6),
                      paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#26303e"))
    return fig


# ==========================================================================
#  NIPUN BHARAT PAGE
# ==========================================================================
def render_nipun():
    _css()
    df, rq, err = load_context()
    if err:
        st.title("🧮 NIPUN Bharat")
        st.info(err)
        return
    sig = _sig(df)

    # ---- 1 · executive overview -----------------------------------------
    _header("NIPUN Bharat",
            "Monitoring foundational learning retention across Grades 4–6",
            "Our assessment provides evidence of foundational mathematical "
            "competencies in Grades 4–6. These findings identify persistent "
            "learning gaps that survive past the foundational stage and help "
            "prioritize remediation — this is retention monitoring, not an "
            "official measurement of NIPUN's Grade-3 targets.")

    ca = competency_accuracy(df, sig)
    fnd = ca[ca["competency"].isin(FOUNDATIONAL)]
    fnd_by_year = (fnd.groupby("Year")
                   .apply(lambda x: np.average(x["acc"], weights=x["n"]),
                          include_groups=False).round(1))
    pgaps = persistent_gaps(df, sig)
    gg = gender_gap_by(df, sig)
    full, pct = district_year_standing(df, sig)
    dmove, dcls = movement_classes(pct)
    _fnd_now = float(fnd_by_year.iloc[-1])
    _gapw = float(gg["Gap (G−B)"].abs().max()) if len(gg) else 0

    k = st.columns(5)
    k[0].metric("Foundational competency health",
                f"{_fnd_now:.0f}%",
                help="Latest-year accuracy on number sense, place value, "
                     "addition, subtraction — each child scored on their own "
                     "paper.")
    k[1].metric("Districts requiring attention",
                int((dcls == "Declining ↓").sum()
                    + pgaps["District"].nunique() if len(pgaps) else 0),
                help="Declining standing or a persistent competency gap.")
    k[2].metric("Persistent competency gaps",
                len(pgaps),
                help="District × competency below 50% in every tested year.")
    k[3].metric("3-year foundational trend",
                f"{float(fnd_by_year.iloc[-1] - fnd_by_year.iloc[0]):+.1f} pts",
                help="⚠️ Papers change yearly — read as indicative, with the "
                     "percentile view for movement.")
    k[4].metric("Widest gender gap (any skill)", f"{_gapw:.1f} pts")

    st.divider()

    # ---- 2 · 3-year trajectory ------------------------------------------
    st.markdown("### 📈 3-year learning trajectory")
    c1, c2 = st.columns([1.5, 1])
    with c1:
        tr = (fnd.groupby(["Year", "Grade"])
              .apply(lambda x: np.average(x["acc"], weights=x["n"]),
                     include_groups=False).rename("acc").reset_index())
        tr["Grade"] = "Grade " + tr["Grade"].astype(str)
        fig = px.line(tr, x="Year", y="acc", color="Grade", markers=True,
                      labels={"acc": "Foundational accuracy %"},
                      color_discrete_sequence=[ACCENT, "#10B981", "#67a99a"])
        ov = fnd_by_year.reset_index()
        ov.columns = ["Year", "acc"]
        fig.add_scatter(x=ov["Year"], y=ov["acc"], name="All grades",
                        mode="lines+markers",
                        line=dict(color="#101828", width=3, dash="dot"))
        fig.update_layout(height=340, margin=dict(t=10, b=10),
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("⚠️ A different paper is set every year, so these lines "
                   "mix learning with paper difficulty. They answer "
                   "*'roughly where is foundational learning'*; for who is "
                   "genuinely moving, use District movement below (percentile "
                   "standing, paper-proof).")
    with c2:
        st.markdown("**Foundational skills = the NIPUN core**")
        st.markdown("- number sense\n- place value\n- addition\n- subtraction")
        st.caption("Tracked separately from operations "
                   "(×, ÷, fractions) and applied skills (measurement, "
                   "shapes, data). Foundational weakness that persists into "
                   "Grades 4–6 is exactly what NIPUN aims to prevent.")
        _worst_f = (fnd.groupby("competency")
                    .apply(lambda x: np.average(x["acc"], weights=x["n"]),
                           include_groups=False).sort_values())
        st.metric("Weakest foundational skill",
                  _worst_f.index[0], f"{_worst_f.iloc[0]:.0f}% accuracy")

    st.divider()

    # ---- 3 · competency health heatmap ----------------------------------
    st.markdown("### 🔥 Competency health — persistence at a glance")
    hm = (ca.groupby(["competency", "Year"])
          .apply(lambda x: np.average(x["acc"], weights=x["n"]),
                 include_groups=False).rename("acc").reset_index()
          .pivot(index="competency", columns="Year", values="acc"))
    order = [c for c in FOUNDATIONAL + OPERATIONS + APPLIED if c in hm.index]
    hm = hm.reindex(order).round(0)
    fig = px.imshow(hm, color_continuous_scale="RdYlGn", range_color=[30, 80],
                    aspect="auto", text_auto=True, height=420,
                    labels={"color": "% correct"})
    fig.update_layout(margin=dict(t=6, b=6), coloraxis_showscale=False,
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#26303e"))
    st.plotly_chart(fig, use_container_width=True)
    _low = hm.min(axis=1).sort_values()
    st.caption(f"Blank = not tested that year (the paper changes yearly). A "
               f"row that stays red is a persistent weakness — worst here: "
               f"**{_low.index[0]}**, never above "
               f"{hm.loc[_low.index[0]].max():.0f}%.")

    st.divider()

    # ---- 4 · foundational learning leakage ------------------------------
    st.markdown("### 🕳️ Foundational learning leakage — the early-warning "
                "chain")
    _fa = float(fnd_by_year.iloc[-1])
    ops = ca[ca["competency"].isin(OPERATIONS)]
    _oa = float(np.average(ops["acc"], weights=ops["n"]))
    _flow([f"Foundational gap ({100 - _fa:.0f}% of answers wrong)",
           "persists across years",
           f"operations weakness observed ({100 - _oa:.0f}% wrong on ×, ÷, "
           f"fractions)",
           "EARLY INTERVENTION SIGNAL"])
    st.caption("An indicator chain, not a causal claim: where foundational "
               "accuracy is low and stays low, weakness in operations is "
               "also observed. Both are measured; the arrow is the early-"
               "warning logic, not proof of causation.")

    st.divider()

    # ---- 5 · geographic priority ----------------------------------------
    st.markdown("### 🗺️ Geographic priority — where are the gaps "
                "concentrated?")
    fnd_d = competency_accuracy(df, sig, by=("District",))
    fnd_d = fnd_d[fnd_d["competency"].isin(FOUNDATIONAL)]
    fd = (fnd_d.groupby("District")
          .apply(lambda x: np.average(x["acc"], weights=x["n"]),
                 include_groups=False).rename("Foundational %").reset_index())
    fd = fd.merge(dmove.rename("Δ pctile").reset_index()
                  .rename(columns={"d": "District"}), on="District",
                  how="left")
    q1, q2 = fd["Foundational %"].quantile([.25, .5])

    def _classify(r):
        if r["Foundational %"] < q1 and (r.get("Δ pctile") or 0) <= 0:
            return "Critical"
        if r["Foundational %"] < q1:
            return "Needs Attention"
        if r["Foundational %"] < q2:
            return "Moderate"
        return "Healthy"
    fd["Class"] = fd.apply(_classify, axis=1)
    m1, m2 = st.columns([1.6, 1])
    with m1:
        fig = _choropleth(fd, "Class",
                          dict(color_discrete_map=CLASS_COLORS,
                               category_orders={"Class": list(CLASS_COLORS)}),
                          hover_extra=["Foundational %", "Δ pctile"])
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Critical = bottom-quartile foundational accuracy AND not "
                   "gaining ground vs peers. Blank districts are outside GP-"
                   "contest coverage in the loaded selection.")
    with m2:
        st.markdown("**Top priority regions**")
        pr = fd[fd["Class"].isin(["Critical", "Needs Attention"])] \
            .sort_values("Foundational %").head(6)
        for i, r in enumerate(pr.itertuples(), 1):
            col = CLASS_COLORS[r.Class]
            why = []
            _pg = pgaps[pgaps["District"] == r.District] if len(pgaps) else []
            if len(_pg):
                why.append(f"persistent {_pg.iloc[0]['competency']} gap")
            if (r._3 or 0) < -5:
                why.append(f"standing {r._3:+.0f} pctile pts")
            if not why:
                why.append(f"foundational {r._2:.0f}%")
            st.markdown(
                f"<div style='border:1px solid #e6e9ef;border-left:4px solid "
                f"{col};border-radius:8px;padding:8px 12px;margin:6px 0;"
                f"background:#fff;'><b>{i}. {r.District}</b><br>"
                f"<span style='color:#67707f;font-size:13px;'>"
                f"{' · '.join(why)}</span></div>", unsafe_allow_html=True)

    st.divider()

    # ---- 6 · gender -------------------------------------------------------
    st.markdown("### ⚖️ Gender — persistent, narrowing, or widening?")
    ggy = gender_gap_by(df, sig, key="Year")
    fig = go.Figure()
    fig.add_bar(x=ggy.index, y=ggy["Girls"], name="Girls",
                marker_color="#ff2d78")
    fig.add_bar(x=ggy.index, y=ggy["Boys"], name="Boys",
                marker_color="#00b4d8")
    fig.update_layout(barmode="group", height=300,
                      yaxis_title="Accuracy %", margin=dict(t=10, b=10),
                      legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, use_container_width=True)
    _g0, _g1 = float(ggy["Gap (G−B)"].iloc[0]), float(ggy["Gap (G−B)"].iloc[-1])
    _dirw = ("narrowed" if abs(_g1) < abs(_g0)
             else "widened" if abs(_g1) > abs(_g0) else "held steady")
    st.caption(f"The overall gap {_dirw}: {_g0:+.1f} pts (girls − boys) in "
               f"{ggy.index[0]} → {_g1:+.1f} pts in {ggy.index[-1]}. "
               f"Per-competency gaps live on the PARAKH page.")

    st.divider()

    # ---- 7 · secondary context ------------------------------------------
    st.markdown("### 🔎 District context (association, not causation)")
    m = secondary_context(df, sig)
    if m is None:
        st.info("Secondary dataset not available.")
    else:
        _str_col = next((c for c in m.columns
                         if "student teacher ratio" in c.lower()), None)
        _lit_col = next((c for c in m.columns
                         if "total literacy" in c.lower()
                         and "rural" in c.lower()), None)
        cc1, cc2 = st.columns(2)
        for col, box in ((_str_col, cc1), (_lit_col, cc2)):
            if not col:
                continue
            r, n = pearson(m[col], m["Assessment accuracy (%)"])
            with box:
                fig = px.scatter(m.reset_index(), x=col,
                                 y="Assessment accuracy (%)",
                                 hover_name="District", height=320,
                                 trendline=None,
                                 color_discrete_sequence=[ACCENT])
                fig.update_layout(margin=dict(t=8, b=8))
                st.plotly_chart(fig, use_container_width=True)
                if r is not None:
                    st.caption(f"r = {r:+.2f} across {n} districts — "
                               f"districts with "
                               f"{'higher' if r > 0 else 'lower'} {col} "
                               "also show higher assessment performance. "
                               "**Association only** — this may warrant "
                               "further investigation, and cannot explain "
                               "any individual GP.")

    st.divider()

    # ---- 8 · district movement ------------------------------------------
    st.markdown("### 🧭 District movement — improving, stable, declining")
    st.caption("Percentile standing among the districts covered in every "
               "year — the paper-proof way to read cross-year movement.")
    mv = pd.DataFrame({"Δ pctile": dmove, "Class": dcls}).sort_values(
        "Δ pctile", ascending=False)
    c1, c2, c3 = st.columns(3)
    for box, label, emoji in ((c1, "Improving ↑", "🟢"),
                              (c2, "Stable →", "🟡"),
                              (c3, "Declining ↓", "🔴")):
        sub = mv[mv["Class"] == label]
        box.markdown(f"**{emoji} {label} ({len(sub)})**")
        for d, r in sub.head(6).iterrows():
            box.markdown(f"- {d} ({r['Δ pctile']:+.0f})")

    st.divider()

    # ---- 9 · early-intervention engine -----------------------------------
    st.markdown("### 🚨 Early-intervention engine")
    e1, e2 = st.columns(2)
    thr = e1.slider("Weakness threshold (accuracy below…)", 30, 65, 50, 5,
                    key="nipun_thr")
    st.caption("Deterministic rule: accuracy below the threshold in **every** "
               "year the competency was tested, with minimal improvement or "
               "falling standing → intervention signal. No model, fully "
               "auditable.")
    sigs = intervention_signals(df, sig, float(thr))
    if not sigs:
        st.success("No district × competency pair meets the persistence rule "
                   "at this threshold.")
    for s in sigs[:8]:
        col = "#d73027" if s["priority"] == "HIGH" else "#fc8d59"
        with st.container(border=True):
            st.markdown(
                f"<span class='mi-tag' style='background:{col}22;color:{col};"
                f"'>{s['priority']} PRIORITY</span>  **{s['district']}** — "
                f"{s['signal']}", unsafe_allow_html=True)
            st.markdown(f"Evidence: {s['evidence']}  ·  Trend: {s['trend']}")
            st.caption(f"Suggested intervention: {s['suggestion']}")
    if len(sigs) > 8:
        with st.expander(f"All {len(sigs)} signals"):
            st.dataframe(pd.DataFrame(sigs), use_container_width=True,
                         hide_index=True)

    st.divider()

    # ---- 10 · policy recommendation -------------------------------------
    st.markdown("### 📋 Policy recommendation — evidence to monitoring")
    for s in sigs[:3]:
        with st.container(border=True):
            _flow([f"DATA: {s['competency']} at {s['latest']:.0f}% in "
                   f"{s['district']}",
                   f"INSIGHT: below {thr}% in every tested year",
                   "RISK: children enter higher grades with the gap "
                   "unresolved",
                   f"RECOMMEND: {s['suggestion'].split(' — ')[0]}",
                   f"MONITOR: {s['competency']} accuracy next cycle"])

    # ---- 11 + 12 · alignment + next steps --------------------------------
    st.divider()
    st.markdown("### 🎯 How Learning Insights supports NIPUN")
    _flow(["NIPUN objective", "Foundational numeracy",
           "Assessment evidence (1.38M responses)", "Competency gap detection",
           "Geographic prioritization", "Targeted intervention",
           "Repeat assessment", "Monitor improvement"])
    st.markdown("### ⏭️ Recommended next steps")
    st.markdown(
        "1. Confirm the priority regions above with district officers\n"
        "2. Target the weakest foundational competencies first\n"
        "3. Verify persistence at Block/GP level in the main dashboard\n"
        "4. Deploy targeted remediation in Critical districts\n"
        "5. Re-assess and recompute these signals next cycle\n"
        "6. Re-prioritize from the new evidence")


# ==========================================================================
#  PARAKH PAGE
# ==========================================================================
def render_parakh():
    _css()
    df, rq, err = load_context()
    if err:
        st.title("🧭 PARAKH")
        st.info(err)
        return
    sig = _sig(df)

    _header("PARAKH",
            "Competency-based assessment — evidence of what students can "
            "actually demonstrate",
            "Every question in the assessment maps to a competency, so a "
            "result is not an examination percentage — it is evidence of "
            "which skills a child can demonstrate. This page turns that "
            "evidence into instructional decisions.")

    ca = competency_accuracy(df, sig)
    order = [c for c in FOUNDATIONAL + OPERATIONS + APPLIED
             if c in set(ca["competency"])]

    # ---- core: competency × grade matrix ---------------------------------
    st.markdown("### 🧩 Competency performance matrix")
    hm = (ca.groupby(["competency", "Grade"])
          .apply(lambda x: np.average(x["acc"], weights=x["n"]),
                 include_groups=False).rename("acc").reset_index()
          .pivot(index="competency", columns="Grade", values="acc")
          .reindex(order).round(0))
    hm.columns = [f"Grade {int(c)}" for c in hm.columns]
    fig = px.imshow(hm, color_continuous_scale="RdYlGn", range_color=[30, 80],
                    aspect="auto", text_auto=True, height=420,
                    labels={"color": "% correct"})
    fig.update_layout(margin=dict(t=6, b=6), coloraxis_showscale=False,
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#26303e"))
    st.plotly_chart(fig, use_container_width=True)
    _flat = (hm.max(axis=1) - hm.min(axis=1)).sort_values()
    _worst = hm.min(axis=1).sort_values()
    st.caption(f"Rows = skills, columns = grades, each cell = % of that "
               f"grade demonstrating the skill (each child scored on their "
               f"own year's paper). **{_worst.index[0]}** is the weakest "
               f"skill; **{_flat.index[0]}** barely moves across grades "
               f"({_flat.iloc[0]:.0f}-pt range) — a skill children are not "
               f"gaining as they progress.")

    st.divider()

    # ---- yearly trend with selector --------------------------------------
    st.markdown("### 📅 Competency trend over the three years")
    picks = st.multiselect("Competencies", order,
                           default=[c for c in ("division", "number sense",
                                                "measurement") if c in order],
                           key="parakh_comps")
    if picks:
        tr = (ca[ca["competency"].isin(picks)]
              .groupby(["competency", "Year"])
              .apply(lambda x: np.average(x["acc"], weights=x["n"]),
                     include_groups=False).rename("acc").reset_index())
        fig = px.line(tr, x="Year", y="acc", color="competency", markers=True,
                      labels={"acc": "Accuracy %"})
        fig.update_layout(height=330, margin=dict(t=8, b=8),
                          legend=dict(orientation="h", y=-0.28))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("⚠️ The paper changes yearly — trends are indicative; a "
                   "skill low in *every* year (like division) is the robust "
                   "finding, not small year-to-year wiggles.")

    st.divider()

    # ---- geographic competency gap ---------------------------------------
    st.markdown("### 🗺️ Where is a specific competency weakest?")
    comp_pick = st.selectbox("Competency to map", order,
                             index=order.index("division")
                             if "division" in order else 0,
                             key="parakh_map_comp")
    cad = competency_accuracy(df, sig, by=("District",))
    cd = (cad[cad["competency"] == comp_pick]
          .groupby("District")
          .apply(lambda x: np.average(x["acc"], weights=x["n"]),
                 include_groups=False).rename("Accuracy %").reset_index())
    fig = _choropleth(cd, "Accuracy %",
                      dict(color_continuous_scale="RdYlGn",
                           range_color=[float(cd["Accuracy %"].min()) - 1,
                                        float(cd["Accuracy %"].max()) + 1]))
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    _w = cd.sort_values("Accuracy %")
    st.caption(f"**{comp_pick}** runs from "
               f"{_w.iloc[0]['Accuracy %']:.0f}% in {_w.iloc[0]['District']} "
               f"to {_w.iloc[-1]['Accuracy %']:.0f}% in "
               f"{_w.iloc[-1]['District']} — a "
               f"{_w.iloc[-1]['Accuracy %'] - _w.iloc[0]['Accuracy %']:.0f}-"
               f"point spread on one skill. This is where instructional "
               f"support for {comp_pick} belongs first.")

    st.divider()

    # ---- grade comparison + gender ---------------------------------------
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("### 🎓 Grade progression")
        gp = (ca.groupby(["competency", "Grade"])
              .apply(lambda x: np.average(x["acc"], weights=x["n"]),
                     include_groups=False).rename("acc").reset_index())
        gp = gp[gp["competency"].isin(order)]
        gp["Grade"] = "Grade " + gp["Grade"].astype(str)
        fig = px.bar(gp, x="competency", y="acc", color="Grade",
                     barmode="group", height=360,
                     labels={"acc": "Accuracy %", "competency": ""},
                     color_discrete_sequence=[ACCENT, "#10B981", "#8fd6c0"])
        fig.update_layout(margin=dict(t=8, b=8),
                          legend=dict(orientation="h", y=-0.5))
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Bars that don't rise with grade = skills not being "
                   "gained through progression.")
    with g2:
        st.markdown("### ⚖️ Gender by competency")
        gg = gender_gap_by(df, sig).reindex(order).dropna()
        gg2 = gg.sort_values("Gap (G−B)")
        fig = px.bar(gg2.reset_index(), x="Gap (G−B)", y="competency",
                     orientation="h", height=360,
                     color="Gap (G−B)", color_continuous_scale="RdBu",
                     color_continuous_midpoint=0,
                     labels={"competency": ""})
        fig.update_layout(coloraxis_showscale=False, margin=dict(t=8, b=8))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Blue right = girls demonstrate the skill more often; "
                   "red left = boys. Values in percentage points.")

    st.divider()

    # ---- data → decision pipeline ----------------------------------------
    st.markdown("### 🔁 Why competency-based assessment matters")
    _flow(["Assessment question", "Competency", "Performance evidence",
           "Gap detection", "Geographic identification",
           "Instructional / policy intervention", "Reassessment"])
    st.caption("The pipeline PARAKH stands for: an exam answer becomes skill "
               "evidence, evidence becomes a located gap, and the gap "
               "becomes a monitorable intervention.")

    st.divider()

    # ---- recommendation ---------------------------------------------------
    st.markdown("### 📋 Assessment-driven recommendations")
    overall = (ca.groupby("competency")
               .apply(lambda x: np.average(x["acc"], weights=x["n"]),
                      include_groups=False).sort_values())
    for comp in overall.index[:3]:
        cd2 = (cad[cad["competency"] == comp].groupby("District")
               .apply(lambda x: np.average(x["acc"], weights=x["n"]),
                      include_groups=False).sort_values())
        _where = ", ".join(cd2.index[:3])
        with st.container(border=True):
            _flow([f"Competency: {comp}",
                   f"Observed: {overall[comp]:.0f}% statewide",
                   f"Concentrated: {_where}",
                   "Persistent across years"
                   if comp in set(persistent_gaps(df, sig)["competency"])
                   else "Present in the latest cycle",
                   f"Recommend: {_suggest(comp).split(' — ')[0]}",
                   f"Monitor: {comp} accuracy next assessment"])