"""
Datathon demo — NO LLM VERSION.

The Datathon rules prohibit all generative AI, including local LLMs. Every
layer here is deterministic: SQL aggregation, statistical tests, classical
ML (KMeans / logistic regression) and template-based text generation.
Same input always produces the same output, which the fixed-seed
reproducibility rule requires.

This is the standalone explainer that walks through each layer. The 17-tab
dashboard in streamlit_app/ is the actual deliverable.

Run:   streamlit run app.py     (from the project root, next to this file)
"""
import os, sys, time
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gen_data, analytics, verbalize, models_ml
import insights, brief, playbook, charts, competency

st.set_page_config(page_title="Datathon Demo — No-LLM Pipeline", layout="wide")

# ---------------- sidebar ----------------
st.sidebar.title("⚙️ Settings")
st.sidebar.success("**No LLM used.**\n\nEvery output below is computed by "
                   "SQL, statistics or classical ML — compliant with the "
                   "Datathon rules.")
st.sidebar.divider()
spg = st.sidebar.slider("Students per (block × grade)", 100, 1200, 300, step=100)
if st.sidebar.button("🔄 Generate / regenerate data"):
    with st.spinner("Generating synthetic dataset..."):
        path, n = gen_data.build(students_per_group=spg)
        if os.path.exists(analytics.PARQUET):
            os.remove(analytics.PARQUET)
        analytics.to_parquet()
    st.session_state.pop("results", None)
    st.sidebar.success(f"{n:,} rows generated")

# ---------------- header ----------------
st.title("🎓 Datathon Pipeline — Fully Deterministic (No AI)")
st.caption("Aggregation → statistics → classical ML → template-based reporting. "
           "Reproducible, auditable, and rule-compliant.")

if not os.path.exists(analytics.CSV):
    st.warning("No data yet — click **Generate / regenerate data** in the sidebar.")
    st.stop()

if not os.path.exists(analytics.PARQUET):
    analytics.to_parquet()

@st.cache_data(show_spinner=False)
def load_agg(mtime):
    return analytics.aggregate()

agg = load_agg(os.path.getmtime(analytics.CSV))
raw_rows = analytics.counts()

c1, c2, c3 = st.columns(3)
c1.metric("Raw rows (on disk)", f"{raw_rows:,}")
c2.metric("Aggregated rows (in memory)", f"{len(agg):,}")
c3.metric("Reduction", f"{raw_rows/max(len(agg),1):,.0f}×")

with st.expander("ℹ️ How the whole pipeline works (background)"):
    st.markdown("""
1. **Reduce** — DuckDB aggregates raw student rows into a few hundred summary
   rows on the laptop (no server, no cloud, no data leaves the machine).
2. **Test** — every claimed gap is checked with a two-proportion z-test, so
   noise is suppressed and only real differences are reported.
3. **Model** — scikit-learn clusters blocks into archetypes and fits a risk
   model. These are *classical* ML, not generative AI.
4. **Report** — findings are phrased by Python templates. Every number traces
   back to an exact SQL aggregate, which is what `claims.json` requires.

**No LLM is used anywhere.** The same input always produces the same output.
""")

with st.expander("🔄 **What changed when the LLM was removed** (open this in the meeting)"):
    st.markdown("""
The LLM never calculated anything — it only turned our numbers into sentences.
So we write the sentences in Python instead. **Layers 1, 2, 3 and the risk model
did not change at all.**
""")
    st.dataframe(pd.DataFrame([
        {"Layer": "1 — Aggregate → sentences", "Before": "DuckDB + templates",
         "After": "unchanged", "Status": "✅ no change"},
        {"Layer": "2 — Learning archetypes", "Before": "scikit-learn KMeans",
         "After": "unchanged", "Status": "✅ no change (classical ML is allowed)"},
        {"Layer": "3 — Health status", "Before": "threshold rules",
         "After": "unchanged", "Status": "✅ no change"},
        {"Layer": "4 — Insights", "Before": "LLM decides what is interesting",
         "After": "8 generators + z-test, ranked by effect size", "Status": "🔄 replaced"},
        {"Layer": "5 — Visualisation", "Before": "skipped",
         "After": "4 Plotly charts", "Status": "➕ added (15% of score)"},
        {"Layer": "6a — Risk model", "Before": "LogisticRegression",
         "After": "unchanged", "Status": "⚠️ works, but label is circular"},
        {"Layer": "6b — What-If", "Before": "invented 0.6 effectiveness",
         "After": "rates from observed history, minus rebound", "Status": "🔧 rebuilt"},
        {"Layer": "7 — District brief", "Before": "LLM writes a paragraph",
         "After": "conditional f-string clauses", "Status": "🔄 replaced"},
        {"Layer": "8 — Recommendations", "Before": "LLM invents advice",
         "After": "auditable decision table", "Status": "🔄 replaced"},
        {"Layer": "9 — Survey questions", "Before": "LLM writes questions",
         "After": "limitations & caveats section", "Status": "❌ dropped"},
    ]), use_container_width=True, hide_index=True)
    st.caption("Network calls made by this app: **0**.  Dependencies removed: `requests`, `ollama`.")

districts = sorted(agg["district"].unique())
district = st.selectbox("Choose a district to analyze", districts)
run = st.button("▶  Run all layers", type="primary")

# ---------------- helpers ----------------
def layer_header(num, title, tool, ms=None):
    st.divider()
    st.subheader(f"Layer {num} — {title}")
    cap = f"🔧 {tool}"
    if ms is not None:
        cap += f"   ·   ⏱ {ms:.0f} ms"
    st.caption(cap)

def how(md):
    with st.expander("ℹ️ How this is computed "):
        st.markdown(md)

# ---------------- run pipeline ----------------
if run:
    res, prog = {}, st.progress(0.0, "Starting...")

    t = time.perf_counter()
    sents, vtext = verbalize.verbalize_district(agg, district)
    ctable = verbalize.competency_table(agg, district)
    res["L1"] = {"ms": (time.perf_counter()-t)*1000, "sents": sents, "ctable": ctable}
    prog.progress(0.12, "Layer 1 done")

    t = time.perf_counter()
    clusters, names = models_ml.cluster_blocks(agg, k=3)
    res["L2"] = {"ms": (time.perf_counter()-t)*1000, "clusters": clusters}
    prog.progress(0.24, "Layer 2 done")

    t = time.perf_counter()
    h = ctable.copy()
    h["status"] = h["below_pct"].apply(models_ml.health_status)
    res["L3"] = {"ms": (time.perf_counter()-t)*1000, "table": h}
    prog.progress(0.36, "Layer 3 done")

    t = time.perf_counter()
    res["L4"] = {"ms": (time.perf_counter()-t)*1000,
                 "items": insights.generate(agg, district)}
    res["L4"]["ms"] = (time.perf_counter()-t)*1000
    prog.progress(0.5, "Layer 4 done")

    t = time.perf_counter()
    res["L5"] = {"ms": (time.perf_counter()-t)*1000}
    prog.progress(0.6, "Layer 5 done")

    t = time.perf_counter()
    _, risk = models_ml.train_risk(agg)
    risk_top = (risk[risk["district"] == district]
                .sort_values("risk", ascending=False)
                [["block", "competency", "below_pct", "risk"]].head(6))
    weakest = ctable.iloc[0]["competency"]
    whatif = models_ml.what_if(agg, district, weakest, n_blocks=10)
    res["L6"] = {"ms": (time.perf_counter()-t)*1000, "risk": risk_top,
                 "whatif": whatif, "weakest": weakest}
    prog.progress(0.75, "Layer 6 done")

    t = time.perf_counter()
    worst_blk = (ctable is not None) and district
    res["L7"] = {"briefs": brief.build_all(agg, district)}
    res["L7"]["ms"] = (time.perf_counter()-t)*1000
    prog.progress(0.88, "Layer 7 done")

    t = time.perf_counter()
    res["L8"] = {"ms": (time.perf_counter()-t)*1000,
                 "recs": playbook.recommend(agg, district)}
    res["L8"]["ms"] = (time.perf_counter()-t)*1000
    prog.progress(1.0, "All layers complete ✅")

    st.session_state["results"] = res
    st.session_state["district"] = district

# ---------------- render ----------------
res = st.session_state.get("results")
if res and st.session_state.get("district") == district:

    # ---- Layer 1
    L = res["L1"]
    layer_header(1, "Data → Aggregates → Sentences", "DuckDB + templates", L["ms"])
    st.dataframe(L["ctable"], use_container_width=True, hide_index=True)
    st.code("\n".join(L["sents"][:5]), language=None)
    how(f"""
**Input:** {raw_rows:,} raw student rows.

**Reduce:** one SQL `GROUP BY block × grade × competency × year` computes
`below_pct = 100 × below ÷ total`, turning **{raw_rows:,} rows → {len(agg)} summary
rows** in ~30 ms. *(analytics.py → aggregate())*

**Verbalize:** each summary row fills a fixed sentence template — **{len(L['sents'])}
sentences** for this district. *(verbalize.py → verbalize_district())*

Numbers come from SQL, so they are always exact. Nothing is generated by a model.
""")

    # ---- Layer 2
    L = res["L2"]
    layer_header(2, "Learning Archetypes", "scikit-learn KMeans", L["ms"])
    st.dataframe(L["clusters"], use_container_width=True, hide_index=True)
    how("""
**Input:** a grid of blocks × competencies (each cell = that block's below%).

**Computation:** **KMeans** groups blocks by how similar their weakness patterns
are. Unsupervised — no labels, no pre-trained model, fits live in ~1 s.
*(models_ml.py → cluster_blocks())*

**Why it matters:** one intervention plan can serve a whole group of blocks that
share the same problem, instead of designing a separate plan for each.

*KMeans is classical machine learning, not generative AI — permitted under the rules.*
""")

    # ---- Layer 3
    L = res["L3"]
    layer_header(3, "Learning-Health Status", "threshold rules", L["ms"])
    st.dataframe(L["table"], use_container_width=True, hide_index=True)
    how("""
**Computation:** fixed thresholds, no model at all — below ≥ 60 → **Critical**,
≥ 45 → **At-risk**, else **Strong**. *(models_ml.py → health_status())*

Fully explainable: "why Critical?" → "because 63% ≥ 60%, that is the rule."
""")

    # ---- Layer 3b — Competency Intelligence Report (drill-down)
    st.divider()
    st.subheader("Layer 3b — Competency Intelligence Report")
    st.caption("🔧 aggregation + significance tests   ·   "
               "reproduces the original Groq backend's competency report, computed")

    comps = list(res["L3"]["table"]["competency"])
    pick = st.selectbox("Drill into a competency", comps, key="comp_pick")
    rep = competency.report(agg, district, pick)

    if rep:
        o = rep["overview"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Below grade", f"{o['below_pct']}%", help=f"{o['children_below']:,} children")
        k2.metric("At grade", f"{o['at_pct']}%")
        k3.metric("Above grade", f"{o['above_pct']}%")
        k4.metric("Rank in district", o["rank"].split(" (")[0])

        st.markdown(f"**Status:** {o['performance_level']}  ·  "
                    f"**{o['children_below']:,}** of **{o['students_assessed']:,}** "
                    f"students below grade level")

        s1, s2 = st.columns(2)
        with s1:
            st.markdown("**Gender**"); st.info(rep["gender"]["summary"])
            st.markdown("**Grade progression**"); st.info(rep["grade"]["summary"])
        with s2:
            st.markdown("**Geographic spread**"); st.info(rep["geography"]["summary"])
            st.markdown("**Trend**"); st.info(rep["trend"]["summary"])

        b1, b2 = st.columns(2)
        with b1:
            st.plotly_chart(charts.competency_block_bars(rep), use_container_width=True)
        with b2:
            corr = competency.correlation_matrix(agg)
            f = charts.competency_correlation(corr)
            if f:
                st.plotly_chart(f, use_container_width=True)
                pairs = competency.strongest_pairs(corr)
                if pairs:
                    st.caption("Most-linked: " + "  ·  ".join(
                        f"{p['pair']} (r={p['r']})" for p in pairs))

        st.markdown("**Risk assessment**")
        st.warning(f"{rep['risk']['level']} — {rep['risk']['blocks_critical']} blocks "
                   f"critical. {rep['risk']['summary']}")
        st.markdown("**Competency summary**")
        st.success(rep["summary"])

    how("""
**This replaces the original backend's LLM competency report.** That version sent
every student record to Groq and asked it to describe patterns. Every section it
requested is pure aggregation:

| Original spec asked for | How we compute it |
|---|---|
| Overall performance + level | weighted mean of below/at/above %, ranked vs other competencies |
| Performance distribution | student counts by level |
| Demographic analysis (gender) | weighted f_below vs m_below **+ two-proportion z-test** |
| Demographic analysis (grade) | weighted mean by grade **+ linear slope per grade** |
| Demographic analysis (geography) | weighted mean by block, spread, coefficient of variation |
| Learning patterns | **cross-competency correlation** — which competencies fail together |
| Risk analysis | threshold rules + count of critical blocks |
| Summary | template |

*(competency.py)*

**Two things the LLM version could not do:** test whether a gender gap is
statistically real, and correlate competencies to find shared root causes.
""")

    # ---- Layer 4  (was LLM)
    L = res["L4"]
    layer_header(4, "Ranked Insights", "statistical insight engine", L["ms"])
    st.caption("🟢 computed — previously an LLM layer, now deterministic")
    for i, item in enumerate(L["items"], 1):
        st.markdown(f"**{i}. [{item['category']}]** {item['text']}")
        st.caption(f"↳ {item['evidence']}")

    st.markdown("##### 📋 The generator registry — what replaced the LLM's judgement")
    st.caption("Each generator answers a question the handbook explicitly asks. "
               "Nothing here was chosen arbitrarily.")
    st.dataframe(pd.DataFrame(insights.describe(agg, district)),
                 use_container_width=True, hide_index=True)
    fired = sum(1 for r in insights.describe(agg, district) if r["Fired?"] == "✅ yes")
    st.caption(f"**{fired} of {len(insights.GENERATORS)} generators found something "
               f"in {district}.** A generator that finds nothing stays silent — "
               f"we never pad the list to hit a count.")

    st.markdown("##### 🔍 How the ranking works")
    g1, g2 = st.columns(2)
    with g1:
        f = charts.insight_scores(L["items"])
        if f: st.plotly_chart(f, use_container_width=True)
        st.caption("Each generator produces a finding and a **score**. "
                   "Code sorts them — no model decides what matters.")
    with g2:
        f = charts.significance_volcano(agg, district)
        if f: st.plotly_chart(f, use_container_width=True)
        st.caption("**Red** gaps are reported; **grey** ones are suppressed as noise. "
                   "An LLM would have reported all of them.")
    how("""
**Replaces the old LLM layer.** Instead of asking a model what is interesting,
we run **8 generators** — one per question the handbook actually asks — over
*every* aggregated row, score each candidate finding by magnitude, and surface
the strongest of each type. *(insights.py)*

Generators: weakest competency · largest scale · gender gap · steepest decline ·
negative outlier · learning progression · bright spot · worst block.

**Gender gaps are filtered by a two-proportion z-test** — a 7-point gap on 150
students is noise (p=0.20) while the same gap on 1,500 students is real
(p<0.001). An LLM would report both. *(stats_tests.py → two_proportion_z())*
""")

    # ---- Layer 5  (was skipped)
    L = res["L5"]
    layer_header(5, "Visualisation", "Plotly", L["ms"])
    st.caption("Previously skipped — now included (15% of the Datathon score)")
    v1, v2 = st.columns(2)
    with v1:
        st.plotly_chart(charts.gap_heatmap(agg, district), use_container_width=True)
        st.plotly_chart(charts.gender_gap_bars(agg, district), use_container_width=True)
    with v2:
        st.plotly_chart(charts.trend_lines(agg, district), use_container_width=True)
        st.plotly_chart(charts.severity_scale_scatter(agg, district), use_container_width=True)
    how("""
Four code-based Plotly charts (Tableau and Power BI are banned by the rules):

- **Gap heatmap** — block × competency, spot problem clusters instantly
- **Trend lines** — is each competency improving or sliding
- **Gender gap bars** — diverging, so direction is unmistakable
- **Severity vs scale** — the reframing chart: the highest *percentage* is often
  not the highest *number of children*, which is what a policymaker acts on
""")

    # ---- Layer 6
    L = res["L6"]
    layer_header(6, "Risk Model + What-If Scenarios", "scikit-learn + arithmetic", L["ms"])
    st.write(f"**Risk model** — blocks most likely to stay below grade in **{district}**:")
    st.dataframe(L["risk"], use_container_width=True, hide_index=True)

    w = L["whatif"]
    if w:
        st.write(f"**What-If** — targeting the {w['blocks_targeted']} weakest blocks "
                 f"for **{w['competency']}** ({w['students_covered']:,} students):")
        st.caption(f"Blocks: {', '.join(w['block_names'])}  ·  "
                   f"current average {w['before_below_pct']}% below grade")
        st.dataframe(pd.DataFrame(w["scenarios"]), use_container_width=True, hide_index=True)
        st.info(f"⚠️ Natural rebound of **{w['natural_rebound_pts']} points** has been "
                f"subtracted from every scenario — the worst blocks improve somewhat "
                f"on their own (regression to the mean), and we do not claim that "
                f"as our impact.")
    how(f"""
**Risk model:** logistic regression trained live on rows that have a previous
year. Features = [current below%, year-over-year change].
*(models_ml.py → train_risk())*

**What-If — rebuilt to be defensible.** The earlier version assumed a made-up
60% recovery rate, which a judge would immediately challenge. Now:

1. Improvement rates come **from the data itself** — the 50th/75th/90th
   percentile of gains actually observed across
   {w['benchmark_sample'] if w else 0} year-over-year improvements.
2. The **regression-to-the-mean rebound is subtracted** — we target the worst
   blocks, some of which would improve anyway, so we only claim the net gain.
3. Results are a **range of scenarios**, never one false-precision number.
4. Two bugs fixed: the district filter is now actually applied, and grades are
   rolled up so "blocks" means real blocks.

*(models_ml.py → improvement_benchmarks(), natural_rebound(), what_if())*

This is a **prioritisation tool**, not a prediction. The ranking of which blocks
to target is defensible even where the absolute effect size is uncertain.
""")

    # ---- Layer 7  (was LLM)
    L = res["L7"]
    layer_header(7, "Role-Based Briefs", "template-based NLG", L["ms"])
    st.caption("🟢 computed — previously an LLM layer, now deterministic")

    role_key = st.radio(
        "Who is reading this brief?",
        list(brief.ROLES.keys()), horizontal=True,
        format_func=lambda r: f"{brief.ROLES[r]['icon']} {brief.ROLES[r]['label']}",
    )
    b = L["briefs"][role_key]
    meta = brief.ROLES[role_key]
    st.caption(f"**Scope:** {b['scope']}  ·  **Sees:** {meta['scope']}  ·  "
               f"**Acts over:** {meta['horizon']}")
    st.info(b["text"])
    st.json(b["metrics"], expanded=False)

    with st.expander("📄 See all three roles side by side"):
        for rk, bb in L["briefs"].items():
            m = brief.ROLES[rk]
            st.markdown(f"**{m['icon']} {m['label']}** — *{bb['scope']}*")
            st.write(bb["text"])
            st.divider()
        st.caption("Same data, three different readers. A block officer is never told "
                   "to reform curriculum; a policy maker is never told to run a camp.")

    st.markdown("##### 🔍 What replaced the LLM here — every clause traced to a computation")
    st.dataframe(pd.DataFrame(brief.build_breakdown(agg, district)),
                 use_container_width=True, hide_index=True)
    st.caption("The paragraph above is these five clauses joined. "
               "Change the data and the wording changes with it — but never randomly.")
    how("""
**Replaces the old LLM layer.** Conditional f-string clauses assembled from
computed facts — status, weakest/strongest competency, year-on-year direction,
count of *statistically significant* gender gaps, children affected.
*(brief.py → build())*

Reads like written prose but produces byte-identical output every run, which
the fixed-seed reproducibility rule requires.
""")

    # ---- Layer 8  (was LLM)
    L = res["L8"]
    layer_header(8, "Intervention Plan", "rule-based playbook", L["ms"])
    st.caption("🟢 computed — previously an LLM layer, now deterministic")
    for r in L["recs"]:
        st.markdown(f"**{r['priority']}** — {r['recommendation']}")
        st.caption(f"↳ rule fired: `{r['rule_fired']}`  ·  ~{r['children']:,} children")

    cov = playbook.coverage_stats(agg, district)
    if cov:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Recommendations", cov["recommendations_generated"])
        k2.metric("Unique rule combos", cov["unique_rule_combinations"])
        k3.metric("Possible combinations", f"{cov['theoretical_combinations']:,}")
        k4.metric("With a peer model", cov["with_peer_model"])
        st.caption(f"{cov['base_actions_defined']} base actions × "
                   f"{cov['modifier_clauses']} modifier clauses, composed — not "
                   f"{cov['theoretical_combinations']:,} hand-written templates.")

    st.markdown("##### 🔍 What replaced the LLM here — the decision table itself")
    p1, p2 = st.columns([1.15, 1])
    with p1:
        st.plotly_chart(charts.playbook_grid(agg, district), use_container_width=True)
        st.caption("Every block × competency falls into exactly one cell. "
                   "Empty cells fire **no** recommendation — restraint is deliberate.")
    with p2:
        st.markdown("""
**Why two axes and not one?**

| Block | below% | trend | Response |
|---|---|---|---|
| A | 64% | worsening | 🚨 emergency |
| B | 66% | improving | ✅ don't disrupt |

Same severity, **opposite action**. Ranking by below% alone would treat
them identically.

**Classifiers:**
```
severity   ≥60 Critical · ≥45 At-risk · else Strong
trajectory Δ>+1 Declining · Δ<−1 Improving · else Stagnant
equity     |gap|≥4 AND p<0.05  → adds a clause
```
The ±1 dead-band stops random wobble reading as a trend.
""")
    how("""
**Replaces the old LLM layer.** An explicit decision table the team authors
once, applied consistently by code: `(severity, trajectory, equity)` →
`(priority, action)`. *(playbook.py → PLAYBOOK, recommend())*

Every recommendation shows **which rule fired**. An education officer can ask
"why this?" and be shown the exact condition — an auditability an LLM cannot
offer. Gender-responsive clauses attach only where the gap passes the z-test.
""")

    # ---- Layer 9 replaced with limitations
    st.divider()
    st.subheader("Layer 9 — Limitations & Caveats")
    st.caption("Replaces the old survey-generation layer (no longer useful)")
    st.markdown("""
- Assessment data measures **learning evidence**, not teaching quality; low scores
  do not automatically imply poor schools.
- Geographic patterns often reflect **socio-economic conditions**. Joining Census /
  NFHS / ASER data at district level is required before drawing conclusions.
- All relationships shown are **associations, not causation**.
- The dataset carries **no school identifier**, so no analysis below Gram
  Panchayat level is possible.
- What-If figures are **scenarios under stated assumptions**, not forecasts.
""")
    how("""
The handbook explicitly rewards *"acknowledge limitations"* under Depth of
Analysis (20%). The old Layer 9 generated survey questions, which cannot be
used within a one-day event — this slot is better spent on scored criteria.
""")

    st.divider()
    total = sum(res[k]["ms"] for k in res)
    st.success(f"✅ All layers executed for {district}  ·  total **{total:.0f} ms**  ·  "
               f"no LLM, no network calls, fully reproducible")
else:
    st.info("Pick a district and click **Run all layers**.")
