"""
Layer 7 — ROLE-BASED narrative briefs (no LLM).

The same data, read by three different people who can act on different things:

  BLOCK OFFICER    -> one block. Classroom-level. "What do I do in my schools?"
  DISTRICT OFFICER -> all blocks in a district. "Where do I send resources?"
  POLICY MAKER     -> all districts. "Is this systemic or local?"

Each role gets a different SCOPE of data, a different level of DETAIL, and a
different VOCABULARY of action — a block officer is never told to reform
curriculum, and a policy maker is never told to run a remedial camp.

Template-based NLG: identical output every run, as the reproducibility rule
requires.

------------------------------------------------------------------------------
A BRIEF IS THE THING SOMEONE ACTUALLY READS
------------------------------------------------------------------------------
So it must never contradict the Insights tab, and it must never quote a figure
the rest of the system has already suppressed as noise. v1 did both:

  * NO SIZE FLOOR ANYWHERE. Asked for "the block that needs a brief" it picked
    the WORST block by percentage, which is reliably the smallest one — it
    wrote a full brief about three children. The Insights tab, correctly,
    ignored the same block entirely.
  * UNWEIGHTED YEAR-ON-YEAR CHANGE. A 4,000-child block improving 1 point and a
    10-child block collapsing 60 produced "district-wide performance has
    deteriorated by 30 points". The size-weighted truth was -0.85 — a slight
    improvement.
  * UNCORRECTED GENDER COUNTS. "Significant gender gaps in N combinations" was
    an uncorrected count: 150 blocks drawn from one distribution with no gap in
    it reported 5.
  * IT ASKED FOR AN ANALYSIS WE ALREADY RUN. The policy brief ended "before
    attributing this to school quality it should be tested against
    socio-economic indicators" — which is exactly what secondary.py does. It
    now reports the answer.

Every threshold here is the same one insights.py and playbook.py use, so the
three layers cannot disagree about what counts as real.
"""
import numpy as np
import pandas as pd

from stats_tests import two_proportion_z, proportion_test
from units import children_below, eff_n, headcount
# one definition of "bigger than the noise", shared with the other layers
from insights import _se_pct, _evidence, _score_diff


MIN_N = 30

ROLES = {
    "block":    {"label": "Block Education Officer",  "icon": "🏫",
                 "scope": "one block",   "horizon": "this term"},
    "district": {"label": "District Education Officer", "icon": "🏛️",
                 "scope": "all blocks in the district", "horizon": "this academic year"},
    "policy":   {"label": "State Policy Maker",       "icon": "🏢",
                 "scope": "all districts", "horizon": "1–3 years"},
}


# ------------------------------------------------------------------ helpers
def _wmean(g, col="below_pct"):
    """Student-weighted mean — a block with 3,000 kids counts more than one with 300."""
    if len(g) == 0:
        return float("nan")
    w = pd.to_numeric(g["n"], errors="coerce").fillna(0.0)
    v = pd.to_numeric(g[col], errors="coerce")
    ok = v.notna() & (w > 0)
    if not ok.any():
        return float("nan")
    return float((v[ok] * w[ok]).sum() / w[ok].sum())


def _wdelta(rows):
    """
    Size-weighted year-on-year change.

    The unweighted mean of per-row deltas let a 10-child block swing an entire
    district by 30 points. Only rows that HAVE a previous value contribute, and
    each contributes in proportion to its children.
    """
    p = rows.dropna(subset=["prev_pct"])
    if p.empty:
        return None, 0
    w = pd.to_numeric(p["n"], errors="coerce").fillna(0.0)
    if w.sum() <= 0:
        return None, 0
    d = float((((p["below_pct"] - p["prev_pct"]) * w).sum()) / w.sum())
    return d, int(w.sum())


def _status(below, n=None):
    """Band on the evidence when a size is available, on the value when not."""
    b = below if n is None else below - 1.96 * _se_pct(below, n)
    if b >= 60: return "critical"
    if b >= 45: return "at-risk"
    if b >= 30: return "developing"
    return "broadly healthy"


def _big(rows, by, min_n):
    """Group `rows` by `by`, keep only groups with enough children."""
    g = (rows.groupby(by)
              .apply(lambda s: pd.Series({"v": _wmean(s), "n": s["n"].sum()}),
                     include_groups=False)
              .dropna())
    return g[g["n"] >= min_n]


def _sig_gender(rows):
    """
    Weighted gender gap + significance for a slice of rows.

    The z-test is sized by CHILDREN, not responses. One child's 20 answers are
    one child measured 20 times, not 20 independent observations; using the
    response count shrinks the standard error by sqrt(20) and reports noise as
    significant.
    """
    if len(rows) == 0:
        return 0.0, 1.0, False
    kids = eff_n(rows)
    f = _wmean(rows, "f_below")
    m = _wmean(rows, "m_below")
    if not (np.isfinite(f) and np.isfinite(m)):
        return 0.0, 1.0, False
    # Split those children into arms using the observed girl share, rather than
    # assuming 50/50.
    if "f_n" in rows.columns and rows["n"].sum() > 0:
        share = float(rows["f_n"].sum()) / float(rows["n"].sum())
    else:
        share = 0.5
    nf = max(int(round(kids * share)), 1)
    nm = max(kids - nf, 1)
    _, p, _method = proportion_test(f, m, nf, nm)
    return f - m, p, (abs(f - m) >= 4 and p < 0.05)


def _bh(pvals):
    """Benjamini-Hochberg adjusted p-values, order preserved."""
    m = len(pvals)
    if not m:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, prev = [1.0] * m, 1.0
    for rank, i in enumerate(reversed(order), start=1):
        prev = min(prev, pvals[i] * m / (m - rank + 1))
        adj[i] = prev
    return adj


def _count_gender_gaps(cur, min_n):
    """
    How many block x competency cells show a REAL gender gap.

    Every cell is a separate test, so an uncorrected count is guaranteed to be
    positive on a big district whether or not any gap exists.
    """
    cells, ps = [], []
    for key, rows in cur.groupby(["block", "competency"]):
        if int(rows["n"].sum()) < min_n:
            continue
        gap, p, _ = _sig_gender(rows)
        cells.append((key, gap))
        ps.append(p)
    if not cells:
        return 0, 0
    adj = _bh(ps)
    return sum(1 for (_, gap), a in zip(cells, adj)
               if abs(gap) >= 4 and a < 0.05), len(cells)


def _latest(d, year):
    return d[d["year"] == (year if year is not None else d["year"].max())]


# ------------------------------------------------- district context (layer 6)
def _ctx_for(context, unit):
    """
    What the cross-dataset join says about this district — plain fields.

    Returns None when no context was supplied or it cannot speak about this
    unit, so every caller degrades to a primary-only sentence.
    """
    if not context:
        return None
    try:
        import insights_cross as X
    except Exception:
        return None
    fit = context.get("fit") or {}
    out = {"level": context.get("level", "District"),
           "n_units": context.get("n_units", 0),
           "over_under": None, "explains": None, "no_link": False,
           "n_tested": 0, "descriptive_only": context.get("descriptive_only")}
    tab = context.get("table")
    if tab is not None and not tab.empty:
        tested = tab[~tab["derived"]]
        out["n_tested"] = int(len(tested))
        out["no_link"] = bool(len(tested) and not (
            tested["verdict"] == "significant after FDR correction").any())
    if fit.get("ok") and fit.get("usable"):
        out["explains"] = float(fit.get("adj_r2") or 0.0)
        if unit is not None:
            val, se = X._loo_residual(context, unit)
            if val is not None and X._evidence(val, se) > 0:
                out["over_under"] = float(val)
    return out


def _ctx_sentence(c, unit, role):
    """One sentence of district context, phrased for the role."""
    if not c:
        return None
    if c["over_under"] is not None:
        better = c["over_under"] < 0
        mag = abs(c["over_under"])
        if role == "block":
            return (f"For context, {unit} as a district performs {mag:.1f} points "
                    f"{'better' if better else 'worse'} than its income, literacy "
                    f"and staffing would predict — "
                    + ("so the gaps in your block are local rather than a symptom "
                       "of district-wide disadvantage."
                       if better else
                       "so expect district-level constraints alongside anything "
                       "you can fix in your own schools."))
        return (f"Allowing for income, literacy and staffing, {unit} performs "
                f"{mag:.1f} points {'better' if better else 'worse'} than "
                f"comparable districts — "
                + ("evidence that local practice is adding value, and worth "
                   "documenting before it is lost."
                   if better else
                   "which points at delivery rather than circumstances."))
    if c["no_link"] and c["n_tested"]:
        return (f"None of the {c['n_tested']} district indicators tested — income, "
                f"literacy, teacher numbers, infrastructure — explains where "
                f"districts land, so differences here are unlikely to be closed "
                f"by resourcing alone.")
    if c["explains"] is not None and c["explains"] > 0.05:
        return (f"District circumstances account for about "
                f"{100 * c['explains']:.0f}% of the variation between districts; "
                f"the rest is what schools and teaching can move.")
    return None


# =============================================================== BLOCK OFFICER
def _block_brief(agg, district, block, year=None, min_n=MIN_N, context=None):
    d = agg[(agg["district"] == district) & (agg["block"] == block)]
    if d.empty:
        return None
    yr = year if year is not None else d["year"].max()
    cur = _latest(d, yr)
    n_here = int(cur["n"].sum())
    if n_here < min_n:
        return None                     # too few children to write a brief about

    below = _wmean(cur)
    kids = children_below(cur, below)

    comp = _big(cur, "competency", min_n)["v"].sort_values()
    if len(comp) < 1:
        return None
    worst, best = comp.index[-1], comp.index[0]

    # how do I compare with my own district?
    dist_cur = _latest(agg[agg["district"] == district], yr)
    dist_below = _wmean(dist_cur)
    vs = below - dist_below

    # which grade needs the attention — only grades big enough to act on
    gr = _big(cur, "grade", min_n)["v"]
    worst_grade = int(gr.idxmax()) if len(gr) >= 2 else None

    delta, _dn = _wdelta(cur)
    gap, p, sig = _sig_gender(cur)
    sig = sig and eff_n(cur) >= min_n

    paras = []
    paras.append(
        f"{block} has {below:.0f}% of children below grade level in {int(yr)} — "
        + (f"{abs(vs):.0f} points {'above' if vs > 0 else 'below'} the {district} "
           f"average of {dist_below:.0f}%." if abs(vs) >= 1 else
           f"in line with the {district} average.")
    )
    if len(comp) >= 2:
        paras.append(
            f"Your weakest area is {worst} ({comp[worst]:.0f}% below grade) and your "
            f"strongest is {best} ({comp[best]:.0f}%)."
            + (f" Grade {worst_grade} needs the most classroom attention, at "
               f"{gr[worst_grade]:.0f}% below grade level." if worst_grade else "")
        )
    if delta is not None and abs(delta) >= 1:
        paras.append(f"Compared with last year the block has "
                     f"{'slipped' if delta > 0 else 'improved'} by {abs(delta):.0f} points.")
    else:
        paras.append("Performance is broadly unchanged from last year.")
    if sig:
        paras.append(f"{'Girls' if gap > 0 else 'Boys'} in this block trail by "
                     f"{abs(gap):.0f} points — worth addressing through seating, "
                     f"grouping and participation practices in class.")
    cs = _ctx_sentence(_ctx_for(context, district), district, "block")
    if cs:
        paras.append(cs)
    paras.append(f"Roughly {kids:,} children in your schools need additional support "
                 f"this term.")

    return {"role": "block", "scope": f"{block}, {district}",
            "headline": f"{block} — {_status(below, n_here)}",
            "text": " ".join(paras),
            "metrics": {"below_pct": round(below, 1), "children": kids,
                        "students": n_here,
                        "vs_district_pts": round(vs, 1),
                        "weakest": worst, "focus_grade": worst_grade}}


# ============================================================ DISTRICT OFFICER
def _district_brief(agg, district, year=None, min_n=MIN_N, context=None):
    d = agg[agg["district"] == district]
    if d.empty:
        return None
    yr = year if year is not None else d["year"].max()
    cur = _latest(d, yr)

    below = _wmean(cur)
    students = headcount(cur)
    kids = children_below(cur, below)

    comp = _big(cur, "competency", min_n)["v"].sort_values()
    blk = _big(cur, "block", min_n)["v"].sort_values()
    spread = float(blk.max() - blk.min()) if len(blk) >= 2 else 0.0

    # where the children actually are — priority is burden, not percentage.
    burden = (cur.groupby("block")
                 .apply(lambda g: children_below(g, _wmean(g)), include_groups=False)
                 .sort_values(ascending=False))
    burden = burden[burden.index.isin(blk.index)] if len(blk) else burden
    top_blocks = list(burden.index[:3])

    delta, _dn = _wdelta(cur)
    n_sig, n_tested = _count_gender_gaps(cur, min_n)

    paras = []
    paras.append(
        f"{district} is {_status(below, int(cur['n'].sum()))}, with {below:.0f}% of "
        f"{students:,} assessed children below grade level across "
        f"{cur['block'].nunique()} blocks ({kids:,} children)."
    )
    if len(comp) >= 2:
        paras.append(
            f"{comp.index[-1]} is the district's weakest competency at "
            f"{comp.iloc[-1]:.0f}% below grade level; {comp.index[0]} is strongest "
            f"at {comp.iloc[0]:.0f}%."
        )
    if len(blk) >= 2:
        paras.append(
            f"Performance varies {spread:.0f} points between blocks — from "
            f"{blk.index[0]} at {blk.iloc[0]:.0f}% to {blk.index[-1]} at "
            f"{blk.iloc[-1]:.0f}% — so a single district figure conceals "
            f"substantial internal inequity."
            if spread >= 8 else
            f"Performance is fairly uniform across blocks (a {spread:.0f}-point range)."
        )
    if top_blocks:
        paras.append(
            f"By absolute number of children affected, resources should concentrate "
            f"first on {', '.join(top_blocks)}."
        )
    if delta is not None and abs(delta) >= 1:
        paras.append(f"District-wide performance has "
                     f"{'deteriorated' if delta > 0 else 'improved'} "
                     f"by {abs(delta):.0f} points year-on-year.")
    if n_sig:
        paras.append(f"Statistically significant gender gaps were found in {n_sig} of "
                     f"{n_tested} block-competency combinations, after correcting for "
                     f"the number of comparisons, and warrant targeted review.")
    cs = _ctx_sentence(_ctx_for(context, district), district, "district")
    if cs:
        paras.append(cs)

    return {"role": "district", "scope": district,
            "headline": f"{district} — {_status(below, int(cur['n'].sum()))}",
            "text": " ".join(paras),
            "metrics": {"below_pct": round(below, 1), "children": kids,
                        "block_spread_pts": round(spread, 1),
                        "weakest": comp.index[-1] if len(comp) else None,
                        "priority_blocks": top_blocks,
                        "significant_gender_gaps": n_sig,
                        "gender_cells_tested": n_tested}}


# =============================================================== POLICY MAKER
def _policy_brief(agg, year=None, min_n=MIN_N, context=None):
    yr = year if year is not None else agg["year"].max()
    cur = _latest(agg, yr)

    below = _wmean(cur)
    students = headcount(cur)
    kids = children_below(cur, below)

    dist = _big(cur, "district", min_n)["v"].sort_values()
    comp = _big(cur, "competency", min_n)["v"].sort_values()
    if not len(comp):
        return None
    worst_comp = comp.index[-1]

    # SYSTEMIC test: is the same competency weakest almost everywhere?
    per = (cur.groupby(["district", "competency"])
              .apply(lambda s: pd.Series({"v": _wmean(s), "n": s["n"].sum()}),
                     include_groups=False)
              .dropna())
    per = per[per["n"] >= min_n]
    weakest_per_district = per["v"].groupby(level=0).idxmax().apply(lambda t: t[1])
    share = float((weakest_per_district == worst_comp).mean()) if len(
        weakest_per_district) else 0.0
    systemic = share >= 0.6

    delta, _dn = _wdelta(cur)
    gap, p, sig = _sig_gender(cur)

    paras = []
    paras.append(
        f"Across {cur['district'].nunique()} districts, {below:.0f}% of {students:,} "
        f"assessed children are below grade level in {int(yr)} — approximately "
        f"{kids:,} children statewide."
    )
    if systemic:
        paras.append(
            f"{worst_comp} is the weakest competency in {share*100:.0f}% of districts. "
            f"A weakness this uniform points to a curriculum, teaching-material or "
            f"teacher-preparation issue rather than local implementation, and is best "
            f"addressed through state-level instructional design."
        )
    else:
        paras.append(
            f"{worst_comp} is weakest overall at {comp.iloc[-1]:.0f}%, but the "
            f"weakest competency differs by district, indicating local implementation "
            f"factors rather than a single systemic cause."
        )

    # The spread, and then — crucially — what the cross-dataset join says about
    # it. v1 ended this paragraph by RECOMMENDING the socio-economic test we
    # already run, which reads as though the analysis stopped short.
    if len(dist) >= 2:
        spread_txt = (
            f"District performance ranges from {dist.index[0]} at {dist.iloc[0]:.0f}% "
            f"to {dist.index[-1]} at {dist.iloc[-1]:.0f}% below grade level — a "
            f"{dist.max() - dist.min():.0f}-point spread.")
        c = _ctx_for(context, None)
        if c and c.get("descriptive_only"):
            spread_txt += (
                f" At {c['level'].lower()} level there are only {c['n_units']} units, "
                f"too few to test that spread against socio-economic indicators; the "
                f"district-level view does support the test.")
        elif c and c["no_link"] and c["n_tested"]:
            spread_txt += (
                f" Tested against {c['n_tested']} socio-economic indicators — income, "
                f"literacy, teacher numbers, infrastructure — none explains where a "
                f"district lands. The spread is therefore not attributable to district "
                f"circumstances, which points at school and classroom factors.")
        elif c and c["explains"] is not None:
            spread_txt += (
                f" Tested against {c['n_tested']} socio-economic indicators, district "
                f"circumstances account for about {100 * c['explains']:.0f}% of it; the "
                f"remainder is what schooling can move.")
        else:
            spread_txt += (" Before attributing this to school quality it should be "
                           "tested against socio-economic indicators.")
        paras.append(spread_txt)

    if delta is not None and abs(delta) >= 1:
        paras.append(f"The statewide trend is "
                     f"{'negative' if delta > 0 else 'positive'}, moving "
                     f"{abs(delta):.0f} points year-on-year.")
    if sig:
        paras.append(f"A statewide gender gap of {abs(gap):.0f} points is present "
                     f"({'girls' if gap > 0 else 'boys'} trailing), which is consistent "
                     f"enough to justify a state-level equity programme.")

    return {"role": "policy", "scope": "All districts",
            "headline": f"Statewide — {_status(below, int(cur['n'].sum()))}",
            "text": " ".join(paras),
            "metrics": {"below_pct": round(below, 1), "children": kids,
                        "districts": int(cur["district"].nunique()),
                        "weakest": worst_comp, "systemic": bool(systemic),
                        "systemic_share": round(share, 2),
                        "lowest_district": dist.index[-1] if len(dist) else None}}


# ==================================================================== public
def build(agg, district=None, role="district", block=None, year=None,
          min_n=MIN_N, context=None):
    """
    Role-aware brief. Returns a dict; use .['text'] for the paragraph.

    `context` is the bundle from insights_cross.prepare(). Supplying it lets a
    brief say whether the district beats or falls short of its circumstances —
    the one question a single-table analysis cannot answer.
    """
    if role == "block":
        if block is None:
            # the block that most NEEDS the brief, among blocks big enough to
            # write one about. Picking the raw maximum picked the smallest
            # block every time — v1 produced a full brief on three children.
            b = _latest(agg[agg["district"] == district], year)
            big = _big(b, "block", min_n)
            if not len(big):
                return None
            block = big["v"].idxmax()
        return _block_brief(agg, district, block, year, min_n=min_n,
                            context=context)
    if role == "policy":
        return _policy_brief(agg, year, min_n=min_n, context=context)
    return _district_brief(agg, district, year, min_n=min_n, context=context)


def build_all(agg, district, block=None, year=None, min_n=MIN_N, context=None):
    """All three roles at once — for side-by-side comparison in the demo."""
    return {r: build(agg, district, role=r, block=block, year=year,
                     min_n=min_n, context=context) for r in ROLES}


def build_breakdown(agg, district, year=None, min_n=MIN_N, context=None):
    """Clause-by-clause traceability for the district brief."""
    d = _latest(agg[agg["district"] == district], year)
    if d.empty:
        return []
    overall = _wmean(d)
    comp = _big(d, "competency", min_n)["v"]
    blk = _big(d, "block", min_n)["v"]
    affected = int(round((d["n"] * d["below_pct"] / 100).sum()))
    delta, dn = _wdelta(d)
    n_sig, n_tested = _count_gender_gaps(d, min_n)
    c = _ctx_for(context, district)
    rows = [
        {"clause": "Overall status",
         "computation": "student-weighted mean(below_pct), banded on the lower "
                        "95% bound",
         "value": f"{overall:.1f}% → {_status(overall, int(d['n'].sum()))}"},
        {"clause": "Weakest / strongest",
         "computation": f"weighted mean by competency, ≥{min_n} students → "
                        f"idxmax / idxmin",
         "value": (f"{comp.idxmax()} {comp.max():.0f}%  /  "
                   f"{comp.idxmin()} {comp.min():.0f}%") if len(comp) else "—"},
        {"clause": "Block spread",
         "computation": f"max(block) − min(block), blocks with ≥{min_n} students",
         "value": f"{blk.max() - blk.min():.0f} pts" if len(blk) >= 2 else "—"},
        {"clause": "Year-on-year",
         "computation": "student-weighted mean(below_pct − prev_pct), fires if "
                        "|Δ| ≥ 1",
         "value": f"{delta:+.1f} pts over {dn:,} students"
                  if delta is not None else "no prior year"},
        {"clause": "Gender gaps",
         "computation": "|gap| ≥ 4 AND Benjamini-Hochberg adjusted p < 0.05 "
                        "across every cell tested",
         "value": f"{n_sig} of {n_tested} cells"},
        {"clause": "Children affected",
         "computation": "Σ(n × below_pct ÷ 100)",
         "value": f"{affected:,}"},
    ]
    if c:
        rows.append({
            "clause": "District context",
            "computation": "least-squares residual on context controls "
                           "(leave-one-out), plus FDR-corrected correlations",
            "value": (f"{c['over_under']:+.1f} pts vs circumstances"
                      if c["over_under"] is not None else
                      (f"no link among {c['n_tested']} indicators"
                       if c["no_link"] else "context model not usable"))})
    return rows
