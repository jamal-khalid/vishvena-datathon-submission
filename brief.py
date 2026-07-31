"""
Layer 7 — ROLE-BASED narrative briefs (no LLM).

The same data, read by three different people who can act on different things:

  BLOCK OFFICER    -> one block. Classroom-level. "What do I do in my schools?"
  DISTRICT OFFICER -> all blocks in a district. "Where do I send resources?"
  POLICY MAKER     -> all districts. "Is this systemic or local?"

Each role gets a different SCOPE of data, a different level of DETAIL, and a
different VOCABULARY of action — a block officer is never told to reform
curriculum, and a policy maker is never told to run a remedial camp.

Template-based NLG: identical output every run, as the reproducibility rule requires.
"""
import numpy as np
import pandas as pd
from stats_tests import two_proportion_z, proportion_test
from units import children_below, eff_n, headcount

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
    n = g["n"].sum()
    return float((g[col] * g["n"]).sum() / n) if n else float("nan")


def _status(below):
    if below >= 60: return "critical"
    if below >= 45: return "at-risk"
    if below >= 30: return "developing"
    return "broadly healthy"


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


def _latest(d, year):
    return d[d["year"] == (year if year is not None else d["year"].max())]


# =============================================================== BLOCK OFFICER
def _block_brief(agg, district, block, year=None):
    d = agg[(agg["district"] == district) & (agg["block"] == block)]
    if d.empty:
        return None
    yr = year if year is not None else d["year"].max()
    cur = _latest(d, yr)

    below = _wmean(cur)
    kids = children_below(cur, below)

    comp = cur.groupby("competency").apply(_wmean, include_groups=False).sort_values()
    worst, best = comp.index[-1], comp.index[0]

    # how do I compare with my own district?
    dist_cur = _latest(agg[agg["district"] == district], yr)
    dist_below = _wmean(dist_cur)
    vs = below - dist_below

    # which grade needs the attention
    gr = cur.groupby("grade").apply(_wmean, include_groups=False)
    worst_grade = int(gr.idxmax())

    prev = cur.dropna(subset=["prev_pct"])
    delta = float((prev["below_pct"] - prev["prev_pct"]).mean()) if not prev.empty else None

    gap, p, sig = _sig_gender(cur)

    paras = []
    paras.append(
        f"{block} has {below:.0f}% of children below grade level in {int(yr)} — "
        + (f"{abs(vs):.0f} points {'above' if vs > 0 else 'below'} the {district} "
           f"average of {dist_below:.0f}%." if abs(vs) >= 1 else
           f"in line with the {district} average.")
    )
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
    paras.append(f"Roughly {kids:,} children in your schools need additional support "
                 f"this term.")

    return {"role": "block", "scope": f"{block}, {district}",
            "headline": f"{block} — {_status(below)}",
            "text": " ".join(paras),
            "metrics": {"below_pct": round(below, 1), "children": kids,
                        "vs_district_pts": round(vs, 1),
                        "weakest": worst, "focus_grade": worst_grade}}


# ============================================================ DISTRICT OFFICER
def _district_brief(agg, district, year=None):
    d = agg[agg["district"] == district]
    if d.empty:
        return None
    yr = year if year is not None else d["year"].max()
    cur = _latest(d, yr)

    below = _wmean(cur)
    students = headcount(cur)
    kids = children_below(cur, below)

    comp = cur.groupby("competency").apply(_wmean, include_groups=False).sort_values()
    worst, best = comp.index[-1], comp.index[0]

    blk = cur.groupby("block").apply(_wmean, include_groups=False).sort_values()
    spread = float(blk.max() - blk.min())

    # where the children actually are — priority is burden, not percentage.
    # Ranking only needs relative burden, but report it in real children.
    burden = (cur.groupby("block")
                 .apply(lambda g: children_below(g, _wmean(g)), include_groups=False)
                 .sort_values(ascending=False))
    top_blocks = list(burden.index[:3])

    prev = cur.dropna(subset=["prev_pct"])
    delta = float((prev["below_pct"] - prev["prev_pct"]).mean()) if not prev.empty else None

    n_sig = 0
    for _, rows in cur.groupby(["block", "competency"]):
        if _sig_gender(rows)[2]:
            n_sig += 1

    paras = []
    paras.append(
        f"{district} is {_status(below)}, with {below:.0f}% of {students:,} assessed "
        f"children below grade level across {cur['block'].nunique()} blocks "
        f"({kids:,} children)."
    )
    paras.append(
        f"{worst} is the district's weakest competency at {comp[worst]:.0f}% below "
        f"grade level; {best} is strongest at {comp[best]:.0f}%."
    )
    paras.append(
        f"Performance varies {spread:.0f} points between blocks — from {blk.index[0]} "
        f"at {blk.iloc[0]:.0f}% to {blk.index[-1]} at {blk.iloc[-1]:.0f}% — so a single "
        f"district figure conceals substantial internal inequity."
        if spread >= 8 else
        f"Performance is fairly uniform across blocks (a {spread:.0f}-point range)."
    )
    paras.append(
        f"By absolute number of children affected, resources should concentrate first "
        f"on {', '.join(top_blocks)}."
    )
    if delta is not None and abs(delta) >= 1:
        paras.append(f"District-wide performance has "
                     f"{'deteriorated' if delta > 0 else 'improved'} "
                     f"by {abs(delta):.0f} points year-on-year.")
    if n_sig:
        paras.append(f"Statistically significant gender gaps were found in {n_sig} "
                     f"block-competency combinations and warrant targeted review.")

    return {"role": "district", "scope": district,
            "headline": f"{district} — {_status(below)}",
            "text": " ".join(paras),
            "metrics": {"below_pct": round(below, 1), "children": kids,
                        "block_spread_pts": round(spread, 1),
                        "weakest": worst, "priority_blocks": top_blocks,
                        "significant_gender_gaps": n_sig}}


# =============================================================== POLICY MAKER
def _policy_brief(agg, year=None):
    yr = year if year is not None else agg["year"].max()
    cur = _latest(agg, yr)

    below = _wmean(cur)
    students = headcount(cur)
    kids = children_below(cur, below)

    dist = cur.groupby("district").apply(_wmean, include_groups=False).sort_values()
    comp = cur.groupby("competency").apply(_wmean, include_groups=False).sort_values()
    worst_comp = comp.index[-1]

    # SYSTEMIC test: is the same competency weakest almost everywhere?
    weakest_per_district = (cur.groupby(["district", "competency"])
                              .apply(_wmean, include_groups=False)
                              .groupby(level=0).idxmax().apply(lambda t: t[1]))
    share = float((weakest_per_district == worst_comp).mean())
    systemic = share >= 0.6

    prev = cur.dropna(subset=["prev_pct"])
    delta = float((prev["below_pct"] - prev["prev_pct"]).mean()) if not prev.empty else None

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
            f"{worst_comp} is weakest overall at {comp[worst_comp]:.0f}%, but the "
            f"weakest competency differs by district, indicating local implementation "
            f"factors rather than a single systemic cause."
        )
    paras.append(
        f"District performance ranges from {dist.index[0]} at {dist.iloc[0]:.0f}% to "
        f"{dist.index[-1]} at {dist.iloc[-1]:.0f}% below grade level — a "
        f"{dist.max()-dist.min():.0f}-point spread. Before attributing this to school "
        f"quality it should be tested against socio-economic indicators."
    )
    if delta is not None and abs(delta) >= 1:
        paras.append(f"The statewide trend is "
                     f"{'negative' if delta > 0 else 'positive'}, moving "
                     f"{abs(delta):.0f} points year-on-year.")
    if sig:
        paras.append(f"A statewide gender gap of {abs(gap):.0f} points is present "
                     f"({'girls' if gap > 0 else 'boys'} trailing), which is consistent "
                     f"enough to justify a state-level equity programme.")

    return {"role": "policy", "scope": "All districts",
            "headline": f"Statewide — {_status(below)}",
            "text": " ".join(paras),
            "metrics": {"below_pct": round(below, 1), "children": kids,
                        "districts": int(cur["district"].nunique()),
                        "weakest": worst_comp, "systemic": bool(systemic),
                        "systemic_share": round(share, 2),
                        "lowest_district": dist.index[-1]}}


# ==================================================================== public
def build(agg, district=None, role="district", block=None, year=None):
    """Role-aware brief. Returns a dict; use .['text'] for the paragraph."""
    if role == "block":
        if block is None:
            b = agg[agg["district"] == district]
            block = b.groupby("block").apply(_wmean, include_groups=False).idxmax()
        return _block_brief(agg, district, block, year)
    if role == "policy":
        return _policy_brief(agg, year)
    return _district_brief(agg, district, year)


def build_all(agg, district, block=None, year=None):
    """All three roles at once — for side-by-side comparison in the demo."""
    return {r: build(agg, district, role=r, block=block, year=year) for r in ROLES}


def build_breakdown(agg, district, year=None):
    """Clause-by-clause traceability for the district brief (kept from v1)."""
    d = _latest(agg[agg["district"] == district], year)
    if d.empty:
        return []
    overall = _wmean(d)
    comp = d.groupby("competency").apply(_wmean, include_groups=False)
    blk = d.groupby("block").apply(_wmean, include_groups=False)
    affected = int(round((d["n"] * d["below_pct"] / 100).sum()))
    prev = d.dropna(subset=["prev_pct"])
    delta = (prev["below_pct"] - prev["prev_pct"]).mean() if not prev.empty else None
    n_sig = sum(1 for _, r in d.groupby(["block", "competency"]) if _sig_gender(r)[2])
    return [
        {"clause": "Overall status",
         "computation": "student-weighted mean(below_pct) → threshold rule",
         "value": f"{overall:.1f}% → {_status(overall)}"},
        {"clause": "Weakest / strongest",
         "computation": "weighted mean by competency → idxmax / idxmin",
         "value": f"{comp.idxmax()} {comp.max():.0f}%  /  {comp.idxmin()} {comp.min():.0f}%"},
        {"clause": "Block spread",
         "computation": "max(block) − min(block)",
         "value": f"{blk.max()-blk.min():.0f} pts"},
        {"clause": "Year-on-year",
         "computation": "mean(below_pct − prev_pct), fires if |Δ| ≥ 1",
         "value": f"{delta:+.1f} pts" if delta is not None else "no prior year"},
        {"clause": "Gender gaps",
         "computation": "|gap| ≥ 4 AND two-proportion z-test p < 0.05",
         "value": f"{n_sig} significant"},
        {"clause": "Children affected",
         "computation": "Σ(n × below_pct ÷ 100)",
         "value": f"{affected:,}"},
    ]
