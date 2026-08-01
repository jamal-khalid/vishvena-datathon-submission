"""
Layer 8 — Intervention Playbook v3 (no LLM).

DESIGN: composition, not enumeration.

Writing 300 hand-written recommendations is unmaintainable. Instead we compose:

    BASE ACTION      (severity x trajectory)      -> 12 cells, all filled
  + MODIFIER CLAUSES (each fires independently)   -> 10 optional clauses
  + PEER MODEL       (positive-deviance matching) -> "replicate Block Y"
  + DISTRICT CONTEXT (the cross-dataset join)     -> "...but the district
                                                     already beats its
                                                     circumstances, so this is
                                                     a local problem"

    12 base x C(10, 0..3) clause subsets = 12 x 176 = 2,112 distinct outputs.

`coverage_stats()` reports that same number — computed, never hardcoded, so
the count in the UI cannot drift from what the code can actually produce.

------------------------------------------------------------------------------
WHY A RECOMMENDATION IS HARDER TO GET RIGHT THAN AN INSIGHT
------------------------------------------------------------------------------
A wrong insight misinforms. A wrong recommendation sends a resource teacher to
the wrong block. So every rule here is gated the same way insights.py is:

  * NEVER act on a group below `min_n`. v2 issued "P1 — Immediate: launch an
    intensive remediation camp" for a block of TWO children.
  * SEVERITY AND TRAJECTORY ARE EVIDENCE-BASED. v2 called any move over 1
    point "Declining"; a 1.5-point drift across 40 children is noise, and it
    triggered the highest priority in the system.
  * CLAIM ONLY WHAT WAS CHECKED. v2 said "N consecutive years" after counting
    distinct years — 2020 and 2024 was reported as consecutive. It also picked
    the worst grade from an UNWEIGHTED mean, so a 20-child grade outranked a
    1,000-child one for delivery focus.
  * CORRECT FOR MULTIPLE TESTS. v2 attached a gender-equity clause to 7 of 200
    blocks generated from a single 50% distribution with no gap in it.
  * A PEER MODEL MUST BE WORTH COPYING. v2 could tell a 4,000-student block to
    replicate a 3-student one.
"""
import math

import numpy as np
import pandas as pd

from stats_tests import two_proportion_z, proportion_test
from units import children_below, eff_n, headcount
# One definition of "is this bigger than the noise", shared with the insight
# engine so the two layers cannot drift apart on what counts as evidence.
from insights import _se_pct, _evidence, _score_diff, _wmean


# Never recommend an intervention for a group smaller than this. The dashboard
# passes its own sidebar value.
MIN_N = 30


# ================================================================ classifiers

def severity(below_pct, n=None):
    """
    Banding on the EVIDENCE, not the point estimate.

    With n supplied a block is only "Critical" when we are confident it is
    genuinely at or past 60% — the lower bound of its interval has to clear the
    line. Without n the old point-estimate behaviour is kept so existing
    callers and the decision-grid chart still work.
    """
    if n is None:
        if below_pct >= 60: return "Critical"
        if below_pct >= 45: return "At-risk"
        return "Strong"
    lo = float(below_pct) - 1.96 * _se_pct(below_pct, n)
    if lo >= 60: return "Critical"
    if lo >= 45: return "At-risk"
    return "Strong"


def trajectory(below_pct, prev_pct, n=None):
    """
    Direction of travel, gated on evidence.

    A fixed 1-point threshold made noise look like decline: 61.0 -> 62.5 across
    40 children was "Declining" and earned P1 — Immediate. The change now has
    to exceed its own confidence interval.
    """
    if pd.isna(prev_pct):
        return "Unknown"
    d = float(below_pct) - float(prev_pct)
    if n is None:
        if d > 1:  return "Declining"
        if d < -1: return "Improving"
        return "Stagnant"
    if _score_diff(below_pct, n, prev_pct, n) <= 0:
        return "Stagnant"
    return "Declining" if d > 0 else "Improving"


def scale_band(children):
    if children >= 1000: return "Large"
    if children >= 300:  return "Medium"
    return "Small"


def _consecutive(years):
    """Longest run of consecutive years present. 2020 and 2024 is a run of 1."""
    ys = sorted({int(y) for y in years if pd.notna(y)})
    if not ys:
        return 0
    best = run = 1
    for a, b in zip(ys, ys[1:]):
        run = run + 1 if b == a + 1 else 1
        best = max(best, run)
    return best


# ============================================================== base actions
# All 12 cells of the severity x trajectory grid are filled. v2 left the three
# "Strong" combinations empty and silently `continue`d, so a block that is
# doing well and improving produced nothing at all — no way to tell "no action
# needed" apart from "the engine had nothing to say".

BASE_ACTIONS = {
    ("Critical", "Declining"): ("P1 — Immediate",
        "Launch an intensive {comp} remediation camp in {blk}, where {pct}% of children "
        "are below grade level and performance fell a further {yoy} points this year"),
    ("Critical", "Stagnant"): ("P1 — Immediate",
        "Assign a dedicated {comp} resource teacher to {blk}, where {pct}% remain below "
        "grade level with no measurable movement over the past year"),
    ("Critical", "Improving"): ("P2 — Sustain",
        "Protect and document the current {comp} approach in {blk} — it gained {gain} "
        "points this year, though {pct}% are still below grade level"),
    ("Critical", "Unknown"): ("P1 — Immediate",
        "Prioritise {comp} in {blk} for immediate diagnostic review — {pct}% are below "
        "grade level and no prior-year baseline exists to judge direction"),

    ("At-risk", "Declining"): ("P2 — Act this term",
        "Provide focused {comp} teacher mentoring in {blk}, which slipped {yoy} points "
        "to {pct}% below grade level — acting now prevents escalation to critical"),
    ("At-risk", "Stagnant"): ("P3 — Plan",
        "Add {comp} to the monthly cluster academic review for {blk}, where {pct}% "
        "remain below grade level and performance has not shifted"),
    ("At-risk", "Improving"): ("P4 — Monitor",
        "Continue the current {comp} practice in {blk} — it improved {gain} points, "
        "with {pct}% still below grade level; review again next cycle"),
    ("At-risk", "Unknown"): ("P3 — Plan",
        "Establish a {comp} baseline review for {blk} at {pct}% below grade level"),

    ("Strong", "Declining"): ("P4 — Monitor",
        "Place {blk} on the {comp} early-warning watchlist — still {pct}% below grade "
        "level but slipping {yoy} points; schedule classroom observation"),
    ("Strong", "Stagnant"): ("P5 — Hold",
        "No new {comp} intervention needed in {blk} at {pct}% below grade level — hold "
        "the current approach and keep it in the routine review cycle"),
    ("Strong", "Improving"): ("P5 — Learn from",
        "Document what {blk} is doing in {comp} — it improved {gain} points to {pct}% "
        "below grade level and is a candidate source of practice for weaker blocks"),
    ("Strong", "Unknown"): ("P5 — Hold",
        "No action indicated for {comp} in {blk} at {pct}% below grade level; a "
        "prior-year baseline would confirm the direction of travel"),
}


# ============================================================ modifier clauses
# Each returns a sentence or None. They compose onto the base action.

def _mod_equity(f):
    if not f["equity"]:
        return None
    lag = "girls" if f["gender_gap"] > 0 else "boys"
    return (f"Pair this with gender-responsive pedagogy and structured peer-learning "
            f"groups, as {lag} trail by {abs(f['gender_gap']):.0f} points here "
            f"(significant after correcting for the {f['equity_tests']} block-competency "
            f"comparisons in this district, adjusted p={f['p_adj']:.3f})")


def _mod_chronic(f):
    if not f["chronic"]:
        return None
    run = f["consecutive_years"]
    span = (f"{run} consecutive years" if run >= 2
            else f"every one of the {f['years_bad']} years measured")
    return (f"Treat this as structural rather than cyclical — {f['competency']} has been "
            f"below 50% mastery in {span} in this block")


def _mod_scale(f):
    # relative, not absolute: only the top slice of the district qualifies
    if not f.get("scale_top"):
        return None
    return (f"This is one of the largest single concentrations of below-grade children "
            f"in the district ({f['children']:,}), so it should rank high for "
            f"resource allocation")


def _mod_grade(f):
    if f["worst_grade"] is None:
        return None
    return (f"Concentrate delivery on Grade {f['worst_grade']}, which is "
            f"{f['grade_delta']:.0f} points weaker than the other grades in this block "
            f"across {f['worst_grade_n']:,} children")


def _mod_peer(f):
    p = f["peer"]
    if p is None:
        return None
    return (f"Study and replicate the approach used in {p['block']}, which has a very "
            f"similar overall learning profile across {p['their_n']:,} children but "
            f"achieves {p['their_pct']:.0f}% below grade in {f['competency']} versus "
            f"{p['our_pct']:.0f}% here (a {p['edge']:.0f}-point advantage)")


def _mod_bundle(f):
    if not f["bundle_with"]:
        return None
    return (f"Deliver this jointly with {f['bundle_with']}, which fails in the same "
            f"blocks (r={f['bundle_r']:.2f} across {f['bundle_n']} blocks, p="
            f"{f['bundle_p']:.3f}) and likely shares a root cause")


# ---- cross-dataset clauses: what the district CONTEXT adds ------------------
# These are the reason to join a second dataset at all. They change what the
# action should be, not just how it is described: a block inside a district
# that already outperforms its circumstances has a local problem, while one in
# a district falling short of its circumstances needs district-level attention
# too, or the block-level fix will not hold.

def _mod_ctx_over(f):
    c = f.get("ctx") or {}
    if c.get("over_under") is None or c["over_under"] >= 0:
        return None
    return (f"Note that {c['unit']} as a whole already performs "
            f"{abs(c['over_under']):.1f} points BETTER than its income, literacy and "
            f"staffing predict — so this block is a local exception, not a symptom of "
            f"district-wide disadvantage, and a block-level fix should hold")


def _mod_ctx_under(f):
    c = f.get("ctx") or {}
    if c.get("over_under") is None or c["over_under"] <= 0:
        return None
    return (f"Escalate alongside a district-level review: {c['unit']} performs "
            f"{c['over_under']:.1f} points WORSE than its circumstances predict, so "
            f"fixing this block alone is unlikely to hold while the wider system "
            f"under-delivers")


def _mod_ctx_no_link(f):
    c = f.get("ctx") or {}
    if not c.get("no_link"):
        return None
    return (f"Resourcing alone is unlikely to close this: across {c['n_units']} "
            f"{c['level'].lower()}s none of the {c['n_tested']} context indicators "
            f"— income, literacy, teacher numbers, infrastructure — explains where "
            f"results land, so the lever here is classroom practice rather than inputs")


def _mod_ctx_peer_unit(f):
    c = f.get("ctx") or {}
    if not c.get("peer_units"):
        return None
    return (f"Benchmark against {c['peer_units']}, the {c['level'].lower()}s with the "
            f"most similar income, literacy and staffing to {c['unit']} — they average "
            f"{c['peer_pct']:.1f}% below grade level against {c['unit_pct']:.1f}% here")


MODIFIERS = [
    ("peer",         _mod_peer),
    ("chronic",      _mod_chronic),
    ("equity",       _mod_equity),
    ("ctx_under",    _mod_ctx_under),
    ("ctx_over",     _mod_ctx_over),
    ("grade",        _mod_grade),
    ("ctx_peer",     _mod_ctx_peer_unit),
    ("bundle",       _mod_bundle),
    ("ctx_no_link",  _mod_ctx_no_link),
    ("scale",        _mod_scale),
]


# ========================================================= positive deviance

def _peer_matrix(agg, year, min_n=MIN_N):
    """
    Block x competency grid of size-weighted below%, with a matching grid of
    child counts so a peer that is too small to learn from can be excluded.
    """
    d = agg[agg["year"] == year]
    if d.empty:
        return None, None
    g = (d.groupby(["district", "block", "competency"])
           .apply(lambda s: pd.Series({"below": _wmean(s), "n": s["n"].sum()}),
                  include_groups=False)
           .reset_index())
    piv = g.pivot_table(index=["district", "block"], columns="competency",
                        values="below")
    cnt = g.pivot_table(index=["district", "block"], columns="competency",
                        values="n")
    return piv, cnt


def find_peer_model(piv, block, competency, min_edge=8.0, cnt=None,
                    min_n=MIN_N):
    """
    Positive deviance: find a block with a SIMILAR profile on every other
    competency, but which performs notably better on THIS one.

    That similarity constraint is what makes the comparison fair — we are not
    telling a struggling rural block to copy a well-resourced urban one. The
    size constraint is what makes it actionable: v2 could recommend copying a
    three-student block, whose 0% below-grade figure is an accident of having
    three children rather than a practice worth spreading.
    """
    if piv is None or competency not in piv.columns:
        return None
    rows = [i for i in piv.index if i[1] == block]
    if not rows:
        return None
    me = piv.loc[rows[0]]
    others = piv.drop(rows[0])
    if others.empty:
        return None

    other_cols = [c for c in piv.columns if c != competency]
    if not other_cols:
        return None

    # must beat us by a clear margin on the target competency
    better = others[others[competency] <= me[competency] - min_edge]
    # ...and be large enough that its result means something
    if cnt is not None and competency in cnt.columns:
        sizes = cnt[competency].reindex(better.index)
        better = better[sizes.fillna(0) >= min_n]
    if better.empty:
        return None

    # among those, the most similar overall profile
    dist = np.sqrt(((better[other_cols] - me[other_cols]) ** 2).sum(axis=1))
    dist = dist.dropna()
    if dist.empty:
        return None
    best = dist.idxmin()
    their_n = (int(cnt.loc[best, competency])
               if cnt is not None and competency in cnt.columns
               and pd.notna(cnt.loc[best, competency]) else 0)
    return {"block": best[1], "district": best[0],
            "their_pct": float(better.loc[best, competency]),
            "our_pct": float(me[competency]),
            "edge": float(me[competency] - better.loc[best, competency]),
            "their_n": their_n,
            "profile_distance": float(dist.loc[best])}


MAX_CLAUSES = 3          # keep recommendations readable and varied


def _bundle_partner(agg, competency, year, min_r=0.90, alpha=0.05,
                    min_blocks=8, min_n=MIN_N):
    """
    Which other competency fails in the same blocks?

    v2 accepted any r >= 0.90 with no test and no minimum block count. Across
    four blocks of pure noise that fired on 5 of 40 random datasets — a
    one-in-eight chance of inventing a shared root cause. Now the correlation
    must also be significant and rest on enough blocks to mean anything.
    """
    piv, cnt = _peer_matrix(agg, year, min_n=min_n)
    if piv is None or competency not in piv.columns:
        return None, None, None, 0
    use = piv
    if cnt is not None:
        big = (cnt.fillna(0) >= min_n).all(axis=1)
        use = piv[big]
    use = use.dropna(axis=0, how="any")
    if len(use) < min_blocks:
        return None, None, None, len(use)
    corr = use.corr()[competency].drop(competency, errors="ignore")
    if corr.empty:
        return None, None, None, len(use)
    partner = corr.idxmax()
    r = float(corr[partner])
    if r < min_r:
        return None, None, None, len(use)
    _, p = _corr_p(r, len(use))
    if p >= alpha:
        return None, None, None, len(use)
    return partner, r, p, len(use)


def _corr_p(r, n):
    """Two-sided p for a Pearson r, via the same Student's t used elsewhere."""
    import secondary as _S
    if n < 3 or abs(r) >= 1:
        return 0.0, 0.0 if abs(r) >= 1 else 1.0
    t = r * math.sqrt((n - 2) / max(1 - r * r, 1e-15))
    return t, _S.t_two_sided_p(t, n - 2)


# ==================================================================== engine

def _features(agg, district, competency, block, cur_rows, all_rows, piv, cnt,
              bundle, min_n=MIN_N, ctx=None):
    below = _wmean(cur_rows)
    prev_series = cur_rows["prev_pct"].dropna()
    prev = (float((cur_rows.loc[prev_series.index, "prev_pct"]
                   * cur_rows.loc[prev_series.index, "n"]).sum()
                  / cur_rows.loc[prev_series.index, "n"].sum())
            if not prev_series.empty else np.nan)
    n = int(cur_rows["n"].sum())
    children = children_below(cur_rows, below)

    # gender, tested — sized by children, not by assessment responses (units.py),
    # split into real arms, and via a test that stays valid on small cells
    f_b = _wmean(cur_rows, "f_below")
    m_b = _wmean(cur_rows, "m_below")
    kids = eff_n(cur_rows)
    share = (float(cur_rows["f_n"].sum()) / n) if "f_n" in cur_rows.columns and n else 0.5
    nf = max(int(round(kids * share)), 1)
    nm = max(kids - nf, 1)
    _, p, _method = proportion_test(f_b, m_b, nf, nm)

    # chronic: below 50 in every year we have, weighted within each year
    yrs = all_rows.groupby("year").apply(
        lambda g: _wmean(g), include_groups=False)
    chronic = bool(len(yrs) >= 2 and (yrs >= 50).all())

    # worst grade, if clearly worse than the rest — WEIGHTED, and only over
    # grades big enough to act on. An unweighted mean sent delivery at a
    # 20-child grade ahead of a 1,000-child one.
    worst_grade, grade_delta, worst_grade_n = None, 0.0, 0
    gr = (cur_rows.groupby("grade")
          .apply(lambda g: pd.Series({"below": _wmean(g), "n": g["n"].sum()}),
                 include_groups=False)
          .dropna())
    gr = gr[gr["n"] >= min_n]
    if len(gr) >= 2:
        top = gr["below"].idxmax()
        rest = gr.drop(index=top)
        ref = float((rest["below"] * rest["n"]).sum() / rest["n"].sum())
        delta = float(gr.loc[top, "below"] - ref)
        if delta >= 5 and _score_diff(gr.loc[top, "below"], gr.loc[top, "n"],
                                      ref, rest["n"].sum()) > 0:
            worst_grade = int(top)
            grade_delta = delta
            worst_grade_n = int(gr.loc[top, "n"])

    partner, b_r, b_p, b_n = bundle
    return {
        "block": block, "competency": competency, "district": district,
        "below_pct": below, "prev_pct": prev, "n": n, "children": children,
        "severity": severity(below, n), "trajectory": trajectory(below, prev, n),
        "scale": scale_band(children),
        "gender_gap": f_b - m_b, "p": p, "p_adj": 1.0, "equity": False,
        "equity_tests": 0, "nf": nf, "nm": nm,
        "chronic": chronic, "years_bad": int(len(yrs)),
        "consecutive_years": _consecutive(yrs.index),
        "worst_grade": worst_grade, "grade_delta": grade_delta,
        "worst_grade_n": worst_grade_n,
        "peer": find_peer_model(piv, block, competency, cnt=cnt, min_n=min_n),
        "bundle_with": partner, "bundle_r": b_r or 0.0,
        "bundle_p": b_p if b_p is not None else 1.0, "bundle_n": b_n,
        "ctx": ctx,
    }


PRIORITY_ORDER = {"P1 — Immediate": 0, "P2 — Sustain": 1, "P2 — Act this term": 1,
                  "P3 — Plan": 2, "P4 — Monitor": 3, "P5 — Hold": 4,
                  "P5 — Learn from": 4}


def _context_features(ctx, district):
    """Pull this district's cross-dataset situation into plain fields."""
    if not ctx:
        return None
    try:
        import insights_cross as X
    except Exception:
        return None
    key, out_col = ctx.get("key", "District"), ctx.get("outcome_label")
    m = ctx.get("merged")
    unit = str(district)
    info = {"unit": unit, "level": ctx.get("level", "District"),
            "n_units": ctx.get("n_units", 0), "over_under": None,
            "no_link": False, "n_tested": 0,
            "peer_units": None, "peer_pct": None, "unit_pct": None}

    tab = ctx.get("table")
    if tab is not None and not tab.empty:
        tested = tab[~tab["derived"]]
        info["n_tested"] = int(len(tested))
        info["no_link"] = bool(
            len(tested) and
            not (tested["verdict"] == "significant after FDR correction").any())

    row = X._resid_row(ctx, district)
    fit = ctx.get("fit") or {}
    if row is not None and fit.get("ok") and fit.get("usable"):
        val, se = X._loo_residual(ctx, district)
        if val is not None and X._evidence(val, se) > 0:
            info["over_under"] = -float(val)      # positive = worse than context

    # nearest peers by context, for the benchmark clause
    if m is not None and out_col in m.columns:
        ctrls = [c for c in fit.get("controls", []) if c in m.columns]
        if len(ctrls) >= 2:
            d = m[[key, out_col] + ctrls].dropna()
            hit = d[d[key].astype(str).str.strip().str.lower() == unit.lower()]
            if not hit.empty and len(d) >= 6:
                i = hit.index[0]
                Z = d[ctrls].astype(float)
                Z = (Z - Z.mean()) / Z.std().replace(0, 1.0)
                dist = ((Z - Z.loc[i]) ** 2).sum(axis=1) ** 0.5
                peers = dist.drop(index=i).nsmallest(3).index
                if len(peers) >= 2:
                    info["peer_units"] = " and ".join(
                        str(x) for x in d.loc[peers, key])
                    info["peer_pct"] = float(d.loc[peers, out_col].mean())
                    info["unit_pct"] = float(d.loc[i, out_col])
    return info


def _bh(pvals):
    """Benjamini-Hochberg adjusted p-values, order preserved."""
    m = len(pvals)
    if not m:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, prev = [1.0] * m, 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = m - rank + 1
        prev = min(prev, pvals[i] * m / k)
        adj[i] = prev
    return adj


def recommend(agg, district, year=None, limit=10, min_n=MIN_N, context=None):
    """
    Ranked, actionable recommendations for one district.

    `context` is the bundle from insights_cross.prepare(). When supplied, the
    district's standing against comparable districts changes the advice, not
    just its wording — see the _mod_ctx_* clauses.
    """
    d = agg[agg["district"] == district]
    if d.empty:
        return []
    if year is None:
        year = d["year"].max()
    cur = d[d["year"] == year]
    if cur.empty:
        return []

    piv, cnt = _peer_matrix(agg, year, min_n=min_n)
    bundles = {c: _bundle_partner(agg, c, year, min_n=min_n)
               for c in cur["competency"].unique()}
    ctx = _context_features(context, district)

    # "large scale" is relative to this district — top quartile of children affected
    burden = (cur.assign(aff=cur["n"] * cur["below_pct"] / 100)
                 .groupby(["block", "competency"])["aff"].sum())
    scale_cut = float(burden.quantile(0.75)) if len(burden) > 3 else float("inf")

    # ---- pass 1: features for every cell that is big enough to act on ------
    feats = []
    for (blk, comp), rows in cur.groupby(["block", "competency"]):
        if int(rows["n"].sum()) < min_n:
            continue                     # never recommend into a tiny group
        all_rows = d[(d["block"] == blk) & (d["competency"] == comp)]
        f = _features(agg, district, comp, blk, rows, all_rows, piv, cnt,
                      bundles[comp], min_n=min_n, ctx=ctx)
        f["scale_top"] = burden.get((blk, comp), 0) >= scale_cut
        feats.append(f)
    if not feats:
        return []

    # ---- gender: one family of tests, corrected together -------------------
    # Every block x competency cell is a separate comparison. Testing each at
    # 0.05 and attaching a clause to whichever passes guarantees false
    # positives: 200 blocks drawn from one distribution produced 7 of them.
    adj = _bh([f["p"] for f in feats])
    for f, pa in zip(feats, adj):
        f["p_adj"] = float(pa)
        f["equity_tests"] = len(feats)
        f["equity"] = bool(abs(f["gender_gap"]) >= 4 and pa < 0.05
                           and min(f["nf"], f["nm"]) >= min_n)

    out = []
    for f in feats:
        base = BASE_ACTIONS.get((f["severity"], f["trajectory"]))
        if base is None:
            continue
        priority, template = base

        yoy = (f["below_pct"] - f["prev_pct"]) if pd.notna(f["prev_pct"]) else 0.0
        sentence = template.format(
            comp=f["competency"], blk=f["block"], pct=f"{f['below_pct']:.0f}",
            yoy=f"{abs(yoy):.0f}", gain=f"{abs(yoy):.0f}")

        # modifiers are tried in priority order and capped, so the output stays
        # readable and different blocks get genuinely different advice
        fired = [f"severity={f['severity']}", f"trajectory={f['trajectory']}"]
        clauses, skipped = [], []
        for name, fn in MODIFIERS:
            c = fn(f)
            if not c:
                continue
            if len(clauses) < MAX_CLAUSES:
                clauses.append(c)
                fired.append(name)
            else:
                skipped.append(name)

        text = sentence + "." + ("  " + ".  ".join(clauses) + "." if clauses else "")

        out.append({
            "priority": priority,
            "block": f["block"], "competency": f["competency"],
            "children": f["children"], "students": f["n"],
            "severity": f["severity"], "trajectory": f["trajectory"],
            "modifiers": len(clauses),
            "also_applies": ", ".join(skipped) if skipped else "",
            "rule_fired": " · ".join(fired),
            "peer_model": f["peer"]["block"] if f["peer"] else None,
            "uses_context": any(c.startswith("ctx_") for c in fired),
            "recommendation": text,
        })

    out.sort(key=lambda x: (PRIORITY_ORDER.get(x["priority"], 9),
                            -x["modifiers"], -x["children"]))
    return out[:limit]


# ---------------------------------------------------------------- for the demo
def combination_space():
    """
    How many distinct recommendations this rule set can produce.

    Computed, never hardcoded: v2 stated 432 in its docstring, reported 576
    from coverage_stats() and could actually reach 378 — three numbers for one
    quantity, none of which matched the code.
    """
    subsets = sum(math.comb(len(MODIFIERS), i) for i in range(MAX_CLAUSES + 1))
    return {"base_actions": len(BASE_ACTIONS),
            "modifier_clauses": len(MODIFIERS),
            "max_clauses_shown": MAX_CLAUSES,
            "clause_subsets": subsets,
            "distinct_outputs": len(BASE_ACTIONS) * subsets}


def coverage_stats(agg, district, year=None, min_n=MIN_N, context=None):
    """How much of the rule space this district actually exercises."""
    recs = recommend(agg, district, year, limit=10_000, min_n=min_n,
                     context=context)
    space = combination_space()
    if not recs:
        return dict(space, recommendations_generated=0,
                    unique_rule_combinations=0, with_peer_model=0,
                    using_district_context=0)
    combos = {r["rule_fired"] for r in recs}
    return dict(space,
                recommendations_generated=len(recs),
                unique_rule_combinations=len(combos),
                with_peer_model=sum(1 for r in recs if r["peer_model"]),
                using_district_context=sum(1 for r in recs
                                           if r["uses_context"]))


def _match(sev, traj):
    """Kept for the decision-grid chart."""
    b = BASE_ACTIONS.get((sev, traj))
    return {"priority": b[0], "action": b[1]} if b else None
