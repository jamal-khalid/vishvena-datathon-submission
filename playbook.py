"""
Layer 8 — Intervention Playbook v2 (no LLM).

DESIGN: composition, not enumeration.

Writing 200 hand-written recommendations is unmaintainable. Instead we compose:

    BASE ACTION      (severity x trajectory)          -> 9 possibilities
  + MODIFIER CLAUSES (each fires independently)       -> 6 optional clauses
  + PEER MODEL       (positive-deviance matching)     -> "replicate Block Y"

    9 base x 2 equity x 2 chronic x 3 scale x 2 grade x 2 peer  =  432 distinct outputs

Every clause is authored by the team and every one is auditable: the output
records exactly which rules fired.
"""
import numpy as np
import pandas as pd
from stats_tests import two_proportion_z, proportion_test
from units import children_below, eff_n


# ================================================================ classifiers

def severity(below_pct):
    if below_pct >= 60: return "Critical"
    if below_pct >= 45: return "At-risk"
    return "Strong"


def trajectory(below_pct, prev_pct):
    if pd.isna(prev_pct): return "Unknown"
    d = below_pct - prev_pct
    if d > 1:  return "Declining"
    if d < -1: return "Improving"
    return "Stagnant"


def scale_band(children):
    if children >= 1000: return "Large"
    if children >= 300:  return "Medium"
    return "Small"


# ============================================================== base actions
# All 9 cells filled (v1 had only 6).

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
}


# ============================================================ modifier clauses
# Each returns a sentence or None. They compose onto the base action.

def _mod_equity(f):
    if not f["equity"]:
        return None
    lag = "girls" if f["gender_gap"] > 0 else "boys"
    return (f"Pair this with gender-responsive pedagogy and structured peer-learning "
            f"groups, as {lag} trail by {abs(f['gender_gap']):.0f} points here "
            f"(statistically significant, p={f['p']:.3f})")


def _mod_chronic(f):
    if not f["chronic"]:
        return None
    return (f"Treat this as structural rather than cyclical — {f['competency']} has been "
            f"below 50% mastery in {f['years_bad']} consecutive years in this block")


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
            f"{f['grade_delta']:.0f} points weaker than the other grades in this block")


def _mod_peer(f):
    p = f["peer"]
    if p is None:
        return None
    return (f"Study and replicate the approach used in {p['block']}, which has a very "
            f"similar overall learning profile but achieves {p['their_pct']:.0f}% below "
            f"grade in {f['competency']} versus {p['our_pct']:.0f}% here "
            f"(a {p['edge']:.0f}-point advantage)")


def _mod_bundle(f):
    if not f["bundle_with"]:
        return None
    return (f"Deliver this jointly with {f['bundle_with']}, which fails in the same "
            f"blocks (correlation r={f['bundle_r']:.2f}) and likely shares a root cause")


MODIFIERS = [
    ("peer",    _mod_peer),
    ("chronic", _mod_chronic),
    ("equity",  _mod_equity),
    ("grade",   _mod_grade),
    ("bundle",  _mod_bundle),
    ("scale",   _mod_scale),
]


# ========================================================= positive deviance

def _peer_matrix(agg, year):
    d = agg[agg["year"] == year]
    return d.pivot_table(index=["district", "block"], columns="competency",
                         values="below_pct", aggfunc="mean")


def find_peer_model(piv, block, competency, min_edge=8.0):
    """
    Positive deviance: find a block with a SIMILAR profile on every other
    competency, but which performs notably better on THIS one.

    That similarity constraint is what makes the comparison fair — we are not
    telling a struggling rural block to copy a well-resourced urban one.
    """
    rows = [i for i in piv.index if i[1] == block]
    if not rows or competency not in piv.columns:
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
    if better.empty:
        return None

    # among those, the most similar overall profile
    dist = np.sqrt(((better[other_cols] - me[other_cols]) ** 2).sum(axis=1))
    best = dist.idxmin()
    return {"block": best[1], "district": best[0],
            "their_pct": float(better.loc[best, competency]),
            "our_pct": float(me[competency]),
            "edge": float(me[competency] - better.loc[best, competency]),
            "profile_distance": float(dist.loc[best])}


MAX_CLAUSES = 3          # keep recommendations readable and varied


def _bundle_partner(agg, competency, year, min_r=0.90):
    """Which other competency fails in the same blocks?"""
    piv = _peer_matrix(agg, year)
    if piv.shape[0] < 4 or competency not in piv.columns:
        return None, None
    corr = piv.corr()[competency].drop(competency)
    if corr.empty:
        return None, None
    partner = corr.idxmax()
    return (partner, float(corr[partner])) if corr[partner] >= min_r else (None, None)


# ==================================================================== engine

def _features(agg, district, competency, block, cur_rows, all_rows, piv, bundle):
    below = float((cur_rows["below_pct"] * cur_rows["n"]).sum() / cur_rows["n"].sum())
    prev_series = cur_rows["prev_pct"].dropna()
    prev = float(prev_series.mean()) if not prev_series.empty else np.nan
    n = int(cur_rows["n"].sum())
    children = children_below(cur_rows, below)

    # gender, tested — sized by children, not by assessment responses (units.py),
    # split into real arms, and via a test that stays valid on small cells
    f_b = float((cur_rows["f_below"] * cur_rows["n"]).sum() / n)
    m_b = float((cur_rows["m_below"] * cur_rows["n"]).sum() / n)
    kids = eff_n(cur_rows)
    share = (float(cur_rows["f_n"].sum()) / n) if "f_n" in cur_rows.columns and n else 0.5
    nf = max(int(round(kids * share)), 1)
    nm = max(kids - nf, 1)
    _, p, _method = proportion_test(f_b, m_b, nf, nm)
    equity = abs(f_b - m_b) >= 4 and p < 0.05

    # chronic: below 50 in every year we have
    yrs = all_rows.groupby("year").apply(
        lambda g: (g["below_pct"] * g["n"]).sum() / g["n"].sum(), include_groups=False)
    chronic = bool(len(yrs) >= 2 and (yrs >= 50).all())

    # worst grade, if clearly worse than the rest
    worst_grade, grade_delta = None, 0.0
    gr = cur_rows.groupby("grade")["below_pct"].mean()
    if len(gr) >= 2:
        top = gr.idxmax()
        delta = gr[top] - gr.drop(top).mean()
        if delta >= 5:
            worst_grade, grade_delta = int(top), float(delta)

    return {
        "block": block, "competency": competency, "district": district,
        "below_pct": below, "prev_pct": prev, "n": n, "children": children,
        "severity": severity(below), "trajectory": trajectory(below, prev),
        "scale": scale_band(children),
        "gender_gap": f_b - m_b, "p": p, "equity": equity,
        "chronic": chronic, "years_bad": int(len(yrs)),
        "worst_grade": worst_grade, "grade_delta": grade_delta,
        "peer": find_peer_model(piv, block, competency),
        "bundle_with": bundle[0], "bundle_r": bundle[1] or 0.0,
    }


PRIORITY_ORDER = {"P1 — Immediate": 0, "P2 — Sustain": 1, "P2 — Act this term": 1,
                  "P3 — Plan": 2, "P4 — Monitor": 3}


def recommend(agg, district, year=None, limit=10):
    d = agg[agg["district"] == district]
    if year is None:
        year = d["year"].max()
    cur = d[d["year"] == year]
    if cur.empty:
        return []

    piv = _peer_matrix(agg, year)
    bundles = {c: _bundle_partner(agg, c, year) for c in cur["competency"].unique()}

    # "large scale" is relative to this district — top quartile of children affected
    burden = (cur.assign(aff=cur["n"] * cur["below_pct"] / 100)
                 .groupby(["block", "competency"])["aff"].sum())
    scale_cut = float(burden.quantile(0.75)) if len(burden) > 3 else float("inf")

    out = []
    for (blk, comp), rows in cur.groupby(["block", "competency"]):
        all_rows = d[(d["block"] == blk) & (d["competency"] == comp)]
        f = _features(agg, district, comp, blk, rows, all_rows, piv, bundles[comp])
        f["scale_top"] = burden.get((blk, comp), 0) >= scale_cut

        base = BASE_ACTIONS.get((f["severity"], f["trajectory"]))
        if base is None:                      # Strong+Stagnant / Strong+Improving
            continue
        priority, template = base

        yoy = (f["below_pct"] - f["prev_pct"]) if pd.notna(f["prev_pct"]) else 0.0
        sentence = template.format(
            comp=comp, blk=blk, pct=f"{f['below_pct']:.0f}",
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
            "block": blk, "competency": comp,
            "children": f["children"],
            "severity": f["severity"], "trajectory": f["trajectory"],
            "modifiers": len(clauses),
            "also_applies": ", ".join(skipped) if skipped else "",
            "rule_fired": " · ".join(fired),
            "peer_model": f["peer"]["block"] if f["peer"] else None,
            "recommendation": text,
        })

    out.sort(key=lambda x: (PRIORITY_ORDER.get(x["priority"], 9),
                            -x["modifiers"], -x["children"]))
    return out[:limit]


# ---------------------------------------------------------------- for the demo
def coverage_stats(agg, district, year=None):
    """How much of the rule space this district actually exercises."""
    recs = recommend(agg, district, year, limit=10_000)
    if not recs:
        return {}
    combos = {r["rule_fired"] for r in recs}
    return {
        "recommendations_generated": len(recs),
        "unique_rule_combinations": len(combos),
        "base_actions_defined": len(BASE_ACTIONS),
        "modifier_clauses": len(MODIFIERS),
        "theoretical_combinations": len(BASE_ACTIONS) * (2 ** len(MODIFIERS)),
        "with_peer_model": sum(1 for r in recs if r["peer_model"]),
    }


def _match(sev, traj):
    """Kept for the decision-grid chart."""
    b = BASE_ACTIONS.get((sev, traj))
    return {"priority": b[0], "action": b[1]} if b else None
