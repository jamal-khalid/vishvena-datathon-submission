"""
Layer 4 replacement — DETERMINISTIC insight engine (no LLM).

Instead of asking a model "what is interesting?", we run one generator per
question the Datathon handbook actually asks, score every candidate finding
by the EVIDENCE behind it, and surface the strongest of each type.

Handbook questions -> generators:
  "Where are students struggling the most?"        -> worst_competency, worst_block
  "Which competencies need immediate attention?"   -> worst_competency
  "Which districts are improving faster?"          -> steepest_decline, bright_spot
  "Are there persistent gender gaps?"              -> gender_gap
  "Which regions require additional support?"      -> scale, outlier_block
  (plus) learning progression across grades        -> grade_progression

------------------------------------------------------------------------------
THREE RULES EVERY GENERATOR FOLLOWS
------------------------------------------------------------------------------
1. SCORE BY EVIDENCE, NOT BY MAGNITUDE.
   `score` is the lower 95% bound on the effect, in percentage points. A
   100-point swing across 3 children scores near zero because its confidence
   interval spans almost everything; a 12-point gap across 400 children keeps
   most of its 12. Ranking by the raw number surfaced the noisiest cell in the
   district every time — and because noise looks the same everywhere, it also
   made every district read identically.

   A side effect: every generator now scores in the SAME UNIT, so the old
   arbitrary weights (x5, x4, x2, hardcoded 40) are gone and cross-type
   ranking finally means something.

2. NEVER REPORT A GROUP SMALLER THAN `min_n`.
   Applied to the group actually being described, before any extreme is
   picked. Extremes and small samples are the same cells.

3. SAY ONLY WHAT WAS MEASURED.
   Weighted means, not unweighted. Distinct years counted, not rows. No
   longitudinal language on cross-sectional data.
"""
import math
import pandas as pd
from stats_tests import (two_proportion_z, proportion_test, cohens_h,
                         effect_label, sig_marker)
from units import children_below, eff_n, headcount


# Groups smaller than this are never described. The dashboard passes its own
# sidebar value; 30 is the usual floor for a stable percentage.
MIN_N = 30

Z95 = 1.959963985

# Generators that raise are recorded here rather than silently swallowed —
# `except Exception: continue` made a crash indistinguishable from "found
# nothing", so a bug could survive indefinitely.
ERRORS = []


# ----------------------------------------------------------------- evidence
def _se_pct(p, n):
    """
    Standard error of a percentage, Agresti-Coull adjusted.

    The plain formula gives SE = 0 at p = 0% or 100%, which would make
    "3 of 3 children failed" look infinitely precise. Adding two successes
    and two failures keeps small samples honest.
    """
    n = max(int(n or 0), 1)
    x = round(float(p) / 100.0 * n)
    pt = (x + 2.0) / (n + 4.0)
    return 100.0 * math.sqrt(max(pt * (1.0 - pt), 1e-12) / (n + 4.0))


def _evidence(effect, se):
    """Lower 95% bound on |effect|. Zero when the effect sits inside the noise."""
    return max(0.0, abs(float(effect)) - Z95 * float(se))


def _score_rate(p, n, ref=0.0):
    """Evidence that a rate differs from a reference value."""
    return _evidence(float(p) - float(ref), _se_pct(p, n))


def _score_diff(p1, n1, p2, n2):
    """Evidence that two rates differ from each other."""
    se = math.hypot(_se_pct(p1, n1), _se_pct(p2, n2))
    return _evidence(float(p1) - float(p2), se)


def _wmean(g, col="below_pct", w="n"):
    """Size-weighted mean. An unweighted one lets a 3-child block outvote 1,000."""
    if g.empty:
        return float("nan")
    wt = pd.to_numeric(g[w], errors="coerce").fillna(0.0)
    v = pd.to_numeric(g[col], errors="coerce")
    ok = v.notna() & (wt > 0)
    if not ok.any():
        return float("nan")
    return float((v[ok] * wt[ok]).sum() / wt[ok].sum())


def _g(grade):
    """Grade suffix, omitted when grades are pooled (grade == 0)."""
    try:
        return f" (Grade {int(grade)})" if int(grade) else ""
    except (TypeError, ValueError):
        return ""


def _district(agg, district, year=None):
    d = agg[agg["district"] == district]
    if year is None or year not in set(d["year"].dropna()):
        if d.empty:
            return d, year
        year = d["year"].max()
    return d[d["year"] == year], year


def _find(category, score, text, evidence, source, n=None, **extra):
    """One finding. `source` names the generator so any claim can be traced."""
    out = {"category": category, "score": round(float(score), 2), "text": text,
           "evidence": evidence, "source": source}
    if n is not None:
        out["n"] = int(n)
    out.update(extra)
    return out


# ---------------------------------------------------------------- generators

def g_worst_competency(agg, district, year=None, min_n=MIN_N):
    """Which competency is weakest across the district?"""
    d, yr = _district(agg, district, year)
    if d.empty:
        return []
    # weighted: a competency is weak because most CHILDREN are below, not
    # because one small block is
    t = (d.groupby("competency")
           .apply(lambda g: pd.Series({"below": _wmean(g), "n": g["n"].sum()}),
                  include_groups=False)
           .dropna())
    t = t[t["n"] >= min_n]
    if len(t) < 2:
        return []
    worst = t["below"].idxmax()
    row = t.loc[worst]
    rest = t.drop(index=worst)
    ref = float((rest["below"] * rest["n"]).sum() / rest["n"].sum())
    ref_n = float(rest["n"].sum())
    score = _score_diff(row["below"], row["n"], ref, ref_n)
    if score <= 0:
        return []
    affected = int(round(row["n"] * row["below"] / 100.0))
    return [_find(
        "Weakest competency", score,
        f"**{worst}** is the district's weakest competency — "
        f"{row['below']:.0f}% of {int(row['n']):,} students are below grade "
        f"level, affecting about {affected:,} children. The other "
        f"{len(rest)} competencies average {ref:.0f}%.",
        f"size-weighted below% by competency vs the other competencies, {yr}",
        "g_worst_competency", n=int(row["n"])),
    ]


def g_worst_block(agg, district, year=None, min_n=MIN_N):
    """Which single block-competency pairing is in the worst shape?"""
    d, yr = _district(agg, district, year)
    if d.empty:
        return []
    t = (d.groupby(["block", "competency"])
           .apply(lambda g: pd.Series({"below": _wmean(g), "n": g["n"].sum()}),
                  include_groups=False)
           .reset_index().dropna())
    t = t[t["n"] >= min_n]
    if t.empty:
        return []
    r = t.loc[t["below"].idxmax()]
    peers = t[t["competency"] == r["competency"]].drop(index=r.name)
    if peers.empty:
        return []
    ref = float((peers["below"] * peers["n"]).sum() / peers["n"].sum())
    score = _score_diff(r["below"], r["n"], ref, peers["n"].sum())
    if score <= 0:
        return []
    return [_find(
        "Worst block", score,
        f"**{r['block']}** is worst in **{r['competency']}** — "
        f"{r['below']:.0f}% below grade level across {int(r['n']):,} students, "
        f"against {ref:.0f}% in the district's other blocks.",
        f"size-weighted block x competency maximum, {yr}",
        "g_worst_block", n=int(r["n"])),
    ]


def g_outlier_block(agg, district, year=None, min_n=MIN_N):
    """
    Which block deviates most from its district's average?

    The z-score leaves the candidate OUT of the mean and SD it is measured
    against. Including it capped the achievable z at (k-1)/sqrt(k) — with
    three blocks that is 1.155, so the old `z >= 1.5` filter could never fire
    however extreme the outlier was.
    """
    d, yr = _district(agg, district, year)
    if d.empty:
        return []
    out = []
    for comp, sub in d.groupby("competency"):
        b = (sub.groupby("block")
                .apply(lambda g: pd.Series({"below": _wmean(g),
                                            "n": g["n"].sum()}),
                       include_groups=False)
                .dropna())
        b = b[b["n"] >= min_n]
        if len(b) < 4:                      # 3 peers minimum to define "normal"
            continue
        blk = b["below"].idxmax()
        peers = b.drop(index=blk)
        mu, sd = peers["below"].mean(), peers["below"].std()
        if not sd or pd.isna(sd) or sd == 0:
            continue
        z = (b.loc[blk, "below"] - mu) / sd
        if z < 2.0:
            continue
        score = _score_diff(b.loc[blk, "below"], b.loc[blk, "n"],
                            mu, peers["n"].sum())
        if score <= 0:
            continue
        out.append(_find(
            "Negative outlier", score,
            f"**{blk}** is a negative outlier in **{comp}** — "
            f"{b.loc[blk, 'below']:.0f}% below grade versus {mu:.0f}% across "
            f"the district's other {len(peers)} blocks, which is {z:.1f} "
            f"standard deviations worse than any of them.",
            f"leave-one-out z-score vs the other blocks, {yr}",
            "g_outlier_block", n=int(b.loc[blk, "n"])))
    return sorted(out, key=lambda x: -x["score"])[:1]


def g_gender_gap(agg, district, year=None, min_n=MIN_N):
    """
    Are there statistically significant gender gaps?

    Every cell in the district is a separate test, so testing at p < 0.05 and
    reporting the most extreme survivor is guaranteed to find something: with
    300 cells and no real gap anywhere, ~15 cross the line by luck and the
    biggest of those is the most likely to be pure noise. Benjamini-Hochberg
    controls the false-discovery rate across the whole family.
    """
    d, yr = _district(agg, district, year)
    if d.empty:
        return []
    cand = []
    for r in d.itertuples():
        if pd.isna(r.gender_gap) or abs(r.gender_gap) < 4:
            continue
        nf = int(getattr(r, "f_n", 0) or 0)
        nm = int(getattr(r, "m_n", 0) or 0)
        if not nf or not nm:
            nf = max(int(r.n) // 2, 1)
            nm = max(int(r.n) - nf, 1)
        if min(nf, nm) < min_n:
            continue
        z, p, method = proportion_test(r.f_below, r.m_below, nf, nm)
        cand.append((p, z, method, r, nf, nm))
    if not cand:
        return []

    # Benjamini-Hochberg across every cell tested in this district
    m = len(cand)
    order = sorted(range(m), key=lambda i: cand[i][0])
    adj, prev = [0.0] * m, 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = m - rank + 1
        prev = min(prev, cand[i][0] * m / k)
        adj[i] = prev
    keep = [(adj[i], *cand[i]) for i in range(m) if adj[i] < 0.05]
    if not keep:
        return []

    out = []
    for p_adj, p, z, method, r, nf, nm in keep:
        score = _score_diff(r.f_below, nf, r.m_below, nm)
        if score <= 0:
            continue
        h = cohens_h(r.f_below, r.m_below)
        lag = "girls" if r.gender_gap > 0 else "boys"
        how = (f"two-proportion z-test, z={z:.2f}" if method == "z"
               else "Fisher's exact test (cells too small for a z-test)")
        out.append(_find(
            "Gender gap", score,
            f"In **{r.block}**{_g(r.grade)} — {r.competency}: "
            f"**{lag} trail by {abs(r.gender_gap):.0f} points** "
            f"({r.f_below:.0f}% of {nf} girls vs {r.m_below:.0f}% of {nm} "
            f"boys below grade). Significant after correcting for the "
            f"{m} comparisons made in this district "
            f"({sig_marker(p_adj)} adjusted), {effect_label(h)} effect.",
            f"{how}, Benjamini-Hochberg over {m} cells, {yr}",
            "g_gender_gap", n=nf + nm))
    return sorted(out, key=lambda x: -x["score"])[:1]


def g_steepest_decline(agg, district, year=None, min_n=MIN_N):
    """What is getting worse fastest?"""
    d, yr = _district(agg, district, year)
    d = d.dropna(subset=["prev_pct"])
    d = d[d["n"] >= min_n]
    if d.empty:
        return []
    d = d.assign(yoy=d["below_pct"] - d["prev_pct"])
    r = d.loc[d["yoy"].idxmax()]
    if r["yoy"] <= 0:
        return []
    score = _score_diff(r["below_pct"], r["n"], r["prev_pct"], r["n"])
    if score <= 0:
        return []
    return [_find(
        "Steepest decline", score,
        f"**{r['competency']} in {r['block']}{_g(r['grade'])}** "
        f"deteriorated fastest — from {r['prev_pct']:.0f}% to "
        f"{r['below_pct']:.0f}% below grade level "
        f"({r['yoy']:+.0f} points in one year, {int(r['n']):,} students).",
        f"year-over-year change into {yr}",
        "g_steepest_decline", n=int(r["n"])),
    ]


def g_bright_spot(agg, district, year=None, min_n=MIN_N):
    """What improved most? (judges reward finding what works, not just failures)"""
    d, yr = _district(agg, district, year)
    d = d.dropna(subset=["prev_pct"])
    d = d[d["n"] >= min_n]
    if d.empty:
        return []
    d = d.assign(gain=d["prev_pct"] - d["below_pct"])
    r = d.loc[d["gain"].idxmax()]
    if r["gain"] <= 0:
        return []
    score = _score_diff(r["prev_pct"], r["n"], r["below_pct"], r["n"])
    if score <= 0:
        return []
    return [_find(
        "Bright spot", score,
        f"**{r['block']} improved {r['competency']}{_g(r['grade'])} "
        f"by {r['gain']:.0f} points** — from {r['prev_pct']:.0f}% to "
        f"{r['below_pct']:.0f}% below grade across {int(r['n']):,} students. "
        f"Worth studying and replicating.",
        f"largest year-over-year gain into {yr}",
        "g_bright_spot", n=int(r["n"])),
    ]


def g_scale(agg, district, year=None, min_n=MIN_N):
    """Where are the most CHILDREN affected? (severity != scale)"""
    d, yr = _district(agg, district, year)
    if d.empty:
        return []
    # Summing n across competencies would count each child once per question,
    # so the burden is (children in the block) x (their mean below%).
    t = (d.groupby("block")
           .apply(lambda g: pd.Series({
               "below": _wmean(g),
               "kids": headcount(g),
               "aff": children_below(g, _wmean(g))}),
               include_groups=False)
           .dropna())
    t = t[t["kids"] >= min_n]
    if len(t) < 2:
        return []
    blk = t["aff"].idxmax()
    rest = t.drop(index=blk)
    ref = float((rest["below"] * rest["kids"]).sum() / rest["kids"].sum())
    # scored in the same unit as everything else: how sure are we that this
    # block's rate is worse than the rest, given how many children it covers
    score = _score_diff(t.loc[blk, "below"], t.loc[blk, "kids"],
                        ref, rest["kids"].sum())
    share = 100.0 * t.loc[blk, "aff"] / max(t["aff"].sum(), 1)
    return [_find(
        "Largest scale", score,
        f"**{blk}** carries the largest absolute burden — about "
        f"**{int(t.loc[blk, 'aff']):,} children** below grade level across all "
        f"competencies ({share:.0f}% of the district's total, mean "
        f"{t.loc[blk, 'below']:.0f}% against {ref:.0f}% elsewhere). "
        f"Highest percentage is not the same as highest need.",
        f"children in block x size-weighted mean below%, {yr}",
        "g_scale", n=int(t.loc[blk, "kids"])),
    ]


def g_grade_progression(agg, district, year=None, min_n=MIN_N):
    """
    Which grade is struggling most, and does the pattern rise with grade level?

    NOTE ON WORDING: these are different children measured in the same year,
    not one cohort followed over time. The old text said gaps were
    "compounding", which is a longitudinal claim this data cannot support.
    """
    d, yr = _district(agg, district, year)
    if d.empty:
        return []
    t = (d.groupby("grade")
           .apply(lambda g: pd.Series({"below": _wmean(g),
                                       "kids": headcount(g)}),
                  include_groups=False)
           .dropna())
    t = t[t["kids"] >= min_n]
    if len(t) < 2:
        return []
    worst, best = t["below"].idxmax(), t["below"].idxmin()
    score = _score_diff(t.loc[worst, "below"], t.loc[worst, "kids"],
                        t.loc[best, "below"], t.loc[best, "kids"])
    if score <= 0:
        return []
    lo, hi = t.index.min(), t.index.max()
    monotone = (t["below"].is_monotonic_increasing and worst == hi)
    trend = (" The pattern rises steadily with grade level."
             if monotone else
             f" This is not a steady rise — Grade {int(lo)} sits at "
             f"{t.loc[lo, 'below']:.0f}% and Grade {int(hi)} at "
             f"{t.loc[hi, 'below']:.0f}%.")
    return [_find(
        "Learning progression", score,
        f"**Grade {int(worst)} is the weakest year group** at "
        f"{t.loc[worst, 'below']:.0f}% below grade level, against "
        f"{t.loc[best, 'below']:.0f}% in Grade {int(best)} "
        f"({t.loc[worst, 'below'] - t.loc[best, 'below']:+.0f} points)."
        + trend
        + " These are different children measured in the same year, not one "
          "cohort followed over time.",
        f"size-weighted below% by grade, {yr}",
        "g_grade_progression", n=int(t["kids"].sum())),
    ]


def g_persistence(agg, district, year=None, min_n=MIN_N):
    """
    Chronically weak in EVERY year — a different policy problem from a one-year
    dip. A block failing three years running has a structural issue, not bad
    luck.

    The year check counts DISTINCT YEARS. Counting rows meant a block present
    in 2 of 3 years but split across 2 grades scored 4 rows >= 3 years and the
    sentence claimed "every one of the 3 years" — which was simply false.
    """
    d = agg[agg["district"] == district]
    if d.empty:
        return []
    years = d["year"].nunique()
    if years < 2:
        return []
    g = (d.groupby(["block", "competency"])
           .apply(lambda s: pd.Series({
               "best_year": s["below_pct"].min(),   # min below% = its BEST year
               "avg": _wmean(s),
               "years_seen": s["year"].nunique(),
               "n": s["n"].sum() / max(s["year"].nunique(), 1)}),
               include_groups=False))
    chronic = g[(g["best_year"] >= 50) & (g["years_seen"] >= years)
                & (g["n"] >= min_n)]
    if chronic.empty:
        return []
    blk, comp = chronic["avg"].idxmax()
    row = chronic.loc[(blk, comp)]
    score = _score_rate(row["avg"], row["n"], ref=50.0)
    if score <= 0:
        return []
    return [_find(
        "Chronic weakness", score,
        f"**{comp} in {blk}** has been below 50% mastery in **every one of "
        f"the {years} years** measured (average {row['avg']:.0f}% below grade, "
        f"about {int(row['n']):,} students a year). This is a structural gap, "
        f"not a one-year fluctuation.",
        f"min below_pct across all {years} distinct years >= 50",
        "g_persistence", n=int(row["n"])),
    ]


def g_within_district_spread(agg, district, year=None, min_n=MIN_N):
    """
    Inequity INSIDE the district — the handbook opens with exactly this:
    "One cluster may consistently outperform its neighbours."
    """
    d, yr = _district(agg, district, year)
    if d.empty:
        return []
    out = []
    for comp, sub in d.groupby("competency"):
        b = (sub.groupby("block")
                .apply(lambda g: pd.Series({"below": _wmean(g),
                                            "n": g["n"].sum()}),
                       include_groups=False)
                .dropna())
        b = b[b["n"] >= min_n]
        if len(b) < 2:
            continue
        top, bot = b["below"].idxmax(), b["below"].idxmin()
        spread = b.loc[top, "below"] - b.loc[bot, "below"]
        # the old version appended unconditionally, so it fired in every
        # district even when the spread rounded to zero
        score = _score_diff(b.loc[top, "below"], b.loc[top, "n"],
                            b.loc[bot, "below"], b.loc[bot, "n"])
        if score <= 0 or spread < 5:
            continue
        out.append(_find(
            "Within-district inequity", score,
            f"**{comp} varies by {spread:.0f} points across blocks** within "
            f"{district} — from {bot} at {b.loc[bot, 'below']:.0f}% to "
            f"{top} at {b.loc[top, 'below']:.0f}% below grade level. "
            f"A district average would hide this entirely.",
            f"max block - min block among blocks with >= {min_n} students, {yr}",
            "g_within_district_spread",
            n=int(b.loc[top, "n"] + b.loc[bot, "n"])))
    return sorted(out, key=lambda x: -x["score"])[:1]


def g_gender_by_competency(agg, district, year=None, min_n=MIN_N):
    """
    Is the gender gap concentrated in particular competencies?
    Handbook: "Girls may outperform boys in foundational competencies
    but lag in higher-order problem solving."
    """
    d, yr = _district(agg, district, year)
    if d.empty:
        return []
    t = (d.groupby("competency")
           .apply(lambda g: pd.Series({
               "gap": _wmean(g, "gender_gap"),
               "nf": pd.to_numeric(g.get("f_n"), errors="coerce").fillna(0).sum(),
               "nm": pd.to_numeric(g.get("m_n"), errors="coerce").fillna(0).sum()}),
               include_groups=False)
           .dropna())
    t = t[(t["nf"] >= min_n) & (t["nm"] >= min_n)]
    if len(t) < 2:
        return []
    hi, lo = t["gap"].idxmax(), t["gap"].idxmin()
    spread = t.loc[hi, "gap"] - t.loc[lo, "gap"]
    if spread < 4:
        return []
    # both ends are differences of proportions, so the spread's evidence is
    # the evidence that the two gaps differ from each other
    score = _score_diff(t.loc[hi, "gap"], min(t.loc[hi, "nf"], t.loc[hi, "nm"]),
                        t.loc[lo, "gap"], min(t.loc[lo, "nf"], t.loc[lo, "nm"]))
    if score <= 0:
        return []

    def _phrase(v):
        # gender_gap = girls' below% - boys' below%, so POSITIVE means girls
        # are further behind. "girls trail by -45" was arithmetically true and
        # unreadable.
        return (f"girls trail by {v:.1f}" if v > 0 else
                f"boys trail by {abs(v):.1f}" if v < 0 else "they are level")
    return [_find(
        "Gender x competency", score,
        f"The gender gap is **not uniform across competencies** — in "
        f"**{hi}** {_phrase(t.loc[hi, 'gap'])} points, while in **{lo}** "
        f"{_phrase(t.loc[lo, 'gap'])} points (a {spread:.0f}-point "
        f"difference). A single district-wide gender figure would average "
        f"these away.",
        f"size-weighted mean gender_gap by competency, {yr}",
        "g_gender_by_competency",
        n=int(min(t.loc[hi, "nf"], t.loc[hi, "nm"]))),
    ]


GENERATORS = [
    g_worst_competency,
    g_scale,
    g_gender_gap,
    g_steepest_decline,
    g_outlier_block,
    g_grade_progression,
    g_bright_spot,
    g_worst_block,
    g_persistence,
    g_within_district_spread,
    g_gender_by_competency,
]


# ---------------------------------------------------------------- self-documentation
# Every generator traces to a specific line in the Datathon handbook.
# This drives the "generator registry" panel in the demo.
REGISTRY = {
    "g_worst_competency": dict(
        name="Weakest competency", answers="Which competencies require immediate attention?",
        formula="size-weighted below% by competency vs the rest → idxmax",
        filters=f"≥{MIN_N} students, evidence > 0"),
    "g_worst_block": dict(
        name="Worst block", answers="Where are students struggling the most?",
        formula="size-weighted below% by block × competency → idxmax",
        filters=f"≥{MIN_N} students, evidence > 0"),
    "g_outlier_block": dict(
        name="Negative outlier", answers="Which regions require additional support?",
        formula="leave-one-out z = (block − mean of OTHER blocks) / σ",
        filters=f"z ≥ 2.0, ≥4 blocks of ≥{MIN_N} students"),
    "g_gender_gap": dict(
        name="Gender gap", answers="Are there persistent gender gaps?",
        formula="f_below − m_below, two-proportion z or Fisher's exact",
        filters=f"|gap| ≥ 4 pts, ≥{MIN_N} per arm, BH-adjusted p < 0.05"),
    "g_steepest_decline": dict(
        name="Steepest decline", answers="Which districts are improving faster than others?",
        formula="below_pct − prev_pct → idxmax",
        filters=f"≥{MIN_N} students, evidence > 0"),
    "g_bright_spot": dict(
        name="Bright spot", answers="Which districts are improving faster than others?",
        formula="prev_pct − below_pct → idxmax",
        filters=f"≥{MIN_N} students, evidence > 0"),
    "g_scale": dict(
        name="Largest scale", answers="Which regions require additional support?",
        formula="children × below% by block → idxmax",
        filters=f"≥{MIN_N} children per block"),
    "g_grade_progression": dict(
        name="Learning progression", answers="Problem statement: Learning progression",
        formula="size-weighted below% by grade → worst vs best grade",
        filters=f"≥{MIN_N} children per grade, evidence > 0"),
    "g_persistence": dict(
        name="Chronic weakness", answers="Depth: structural vs one-year problems",
        formula="min(below_pct) across ALL years ≥ 50",
        filters="present in every DISTINCT year, ≥{} students/yr".format(MIN_N)),
    "g_within_district_spread": dict(
        name="Within-district inequity", answers="Problem statement: Geographic inequity",
        formula="max(block) − min(block) per competency",
        filters=f"spread ≥ 5 pts, ≥{MIN_N} students, evidence > 0"),
    "g_gender_by_competency": dict(
        name="Gender × competency", answers="Handbook: girls lag in higher-order skills",
        formula="spread of size-weighted gender_gap across competencies",
        filters=f"spread ≥ 4 pts, ≥{MIN_N} per arm, evidence > 0"),
}


# Themes group generators so we never rank a gender gap against a failure rate.
THEMES = {
    "📉 Performance & Learning Gaps": ["g_worst_competency", "g_worst_block",
                                       "g_persistence"],
    "⚖️ Equity":                      ["g_gender_gap", "g_gender_by_competency"],
    "📈 Trends Over Time":            ["g_steepest_decline", "g_bright_spot"],
    "🗺️ Geographic Variation":        ["g_outlier_block", "g_within_district_spread",
                                       "g_scale"],
    "🎓 Learning Progression":         ["g_grade_progression"],
}
_FN = {f.__name__: f for f in GENERATORS}
_THEME_OF = {fn: th for th, names in THEMES.items() for fn in names}
# cross-dataset generators live in their own theme so they neither crowd out
# the primary findings nor get crowded out by them
_THEME_OF.update({
    "x_over_under": "🔗 District Context",
    "x_rank_shift": "🔗 District Context",
    "x_peer_comparison": "🔗 District Context",
    "x_context_explains": "🔗 District Context",
    "x_strongest_link": "🔗 District Context",
    "x_no_link": "🔗 District Context",
    "x_redundant_context": "🔗 District Context",
    "x_unit_spread": "🔗 District Context",
    "x_context_contrast": "🔗 District Context",
    "x_too_few_units": "🔗 District Context",
})


def _run(fn, agg, district, year, min_n):
    """Call one generator, recording rather than swallowing any exception."""
    try:
        return fn(agg, district, year, min_n=min_n) or []
    except Exception as exc:                       # noqa: BLE001 - reported below
        ERRORS.append({"generator": fn.__name__, "district": district,
                       "error": f"{type(exc).__name__}: {exc}"})
        return None                                # None = crashed, [] = nothing


def generate_by_theme(agg, district, year=None, per_theme=2, min_n=MIN_N):
    """
    Findings grouped by theme instead of one flat ranked list.

    Scores share a unit across every generator now, but grouping still keeps a
    gender finding from crowding out a geography one.
    """
    out = {}
    for theme, fn_names in THEMES.items():
        items = []
        for name in fn_names:
            fn = _FN.get(name)
            if fn is None:
                continue
            items.extend(_run(fn, agg, district, year, min_n) or [])
        items.sort(key=lambda x: -x["score"])
        out[theme] = items[:per_theme]
    return out


def describe(agg=None, district=None, year=None, min_n=MIN_N):
    """Registry rows, optionally annotated with what each generator found THIS run."""
    rows = []
    for fn in GENERATORS:
        meta = REGISTRY.get(fn.__name__, {})
        row = {"Generator": meta.get("name", fn.__name__),
               "Answers": meta.get("answers", "—"),
               "Formula": meta.get("formula", "—"),
               "Filter": meta.get("filters", "—")}
        if agg is not None:
            hits = _run(fn, agg, district, year, min_n)
            if hits is None:                       # crashed — never show as "nothing"
                row["Fired?"] = "⚠️ error"
                row["Score"] = None
            else:
                row["Fired?"] = "✅ yes" if hits else "— nothing found"
                row["Score"] = round(hits[0]["score"], 1) if hits else None
        rows.append(row)
    return rows


def generate(agg, district, year=None, limit=7, min_n=MIN_N, per_theme=2,
             context=None):
    """
    The strongest well-evidenced finding of each type, spread across themes.

    Ranked by evidence (lower 95% bound of the effect in percentage points),
    then capped per theme so one busy theme cannot fill the whole list and
    make every district read the same.

    `context` is the dict returned by insights_cross.prepare(). When supplied,
    cross-dataset findings — how the district performs against districts in
    similar circumstances — join the same ranked list. They score in the same
    unit (points of the outcome), so the comparison is meaningful. Omit it and
    behaviour is unchanged.
    """
    found = []
    for gen in GENERATORS:
        found.extend(_run(gen, agg, district, year, min_n) or [])
    if context:
        try:
            import insights_cross as _cross
            found.extend(_cross.generate(context, district, limit=limit))
        except Exception as exc:                   # noqa: BLE001 — recorded
            ERRORS.append({"generator": "insights_cross.generate",
                           "district": district,
                           "error": f"{type(exc).__name__}: {exc}"})
    found.sort(key=lambda x: -x["score"])

    picked, per = [], {}
    for f in found:
        th = _THEME_OF.get(f.get("source"), "other")
        if per.get(th, 0) >= per_theme:
            continue
        per[th] = per.get(th, 0) + 1
        picked.append(f)
        if len(picked) >= limit:
            break
    return picked
