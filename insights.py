"""
Layer 4 replacement — DETERMINISTIC insight engine (no LLM).

Instead of asking a model "what is interesting?", we run one generator per
question the Datathon handbook actually asks, score every candidate finding
by effect size, and surface the strongest of each type.

Handbook questions -> generators:
  "Where are students struggling the most?"        -> worst_competency, worst_block
  "Which competencies need immediate attention?"   -> worst_competency
  "Which districts are improving faster?"          -> steepest_decline, bright_spot
  "Are there persistent gender gaps?"              -> gender_gap
  "Which regions require additional support?"      -> scale, outlier_block
  (plus) learning progression across grades        -> grade_progression
"""
import pandas as pd
from stats_tests import (two_proportion_z, proportion_test, cohens_h,
                         effect_label, sig_marker)
from units import children_below, eff_n, headcount



def _g(grade):
    """Grade suffix, omitted when grades are pooled (grade == 0)."""
    try:
        return f" (Grade {int(grade)})" if int(grade) else ""
    except (TypeError, ValueError):
        return ""


def _district(agg, district, year=None):
    d = agg[agg["district"] == district]
    if year is None:
        year = d["year"].max()
    return d[d["year"] == year], year


# ---------------------------------------------------------------- generators

def g_worst_competency(agg, district, year=None):
    """Which competency is weakest across the district?"""
    d, yr = _district(agg, district, year)
    if d.empty: return []
    t = d.groupby("competency").agg(below=("below_pct", "mean"), n=("n", "sum"))
    worst = t["below"].idxmax()
    row = t.loc[worst]
    affected = int(round(row["n"] * row["below"] / 100))
    return [{
        "category": "Weakest competency",
        "score": float(row["below"]),
        "text": (f"**{worst}** is the district's weakest competency — "
                 f"{row['below']:.0f}% of {int(row['n']):,} students are below grade "
                 f"level, affecting about {affected:,} children."),
        "evidence": f"mean below% across all blocks and grades, {yr}",
    }]


def g_worst_block(agg, district, year=None):
    """Which single block-competency pairing is in the worst shape?"""
    d, yr = _district(agg, district, year)
    if d.empty: return []
    t = (d.groupby(["block", "competency"])
           .agg(below=("below_pct", "mean"), n=("n", "sum")).reset_index())
    r = t.loc[t["below"].idxmax()]
    return [{
        "category": "Worst block",
        "score": float(r["below"]),
        "text": (f"**{r['block']}** is worst in **{r['competency']}** — "
                 f"{r['below']:.0f}% below grade level across {int(r['n']):,} students."),
        "evidence": f"block x competency maximum, {yr}",
    }]


def g_outlier_block(agg, district, year=None):
    """Which block deviates most from its own district's average?"""
    d, yr = _district(agg, district, year)
    if d.empty: return []
    out = []
    for comp, sub in d.groupby("competency"):
        b = sub.groupby("block")["below_pct"].mean()
        if len(b) < 3: continue
        mu, sd = b.mean(), b.std()
        if not sd or pd.isna(sd) or sd == 0: continue
        z = (b - mu) / sd
        blk = z.idxmax()
        if z[blk] < 1.5: continue
        out.append({
            "category": "Negative outlier",
            "score": float(z[blk]) * 10,
            "text": (f"**{blk}** is a negative outlier in **{comp}** — "
                     f"{b[blk]:.0f}% below grade versus a district average of "
                     f"{mu:.0f}%, which is {z[blk]:.1f} standard deviations worse."),
            "evidence": f"z-score vs district mean, {yr}",
        })
    return sorted(out, key=lambda x: -x["score"])[:1]


def g_gender_gap(agg, district, year=None):
    """Are there statistically significant gender gaps?"""
    d, yr = _district(agg, district, year)
    if d.empty: return []
    out = []
    for r in d.itertuples():
        if pd.isna(r.gender_gap) or abs(r.gender_gap) < 4:
            continue
        # Real arm sizes, and a test that stays valid when those arms are tiny.
        nf = int(getattr(r, "f_n", 0)) or max(int(r.n) // 2, 1)
        nm = int(getattr(r, "m_n", 0)) or max(int(r.n) - nf, 1)
        z, p, method = proportion_test(r.f_below, r.m_below, nf, nm)
        if p >= 0.05:                              # suppress noise
            continue
        h = cohens_h(r.f_below, r.m_below)
        lag = "girls" if r.gender_gap > 0 else "boys"
        how = (f"two-proportion z-test, z={z:.2f}" if method == "z"
               else "Fisher's exact test (cells too small for a z-test)")
        out.append({
            "category": "Gender gap",
            "score": abs(float(r.gender_gap)),
            "text": (f"In **{r.block}**{_g(r.grade)} — {r.competency}: "
                     f"**{lag} trail by {abs(r.gender_gap):.0f} points** "
                     f"({r.f_below:.0f}% of {nf} girls vs {r.m_below:.0f}% of {nm} "
                     f"boys below grade). Statistically significant, {sig_marker(p)}, "
                     f"{effect_label(h)} effect."),
            "evidence": f"{how}, {yr}",
        })
    return sorted(out, key=lambda x: -x["score"])[:1]


def g_steepest_decline(agg, district, year=None):
    """What is getting worse fastest?"""
    d, yr = _district(agg, district, year)
    d = d.dropna(subset=["prev_pct"])
    if d.empty: return []
    d = d.assign(yoy=d["below_pct"] - d["prev_pct"])
    r = d.loc[d["yoy"].idxmax()]
    if r["yoy"] < 1: return []
    return [{
        "category": "Steepest decline",
        "score": float(r["yoy"]) * 5,
        "text": (f"**{r['competency']} in {r['block']}{_g(r['grade'])}** "
                 f"deteriorated fastest — from {r['prev_pct']:.0f}% to "
                 f"{r['below_pct']:.0f}% below grade level "
                 f"({r['yoy']:+.0f} points in one year)."),
        "evidence": f"year-over-year change into {yr}",
    }]


def g_bright_spot(agg, district, year=None):
    """What improved most? (judges reward finding what works, not just failures)"""
    d, yr = _district(agg, district, year)
    d = d.dropna(subset=["prev_pct"])
    if d.empty: return []
    d = d.assign(gain=d["prev_pct"] - d["below_pct"])
    r = d.loc[d["gain"].idxmax()]
    if r["gain"] < 1: return []
    return [{
        "category": "Bright spot",
        "score": float(r["gain"]) * 5,
        "text": (f"**{r['block']} improved {r['competency']}{_g(r['grade'])} "
                 f"by {r['gain']:.0f} points** — from {r['prev_pct']:.0f}% to "
                 f"{r['below_pct']:.0f}% below grade. Worth studying and replicating."),
        "evidence": f"largest year-over-year gain into {yr}",
    }]


def g_scale(agg, district, year=None):
    """Where are the most CHILDREN affected? (severity != scale)"""
    d, yr = _district(agg, district, year)
    if d.empty: return []
    # Summing n across competencies would count each child once per question,
    # so the burden is (children in the block) x (their mean below%).
    t = (d.groupby("block")
           .apply(lambda g: pd.Series({
               "below": float((g["below_pct"] * g["n"]).sum() / g["n"].sum()),
               "aff": children_below(
                   g, (g["below_pct"] * g["n"]).sum() / g["n"].sum())}),
               include_groups=False))
    blk = t["aff"].idxmax()
    return [{
        "category": "Largest scale",
        "score": 40.0,
        "text": (f"**{blk}** carries the largest absolute burden — about "
                 f"**{int(t.loc[blk,'aff']):,} children** below grade level across all "
                 f"competencies (mean {t.loc[blk,'below']:.0f}%). "
                 f"Highest percentage is not the same as highest need."),
        "evidence": f"children in block x weighted mean below%, {yr}",
    }]


def g_grade_progression(agg, district, year=None):
    """Do gaps widen as children move up grades?"""
    d, yr = _district(agg, district, year)
    if d.empty: return []
    t = d.groupby("grade")["below_pct"].mean()
    if len(t) < 2: return []
    lo, hi = t.index.min(), t.index.max()
    delta = t[hi] - t[lo]
    if abs(delta) < 2: return []
    word = "widen" if delta > 0 else "narrow"
    return [{
        "category": "Learning progression",
        "score": abs(float(delta)) * 4,
        "text": (f"Learning gaps **{word} with grade level** — Grade {lo} averages "
                 f"{t[lo]:.0f}% below grade while Grade {hi} averages {t[hi]:.0f}% "
                 f"({delta:+.0f} points). Gaps are compounding rather than closing."
                 if delta > 0 else
                 f"Learning gaps **{word} with grade level** — Grade {lo} averages "
                 f"{t[lo]:.0f}% below grade while Grade {hi} averages {t[hi]:.0f}% "
                 f"({delta:+.0f} points)."),
        "evidence": f"mean below% by grade, {yr}",
    }]


def g_persistence(agg, district, year=None):
    """
    Chronically weak in EVERY year — a different policy problem from a one-year dip.
    A block failing three years running has a structural issue, not bad luck.
    """
    d = agg[agg["district"] == district]
    if d.empty: return []
    years = d["year"].nunique()
    if years < 2: return []
    g = (d.groupby(["block", "competency"])["below_pct"]
           .agg(worst_year="min", avg="mean", seen="count"))
    chronic = g[(g["worst_year"] >= 50) & (g["seen"] >= years)]
    if chronic.empty: return []
    blk, comp = chronic["avg"].idxmax()
    row = chronic.loc[(blk, comp)]
    return [{
        "category": "Chronic weakness",
        "score": float(row["avg"]),
        "text": (f"**{comp} in {blk}** has been below 50% mastery in **every one of "
                 f"the {years} years** measured (average {row['avg']:.0f}% below grade). "
                 f"This is a structural gap, not a one-year fluctuation."),
        "evidence": f"min below_pct across all {years} years >= 50",
    }]


def g_within_district_spread(agg, district, year=None):
    """
    Inequity INSIDE the district — the handbook opens with exactly this:
    "One cluster may consistently outperform its neighbours."
    """
    d, yr = _district(agg, district, year)
    if d.empty: return []
    out = []
    for comp, sub in d.groupby("competency"):
        b = sub.groupby("block")["below_pct"].mean()
        if len(b) < 2: continue
        spread = b.max() - b.min()
        out.append({
            "category": "Within-district inequity",
            "score": float(spread),
            "text": (f"**{comp} varies by {spread:.0f} points across blocks** within "
                     f"{district} — from {b.idxmin()} at {b.min():.0f}% to "
                     f"{b.idxmax()} at {b.max():.0f}% below grade level. "
                     f"A district average would hide this entirely."),
            "evidence": f"max block − min block, {yr}",
        })
    return sorted(out, key=lambda x: -x["score"])[:1]


def g_gender_by_competency(agg, district, year=None):
    """
    Is the gender gap concentrated in particular competencies?
    Handbook: "Girls may outperform boys in foundational competencies
    but lag in higher-order problem solving."
    """
    d, yr = _district(agg, district, year)
    if d.empty: return []
    t = d.groupby("competency")["gender_gap"].mean()
    if len(t) < 2: return []
    hi, lo = t.idxmax(), t.idxmin()
    if t[hi] - t[lo] < 4: return []
    return [{
        "category": "Gender × competency",
        "score": float(t[hi] - t[lo]) * 2,
        "text": (f"The gender gap is **not uniform across competencies** — girls trail "
                 f"by {t[hi]:+.1f} points in **{hi}** but {t[lo]:+.1f} in **{lo}** "
                 f"(a {t[hi]-t[lo]:.0f}-point difference). A single district-wide "
                 f"gender figure would average these away."),
        "evidence": f"spread of mean gender_gap across competencies, {yr}",
    }]


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
        formula="groupby(competency).below_pct.mean() → idxmax", filters="—"),
    "g_worst_block": dict(
        name="Worst block", answers="Where are students struggling the most?",
        formula="groupby(block, competency).below_pct.mean() → idxmax", filters="—"),
    "g_outlier_block": dict(
        name="Negative outlier", answers="Which regions require additional support?",
        formula="z = (block − district_mean) / σ  → idxmax", filters="z ≥ 1.5, ≥3 blocks"),
    "g_gender_gap": dict(
        name="Gender gap", answers="Are there persistent gender gaps?",
        formula="f_below − m_below, then two-proportion z-test",
        filters="|gap| ≥ 4 pts AND p < 0.05"),
    "g_steepest_decline": dict(
        name="Steepest decline", answers="Which districts are improving faster than others?",
        formula="below_pct − prev_pct → idxmax", filters="Δ ≥ +1 pt"),
    "g_bright_spot": dict(
        name="Bright spot", answers="Which districts are improving faster than others?",
        formula="prev_pct − below_pct → idxmax", filters="gain ≥ 1 pt"),
    "g_scale": dict(
        name="Largest scale", answers="Which regions require additional support?",
        formula="Σ(n × below_pct ÷ 100) by block → idxmax", filters="—"),
    "g_grade_progression": dict(
        name="Learning progression", answers="Problem statement: Learning progression",
        formula="mean below_pct at highest grade − lowest grade", filters="|Δ| ≥ 2 pts"),
    "g_persistence": dict(
        name="Chronic weakness", answers="Depth: structural vs one-year problems",
        formula="min(below_pct) across ALL years ≥ 50", filters="present in every year"),
    "g_within_district_spread": dict(
        name="Within-district inequity", answers="Problem statement: Geographic inequity",
        formula="max(block) − min(block) per competency", filters="reports widest only"),
    "g_gender_by_competency": dict(
        name="Gender × competency", answers="Handbook: girls lag in higher-order skills",
        formula="spread of mean gender_gap across competencies", filters="spread ≥ 4 pts"),
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


def generate_by_theme(agg, district, year=None, per_theme=2):
    """
    Findings grouped by theme instead of one flat ranked list.

    Scores are only compared WITHIN a theme, where they share units.
    This removes the arbitrary cross-type weighting.
    """
    out = {}
    for theme, fn_names in THEMES.items():
        items = []
        for name in fn_names:
            fn = _FN.get(name)
            if fn is None:
                continue
            try:
                items.extend(fn(agg, district, year))
            except Exception:
                continue
        items.sort(key=lambda x: -x["score"])
        out[theme] = items[:per_theme]
    return out


def describe(agg=None, district=None, year=None):
    """Registry rows, optionally annotated with what each generator found THIS run."""
    rows = []
    for fn in GENERATORS:
        meta = REGISTRY.get(fn.__name__, {})
        row = {"Generator": meta.get("name", fn.__name__),
               "Answers": meta.get("answers", "—"),
               "Formula": meta.get("formula", "—"),
               "Filter": meta.get("filters", "—")}
        if agg is not None:
            try:
                hits = fn(agg, district, year)
            except Exception:
                hits = []
            row["Fired?"] = ("✅ yes" if hits else "— nothing found")
            row["Score"] = round(hits[0]["score"], 1) if hits else None
        rows.append(row)
    return rows


def generate(agg, district, year=None, limit=7):
    """Run every generator, return the strongest finding of each type."""
    found = []
    for gen in GENERATORS:
        try:
            found.extend(gen(agg, district, year))
        except Exception:
            continue                     # a generator that finds nothing is fine
    found.sort(key=lambda x: -x["score"])
    return found[:limit]
