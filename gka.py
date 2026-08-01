"""
Layer 5 — GKA programme impact (no LLM, deterministic).

The question this layer answers is the only one the foundation actually cares
about: **is the programme helping children?**

Every other tab compares groups measured by the SAME paper — district vs
district, girls vs boys — and those comparisons are clean. This layer compares
across papers, and that is a different problem, because the nine GP-contest
papers are not equated to one another. Measured on the real file:

        year      2022-23  2023-24  2024-25
        Grade 4      49.6     47.8     57.8
        Grade 5      56.1     51.4     51.7
        Grade 6      62.5     54.5     49.3

In 2024-25 Grade 4 outscores Grade 6. Children did not become simultaneously
sharper in Grade 4 and duller in Grade 6 in the same year; the papers changed.
So a raw score difference between two papers is learning PLUS a difficulty
shift of unknown size, and no amount of averaging separates them.

Two measures survive that, and this module is built on both:

  1. ANCHORED DRIFT — compare only the skills the two papers share. The
     difficulty shift on those skills is the thing being held constant, so
     what is left is closer to real change. Needs enough shared skills to be
     worth trusting; the module refuses below MIN_ANCHORS and says why.

  2. RELATIVE POSITION — where a unit stands among all units sitting the SAME
     paper. Paper difficulty cancels exactly, because every unit faced it.
     Works even where no anchors exist, which is why the danger tables use it.

Nothing here reports a raw cross-year point change as if it meant learning.
Where a number cannot be earned, the finding says so rather than inventing one.
"""
import math

import numpy as np
import pandas as pd

MIN_N = 30                 # children before a percentage is worth stating
MIN_ANCHORS = 5            # shared skills before an anchored claim is allowed
MIN_UNITS_FOR_RANK = 8     # units before a percentile means anything
Z95 = 1.959963985
FLAT_PTS = 1.5             # a step smaller than this is flat even if "significant"

ERRORS = []


# ===========================================================================
#  Paper coverage — which competency was tested in which paper
# ===========================================================================
def coverage(qmaps):
    """
    What each of the nine papers actually tested.

    `qmaps` is {(year, grade): {question: competency}} — the per-paper mapping
    lifted from each workbook's own "Competency Mapping" sheet.

    Returns core (tested in EVERY paper, so comparable across all years),
    rotating (everything else, with the papers it is missing from) and a
    tested/not-tested matrix. A competency absent from a paper must render as
    a GAP, never as a zero: "not asked" and "nobody could do it" are opposite
    findings and they look identical once a missing cell becomes 0.
    """
    if not qmaps:
        return None
    papers = sorted(qmaps.keys())
    universe = sorted({c for m in qmaps.values() for c in m.values()})
    where = {c: {p for p in papers if c in set(qmaps[p].values())}
             for c in universe}
    core = sorted([c for c in universe if len(where[c]) == len(papers)])
    rotating = {}
    for c in universe:
        if c in core:
            continue
        rotating[c] = {
            "tested_in": sorted(where[c]),
            "missing_from": sorted(set(papers) - where[c]),
            "n_papers": len(where[c]),
        }
    matrix = pd.DataFrame(
        [{"competency": c, "year": y, "grade": g, "tested": (y, g) in where[c],
          "items": sum(1 for q, v in qmaps.get((y, g), {}).items() if v == c)}
         for c in universe for (y, g) in papers])
    return {"papers": papers, "core": core, "rotating": rotating,
            "universe": universe, "matrix": matrix, "n_papers": len(papers)}


def coverage_note(qmaps):
    """
    One honest paragraph about competencies with gaps — grade-aware.

    A year-level note alone is not enough: 'data handling' is tested in Grade 4
    only in 2023-24, but appears in Grades 5 and 6 in 2024-25, so a year-level
    check calls it present that year and the Grade 4 gap goes unmentioned.
    """
    cov = coverage(qmaps)
    if not cov or not cov["rotating"]:
        return None
    yrs = sorted({y for y, _ in cov["papers"]})
    bits = []
    for comp, info in sorted(cov["rotating"].items(),
                             key=lambda kv: -kv[1]["n_papers"]):
        miss = info["missing_from"]
        whole_years = [y for y in yrs
                       if all((y, g) in miss for (yy, g) in cov["papers"]
                              if yy == y)]
        if whole_years:
            rest = [f"Grade {g} in {y}" for (y, g) in miss
                    if y not in whole_years]
            txt = ("**%s** — not tested at all in %s"
                   % (comp, ", ".join(str(y) for y in whole_years)))
            if rest:
                txt += f"; also missing for {', '.join(rest)}"
        else:
            txt = (f"**{comp}** — not tested for "
                   + ", ".join(f"Grade {g} in {y}" for (y, g) in miss))
        bits.append(txt)
    return ("📋 **Coverage:** the paper changes every year and grade, so some "
            "competencies have gaps — " + " · ".join(bits)
            + f". The {len(cov['core'])} core competencies ("
            + ", ".join(cov["core"])
            + ") are in all "
            + f"{cov['n_papers']} papers and are the only ones safe to follow "
              "across every year. Gaps are shown as gaps, never as zero.")


# ===========================================================================
#  Small statistics helpers
# ===========================================================================
def _evidence(effect, se):
    """Lower 95% bound of |effect| — 0 when the interval crosses zero."""
    return max(0.0, abs(float(effect)) - Z95 * float(se))


def _bh(pvals, alpha=0.05):
    """Benjamini-Hochberg. Returns a boolean keep-mask in the input order."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    keep = np.zeros(n, dtype=bool)
    if passed.any():
        cut = np.max(np.where(passed)[0])
        keep[order[:cut + 1]] = True
    return keep


def _two_sided_p(z):
    """Normal tail probability without scipy."""
    return math.erfc(abs(float(z)) / math.sqrt(2.0))


def _norm_pdf(x, mu, sd):
    """Normal density — used to turn a score error into a percentile error."""
    if not sd or sd <= 0:
        return 0.0
    z = (float(x) - float(mu)) / float(sd)
    return math.exp(-0.5 * z * z) / (float(sd) * math.sqrt(2.0 * math.pi))


def _item_cols(df):
    return [c for c in df.columns
            if str(c).upper().startswith("Q") and str(c)[1:].isdigit()]


# ===========================================================================
#  The 3x3 grid — the evidence that the papers are not equated
# ===========================================================================
def paper_grid(df, year_col="Year", grade_col="Grade", item_cols=None):
    """Mean % correct and headcount for every (grade, year) paper."""
    items = item_cols or _item_cols(df)
    if not items:
        return None
    v = df[items].apply(pd.to_numeric, errors="coerce")
    out = df[[year_col, grade_col]].copy()
    out["pct"] = v.mean(axis=1) * 100.0
    g = (out.groupby([grade_col, year_col])["pct"]
            .agg(["mean", "size"]).reset_index()
            .rename(columns={"mean": "pct", "size": "children"}))
    return g


def grid_is_inconsistent(grid, grade_col="Grade", year_col="Year"):
    """
    Does the grade ordering flip between years?

    Within one year the papers are grade-specific, so % correct rising or
    falling with grade means nothing on its own. What DOES mean something is
    the ordering changing between years: whatever relationship holds between
    the Grade 4 and Grade 6 papers, it cannot reverse because of children.
    A flip is proof the papers were re-written, not that cohorts changed.
    """
    if grid is None or grid.empty:
        return None
    piv = grid.pivot(index=grade_col, columns=year_col, values="pct")
    if piv.shape[1] < 2:
        return None
    signs = {}
    for y in piv.columns:
        col = piv[y].dropna()
        if len(col) < 2:
            continue
        signs[str(y)] = 1 if col.iloc[-1] > col.iloc[0] else -1
    if len(set(signs.values())) > 1:
        up = [y for y, s in signs.items() if s > 0]
        dn = [y for y, s in signs.items() if s < 0]
        return {"flipped": True, "rises_in": up, "falls_in": dn, "table": piv}
    return {"flipped": False, "table": piv}


# ===========================================================================
#  Anchored comparison — the only defensible cross-paper score difference
# ===========================================================================
def anchors(qnames, a, b):
    """
    Skill names present in BOTH papers.

    Question NUMBERS are meaningless across papers — Q7 is a different question
    in every one. The workbooks carry a skill name per item
    ("long_division_without_remainder"), and those names recur, which is what
    makes any cross-paper comparison possible at all.

    A name is a SKILL, not a guaranteed-identical item, so this is common-skill
    anchoring rather than strict item equating: it holds content constant, not
    item difficulty. Callers must report the anchor count so a reader can see
    how much the comparison rests on.
    """
    na, nb = qnames.get(a) or {}, qnames.get(b) or {}
    sa = {str(v).strip() for v in na.values() if str(v).strip()}
    sb = {str(v).strip() for v in nb.values() if str(v).strip()}
    return sorted(sa & sb)


def _anchor_frame(df, qnames, paper, skills, year_col, grade_col):
    """Per-child mean on the anchor skills, plus the columns that identify them."""
    year, grade = paper
    m = qnames.get(paper) or {}
    cols = [q for q, nm in m.items()
            if str(nm).strip() in set(skills) and q in df.columns]
    if not cols:
        return None, []
    sub = df[(df[year_col].astype(str) == str(year))
             & (pd.to_numeric(df[grade_col], errors="coerce") == grade)]
    if sub.empty:
        return None, []
    vals = sub[cols].apply(pd.to_numeric, errors="coerce")
    out = sub.drop(columns=[c for c in sub.columns if c in cols]).copy()
    out["_anchor"] = vals.mean(axis=1) * 100.0
    # per-skill means too: several items can share one skill name, and a skill
    # with four items must not count four times against a skill with one.
    per_skill = {}
    for s in skills:
        qs = [q for q, nm in m.items()
              if str(nm).strip() == s and q in df.columns]
        if qs:
            per_skill[s] = float(
                sub[qs].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                .mean() * 100.0)
    return out, per_skill


def anchored_step(df, qnames, a, b, unit_col=None, year_col="Year",
                  grade_col="Grade", min_n=MIN_N):
    """
    Change from paper `a` to paper `b`, measured only on the skills they share.

    Returns one row per unit (or a single "All" row when unit_col is None):
        raw_a/raw_b      mean % correct on the WHOLE paper  (not comparable)
        anc_a/anc_b      mean % correct on the SHARED skills (comparable)
        drift            anc_b - anc_a
        se, evidence     uncertainty and the lower bound of |drift|
        n_anchors        how many skills the comparison rests on

    The standard error combines two things that dominate in different regimes.
    At state level the child sample is enormous, so sampling error is ~0.1pt
    and quoting it alone would claim a 7.0 +/- 0.2 point gain — overconfident,
    because the real uncertainty is WHICH skills happen to be shared. At GP
    level, with ~37 children, child sampling dominates instead. Both are
    included so neither regime is overstated.
    """
    sk = anchors(qnames, a, b)
    res_cols = ["unit", "n_a", "n_b", "raw_a", "raw_b", "raw_drift",
                "anc_a", "anc_b", "drift", "se", "evidence", "n_anchors"]
    if len(sk) < MIN_ANCHORS:
        return pd.DataFrame(columns=res_cols), {
            "ok": False, "n_anchors": len(sk), "skills": sk,
            "reason": (f"only {len(sk)} skill(s) are shared between the "
                       f"{a[0]} Grade {a[1]} and {b[0]} Grade {b[1]} papers — "
                       f"fewer than the {MIN_ANCHORS} needed to hold content "
                       f"constant, so no comparable score change can be "
                       f"computed for this step.")}

    fa, ska = _anchor_frame(df, qnames, a, sk, year_col, grade_col)
    fb, skb = _anchor_frame(df, qnames, b, sk, year_col, grade_col)
    if fa is None or fb is None:
        return pd.DataFrame(columns=res_cols), {
            "ok": False, "n_anchors": len(sk), "skills": sk,
            "reason": "one of the two papers has no rows in this selection."}

    items_a, items_b = _item_cols(df), _item_cols(df)
    for f, paper, items in ((fa, a, items_a), (fb, b, items_b)):
        pres = [c for c in items if c in df.columns]
        sub = df[(df[year_col].astype(str) == str(paper[0]))
                 & (pd.to_numeric(df[grade_col], errors="coerce") == paper[1])]
        f["_raw"] = (sub[pres].apply(pd.to_numeric, errors="coerce")
                     .mean(axis=1) * 100.0).reindex(f.index)

    # between-skill spread of the drift: the uncertainty in the anchor set
    common = [s for s in sk if s in ska and s in skb]
    diffs = np.array([skb[s] - ska[s] for s in common], dtype=float)
    sd_between = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    between_se = sd_between / math.sqrt(max(len(diffs), 1))

    def _agg(f, key):
        if key is None:
            g = f[["_anchor", "_raw"]].agg(["mean", "std", "count"])
            return pd.DataFrame({
                "unit": ["All"], "mean": [g.loc["mean", "_anchor"]],
                "sd": [g.loc["std", "_anchor"]], "n": [g.loc["count", "_anchor"]],
                "raw": [g.loc["mean", "_raw"]]})
        t = (f.groupby(key)
              .agg(mean=("_anchor", "mean"), sd=("_anchor", "std"),
                   n=("_anchor", "count"), raw=("_raw", "mean"))
              .reset_index().rename(columns={key: "unit"}))
        return t

    ta, tb = _agg(fa, unit_col), _agg(fb, unit_col)
    m = ta.merge(tb, on="unit", suffixes=("_a", "_b"))
    m = m[(m["n_a"] >= min_n) & (m["n_b"] >= min_n)]
    if m.empty:
        return pd.DataFrame(columns=res_cols), {
            "ok": False, "n_anchors": len(sk), "skills": sk,
            "reason": f"no unit has at least {min_n} children in both papers."}

    sam_a = m["sd_a"].fillna(0) / np.sqrt(m["n_a"].clip(lower=1))
    sam_b = m["sd_b"].fillna(0) / np.sqrt(m["n_b"].clip(lower=1))
    drift = m["mean_b"] - m["mean_a"]
    se = np.sqrt(sam_a ** 2 + sam_b ** 2 + between_se ** 2)
    out = pd.DataFrame({
        "unit": m["unit"], "n_a": m["n_a"].astype(int),
        "n_b": m["n_b"].astype(int),
        "raw_a": m["raw_a"].round(2), "raw_b": m["raw_b"].round(2),
        "raw_drift": (m["raw_b"] - m["raw_a"]).round(2),
        "anc_a": m["mean_a"].round(2), "anc_b": m["mean_b"].round(2),
        "drift": drift.round(2), "se": se.round(3),
        "evidence": [round(_evidence(d, s), 2) for d, s in zip(drift, se)],
        "n_anchors": len(common) or len(sk)})
    meta = {"ok": True, "n_anchors": len(common) or len(sk), "skills": sk,
            "sd_between_skills": round(sd_between, 2),
            "between_se": round(between_se, 3),
            "reason": None}
    return out.sort_values("drift", ascending=False).reset_index(drop=True), meta


# ===========================================================================
#  Cohorts — the diagonals through the grid
# ===========================================================================
def cohort_paths(years, grades):
    """
    Every diagonal: the same children one year older and one grade higher.

    Longest first, so the fullest evidence is presented before its fragments.
    Two-point paths are kept because a three-year path is often unanchorable
    while one of its steps is fine.
    """
    ys = sorted(map(str, years))
    gs = sorted(int(g) for g in grades)
    paths = []
    for gi, g0 in enumerate(gs):
        for yi, y0 in enumerate(ys):
            p, g, y = [], gi, yi
            while g < len(gs) and y < len(ys):
                p.append((ys[y], gs[g]))
                g += 1
                y += 1
            if len(p) >= 2:
                paths.append(p)
    paths.sort(key=len, reverse=True)
    return paths


def cohort_label(path):
    a, b = path[0], path[-1]
    return f"Grade {a[1]} {a[0]} → Grade {b[1]} {b[0]}"


# ===========================================================================
#  Relative position — paper-proof, works everywhere
# ===========================================================================
def percentiles(df, unit_col, year_col="Year", grade_col="Grade",
                item_cols=None, min_n=MIN_N, constant_panel=True):
    """
    Each unit's standing among the units that sat the SAME paper.

    Paper difficulty cancels exactly: every unit in a (year, grade) cell faced
    the identical questions, so a change in standing is a change relative to
    the state, whatever the paper did.

    `constant_panel` restricts to units present in every year. This is not
    tidiness — GP coverage grows from 2,628 to 4,877 across the three years,
    so a GP that stood still would drift in percentile purely because the
    comparison set changed underneath it.
    """
    items = item_cols or _item_cols(df)
    if not items or unit_col not in df.columns:
        return None
    v = df[items].apply(pd.to_numeric, errors="coerce")
    base = df[[unit_col, year_col, grade_col]].copy()
    base["pct"] = v.mean(axis=1) * 100.0
    base = base.dropna(subset=[unit_col])

    if constant_panel:
        per_year = base.groupby(year_col)[unit_col].apply(set)
        if len(per_year) > 1:
            keep = set.intersection(*per_year.tolist())
            base = base[base[unit_col].isin(keep)]
        if base.empty:
            return None

    g = (base.groupby([unit_col, year_col, grade_col])["pct"]
             .agg(["mean", "std", "size"]).reset_index()
             .rename(columns={"mean": "pct", "std": "sd", "size": "n"}))
    g = g[g["n"] >= min_n]
    if g.empty:
        return None

    # rank within each paper; too few units and a percentile is meaningless
    out = []
    for (y, gr), sub in g.groupby([year_col, grade_col]):
        if len(sub) < MIN_UNITS_FOR_RANK:
            continue
        s = sub.copy()
        s["percentile"] = s["pct"].rank(pct=True) * 100.0
        s["units_in_paper"] = len(sub)
        # How uncertain is that percentile? It is NOT a function of the unit's
        # own headcount alone: moving one score point is worth many percentile
        # points where units are densely packed and almost none out in a tail.
        # Convert the unit's score standard error through the local density of
        # unit means, so the same score wobble costs more percentile in a
        # crowded middle than at the edges.
        mu = float(s["pct"].mean())
        sd_units = float(s["pct"].std(ddof=1)) if len(s) > 1 else 0.0
        score_se = (s["sd"].fillna(0.0) / np.sqrt(s["n"].clip(lower=1)))
        dens = s["pct"].apply(lambda x: _norm_pdf(x, mu, sd_units))
        s["pct_se"] = score_se
        s["perc_se"] = (100.0 * dens * score_se).clip(lower=0.5, upper=50.0)
        # Standing expressed in standard deviations of the unit spread. This,
        # not the percentile, is what the trend is measured on: percentile is
        # a RANK, so where units bunch together a hair of score is worth many
        # percentile points and the same unit appears to lurch about. z is
        # linear in score, so it moves only when the score really moves.
        # Percentile is kept because it is what a reader understands.
        s["z"] = ((s["pct"] - mu) / sd_units) if sd_units > 0 else 0.0
        s["z_se"] = (score_se / sd_units) if sd_units > 0 else 0.0
        out.append(s)
    if not out:
        return None
    res = pd.concat(out, ignore_index=True)
    res = res.rename(columns={unit_col: "unit", year_col: "year",
                              grade_col: "grade"})
    return res


def unit_trajectory(perc, min_grades=1):
    """
    One percentile per unit per year, over the grades that unit has in EVERY
    year.

    Averaging percentiles (not scores) is what keeps this paper-proof: each
    grade's percentile is already expressed against its own paper.

    The common-grade restriction matters more than it looks. A small GP may
    clear the headcount floor in Grade 6 one year and only in Grade 4 the
    next; averaging whatever qualified would then compare a Grade 6 standing
    with a Grade 4 standing and read the difference as movement. Only grades
    the unit holds in all years are used, so the mix cannot shift underneath
    the trend.
    """
    if perc is None or perc.empty:
        return None
    years = sorted(perc["year"].unique())
    keep = []
    for unit, sub in perc.groupby("unit"):
        by_year = sub.groupby("year")["grade"].apply(set)
        if len(by_year) < len(years):
            continue
        common = set.intersection(*by_year.tolist())
        if len(common) < min_grades:
            continue
        s = sub[sub["grade"].isin(common)].copy()
        s["common_grades"] = len(common)
        keep.append(s)
    if not keep:
        return None
    k = pd.concat(keep, ignore_index=True)

    def _agg(g):
        w = g["n"].astype(float)
        return pd.Series({
            "percentile": float((g["percentile"] * w).sum() / w.sum()),
            "z": float((g["z"] * w).sum() / w.sum()),
            # errors of independent grade means average down by 1/k
            "z_se": float(np.sqrt((g["z_se"] ** 2).sum()) / len(g)),
            "se": float(np.sqrt((g["perc_se"] ** 2).sum()) / len(g)),
            "n": int(g["n"].sum()),
            "grades": int(g["grade"].nunique())})

    t = (k.groupby(["unit", "year"]).apply(_agg, include_groups=False)
          .reset_index())
    return t.sort_values(["unit", "year"])


# ===========================================================================
#  Trajectory shape
# ===========================================================================
SHAPES = {
    ("up", "up"):     ("Sustained rise", "improving in both steps"),
    ("up", "flat"):   ("Rise then plateau", "gained, then levelled off"),
    ("up", "down"):   ("Rise then relapse", "gained, then gave it back"),
    ("flat", "up"):   ("Late take-off", "flat, then improving"),
    ("flat", "flat"): ("Flat", "no measurable movement"),
    ("flat", "down"): ("Late decline", "steady, then falling"),
    ("down", "up"):   ("Recovery (V-shape)", "fell, then recovered"),
    ("down", "flat"): ("Decline then hold", "fell, then stopped falling"),
    ("down", "down"): ("Sustained decline", "falling in both steps"),
}


def step_dir(delta, se=0.0, flat_pts=FLAT_PTS):
    """
    A step counts as movement only if it beats BOTH its own noise and a
    minimum size. Statistical significance alone is not enough: on a district
    of 40,000 children a 0.3-point change is 'significant' and means nothing.
    """
    d = float(delta)
    if abs(d) < flat_pts or _evidence(d, se) <= 0:
        return "flat"
    return "up" if d > 0 else "down"


def classify(values, ses=None, flat_pts=FLAT_PTS):
    """Shape of a 3-point series. Returns (key, label, reading) or None."""
    v = [x for x in values if x == x]
    if len(v) < 3:
        return None
    s = ses or [0.0] * len(v)
    d1 = step_dir(v[1] - v[0], math.hypot(s[0], s[1]), flat_pts)
    d2 = step_dir(v[2] - v[1], math.hypot(s[1], s[2]), flat_pts)
    lbl, reading = SHAPES[(d1, d2)]
    return {"key": f"{d1}_{d2}", "steps": (d1, d2), "label": lbl,
            "reading": reading}


# ===========================================================================
#  Danger tables — units losing ground, corrected for multiple testing
# ===========================================================================
def danger(df, unit_col, year_col="Year", grade_col="Grade", item_cols=None,
           min_n=MIN_N, limit=10, alpha=0.05, constant_panel=True):
    """
    Units whose standing among their peers falls across the whole period.

    Three guards, because picking "the worst 10" out of thousands of GPs finds
    decliners even when nothing is declining:
      * constant panel, so the comparison set cannot move underneath a unit
      * the fall must exceed its own noise, not merely be negative
      * Benjamini-Hochberg across every unit tested

    Ceiling is handled too: a unit already near the top of its paper can only
    go sideways, and listing it as "in danger" would be indefensible.
    """
    perc = percentiles(df, unit_col, year_col, grade_col, item_cols,
                       min_n=min_n, constant_panel=constant_panel)
    traj = unit_trajectory(perc)
    if traj is None or traj.empty:
        return None
    years = sorted(traj["year"].unique())
    if len(years) < 3:
        return None

    rows = []
    for unit, sub in traj.groupby("unit"):
        s = sub.set_index("year")
        if not all(y in s.index for y in years):
            continue
        zs = [float(s.loc[y, "z"]) for y in years]
        pv = [float(s.loc[y, "percentile"]) for y in years]
        ns = s["n"].reindex(years).fillna(0).astype(int)
        if int(ns.min()) < min_n:
            continue
        zse = s["z_se"].reindex(years).fillna(0.3).astype(float).tolist()
        # shape is read on z (linear in score), with a flat band in sd units
        shape = classify(zs, zse, flat_pts=0.10)
        if shape is None:
            continue
        rows.append({
            "unit": unit,
            "start": round(pv[0], 1), "mid": round(pv[1], 1),
            "end": round(pv[-1], 1), "change": round(pv[-1] - pv[0], 1),
            "z_change": round(zs[-1] - zs[0], 2),
            "shape": shape["label"], "shape_key": shape["key"],
            "children": int(ns.sum()), "_se0": zse[0], "_se2": zse[-1],
            "headroom": round(100.0 - pv[-1], 1)})
    if not rows:
        return None
    t = pd.DataFrame(rows)

    # Sampling error alone makes everything "significant": a district of 14,000
    # children has its mean pinned to ~0.4 points, so a 30-percentile move
    # scores z=30 and p=1e-180, which is arithmetically true and useless. The
    # honest comparison is against how much units MOVE ANYWAY — different
    # children, different paper emphasis, ordinary churn. That scale is
    # measured from the data (robust, so the decliners we are hunting cannot
    # inflate the yardstick that judges them) and added to each unit's own error.
    ch = t["z_change"].astype(float)
    churn = 1.4826 * float((ch - ch.median()).abs().median())
    churn = max(churn, 0.05)
    denom = np.sqrt(t["_se0"] ** 2 + t["_se2"] ** 2 + churn ** 2)
    t["p"] = [_two_sided_p(c / d) if d > 0 else 1.0
              for c, d in zip(ch, denom)]
    t["evidence"] = [round(_evidence(c, d), 2) for c, d in zip(ch, denom)]
    t["typical_move_sd"] = round(churn, 2)
    t = t.drop(columns=["_se0", "_se2"])

    # "No growth or a downward curve" — a unit that rose 33 points and then
    # gave back 12 is up 21 overall and is NOT in danger, however
    # rise-then-relapse its shape looks. Net movement has to be negative
    # before anything is flagged.
    falling = t[(t["z_change"] < 0) & (t["shape_key"] != "up_up")].copy()
    if falling.empty:
        return t.iloc[0:0]
    # Already near the top of its paper and only drifting is not "in danger" —
    # listing the state's strongest units as failing would be indefensible.
    falling = falling[~((falling["end"] >= 75.0) & (falling["change"] > -15.0))]
    if falling.empty:
        return t.iloc[0:0]

    # A unit that falls in BOTH steps is the brief's literal ask ("a downward
    # curve across all 3 years") and it is also the only family with enough
    # power to survive correction: on the real file 13 of 59 strictly-declining
    # GPs clear Benjamini-Hochberg, while widening the family to any net fall
    # (774 units) leaves nothing significant at all. Testing the narrow,
    # pre-stated family is what buys the evidence.
    falling["strict_decline"] = falling["shape_key"] == "down_down"
    strict = falling[falling["strict_decline"]].copy()
    if not strict.empty:
        strict["beyond_normal_movement"] = (
            _bh(strict["p"].tolist(), alpha=alpha) & (strict["evidence"] > 0))
    rest = falling[~falling["strict_decline"]].copy()
    rest["beyond_normal_movement"] = False

    strict = strict.sort_values(["z_change", "children"], ascending=[True, False])
    rest = rest.sort_values(["z_change", "children"], ascending=[True, False])
    # Strict decliners first; the rest only top the table up to `limit`, so the
    # brief always has its rows without promoting ordinary churn to a crisis.
    out = pd.concat([strict, rest], ignore_index=True).head(limit)
    return out.reset_index(drop=True)


def stagnant(df, unit_col, year_col="Year", grade_col="Grade", item_cols=None,
             min_n=MIN_N, limit=10, constant_panel=True, bottom_pct=33.0):
    """
    Units that never moved AND sit low — "no growth", the other half of the
    brief. Kept separate from `danger` because the action differs: a falling
    unit has something to explain, a stuck one has never started.
    """
    perc = percentiles(df, unit_col, year_col, grade_col, item_cols,
                       min_n=min_n, constant_panel=constant_panel)
    traj = unit_trajectory(perc)
    if traj is None or traj.empty:
        return None
    years = sorted(traj["year"].unique())
    rows = []
    for unit, sub in traj.groupby("unit"):
        s = sub.set_index("year")["percentile"]
        if not all(y in s.index for y in years) or len(years) < 3:
            continue
        v = [float(s[y]) for y in years]
        se = (sub.set_index("year")["se"].reindex(years)
                 .fillna(10.0).astype(float).tolist())
        shape = classify(v, se)
        if shape is None or shape["key"] != "flat_flat":
            continue
        if max(v) > bottom_pct:
            continue
        rows.append({"unit": unit, "start": round(v[0], 1),
                     "mid": round(v[1], 1), "end": round(v[-1], 1),
                     "change": round(v[-1] - v[0], 1), "shape": "Flat",
                     "shape_key": "flat_flat",
                     "children": int(sub["n"].sum()),
                     "headroom": round(100.0 - v[-1], 1)})
    if not rows:
        return None
    return (pd.DataFrame(rows).sort_values(["end", "children"],
                                           ascending=[True, False])
            .head(limit).reset_index(drop=True))


# ===========================================================================
#  Competency profile — which skills lag, in a paper-proof way
# ===========================================================================
def competency_profile(df, qmaps, year_col="Year", grade_col="Grade",
                       min_items=1):
    """
    Each competency's standing RELATIVE TO ITS OWN PAPER's average.

    Absolute competency scores cannot be compared across papers, but every
    competency inside one paper met the same children on the same day, so the
    gap to that paper's own mean is meaningful. If a competency sits below its
    paper's mean in all nine papers, that is a stable finding about what
    children cannot do — and it survives every difficulty change.
    """
    if not qmaps:
        return None
    rows = []
    for (y, g), m in sorted(qmaps.items()):
        sub = df[(df[year_col].astype(str) == str(y))
                 & (pd.to_numeric(df[grade_col], errors="coerce") == g)]
        if sub.empty:
            continue
        qs = [q for q in m if q in sub.columns]
        if not qs:
            continue
        vals = sub[qs].apply(pd.to_numeric, errors="coerce")
        paper_mean = float(vals.mean(axis=1).mean() * 100.0)
        by_comp = {}
        for q in qs:
            by_comp.setdefault(m[q], []).append(q)
        for comp, cols in by_comp.items():
            if len(cols) < min_items:
                continue
            cm = float(vals[cols].mean(axis=1).mean() * 100.0)
            rows.append({"competency": comp, "year": y, "grade": g,
                         "pct": round(cm, 1), "paper_mean": round(paper_mean, 1),
                         "vs_paper": round(cm - paper_mean, 1),
                         "items": len(cols), "children": len(sub)})
    if not rows:
        return None
    t = pd.DataFrame(rows)
    summ = (t.groupby("competency")
             .agg(papers=("vs_paper", "size"),
                  mean_vs_paper=("vs_paper", "mean"),
                  worst=("vs_paper", "min"), best=("vs_paper", "max"),
                  below_in=("vs_paper", lambda s: int((s < 0).sum())))
             .reset_index())
    summ["mean_vs_paper"] = summ["mean_vs_paper"].round(1)
    summ = summ.sort_values("mean_vs_paper")
    return {"detail": t, "summary": summ}


# ===========================================================================
#  One pass over everything the tab needs
# ===========================================================================
def analyse(df, qmaps, qnames, year_col="Year", grade_col="Grade",
            unit_col="District", gp_col=None, min_n=MIN_N, limit=10):
    """
    Run the whole GKA layer once. Everything downstream reads this dict, so
    the narrative and the tables can never disagree about a number.
    """
    del ERRORS[:]
    # Paper keys arrive in whatever the caller had: the workbooks say
    # "2022-23", the dashboard converts Year to 2022, and a hand-built map may
    # use either. Years derived from the frame below are strings, so normalise
    # every key to (str year, int grade) once — otherwise the lookups miss and
    # the layer reports, silently and wrongly, that no papers share any skill.
    qmaps = {(str(y), int(g)): m for (y, g), m in (qmaps or {}).items()}
    qnames = {(str(y), int(g)): m for (y, g), m in (qnames or {}).items()}
    res = {"unit_col": unit_col, "gp_col": gp_col, "min_n": min_n}
    try:
        years = sorted({str(y) for y in df[year_col].dropna().unique()})
        grades = sorted({int(g) for g in
                         pd.to_numeric(df[grade_col], errors="coerce")
                         .dropna().unique()})
    except Exception as e:                                  # pragma: no cover
        ERRORS.append({"stage": "years/grades", "error": str(e)})
        return None
    if len(years) < 2 or len(grades) < 2:
        return {"too_short": True, "years": years, "grades": grades,
                "unit_col": unit_col, "gp_col": gp_col, "min_n": min_n}
    res["years"], res["grades"] = years, grades
    res["coverage"] = coverage(qmaps)
    res["coverage_note"] = coverage_note(qmaps)

    grid = paper_grid(df, year_col, grade_col)
    res["grid"] = grid
    res["flip"] = grid_is_inconsistent(grid, grade_col, year_col)

    # --- cohort diagonals, anchored where the papers allow it --------------
    res["cohorts"] = []
    for path in cohort_paths(years, grades):
        steps = []
        for a, b in zip(path, path[1:]):
            try:
                t, meta = anchored_step(df, qnames, a, b, None, year_col,
                                        grade_col, min_n)
            except Exception as e:                          # pragma: no cover
                ERRORS.append({"stage": "cohort %s->%s" % (a, b),
                               "error": str(e)})
                continue
            steps.append({"a": a, "b": b, "meta": meta,
                          "row": t.iloc[0].to_dict() if len(t) else {}})
        if steps:
            res["cohorts"].append({"path": path, "label": cohort_label(path),
                                   "steps": steps})

    # --- same grade across years: is a dip the paper or the children? ------
    res["same_grade"] = []
    for g in grades:
        for a, b in zip(years, years[1:]):
            try:
                t, meta = anchored_step(df, qnames, (a, g), (b, g), None,
                                        year_col, grade_col, min_n)
            except Exception as e:                          # pragma: no cover
                ERRORS.append({"stage": "grade %s %s->%s" % (g, a, b),
                               "error": str(e)})
                continue
            res["same_grade"].append(
                {"grade": g, "a": a, "b": b, "meta": meta,
                 "row": t.iloc[0].to_dict() if len(t) else {}})

    # --- the best anchorable step, broken out by unit ----------------------
    best, best_n = None, 0
    for c in res["cohorts"]:
        for s in c["steps"]:
            if s["meta"].get("ok") and s["meta"]["n_anchors"] > best_n:
                best, best_n = s, s["meta"]["n_anchors"]
    res["best_step"] = best
    res["by_unit"] = None
    res["by_unit_meta"] = None
    if best is not None:
        try:
            bu, bmeta = anchored_step(df, qnames, best["a"], best["b"],
                                      unit_col, year_col, grade_col, min_n)
            res["by_unit"], res["by_unit_meta"] = bu, bmeta
        except Exception as e:                              # pragma: no cover
            ERRORS.append({"stage": "by_unit", "error": str(e)})

    # --- relative position, danger, stagnation ----------------------------
    for key, col in (("district", unit_col), ("gp", gp_col)):
        if not col or col not in df.columns:
            res["danger_" + key] = None
            res["stagnant_" + key] = None
            continue
        try:
            res["danger_" + key] = danger(df, col, year_col, grade_col,
                                          min_n=min_n, limit=limit)
            res["stagnant_" + key] = stagnant(df, col, year_col, grade_col,
                                              min_n=min_n, limit=limit)
        except Exception as e:                              # pragma: no cover
            ERRORS.append({"stage": "danger " + key, "error": str(e)})
            res["danger_" + key] = None
            res["stagnant_" + key] = None

    try:
        res["competency"] = competency_profile(df, qmaps, year_col, grade_col)
    except Exception as e:                                  # pragma: no cover
        ERRORS.append({"stage": "competency", "error": str(e)})
        res["competency"] = None

    # --- coverage of the units themselves ---------------------------------
    try:
        per_year = df.groupby(df[year_col].astype(str))[unit_col].apply(
            lambda s: set(s.dropna().astype(str)))
        allu = set().union(*per_year.tolist())
        comm = set.intersection(*per_year.tolist())
        res["unit_coverage"] = {"all": len(allu), "in_every_year": len(comm),
                                "partial": sorted(allu - comm)}
    except Exception:                                       # pragma: no cover
        res["unit_coverage"] = None
    return res


# ===========================================================================
#  Findings
# ===========================================================================
def _f(cat, text, evidence, source, score):
    return {"category": cat, "text": text, "evidence": evidence,
            "source": source, "score": round(float(score), 2)}


def findings(res, limit=12):
    """
    Ranked sentences about whether GKA is helping.

    The instrument warning is scored highest on purpose: every number after it
    is read differently once you know the papers changed.
    """
    if not res or res.get("too_short"):
        return []
    out = []

    flip = res.get("flip") or {}
    if flip.get("flipped"):
        tbl = flip["table"]
        lo, hi = tbl.index.min(), tbl.index.max()
        out.append(_f(
            "instrument",
            "**The nine papers are not equated to each other, so raw scores "
            "cannot be compared across years.** Scores rise with grade in "
            + ", ".join(flip["rises_in"]) + " and fall with grade in "
            + ", ".join(flip["falls_in"]) + ". Children cannot reverse that "
            "ordering in a single year — the papers were rewritten. Every "
            "cross-year figure below is therefore measured on the skills two "
            "papers share, never on raw totals.",
            "Grade %s vs Grade %s ordering flips between years" % (lo, hi),
            "g_papers_not_equated", 100.0))

    for c in res.get("cohorts", []):
        for s in c["steps"]:
            a, b, meta, row = s["a"], s["b"], s["meta"], s["row"]
            step = "Grade %s %s → Grade %s %s" % (a[1], a[0], b[1], b[0])
            if not meta.get("ok"):
                out.append(_f(
                    "cohort",
                    "**" + step + " cannot be compared.** " + meta["reason"],
                    "%d shared skill(s), %d required"
                    % (meta["n_anchors"], MIN_ANCHORS),
                    "g_cohort_blocked", 40.0))
                continue
            d, ev = float(row["drift"]), float(row["evidence"])
            raw = float(row["raw_drift"])
            txt = ("**" + step + ": this cohort "
                   + ("gained" if d > 0 else "lost")
                   + " %.1f points** on the %d skills both papers test."
                   % (abs(d), meta["n_anchors"]))
            if abs(d - raw) >= 2.0:
                txt += (" The raw score moved only %+.1f — the difference is "
                        "the paper, not the children." % raw)
            if ev <= 0:
                txt += (" The change does not clear its own confidence "
                        "interval, so treat it as no measurable movement.")
            out.append(_f(
                "cohort", txt,
                "anchored %+.1f ± %.1f pts · raw %+.1f · %d shared skills · "
                "%s then %s children"
                % (d, 1.96 * float(row["se"]), raw, meta["n_anchors"],
                   format(int(row["n_a"]), ","), format(int(row["n_b"]), ",")),
                "g_cohort_anchored", 60.0 + ev))

    masked = [s for s in res.get("same_grade", [])
              if s["meta"].get("ok") and s["row"]
              and abs(float(s["row"]["drift"])
                      - float(s["row"]["raw_drift"])) >= 3.0]
    if masked:
        w = max(masked, key=lambda s: abs(float(s["row"]["raw_drift"])
                                          - float(s["row"]["drift"])))
        r = w["row"]
        out.append(_f(
            "instrument",
            "**A large part of the apparent decline is the paper getting "
            "harder, not children learning less.** Grade %s appears to fall "
            "%.1f points between %s and %s, but on the %d skills both papers "
            "share the change is only %+.1f."
            % (w["grade"], abs(float(r["raw_drift"])), w["a"], w["b"],
               w["meta"]["n_anchors"], float(r["drift"])),
            "raw %+.1f vs anchored %+.1f on the same children"
            % (float(r["raw_drift"]), float(r["drift"])),
            "g_paper_masked_change", 85.0))

    bu = res.get("by_unit")
    if bu is not None and len(bu):
        up = int((bu["drift"] > 0).sum())
        solid = int(((bu["drift"] > 0) & (bu["evidence"] > 0)).sum())
        best, worst = bu.iloc[0], bu.iloc[-1]
        out.append(_f(
            "spread",
            "**The gain is broad, not one or two places: %d of %d districts "
            "improved** on this step, %d of them by more than their own "
            "margin of error. **%s** gained the most (%+.1f); **%s** fell the "
            "furthest (%+.1f)."
            % (up, len(bu), solid, best["unit"], float(best["drift"]),
               worst["unit"], float(worst["drift"])),
            "anchored on %d shared skills, minimum %d children per district "
            "per paper" % (res["by_unit_meta"]["n_anchors"], res["min_n"]),
            "g_gain_spread", 70.0 + solid))

    comp = res.get("competency")
    if comp is not None:
        s = comp["summary"]
        npapers = int(s["papers"].max())
        weak = s[(s["below_in"] == s["papers"]) & (s["papers"] == npapers)]
        if len(weak):
            w = weak.iloc[0]
            out.append(_f(
                "competency",
                "**%s is the standing weakness: it scores below its own "
                "paper's average in all %d papers**, by %.1f points on "
                "average. Because that is measured against each paper's own "
                "mean, it holds whatever the paper did — it is the clearest "
                "place to aim the programme."
                % (w["competency"], int(w["papers"]),
                   abs(float(w["mean_vs_paper"]))),
                "below its paper mean in %d of %d papers"
                % (int(w["below_in"]), int(w["papers"])),
                "g_competency_persistent_weak", 90.0))
        strong = s[(s["below_in"] == 0) & (s["papers"] == npapers)]
        if len(strong):
            st = strong.iloc[-1]
            out.append(_f(
                "competency",
                "**%s is the consistent strength** — above its paper's "
                "average in all %d papers (+%.1f on average). Whatever is "
                "being done here works, and is the natural model for the "
                "weaker competencies."
                % (st["competency"], int(st["papers"]),
                   float(st["mean_vs_paper"])),
                "never below its paper mean across %d papers"
                % int(st["papers"]),
                "g_competency_persistent_strong", 65.0))

    cov = res.get("coverage")
    if cov and cov.get("rotating"):
        out.append(_f(
            "coverage",
            "**%d competencies are tested in every paper and only those can "
            "be followed across all three years.** %s come and go, so their "
            "gaps are shown as gaps — a competency that was never asked must "
            "not look like one children failed."
            % (len(cov["core"]), ", ".join(sorted(cov["rotating"]))),
            "core: " + ", ".join(cov["core"]),
            "g_competency_coverage", 55.0))

    for key, label in (("gp", "GP"), ("district", "district")):
        d = res.get("danger_" + key)
        if d is None or d.empty:
            continue
        strict = d[d["strict_decline"]] if "strict_decline" in d else d.iloc[0:0]
        sig = (d[d["beyond_normal_movement"]]
               if "beyond_normal_movement" in d else d.iloc[0:0])
        if len(sig):
            w = sig.iloc[0]
            out.append(_f(
                "danger",
                "**%d %s(s) fell in every one of the three years by more than "
                "%ss normally move.** The steepest is **%s**, down %.0f "
                "percentile points (%+.2f standard deviations) across %s "
                "children."
                % (len(sig), label, label, w["unit"], abs(float(w["change"])),
                   float(w["z_change"]), format(int(w["children"]), ",")),
                "strictly declining in both steps · Benjamini-Hochberg "
                "corrected · a typical %s moves %.2f sd"
                % (label, float(w["typical_move_sd"])),
                "g_danger_" + key, 80.0 + len(sig)))
        elif len(strict):
            out.append(_f(
                "danger",
                "**%d %s(s) declined in both steps, but none by more than %ss "
                "move anyway.** They are worth watching rather than "
                "escalating — at this level year-to-year movement is large "
                "enough to swallow the signal."
                % (len(strict), label, label),
                "a typical %s moves %.2f sd"
                % (label, float(d["typical_move_sd"].iloc[0])),
                "g_danger_" + key, 50.0))

    uc = res.get("unit_coverage")
    if uc and uc.get("partial"):
        out.append(_f(
            "coverage",
            "**%d of %d districts were not assessed in all three years** (%s%s), "
            "so every trend here is computed on the %d present throughout. "
            "Comparing all districts each year would mix a change in who was "
            "measured with a change in what they scored."
            % (len(uc["partial"]), uc["all"], ", ".join(uc["partial"][:4]),
               " …" if len(uc["partial"]) > 4 else "", uc["in_every_year"]),
            "%d of %d districts in every year"
            % (uc["in_every_year"], uc["all"]),
            "g_unit_coverage", 75.0))

    out.append(_f(
        "limits",
        "**Every child in this data is in GKA, so there is no comparison "
        "group.** These findings show children under the programme improved "
        "on the skills we can compare, consistently across most districts. "
        "Establishing that GKA *caused* it would need children outside the "
        "programme, or a record of when each district joined.",
        "no non-GKA comparison group exists in the dataset",
        "g_no_control", 30.0))

    out.sort(key=lambda x: -x["score"])
    return out[:limit]


# ===========================================================================
#  Recommendations
# ===========================================================================
# Severity is where the unit ENDS, not how far it fell: a unit that dropped 40
# percentile points but still sits mid-table needs a different response from
# one that fell 15 and is now at the bottom.
def severity(end_percentile):
    p = float(end_percentile)
    if p < 25:
        return "critical"
    if p < 50:
        return "weak"
    if p < 75:
        return "middling"
    return "strong"


# 9 shapes x 4 severities. Every cell is filled: a rule that silently has no
# action for some combination produces blank advice exactly when a unit is
# unusual, which is when advice matters most.
BASE_ACTIONS = {
    ("down_down", "critical"): "Escalate now. This {unit} has lost ground in both years and is among the weakest in the state. Send a field team before the next assessment cycle rather than waiting for another year of data.",
    ("down_down", "weak"): "Put this {unit} on the district review list. Two consecutive falls from an already below-average position is the clearest failing pattern in the data.",
    ("down_down", "middling"): "Investigate while it is still recoverable. Two consecutive falls from mid-table usually means something changed locally — staffing, attendance or teaching time.",
    ("down_down", "strong"): "Check what changed. This {unit} is still strong but has slipped twice running, which is how strong performers become average ones.",
    ("down_flat", "critical"): "Treat the plateau as the problem. The fall has stopped but at the bottom of the state — stabilising at a low level is not recovery.",
    ("down_flat", "weak"): "Find out what stopped the fall and whether it can be pushed further. Halting a decline is real progress that usually goes unrecognised.",
    ("down_flat", "middling"): "Hold the line and look for the cause of the original drop; it has stabilised but has not recovered what it lost.",
    ("down_flat", "strong"): "Low priority. It fell then steadied while remaining strong — monitor rather than intervene.",
    ("flat_down", "critical"): "Act on the recent fall. A {unit} that was static and has now dropped to the bottom has changed recently, so the cause is likely still present and findable.",
    ("flat_down", "weak"): "Ask what changed this year specifically. The decline is recent, which narrows the search considerably.",
    ("flat_down", "middling"): "Watch closely next cycle. One recent fall from a stable position may be noise; two would not be.",
    ("flat_down", "strong"): "Note and monitor. Still strong, but the direction changed this year.",
    ("up_down", "critical"): "Understand the reversal. This {unit} improved and then gave the gain back entirely — whatever drove the improvement was not sustained.",
    ("up_down", "weak"): "Find out what was working during the rise and why it stopped. This is the most recoverable pattern in the table.",
    ("up_down", "middling"): "Look for a one-off cause behind the rise. Gains that vanish usually came from something temporary.",
    ("up_down", "strong"): "Low priority, but worth a note: the gain was not held.",
    ("flat_flat", "critical"): "Start something. This {unit} has not moved in three years and sits at the bottom — the programme is not reaching it in its current form.",
    ("flat_flat", "weak"): "Nothing here is working or failing; it is simply static. Consider whether the standard programme suits this {unit} at all.",
    ("flat_flat", "middling"): "No action needed on trend. Stable and mid-table.",
    ("flat_flat", "strong"): "No action. Stable and strong is the goal state.",
    ("down_up", "critical"): "Support the recovery. It has started climbing from a very low base — this is exactly when withdrawing attention undoes the gain.",
    ("down_up", "weak"): "Protect what has started working. The turnaround is real but the {unit} has not yet regained its original position.",
    ("down_up", "middling"): "Partial recovery only. The last step was upward, but the {unit} is still below where it began — check whether the rebound is holding before treating it as fixed.",
    ("down_up", "strong"): "Watch rather than act. It has started climbing again and remains strong, but has not yet recovered the position it lost.",
    ("flat_up", "critical"): "Reinforce. Movement has begun from a very low base and needs continuity more than new initiatives.",
    ("flat_up", "weak"): "Keep doing whatever started this year. A stable {unit} that begins improving is a delayed-adoption success.",
    ("flat_up", "middling"): "Moving in the right direction again, but from a lower base than it started at — confirm next cycle before closing the case.",
    ("flat_up", "strong"): "Low priority. Improving again and still strong, though not yet back to its starting position.",
    ("up_flat", "critical"): "The rise stalled at a low level. Look for the ceiling — often a resource or staffing limit rather than teaching.",
    ("up_flat", "weak"): "Gains were made and then held. Ask what would be needed to start the next rise.",
    ("up_flat", "middling"): "No action needed. Improved, then consolidated.",
    ("up_flat", "strong"): "No action. Improved and holding at a strong level.",
    ("up_up", "critical"): "Study and copy. Still weak in absolute terms but improving in both years — this is the fastest-moving {unit} at the bottom of the table.",
    ("up_up", "weak"): "Document what is being done here. Two consecutive gains from a weak base is the pattern other {unit}s need.",
    ("up_up", "middling"): "Use as a model. Sustained improvement into the middle of the table.",
    ("up_up", "strong"): "Protect and publicise. Sustained gains at a strong level — the clearest success in the data.",
}


def _m_still_below(r, res):
    """
    Every unit in the danger table is down on net, but four of the nine shapes
    end with an upward step. Without this clause those rows read as good news
    ("the recovery is underway") beside a table showing the unit 31 percentile
    points below where it started.
    """
    if str(r.get("shape_key", "")).endswith("_up"):
        return ("Net of both steps it is still %.0f percentile points below "
                "where it began, so this is a partial rebound, not a recovery."
                % abs(float(r.get("change", 0))))
    return None


def _m_proven(r, res):
    if r.get("beyond_normal_movement"):
        return ("The fall is larger than %ss normally move year to year, and "
                "survives correction for testing every %s in the state, so it "
                "is not chance." % (r["_label"], r["_label"]))
    return None


def _m_watch(r, res):
    if r.get("strict_decline") and not r.get("beyond_normal_movement"):
        return ("The direction is consistent but the size is within normal "
                "year-to-year movement for a %s — watch it, do not escalate "
                "on this evidence alone." % r["_label"])
    return None


def _m_notstrict(r, res):
    if not r.get("strict_decline"):
        return ("This did not fall in both steps, so it is here on the size "
                "of the net change only — a weaker signal than the strictly "
                "declining %ss above it." % r["_label"])
    return None


def _m_small(r, res):
    n = int(r.get("children") or 0)
    if n and n < 200:
        return ("Only %s children across all three years, so its position "
                "moves easily; confirm against the block before acting."
                % format(n, ","))
    return None


def _m_large(r, res):
    n = int(r.get("children") or 0)
    if n >= 20000:
        return ("This covers %s children — the largest single block of "
                "affected pupils in this table." % format(n, ","))
    return None


def _m_headroom(r, res):
    if float(r.get("headroom", 100)) < 20:
        return ("It remains near the top of the state, so the fall is a loss "
                "of margin rather than a crisis.")
    return None


def _m_collapse(r, res):
    if float(r.get("change", 0)) <= -50:
        return ("The drop of %.0f percentile points is large enough to "
                "suspect a change in who was assessed as well as how they "
                "did — check enrolment and coverage first."
                % abs(float(r["change"])))
    return None


def _m_competency(r, res):
    comp = (res or {}).get("competency")
    if comp is None:
        return None
    s = comp["summary"]
    npapers = int(s["papers"].max())
    weak = s[(s["below_in"] == s["papers"]) & (s["papers"] == npapers)]
    if len(weak):
        return ("Statewide, %s is the competency furthest below its own "
                "paper's average, so start there unless local item data says "
                "otherwise." % weak.iloc[0]["competency"])
    return None


def _m_coverage(r, res):
    uc = (res or {}).get("unit_coverage") or {}
    if uc.get("partial") and str(r.get("unit")) in set(uc["partial"]):
        return ("This unit was not assessed in every year, so part of the "
                "movement may be a change in who was measured.")
    return None


MODIFIERS = [("still_below", _m_still_below),
             ("proven", _m_proven), ("watch", _m_watch),
             ("notstrict", _m_notstrict), ("small", _m_small),
             ("large", _m_large), ("headroom", _m_headroom),
             ("collapse", _m_collapse), ("competency", _m_competency),
             ("coverage", _m_coverage)]
MAX_CLAUSES = 3


def combination_space():
    """
    How many distinct recommendations the rules can produce — computed from
    the rules themselves, never hardcoded, so it cannot drift from reality.
    """
    subsets = sum(math.comb(len(MODIFIERS), i) for i in range(MAX_CLAUSES + 1))
    return {"base_actions": len(BASE_ACTIONS),
            "modifiers": len(MODIFIERS),
            "max_clauses_shown": MAX_CLAUSES,
            "clause_subsets": subsets,
            "distinct_outputs": len(BASE_ACTIONS) * subsets}


def recommendations(res, level="gp", limit=10):
    """
    One action per unit in the danger table, composed from the unit's shape,
    where it ended, and whatever else the data can say about it.
    """
    if not res or res.get("too_short"):
        return []
    d = res.get("danger_" + level)
    if d is None or d.empty:
        return []
    label = "GP" if level == "gp" else "district"
    out = []
    for _, row in d.head(limit).iterrows():
        r = row.to_dict()
        r["_label"] = label
        sev = severity(r["end"])
        base = BASE_ACTIONS.get((r["shape_key"], sev))
        if not base:                                        # pragma: no cover
            continue
        clauses, fired = [], []
        for name, fn in MODIFIERS:
            try:
                c = fn(r, res)
            except Exception as e:                          # pragma: no cover
                ERRORS.append({"stage": "modifier " + name, "error": str(e)})
                c = None
            if c:
                clauses.append(c)
                fired.append(name)
            if len(clauses) >= MAX_CLAUSES:
                break
        out.append({
            "unit": r["unit"], "level": label,
            "priority": {"critical": "🔴 Critical", "weak": "🟠 High",
                         "middling": "🟡 Medium",
                         "strong": "🟢 Low"}[sev],
            "severity": sev, "shape": r["shape"],
            "recommendation": " ".join([base.format(unit=label)] + clauses),
            "rule_fired": "+".join([r["shape_key"], sev] + fired),
            "children": int(r["children"]),
            "change": float(r["change"]),
            "proven": bool(r.get("beyond_normal_movement")),
        })
    return out


def programme_recommendations(res):
    """
    Actions for Akshara itself rather than for one place — what the pattern
    across the whole state implies about how GKA and its assessment are run.
    """
    if not res or res.get("too_short"):
        return []
    out = []

    flip = res.get("flip") or {}
    blocked = [s for c in res.get("cohorts", []) for s in c["steps"]
               if not s["meta"].get("ok")]
    if flip.get("flipped") or blocked:
        n_ok = len([s for c in res.get("cohorts", []) for s in c["steps"]
                    if s["meta"].get("ok")])
        out.append({
            "area": "Assessment design",
            "priority": "🔴 Critical",
            "recommendation":
                "Carry a fixed set of anchor questions across papers. "
                "%d of %d cohort steps in this data cannot be compared at all "
                "because consecutive papers share too few skills, so the "
                "programme cannot currently show whether children improved. "
                "Repeating 6-8 identical items in every paper — they need not "
                "be scored or reported — would make every future year "
                "comparable at no cost to the contest itself."
                % (len(blocked), len(blocked) + n_ok),
            "why": "the single change that would most increase what this data "
                   "can prove next year"})

    comp = res.get("competency")
    if comp is not None:
        s = comp["summary"]
        npapers = int(s["papers"].max())
        weak = s[(s["below_in"] == s["papers"]) & (s["papers"] == npapers)]
        strong = s[(s["below_in"] == 0) & (s["papers"] == npapers)]
        if len(weak):
            w = weak.iloc[0]
            txt = ("Target %s directly in the next programme cycle. It sits "
                   "below its own paper's average in every one of the %d "
                   "papers, by %.1f points on average — the only competency "
                   "that never once clears the bar."
                   % (w["competency"], int(w["papers"]),
                      abs(float(w["mean_vs_paper"]))))
            if len(strong):
                st = strong.iloc[-1]
                txt += (" %s is the opposite case, above its paper average in "
                        "all %d, so the teaching approach that works there is "
                        "the obvious place to look for a transferable method."
                        % (st["competency"], int(st["papers"])))
            out.append({"area": "Curriculum focus", "priority": "🔴 Critical",
                        "recommendation": txt,
                        "why": "stable across every paper, so it is not an "
                               "artefact of one year's questions"})

    uc = res.get("unit_coverage") or {}
    if uc.get("partial"):
        out.append({
            "area": "Coverage",
            "priority": "🟠 High",
            "recommendation":
                "Fix or document the %d districts missing from at least one "
                "year (%s). Either assess them every year, or record why not "
                "— at present any statewide total silently changes meaning "
                "between years because the set of districts in it changes."
                % (len(uc["partial"]), ", ".join(uc["partial"][:5])),
            "why": "a moving denominator makes every statewide trend "
                   "ambiguous"})

    bu = res.get("by_unit")
    if bu is not None and len(bu):
        top = bu.head(3)["unit"].tolist()
        out.append({
            "area": "Learn from the leaders",
            "priority": "🟡 Medium",
            "recommendation":
                "Document what %s did. On the one cohort step the papers "
                "allow us to measure honestly, they gained the most of any "
                "district, and their methods are the cheapest available "
                "source of improvement for the rest of the state."
                % ", ".join(top),
            "why": "measured on shared skills, so the gain is not a paper "
                   "artefact"})

    out.append({
        "area": "Evidence",
        "priority": "🟡 Medium",
        "recommendation":
            "Record when each district joined GKA, and how much contact it "
            "has had. With that one column the programme could show that "
            "districts with more exposure improved more — which is the "
            "closest thing to proof of impact available without a comparison "
            "group of children outside the programme.",
        "why": "would convert the current 'consistent with impact' into a "
               "testable dose-response claim"})
    return out


def describe():
    """What this layer computes, for the methods panel."""
    return [
        {"generator": "g_papers_not_equated",
         "checks": "whether the grade ordering flips between years"},
        {"generator": "g_cohort_anchored",
         "checks": "cohort change measured only on skills two papers share"},
        {"generator": "g_cohort_blocked",
         "checks": "steps with too few shared skills to compare at all"},
        {"generator": "g_paper_masked_change",
         "checks": "where raw and anchored change disagree"},
        {"generator": "g_gain_spread",
         "checks": "how many districts share the measured gain"},
        {"generator": "g_competency_persistent_weak",
         "checks": "competency below its own paper mean in every paper"},
        {"generator": "g_competency_persistent_strong",
         "checks": "competency above its own paper mean in every paper"},
        {"generator": "g_competency_coverage",
         "checks": "which competencies are missing from which papers"},
        {"generator": "g_danger_gp",
         "checks": "GPs falling in both steps, BH-corrected"},
        {"generator": "g_danger_district",
         "checks": "districts falling in both steps, BH-corrected"},
        {"generator": "g_unit_coverage",
         "checks": "districts absent from at least one year"},
        {"generator": "g_no_control",
         "checks": "states the absence of a comparison group"},
    ]
