"""
Layer 6 — cross-dataset analysis: assessment results x district context data.

The question the handbook opens with is "Bagalkot reports 68% — is that good?".
You cannot answer it from the assessment alone; you need to know what Bagalkot
started with. That is what this layer is for: join district-level context
(income, literacy, teachers, infrastructure) onto the assessment outcome and
ask which of it actually relates to learning.

Everything here is deterministic arithmetic — no model, no LLM. The p-values,
confidence intervals and multiple-comparison correction are implemented in
plain Python/numpy for the same reason stats_tests.py is: the reproducibility
rule wants byte-identical output, and a grader should be able to read the
formula rather than trust a library.

WHY THIS IS MORE CAUTIOUS THAN A PLAIN CORRELATION TABLE
--------------------------------------------------------
A correlation table over ~25 context variables invites three mistakes, and all
three appear in a naive reading:

  1. TAUTOLOGY. "Average correct answers correlates 0.99 with overall
     percentage" is not a finding — one is the other divided by the number of
     questions. `derived_variables()` detects and quarantines these.

  2. MULTIPLE COMPARISONS. Test 25 variables at p<0.05 and you expect about
     one "significant" result from pure chance. `bh_fdr()` corrects for it.

  3. NO VARIANCE. If the outcome is nearly identical in every district there
     is nothing to explain, and every correlation is noise dressed as signal.
     `variance_check()` refuses to let that pass silently.
"""
import math

import numpy as np
import pandas as pd

ALPHA = 0.05
# Below this coefficient of variation the outcome is effectively constant
# across units and no correlation against it can mean anything.
MIN_OUTCOME_CV = 1.0          # percent
# R^2 above which one variable is treated as an algebraic restatement of another
DERIVED_R2 = 0.999


# ------------------------------------------------------------------ stats
def _betacf(a, b, x, itmax=300, eps=3e-14):
    """Continued fraction for the incomplete beta (Numerical Recipes)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-300:
            d = 1e-300
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-300:
            d = 1e-300
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t, dof):
    """P(|T| >= |t|) for Student's t with `dof` degrees of freedom."""
    if dof <= 0 or not np.isfinite(t):
        return 1.0
    return float(_betai(dof / 2.0, 0.5, dof / (dof + t * t)))


def pearson(x, y):
    """(r, p, n) — Pearson correlation, pairwise-complete."""
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    ok = x.notna() & y.notna()
    x, y = x[ok].to_numpy(float), y[ok].to_numpy(float)
    n = len(x)
    if n < 3 or x.std() == 0 or y.std() == 0:
        return float("nan"), 1.0, n
    r = float(np.corrcoef(x, y)[0, 1])
    r = max(min(r, 1.0), -1.0)
    if abs(r) >= 1.0:
        return r, 0.0, n
    t = r * math.sqrt((n - 2) / (1.0 - r * r))
    return r, t_two_sided_p(t, n - 2), n


def spearman(x, y):
    """(rho, p, n) — Pearson on ranks; robust to outliers and non-linearity."""
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    ok = x.notna() & y.notna()
    return pearson(x[ok].rank(), y[ok].rank())


def fisher_ci(r, n, conf=0.95):
    """Confidence interval for r via the Fisher z transform."""
    if not np.isfinite(r) or n < 4 or abs(r) >= 1.0:
        return (float("nan"), float("nan"))
    z = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    crit = 1.959963984540054 if abs(conf - 0.95) < 1e-9 else 2.5758293035489004
    return (math.tanh(z - crit * se), math.tanh(z + crit * se))


def min_detectable_r(n, alpha=ALPHA):
    """
    Smallest |r| that would reach significance at this sample size.

    With 31 districts anything under about 0.35 cannot be distinguished from
    zero, so quoting r = 0.30 as "the strongest relationship" is quoting noise.
    """
    if n < 4:
        return float("nan")
    lo, hi = 0.0, 1.0
    for _ in range(200):                      # bisection on the p-value
        mid = (lo + hi) / 2.0
        t = mid * math.sqrt((n - 2) / max(1.0 - mid * mid, 1e-15))
        if t_two_sided_p(t, n - 2) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def bh_fdr(pvals):
    """
    Benjamini-Hochberg adjusted p-values (false discovery rate).

    Testing 25 context variables at p<0.05 yields about one false positive by
    chance alone. Without this, the single "significant" correlation in a wide
    table is usually that false positive.
    """
    p = np.asarray([1.0 if (v is None or not np.isfinite(v)) else float(v)
                    for v in pvals], dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]   # enforce monotonicity
    out = np.empty(n, dtype=float)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def strength_label(r):
    a = abs(r)
    if not np.isfinite(a):
        return "undefined"
    if a >= 0.7:
        return "strong"
    if a >= 0.5:
        return "moderate"
    if a >= 0.3:
        return "weak"
    return "negligible"


# --------------------------------------------------------- district names
# Karnataka renamed several districts, and every source spells the rest a
# little differently ("Bagalkot" vs "Bagalkote"). A plain join silently drops
# the mismatches, so the analysis quietly runs on a subset — and the map draws
# blank. These are the official renames plus common transliterations.
DISTRICT_ALIASES = {
    "bangalore": "Bengaluru Urban", "bangalore urban": "Bengaluru Urban",
    "bangalore rural": "Bengaluru Rural", "bengaluru": "Bengaluru Urban",
    "belgaum": "Belagavi", "bellary": "Ballari", "bijapur": "Vijayapura",
    "gulbarga": "Kalaburagi", "mysore": "Mysuru", "mangalore": "Dakshina Kannada",
    "shimoga": "Shivamogga", "tumkur": "Tumakuru", "chikmagalur": "Chikkamagaluru",
    "hospet": "Vijayanagara", "north kanara": "Uttara Kannada",
    "south kanara": "Dakshina Kannada",
    # spellings seen in the district-context workbook. Without these the
    # assessment file's canonical "Kalaburagi"/"Yadgir" fail to reach the
    # context file's "kalaburgi"/"yadagiri" — the vowels differ in the middle
    # of the word, so neither the loose key nor difflib bridges them.
    "kalaburgi": "Kalaburagi", "kalburgi": "Kalaburagi",
    "yadagiri": "Yadgir", "yadgiri": "Yadgir", "yadagir": "Yadgir",
    "bagalkot": "Bagalkote", "vijayanagar": "Vijayanagara",
    "chikkamagalur": "Chikkamagaluru", "chamarajanagar": "Chamarajanagara",
}


def _canon(name):
    """The official spelling this name refers to, if the table knows it."""
    n = str(name).strip()
    return DISTRICT_ALIASES.get(n.lower(), n)


def _dkey(name):
    """Loose comparison key: lowercase, letters only, trailing vowel dropped."""
    t = "".join(ch for ch in str(name).lower() if ch.isalpha() or ch == " ")
    t = " ".join(t.split())
    return t[:-1] if t.endswith(("a", "e")) else t


def align_districts(names, targets, cutoff=0.88):
    """
    Map each name onto the closest official spelling in `targets`.

    Exact match first, then the alias table, then a conservative fuzzy match.
    Every substitution is reported — a rename you cannot see is a rename you
    cannot check, and quietly merging two real districts would be far worse
    than leaving one unmatched.
    """
    import difflib
    tgt = [str(t).strip() for t in targets]
    exact = {t.lower(): t for t in tgt}
    keyed = {}
    for t in tgt:
        keyed.setdefault(_dkey(t), t)
    # The alias table is one-way (variant -> official), but EITHER side can be
    # the variant: the assessment file may be canonical while the context file
    # is not. Canonicalising both sides makes the match direction-independent.
    canon = {}
    for t in tgt:
        canon.setdefault(_dkey(_canon(t)), t)

    mapping, changed, unmatched = {}, [], []
    for raw in names:
        n = str(raw).strip()
        if n in tgt:
            mapping[raw] = n
            continue
        if n.lower() in exact:
            mapping[raw] = exact[n.lower()]
            changed.append((raw, exact[n.lower()], "case"))
            continue
        alias = DISTRICT_ALIASES.get(n.lower())
        if alias and alias in tgt:
            mapping[raw] = alias
            changed.append((raw, alias, "known rename"))
            continue
        k = _dkey(n)
        if k in keyed:
            mapping[raw] = keyed[k]
            changed.append((raw, keyed[k], "spelling variant"))
            continue
        ck = _dkey(_canon(n))
        if ck in canon:
            mapping[raw] = canon[ck]
            changed.append((raw, canon[ck], "known rename"))
            continue
        close = difflib.get_close_matches(n, tgt, n=1, cutoff=cutoff)
        if close:
            mapping[raw] = close[0]
            changed.append((raw, close[0], f"fuzzy ≥{cutoff}"))
        else:
            mapping[raw] = n
            unmatched.append(n)
    return mapping, {"changed": changed, "unmatched": unmatched,
                     "matched": len(names) - len(unmatched), "total": len(names)}


def match_district(name, candidates):
    """
    Resolve `name` to whichever string in `candidates` it refers to, using the
    SAME alias table + loose-key rules align_districts() uses for the join —
    not a plain case-insensitive equality.

    Why this exists: prepare()'s join REPLACES the assessment file's district
    spelling with the context file's before merging, so a row the join filed
    under "bagalkot" or "kalaburgi" no longer matches "Bagalkote" or
    "Kalaburagi" — the assessment file's OWN canonical names — under a naive
    `.lower()` check. That check silently failed for 4 of 28 districts here
    (Bagalkote, Kalaburagi, Vijayanagara, Yadgir), each losing its peer
    benchmark, ctx_over/ctx_under clause, and resource-equity section with no
    error — the caller just saw zero context-driven output for that district.
    Two of the four (Kalaburagi/kalaburgi, Yadgir/yadagiri) differ in the
    MIDDLE of the word, so even the loose trailing-vowel key alone would not
    have caught them — only the alias table does, which is why this reuses
    align_districts() rather than a plain _dkey() comparison.
    """
    amap, _ = align_districts([name], candidates)
    return amap.get(name)


# ------------------------------------------------------------------- join
def join(primary, secondary, key="District"):
    """
    Merge assessment results with district context on district name.

    Name mismatches are the usual failure and they fail quietly — a district
    that does not match simply vanishes from the analysis — so the unmatched
    names are returned rather than dropped in silence.
    """
    p, s = primary.copy(), secondary.copy()
    for f in (p, s):
        f[key] = f[key].astype(str).str.strip()
    merged = p.merge(s, on=key, how="inner", suffixes=("", "_sec"))
    report = {
        "primary_rows": len(p), "secondary_rows": len(s),
        "matched": len(merged),
        "only_primary": sorted(set(p[key]) - set(s[key])),
        "only_secondary": sorted(set(s[key]) - set(p[key])),
    }
    report["ok"] = (not report["only_primary"]) and (not report["only_secondary"])
    return merged, report


# ------------------------------------------------- tautology / variance guards
def _linear_r2(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return 0.0
    r = np.corrcoef(x[ok], y[ok])[0, 1]
    return float(r * r) if np.isfinite(r) else 0.0


def _granularity(s):
    """Smallest step actually present in a column — its rounding unit."""
    v = np.unique(pd.to_numeric(pd.Series(s), errors="coerce").dropna().to_numpy(float))
    if len(v) < 2:
        return 0.0
    d = np.diff(v)
    d = d[d > 0]
    return float(d.min()) if len(d) else 0.0


def _rescaled_within_rounding(x, y):
    """
    Is y just a*x + b, to within the precision the data is stored at?

    Published tables are rounded — "average correct answers" is "overall
    percentage" / 5, but with both rounded to 2dp the fit only reaches
    R² = 0.984, under any sane R² threshold. Comparing the fit residuals
    against the column's own rounding step catches it where R² alone cannot.
    """
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    ok = x.notna() & y.notna()
    if ok.sum() < 4:
        return False
    xs, ys = x[ok].to_numpy(float), y[ok].to_numpy(float)
    if xs.std() == 0 or ys.std() == 0:
        return False
    a, b = np.polyfit(xs, ys, 1)
    resid = np.abs(ys - (a * xs + b))
    step = max(_granularity(ys), _granularity(xs) * abs(a))
    # allow a couple of rounding units of slack, and require a tight fit anyway
    return bool(step > 0 and resid.max() <= 3.0 * step
                and _linear_r2(xs, ys) >= 0.95)


def derived_variables(df, outcome, candidates):
    """
    Columns that are algebraic restatements of the outcome, not explanations.

    Three shapes are caught:
      * a rescaling — "average correct answers" is "overall percentage" / 5
      * the same rescaling hidden by rounding in the published table
      * a component — the outcome is the mean of the question-accuracy columns
    Reporting any of them as a discovered relationship is circular.
    """
    y = pd.to_numeric(df[outcome], errors="coerce")
    flagged = {}
    for c in candidates:
        if c == outcome:
            continue
        if _linear_r2(df[c], y) >= DERIVED_R2:
            flagged[c] = "a linear rescaling of the outcome"
        elif _rescaled_within_rounding(df[c], y):
            flagged[c] = ("a linear rescaling of the outcome "
                          "(exact to the rounding of the published figures)")
    # the outcome reconstructed from a family of columns (e.g. Q1..Q20)
    fams = {}
    for c in candidates:
        base = "".join(ch for ch in str(c) if not ch.isdigit())
        fams.setdefault(base, []).append(c)
    for base, cols in fams.items():
        if len(cols) < 3:
            continue
        block = df[cols].apply(pd.to_numeric, errors="coerce")
        if _linear_r2(block.mean(axis=1), y) >= DERIVED_R2:
            for c in cols:
                flagged.setdefault(
                    c, f"a component of the outcome (its family of "
                       f"{len(cols)} columns averages to it)")
    return flagged


def variance_check(df, outcome):
    """Is there anything to explain? A flat outcome makes every r meaningless."""
    y = pd.to_numeric(df[outcome], errors="coerce").dropna()
    if len(y) < 3 or y.mean() == 0:
        return {"ok": False, "reason": "not enough usable values"}
    cv = float(100.0 * y.std() / abs(y.mean()))
    return {
        "ok": cv >= MIN_OUTCOME_CV, "cv": cv, "sd": float(y.std()),
        "spread": float(y.max() - y.min()),
        "min": float(y.min()), "max": float(y.max()), "n": int(len(y)),
        "reason": (
            f"{outcome} ranges only {y.min():.2f}–{y.max():.2f} across "
            f"{len(y)} units (spread {y.max()-y.min():.2f}, "
            f"CV {cv:.2f}%). With essentially no variation to explain, any "
            f"correlation against it is noise."
            if cv < MIN_OUTCOME_CV else "outcome varies enough to model"),
    }


# ------------------------------------------------------------- correlations
def correlate(df, outcome, predictors=None, alpha=ALPHA, exclude_derived=True):
    """
    Correlation table with everything needed to read it honestly.

    Returns one row per predictor: Pearson r, Spearman rho, n, raw p,
    BH-adjusted p, 95% CI, strength label, and a verdict that already accounts
    for multiple comparisons.
    """
    if predictors is None:
        predictors = [c for c in df.columns
                      if c != outcome and pd.api.types.is_numeric_dtype(df[c])]
    derived = derived_variables(df, outcome, predictors)

    rows = []
    for c in predictors:
        r, p, n = pearson(df[c], df[outcome])
        rho, p_s, _ = spearman(df[c], df[outcome])
        lo, hi = fisher_ci(r, n)
        rows.append({
            "variable": c, "r": r, "spearman": rho, "n": n,
            "p_raw": p, "ci_low": lo, "ci_high": hi,
            "strength": strength_label(r),
            "derived": c in derived, "derived_why": derived.get(c, ""),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Correct only across the variables that are genuinely being tested;
    # including tautologies would deflate everyone else's adjusted p.
    testable = ~out["derived"] if exclude_derived else pd.Series(True, out.index)
    out["p_adj"] = np.nan
    if testable.any():
        out.loc[testable, "p_adj"] = bh_fdr(out.loc[testable, "p_raw"].tolist())

    crit = min_detectable_r(int(out["n"].max()) if len(out) else 0, alpha)
    out["verdict"] = np.where(
        out["derived"], "excluded — not an independent variable",
        np.where(out["p_adj"] < alpha, "significant after FDR correction",
                 np.where(out["p_raw"] < alpha,
                          "nominally significant, fails FDR correction",
                          "not distinguishable from zero")))
    out.attrs["min_detectable_r"] = crit
    out.attrs["alpha"] = alpha
    out.attrs["n_tested"] = int(testable.sum())
    return out.sort_values("r", key=lambda s: s.abs(), ascending=False,
                           ignore_index=True)


# --------------------------------------------------- peer-adjusted performance
def peer_adjusted(df, outcome, controls, key="District"):
    """
    "Is 68% good?" — answered by comparing a district against districts with
    similar circumstances rather than against the state average.

    Fits outcome ~ controls by least squares and reports the residual: how far
    above or below its own context a district actually lands. A poor district
    scoring slightly below average may be over-performing; a rich one at the
    average may be under-performing. This is the cross-dataset question that a
    single-table analysis cannot ask.
    """
    use = [c for c in controls if c in df.columns
           and pd.api.types.is_numeric_dtype(df[c])]
    if not use:
        return None, {"ok": False, "reason": "no usable control variables"}

    d = df[[key, outcome] + use].copy()
    for c in [outcome] + use:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()
    if len(d) < len(use) + 3:
        return None, {"ok": False,
                      "reason": f"only {len(d)} complete rows for "
                                f"{len(use)} controls — not enough to fit"}

    X = d[use].to_numpy(float)
    # standardise so coefficients are comparable and the fit is well conditioned
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    y = d[outcome].to_numpy(float)

    beta, *_ = np.linalg.lstsq(Xs, y, rcond=None)
    pred = Xs @ beta
    resid = y - pred
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    k = len(use)
    adj_r2 = (1 - (1 - r2) * (len(d) - 1) / (len(d) - k - 1)
              if len(d) > k + 1 else float("nan"))

    out = pd.DataFrame({
        key: d[key].values,
        "actual": y,
        "expected_given_context": pred,
        "over_under": resid,
    }).sort_values("over_under", ascending=False, ignore_index=True)

    # A residual is only interesting if it is large relative to the outcome's
    # own spread. Ranking "over-performers" by 0.15 points on a measure whose
    # entire range is 0.40 points is ranking rounding noise, however good the
    # R² looks — so the variance of the outcome gates this, not just the fit.
    vc = variance_check(d, outcome)
    info = {
        "ok": True, "n": int(len(d)), "controls": use,
        "r2": r2, "adj_r2": adj_r2,
        "resid_sd": float(resid.std()),
        "outcome_spread": vc.get("spread", float("nan")),
        "outcome_cv": vc.get("cv", float("nan")),
        "outcome_varies": bool(vc.get("ok", False)),
        "coefficients": dict(zip(use, beta[1:].round(4))),
        # A model that explains nothing means context does not predict the
        # outcome here — worth saying out loud rather than ranking residuals
        # that are pure noise.
        "fit_explains_anything": bool(np.isfinite(adj_r2) and adj_r2 > 0.05),
    }
    info["usable"] = bool(info["outcome_varies"] and info["fit_explains_anything"])
    if not info["outcome_varies"]:
        info["warning"] = (
            f"{outcome} spans only {vc.get('spread', float('nan')):.2f} points "
            f"across these {len(d)} districts (CV {vc.get('cv', float('nan')):.2f}%). "
            f"The over/under-performance column below is smaller than the "
            f"measurement itself — do not rank districts on it.")
    elif not info["fit_explains_anything"]:
        info["warning"] = (
            f"District context explains almost none of the variation "
            f"(adjusted R² = {adj_r2:.3f}). Residuals here are mostly noise, "
            f"not hidden over-performance.")
    return out, info


def redundancy(df, predictors, alpha=ALPHA, min_r=0.7):
    """
    Which context variables are really measuring the same thing?

    Two uses. First it is a genuine finding: "teacher count, household count
    and urban literacy all track district SIZE, not district quality" is worth
    saying. Second it is a warning — feeding eight near-identical variables
    into one regression double-counts whatever they share, so the peer-adjusted
    model should use one of each cluster, not all of them.

    Same significance machinery as everything else: raw p, BH-corrected p, and
    a size threshold, so a pair has to be both strong AND survive correction.
    """
    cols = [c for c in predictors if c in df.columns
            and pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r, p, n = pearson(df[a], df[b])
            rows.append({"a": a, "b": b, "r": r, "p_raw": p, "n": n})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_adj"] = bh_fdr(out["p_raw"].tolist())
    out["redundant"] = (out["p_adj"] < alpha) & (out["r"].abs() >= min_r)
    return out.sort_values("r", key=lambda s: s.abs(), ascending=False,
                           ignore_index=True)


# ----------------------------------------------------------------- sentences
def verbalize(table, outcome, var_info=None, limit=6):
    """Deterministic sentences for the correlation table. Templates, no model."""
    if table is None or table.empty:
        return ["No correlations could be computed."]
    crit = table.attrs.get("min_detectable_r", float("nan"))
    lines = []

    sig = table[table["verdict"] == "significant after FDR correction"]
    nominal = table[table["verdict"] == "nominally significant, fails FDR correction"]
    derived = table[table["derived"]]

    lines.append(
        f"Tested {table.attrs.get('n_tested', len(table))} context variables "
        f"against {outcome} across {int(table['n'].max())} districts. "
        f"At this sample size only |r| ≥ {crit:.2f} can be told apart from "
        f"zero, so anything weaker is not evidence of a relationship.")

    if len(derived):
        names = ", ".join(derived["variable"].head(4))
        lines.append(
            f"{len(derived)} variable(s) were excluded as restatements of the "
            f"outcome rather than explanations of it ({names}"
            f"{'…' if len(derived) > 4 else ''}). Correlating a measure with "
            f"itself rescaled is guaranteed, not informative.")

    if len(sig):
        for r in sig.head(limit).itertuples():
            direction = "higher" if r.r > 0 else "lower"
            lines.append(
                f"**{r.variable}**: districts with more of it tend to score "
                f"{direction} (r = {r.r:+.2f}, 95% CI {r.ci_low:+.2f} to "
                f"{r.ci_high:+.2f}, adjusted p = {r.p_adj:.3f}, n = {r.n}). "
                f"{strength_label(r.r).capitalize()} relationship.")
    else:
        lines.append(
            "**No context variable survives correction for multiple testing.** "
            f"{len(nominal)} looked significant on their own, but testing "
            f"{table.attrs.get('n_tested', len(table))} variables at once means "
            f"about {table.attrs.get('n_tested', 0) * ALPHA:.1f} would do so by "
            f"chance. None can be reported as a real relationship.")

    return lines


def describe():
    """What this layer does, for the methodology panel."""
    return [
        {"step": "Join", "what": "assessment results + district context on district name",
         "guard": "unmatched names are listed, never silently dropped"},
        {"step": "Variance check", "what": "does the outcome vary at all across districts?",
         "guard": f"refuses to interpret correlations when CV < {MIN_OUTCOME_CV}%"},
        {"step": "Tautology filter", "what": "drop variables algebraically derived from the outcome",
         "guard": f"linear R² ≥ {DERIVED_R2}, plus component-family detection"},
        {"step": "Correlation", "what": "Pearson + Spearman, exact t-test p-values",
         "guard": "95% CI via Fisher z; Spearman cross-checks non-linearity"},
        {"step": "Multiple comparisons", "what": "Benjamini–Hochberg FDR across all tested variables",
         "guard": "stops the ~1-in-20 false positive being reported as a finding"},
        {"step": "Power", "what": "minimum detectable |r| at this sample size",
         "guard": "weak correlations are labelled unreadable rather than 'weak but present'"},
        {"step": "Peer adjustment", "what": "outcome ~ context, residual = over/under-performance",
         "guard": "reports adjusted R²; refuses to rank residuals if context explains nothing"},
    ]


# ------------------------------------------------------- make any file usable
def to_unit_level(df, unit, keep=None):
    """
    Collapse many rows per unit into one row per unit.

    The engine below correlates one row against another row, so it needs one
    row per district (or block, or school). Raw assessment data is one row per
    CHILD, so it has to be averaged up first. Numeric columns become their
    mean; a `n_rows` column records how many records each unit is built from,
    because a district summarised from 40 children is not the same evidence as
    one summarised from 40,000.
    """
    if unit not in df.columns:
        return None, f"no column called '{unit}' to group by"
    d = df.copy()
    d[unit] = d[unit].astype(str).str.strip()
    num = [c for c in d.columns
           if c != unit and pd.api.types.is_numeric_dtype(d[c])
           and (keep is None or c in keep)]
    if not num:
        return None, "no numeric columns to summarise"
    out = d.groupby(unit, as_index=False)[num].mean()
    out["n_rows"] = d.groupby(unit).size().reindex(out[unit]).to_numpy()
    return out, (f"{len(d):,} rows collapsed to {len(out)} {unit} "
                 f"(numeric columns averaged)")


def combine(frames, unit="District"):
    """
    Join any number of tables on the unit column, aggregating first if needed.

    Each frame may be unit-level already or raw record-level; either works.
    Returns the merged table plus a per-frame note, so it is always visible
    what was aggregated and what was joined.
    """
    prepared, notes = [], []
    for i, f in enumerate(frames):
        if f is None or unit not in f.columns:
            notes.append(f"frame {i+1}: skipped — no '{unit}' column")
            continue
        if f[unit].duplicated().any():
            agg, note = to_unit_level(f, unit)
            if agg is None:
                notes.append(f"frame {i+1}: skipped — {note}")
                continue
            notes.append(f"frame {i+1}: {note}")
            prepared.append(agg)
        else:
            notes.append(f"frame {i+1}: already one row per {unit} "
                         f"({len(f)} rows)")
            prepared.append(f.assign(**{unit: f[unit].astype(str).str.strip()}))

    if not prepared:
        return None, notes
    merged = prepared[0]
    for nxt in prepared[1:]:
        before = len(merged)
        left, right = set(merged[unit]), set(nxt[unit])
        merged = merged.merge(nxt, on=unit, how="inner", suffixes=("", "_dup"))
        notes.append(f"joined -> {len(merged)} {unit} kept "
                     f"(was {before} before this join)")
        # A name mismatch loses rows silently and the analysis then runs on
        # whatever happened to match, so say exactly what fell out. Losing
        # everything usually means two different naming conventions, not two
        # datasets about different places.
        lost_l, lost_r = sorted(left - right), sorted(right - left)
        if lost_l or lost_r:
            notes.append(
                f"⚠️ {len(lost_l)} {unit} in the first table and "
                f"{len(lost_r)} in the second did not match by name. "
                f"e.g. '{(lost_l or ['-'])[0]}' vs '{(lost_r or ['-'])[0]}'")
        if len(merged) == 0:
            notes.append(
                f"🚫 NOTHING matched. The two files name their {unit}s "
                f"differently — fix the spellings (or map them) before "
                f"anything below can be computed.")
    merged = merged[[c for c in merged.columns if not c.endswith("_dup")]]
    return merged, notes


# --------------------------------------------------------------------- report
def analyse(merged, outcome, controls=None, key="District"):
    """
    One call that runs the whole layer and returns everything the UI needs.

    Deliberately returns the guards alongside the numbers: a caller that only
    reads `table` and ignores `variance` will publish noise.
    """
    preds = [c for c in merged.columns
             if c != outcome and c != key
             and pd.api.types.is_numeric_dtype(merged[c])]
    table = correlate(merged, outcome, preds)
    var = variance_check(merged, outcome)
    if controls is None:
        controls = [c for c in preds
                    if c in table[~table["derived"]]["variable"].tolist()][:8]
    resid, fit = peer_adjusted(merged, outcome, controls, key=key)
    tested_cols = table[~table["derived"]]["variable"].tolist()
    return {
        "outcome": outcome,
        "variance": var,
        "table": table,
        "redundancy": redundancy(merged, tested_cols),
        "derived": table[table["derived"]]["variable"].tolist(),
        "residuals": resid,
        "fit": fit,
        "sentences": verbalize(table, outcome),
        "method": describe(),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python secondary.py <merged.xlsx> [secondary.xlsx] "
              "[outcome column]")
        raise SystemExit(1)

    merged_df = pd.read_excel(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2].lower().endswith((".xlsx", ".csv")):
        sec_df = (pd.read_csv(sys.argv[2]) if sys.argv[2].endswith(".csv")
                  else pd.read_excel(sys.argv[2]))
        merged_df, jr = join(merged_df, sec_df)
        print(f"joined: {jr['matched']} districts matched, clean={jr['ok']}")
        if jr["only_primary"]:
            print("  unmatched in primary  :", jr["only_primary"])
        if jr["only_secondary"]:
            print("  unmatched in secondary:", jr["only_secondary"])

    outcome_col = sys.argv[-1] if sys.argv[-1] in merged_df.columns else None
    if outcome_col is None:
        for guess in ("Overall Percentage", "Overall (%)", "below_pct"):
            if guess in merged_df.columns:
                outcome_col = guess
                break
    if outcome_col is None:
        print("could not find an outcome column; pass it as the last argument")
        raise SystemExit(1)

    res = analyse(merged_df, outcome_col)
    print(f"\noutcome: {outcome_col}")
    print(f"varies enough to model: {res['variance']['ok']}")
    print(f"  {res['variance']['reason']}\n")
    cols = ["variable", "r", "spearman", "n", "p_raw", "p_adj", "verdict"]
    print(res["table"][~res["table"]["derived"]][cols]
          .head(15).to_string(index=False))
    print("\nexcluded as restatements of the outcome:")
    print(" ", ", ".join(res["derived"]) or "(none)")
    print("\n" + "\n".join(" • " + s for s in res["sentences"]))
    if res["fit"] and res["fit"].get("warning"):
        print("\nPEER ADJUSTMENT WARNING:", res["fit"]["warning"])
