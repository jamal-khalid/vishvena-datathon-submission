"""
Layer 4b — CROSS-DATASET insight generators (no LLM).

`insights.py` answers questions the assessment file can answer on its own:
which competency is weakest, which block is worst, is there a gender gap.
None of those can answer the question the handbook actually opens with —

    "Bagalkot reports 68%. Is that good?"

You cannot know without knowing what Bagalkot started with. This module joins
district context (income, literacy, teachers, infrastructure) onto the
assessment result and produces findings of a kind the primary file cannot
support:

  * how a district performs against districts in SIMILAR CIRCUMSTANCES,
    rather than against the state average
  * how far its rank moves once circumstances are accounted for
  * whether district context explains anything at all — reported as a finding
    in its own right, because "we checked and it doesn't" is the honest and
    useful answer when it is true

Findings come back in exactly the shape `insights.py` uses, so the two merge
into one ranked list.

EVERYTHING HERE IS GATED ON THE GUARDS secondary.py ALREADY COMPUTES.
If the outcome barely varies across districts, or the context model explains
nothing, these generators stay silent rather than ranking noise. That matters:
on the merged file we were given, the outcome spans 0.40 points across all 31
districts, and every "insight" derivable from it is an artefact.
"""
import math
import pandas as pd

import secondary as S


MIN_DISTRICTS = 10          # below this, "similar districts" is meaningless
PEER_K = 5                  # how many nearest-context districts to compare with
Z95 = 1.959963985

# The context file is DISTRICT-level. Anything coarser (division, state) is
# reached by rolling districts up; anything finer (block, cluster) would copy
# one district's context onto all of its blocks, which manufactures rows
# without manufacturing information — 220 blocks carrying 31 distinct context
# values look like 220 observations to a correlation and are not.
LEVELS = ("District", "Division")

ERRORS = []

# Rolling up needs to know what a column MEANS. "Rural Male Literacy" in this
# file is a COUNT OF PEOPLE (180,920 to 1,203,960), so two districts combine by
# addition. "Per Capita Income" and "Student Teacher Ratio" are RATES, and
# adding them is nonsense — they combine by a size-weighted mean.
_RATE_HINTS = ("ratio", "rate", "per capita", "percent", "percentage", "%",
               "index", "average", "avg", "mean", "density", "score",
               "per 1000", "per capita")


def _is_rate(col):
    c = str(col).lower()
    return any(h in c for h in _RATE_HINTS)


def _weight_column(df):
    """Best available proxy for how big a district is, for weighting rates."""
    for cand in ("Total Household", "Student Enrolment", "Total Households",
                 "Population", "Students"):
        if cand in df.columns and pd.to_numeric(df[cand],
                                                errors="coerce").sum() > 0:
            return cand
    return None


def _ratio_weight(col, df, tol=0.02):
    """
    For a ratio A-per-B, the correct combined value is Σ A / Σ B — which is the
    mean of the per-unit ratios WEIGHTED BY B, the denominator, not by
    population. "Student Teacher Ratio" must combine weighted by teachers;
    weighting by households quietly gives the wrong number (measured: up to
    2.8 students per teacher out on this file).

    The denominator is found FROM THE DATA, not from the column name: search
    every pair of columns for the one whose quotient reproduces this column.
    Naming is unreliable — "Student Teacher Ratio" here is actually
    Student Enrolment / Primary Teachers Total, and a name-based guess picked
    'Primary+Upper Primary Teachers (Total)' instead.

    Returns the denominator column, or None if no pair explains the ratio.
    """
    r = pd.to_numeric(df[col], errors="coerce")
    if r.notna().sum() < 3 or float(r.abs().mean() or 0) == 0:
        return None
    scale = float(r.abs().mean())
    # Published tables are rounded. This file stores the ratio as whole
    # numbers, so even the exactly-correct quotient is off by up to 0.5 and a
    # tight tolerance would reject the true denominator.
    num = [c for c in df.columns
           if c != col and pd.api.types.is_numeric_dtype(df[c])]
    scored = []
    for a in num:
        va = pd.to_numeric(df[a], errors="coerce")
        for b in num:
            if a == b:
                continue
            vb = pd.to_numeric(df[b], errors="coerce").replace(0, float("nan"))
            err = float((va / vb - r).abs().mean())
            if err == err:                       # not NaN
                scored.append((err, b))
    if not scored:
        return None
    scored.sort()
    best_err, best = scored[0]
    # Published columns are rounded and sometimes computed from a slightly
    # different vintage of the same counts, so an EXACT match is too much to
    # ask. Accept when one pair is a decisive winner — comfortably closer than
    # any other and within a sane distance of the column itself. Otherwise the
    # denominator is genuinely unknown and we fall back to a size weight.
    runner = next((e for e, c in scored if c != best), float("inf"))
    decisive = best_err <= 0.35 * runner
    close = best_err <= 0.10 * scale
    return best if (decisive and close) else None


def rollup_secondary(sec, unit_of, key="District", level="Division"):
    """
    Aggregate a district-level context file up to `level`.

    `unit_of` maps a district name to its parent unit. Returns
    (rolled_frame, rules) where `rules` records how each column was combined,
    so the choice is auditable rather than hidden.
    """
    s = sec.copy()
    s["_unit"] = s[key].astype(str).map(unit_of)
    s = s[s["_unit"].notna()]
    if s.empty:
        return None, {}
    wcol = _weight_column(s)
    num = [c for c in s.columns
           if c not in (key, "_unit") and pd.api.types.is_numeric_dtype(s[c])]

    rules, out = {}, {}
    for c in num:
        wc = (_ratio_weight(c, s) or wcol) if _is_rate(c) else None
        if wc and c != wc:
            w = pd.to_numeric(s[wc], errors="coerce").fillna(0.0)
            v = pd.to_numeric(s[c], errors="coerce")
            tmp = s.assign(_v=v * w, _w=w.where(v.notna(), 0.0))
            gp = tmp.groupby("_unit")[["_v", "_w"]].sum()
            out[c] = gp["_v"] / gp["_w"].replace(0, float("nan"))
            rules[c] = f"weighted mean by {wc}"
        else:
            out[c] = s.groupby("_unit")[c].sum(min_count=1)
            rules[c] = "sum"
    rolled = pd.DataFrame(out).reset_index().rename(columns={"_unit": key})
    return rolled, rules


# ------------------------------------------------------------------ preparing
def prepare(agg, secondary_df, outcome_label="Below grade level (%)",
            key="District", level="District"):
    """
    Join the assessment outcome onto the district context file at `level`
    and run the secondary layer.

    level="District"  — 31 units. Correlations are attempted.
    level="Division"  — districts are rolled up. With 4 divisions the smallest
                        detectable |r| is 0.95, so correlations are NOT
                        attempted; the result carries `descriptive_only` and
                        the generators report comparisons instead of links.

    Everything downstream of the sidebar filters flows through `agg`, so
    slicing and dicing narrows this analysis exactly as it narrows the rest of
    the dashboard.

    Returns None when the two files cannot be joined, so callers degrade to
    primary-only insights instead of raising.
    """
    if agg is None or secondary_df is None or agg.empty or secondary_df.empty:
        return None
    if "district" not in agg.columns:
        return None
    level = level if level in LEVELS else "District"
    unit_col = "division" if level == "Division" else "district"
    if unit_col not in agg.columns:
        return None

    # size-weighted outcome — an unweighted mean would let a small block swing
    # a whole district
    prim = (agg.groupby(unit_col)
            .apply(lambda g: (g["below_pct"] * g["n"]).sum() / g["n"].sum(),
                   include_groups=False)
            .rename(outcome_label).reset_index()
            .rename(columns={unit_col: key}))
    prim = prim[prim[key].notna()]
    if len(prim) < 2:
        return None

    sec = secondary_df.copy()
    kcol = next((c for c in sec.columns
                 if str(c).strip().lower() in
                 ("district", "district name", "dist", "districts")), None)
    if kcol is None:
        return None
    sec = sec.rename(columns={kcol: key})

    rules = {}
    if level == "Division":
        # roll the district context up to the division each district sits in,
        # taken from the assessment file itself
        pairs = (agg[["district", "division"]].dropna().drop_duplicates()
                 .astype(str))
        by_key = {S._dkey(d): v for d, v in
                  zip(pairs["district"], pairs["division"])}
        sec, rules = rollup_secondary(
            sec, lambda d: by_key.get(S._dkey(str(d))), key=key, level=level)
        if sec is None or sec.empty:
            return None
    else:
        # Karnataka renamed several districts and every source spells the rest
        # a little differently; a plain join drops those silently.
        amap, rep = S.align_districts(prim[key].tolist(),
                                      sec[key].astype(str).tolist())
        prim[key] = prim[key].map(amap).fillna(prim[key])

    merged, jr = S.join(prim, sec, key=key)
    if merged is None or merged.empty or len(merged) < 2:
        return None

    n_units = len(merged)
    crit = S.min_detectable_r(n_units)
    # Below MIN_DISTRICTS a correlation cannot separate signal from noise at
    # any effect size a real programme could produce, so we do not compute one
    # and say why instead of publishing a number nobody should read.
    descriptive_only = n_units < MIN_DISTRICTS

    if descriptive_only:
        res = {"outcome": outcome_label,
               "variance": S.variance_check(merged, outcome_label),
               "table": None, "redundancy": None, "derived": [],
               "residuals": None,
               "fit": {"ok": False,
                       "reason": f"only {n_units} {level.lower()}s"},
               "sentences": [], "method": S.describe()}
    else:
        res = S.analyse(merged, outcome_label, key=key)

    res.update(merged=merged, key=key, level=level, unit_col=unit_col,
               n_units=n_units, min_detectable_r=crit,
               descriptive_only=descriptive_only, rollup_rules=rules,
               outcome_label=outcome_label)
    if level == "District":
        res["join"] = rep
    return res


def _resid_row(ctx, district):
    """This district's row in the peer-adjusted table, or None."""
    r = ctx.get("residuals")
    if r is None or r.empty:
        return None
    k = ctx["key"]
    hit = r[r[k].astype(str).str.strip().str.lower()
            == str(district).strip().lower()]
    return None if hit.empty else hit.iloc[0]


def _usable(ctx):
    """Is the context model trustworthy enough to make district claims from?"""
    fit = ctx.get("fit") or {}
    return bool(fit.get("ok") and fit.get("usable"))


def _loo_residual(ctx, district):
    """
    This district's residual from a model fitted WITHOUT it.

    A least-squares fit bends toward every point it contains, so the district
    with the most to say shrinks its own residual — measurably: a planted
    9-point over-performer came back at under 4 once four controls had been
    fitted through it. Refitting on the other n-1 districts asks the right
    question — "how far is it from what its PEERS predict?" — and is the same
    leave-one-out correction g_outlier_block uses.

    Returns (residual, se) or (None, None).
    """
    import numpy as np
    m, key = ctx.get("merged"), ctx["key"]
    out_col = ctx["outcome_label"]
    ctrls = [c for c in (ctx.get("fit") or {}).get("controls", [])
             if c in (m.columns if m is not None else [])]
    if m is None or not ctrls:
        return None, None
    d = m[[key, out_col] + ctrls].apply(
        lambda s: pd.to_numeric(s, errors="coerce") if s.name != key else s)
    d = d.dropna()
    mask = (d[key].astype(str).str.strip().str.lower()
            == str(district).strip().lower())
    if not mask.any() or len(d) < len(ctrls) + 4:
        return None, None

    X = d[ctrls].to_numpy(float)
    y = d[out_col].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    keep = ~mask.to_numpy()
    if keep.sum() < len(ctrls) + 2:
        return None, None
    beta, *_ = np.linalg.lstsq(Xs[keep], y[keep], rcond=None)
    i = int(np.flatnonzero(mask.to_numpy())[0])
    resid = float(y[i] - Xs[i] @ beta)
    # spread of the peers' own residuals under the same fit
    peer_resid = y[keep] - Xs[keep] @ beta
    s = float(np.std(peer_resid, ddof=len(ctrls) + 1)) if keep.sum() > len(ctrls) + 1 else float("nan")
    if not s or pd.isna(s):
        return None, None
    return resid, s * math.sqrt(1.0 + 1.0 / keep.sum())


def _evidence(effect, se):
    return max(0.0, abs(float(effect)) - Z95 * float(se))


def _find(category, score, text, evidence, source, **extra):
    out = {"category": category, "score": round(float(score), 2), "text": text,
           "evidence": evidence, "source": source}
    out.update(extra)
    return out


# ---------------------------------------------------------------- generators

def x_over_under(ctx, district, min_n=None):
    """
    Does this district beat, or fall short of, what its circumstances predict?

    The single most useful thing the cross-dataset join buys: a poor district
    scoring slightly below the state average may be over-performing, and a
    rich one at the average may be coasting. A raw ranking cannot tell them
    apart.
    """
    if not _usable(ctx):
        return []
    row = _resid_row(ctx, district)
    if row is None:
        return []
    r = ctx["residuals"]
    # leave-one-out: how far from what its PEERS predict, not from a line
    # fitted through itself
    val, se = _loo_residual(ctx, district)
    if val is None:
        val = float(row["over_under"])
        sd = float(r["over_under"].std())
        if not sd or pd.isna(sd):
            return []
        se = sd * math.sqrt(1.0 + 1.0 / max(len(r), 2))
    score = _evidence(val, se)
    if score <= 0:
        return []
    better = val < 0          # outcome is "% below grade" — lower is better
    rank = int((r["over_under"] < float(row["over_under"])).sum()) + 1
    return [_find(
        "Performance vs circumstances", score,
        f"**{district} {'beats' if better else 'falls short of'} its "
        f"circumstances by {abs(val):.1f} points** — it scores "
        f"{float(row['actual']):.1f}% below grade level where districts with "
        f"its income, literacy and staffing average "
        f"{float(row['actual']) - val:.1f}%. That is "
        f"{'better' if better else 'worse'} than "
        f"{(len(r) - rank) if better else (rank - 1)} of the other "
        f"{len(r) - 1} districts once context is allowed for."
        + ("  Worth studying — this is where practice, not circumstance, is "
           "doing the work." if better else ""),
        f"least-squares residual on {len(ctx['fit']['controls'])} context "
        f"controls, adjusted R² = {ctx['fit']['adj_r2']:.2f}",
        "x_over_under", n=len(r)),
    ]


def x_rank_shift(ctx, district, min_n=None):
    """
    How far does the district move once circumstances are accounted for?

    A district ranked 24th on the raw table and 6th on the context-adjusted
    one is not failing — it is doing well against a hard hand. That gap is the
    finding, and it is invisible in either table alone.
    """
    if not _usable(ctx):
        return []
    row = _resid_row(ctx, district)
    if row is None:
        return []
    r = ctx["residuals"]
    n = len(r)
    if n < MIN_DISTRICTS:
        return []
    raw_rank = int((r["actual"] < float(row["actual"])).sum()) + 1
    adj_rank = int((r["over_under"] < float(row["over_under"])).sum()) + 1
    shift = raw_rank - adj_rank
    if abs(shift) < max(3, n // 8):
        return []
    sd = float(r["over_under"].std())
    se = sd * math.sqrt(1.0 + 1.0 / max(n, 2))
    score = _evidence(float(row["over_under"]), se)
    if score <= 0:
        return []
    return [_find(
        "Rank vs context", score,
        f"**{district} ranks {raw_rank} of {n} on raw results, but "
        f"{adj_rank} of {n} once circumstances are accounted for** — a move "
        f"of {abs(shift)} places {'up' if shift > 0 else 'down'}. "
        + (f"The raw table understates it." if shift > 0 else
           f"The raw table flatters it."),
        f"raw rank vs rank on the context-adjusted residual, {n} districts",
        "x_rank_shift", n=n),
    ]


def x_peer_comparison(ctx, district, min_n=None):
    """
    Compare against the districts most like this one, not against the state.

    Nearest neighbours in standardised context space — the concrete version of
    "is 68% good *for a district like this one*".
    """
    m, key = ctx.get("merged"), ctx["key"]
    out_col = ctx["outcome_label"]
    if m is None or len(m) < MIN_DISTRICTS:
        return []
    if not (ctx.get("variance") or {}).get("ok"):
        return []
    ctrls = [c for c in (ctx.get("fit") or {}).get("controls", [])
             if c in m.columns]
    if len(ctrls) < 2:
        return []
    d = m[[key, out_col] + ctrls].dropna()
    hit = d[d[key].astype(str).str.strip().str.lower()
            == str(district).strip().lower()]
    if hit.empty or len(d) < MIN_DISTRICTS:
        return []
    i = hit.index[0]

    X = d[ctrls].astype(float)
    sd = X.std().replace(0, 1.0)
    Z = (X - X.mean()) / sd
    dist = ((Z - Z.loc[i]) ** 2).sum(axis=1) ** 0.5
    peers = dist.drop(index=i).nsmallest(min(PEER_K, len(d) - 1)).index
    if len(peers) < 3:
        return []

    mine = float(d.loc[i, out_col])
    peer_vals = d.loc[peers, out_col].astype(float)
    gap = mine - float(peer_vals.mean())
    se = float(peer_vals.std()) / math.sqrt(len(peer_vals))
    score = _evidence(gap, se)
    if score <= 0:
        return []
    better = gap < 0
    names = ", ".join(str(x) for x in d.loc[peers, key].head(3))
    return [_find(
        "Peer comparison", score,
        f"**Against the {len(peers)} districts most like it** — {names} and "
        f"others with similar income, literacy and staffing — {district} is "
        f"**{abs(gap):.1f} points {'better' if better else 'worse'}** "
        f"({mine:.1f}% below grade level against a peer average of "
        f"{peer_vals.mean():.1f}%). Comparing it to the whole state would "
        f"mix in districts facing quite different conditions.",
        f"{len(peers)} nearest districts in standardised context space "
        f"({len(ctrls)} indicators)",
        "x_peer_comparison", n=len(peers)),
    ]


def x_context_explains(ctx, district=None, min_n=None):
    """How much of the difference between districts is circumstance at all?"""
    fit, var = ctx.get("fit") or {}, ctx.get("variance") or {}
    if not fit.get("ok") or not var.get("ok"):
        return []
    adj = float(fit.get("adj_r2") or 0.0)
    spread = float(var.get("spread") or 0.0)
    if adj <= 0.05 or spread <= 0:
        return []
    return [_find(
        "Context explains", adj * spread,
        f"**District circumstances account for {100 * adj:.0f}% of the "
        f"variation between districts** — about {adj * spread:.1f} of the "
        f"{spread:.1f} points separating the best and worst. The remaining "
        f"{(1 - adj) * spread:.1f} points are not explained by income, "
        f"literacy, staffing or infrastructure, which is the part schools "
        f"and teaching can actually move.",
        f"adjusted R² of {len(fit.get('controls', []))} context controls "
        f"over {var.get('n', '?')} districts",
        "x_context_explains"),
    ]


def x_no_link(ctx, district=None, min_n=None):
    """
    Nothing in the context file explains the results — reported as a finding.

    A system that only speaks when it finds something is fishing. This is the
    honest headline when 20-odd indicators are tested and none survives, and
    it comes with the detection floor so nobody reads it as "we proved there
    is no effect".
    """
    tab, var = ctx.get("table"), ctx.get("variance") or {}
    if tab is None or tab.empty:
        return []
    tested = tab[~tab["derived"]]
    if tested.empty:
        return []
    n_sig = int((tested["verdict"] == "significant after FDR correction").sum())
    if n_sig:
        return []
    if not var.get("ok"):
        # nothing to explain in the first place — a different, louder finding
        return [_find(
            "Nothing to explain", 0.0,
            f"**Every district scored almost identically** — a spread of only "
            f"{var.get('spread', float('nan')):.2f} points across "
            f"{var.get('n', '?')} districts. No correlation against a flat "
            f"outcome can mean anything, so no district-context finding "
            f"below should be trusted.",
            "variance check on the district outcome",
            "x_no_link")]
    crit = tab.attrs.get("min_detectable_r", float("nan"))
    top = tested.iloc[(tested["r"].abs()).idxmax()] if len(tested) else None
    spread = float(var.get("spread") or 0.0)
    return [_find(
        "No context link", spread * 0.5,
        f"**None of the {len(tested)} district indicators explains learning "
        f"outcomes.** Income, literacy, teacher numbers and infrastructure "
        f"were all tested; the strongest was "
        f"{top['variable']} at r = {top['r']:+.2f}, below the {crit:.2f} "
        f"needed to be distinguishable from zero with "
        f"{int(tested['n'].max())} districts. This is not proof there is no "
        f"effect — it means an effect this size cannot be detected at this "
        f"number of districts. Where a district sits is therefore not "
        f"predicted by its circumstances.",
        f"Pearson r with Benjamini-Hochberg over {len(tested)} indicators, "
        f"detection floor |r| ≥ {crit:.2f}",
        "x_no_link", n=int(tested["n"].max())),
    ]


def x_strongest_link(ctx, district=None, min_n=None):
    """The indicators that DID survive correction, stated with their range."""
    tab = ctx.get("table")
    if tab is None or tab.empty or not (ctx.get("variance") or {}).get("ok"):
        return []
    tested = tab[~tab["derived"]]
    sig = tested[tested["verdict"] == "significant after FDR correction"]
    if sig.empty:
        return []
    out = []
    spread = float((ctx.get("variance") or {}).get("spread") or 0.0)
    for r in sig.head(2).itertuples():
        direction = "fewer" if r.r < 0 else "more"
        out.append(_find(
            "Context link", abs(r.r) * spread,
            f"**{r.variable} is linked to results** — districts with more of "
            f"it have {direction} children below grade level "
            f"(r = {r.r:+.2f}, plausible range {r.ci_low:+.2f} to "
            f"{r.ci_high:+.2f}, corrected p = {r.p_adj:.3f}). "
            f"{S.strength_label(r.r).capitalize()} relationship across "
            f"{int(r.n)} districts. This is an association, not proof that "
            f"changing it would change results.",
            f"Pearson r, Fisher z interval, Benjamini-Hochberg corrected",
            "x_strongest_link", n=int(r.n)))
    return out


def x_redundant_context(ctx, district=None, min_n=None):
    """
    Most of a district file measures SIZE, not quality — worth saying, because
    it explains why twenty indicators carry far less than twenty indicators'
    worth of information.
    """
    rd = ctx.get("redundancy")
    if rd is None or rd.empty:
        return []
    n_red = int(rd["redundant"].sum())
    if not n_red:
        return []
    tab = ctx.get("table")
    tested = 0 if tab is None else int((~tab["derived"]).sum())
    p0 = rd[rd["redundant"]].iloc[0]
    spread = float((ctx.get("variance") or {}).get("spread") or 0.0)
    return [_find(
        "Redundant indicators", spread * 0.25,
        f"**{n_red} pairs of context indicators measure nearly the same "
        f"thing** — for example {p0['a']} and {p0['b']} move together almost "
        f"perfectly (r = {p0['r']:+.2f}). Most of this file tracks district "
        f"**size** rather than district **quality**, so its {tested} columns "
        f"hold far fewer than {tested} independent pieces of information.",
        f"pairwise Pearson r ≥ 0.7 among context indicators, "
        f"Benjamini-Hochberg corrected",
        "x_redundant_context"),
    ]


# ------------------------------------------------- descriptive (small n)
# At division level there are 4 units. Nothing inferential is possible, but
# plain comparison still is — and saying "these four divisions differ by 6
# points" is a real, checkable statement that needs no p-value.

def x_unit_spread(ctx, district=None, min_n=None):
    """How far apart are the units at this level, and which are the ends?"""
    m, key = ctx.get("merged"), ctx["key"]
    out_col = ctx["outcome_label"]
    if m is None or len(m) < 2:
        return []
    v = pd.to_numeric(m[out_col], errors="coerce")
    d = m.assign(_v=v).dropna(subset=["_v"])
    if len(d) < 2:
        return []
    hi = d.loc[d["_v"].idxmax()]
    lo = d.loc[d["_v"].idxmin()]
    spread = float(hi["_v"] - lo["_v"])
    if spread <= 0:
        return []
    lvl = ctx.get("level", "District").lower()
    return [_find(
        f"{ctx.get('level', 'District')} spread", spread,
        f"**Across the {len(d)} {lvl}s, results range from "
        f"{lo['_v']:.1f}% to {hi['_v']:.1f}% below grade level** — a "
        f"{spread:.1f}-point gap. **{lo[key]}** is strongest and "
        f"**{hi[key]}** weakest. "
        + (f"That gap is the thing any explanation has to account for."
           if spread >= 2 else
           f"That gap is small enough that the {lvl}s are effectively "
           f"level with each other."),
        f"size-weighted outcome by {lvl}, {len(d)} units",
        "x_unit_spread", n=len(d)),
    ]


def x_context_contrast(ctx, district=None, min_n=None):
    """
    Does the unit with the most resources also get the best results?

    A plain two-way comparison, not a correlation — it needs no sample size
    to be true, and it is the question people actually ask.
    """
    m, key = ctx.get("merged"), ctx["key"]
    out_col = ctx["outcome_label"]
    if m is None or len(m) < 3:
        return []
    pick = next((c for c in ("Per Capita Income", "Student Teacher Ratio",
                             "Total Libraries")
                 if c in m.columns), None)
    if pick is None:
        nums = [c for c in m.columns
                if c not in (key, out_col)
                and pd.api.types.is_numeric_dtype(m[c])]
        if not nums:
            return []
        pick = nums[0]
    d = m[[key, out_col, pick]].apply(
        lambda s: pd.to_numeric(s, errors="coerce") if s.name != key else s
    ).dropna()
    if len(d) < 3:
        return []
    lvl = ctx.get("level", "District").lower()
    top = d.loc[d[pick].idxmax()]
    # rank on the outcome: 1 = best = fewest below grade level
    rank = int((d[out_col] < float(top[out_col])).sum()) + 1
    others = d[d[key] != top[key]][out_col].mean()
    gap = float(top[out_col]) - float(others)
    return [_find(
        "Resources vs results", abs(gap),
        f"**{top[key]} has the highest {pick} of any {lvl}** "
        f"({float(top[pick]):,.0f}) but ranks **{rank} of {len(d)}** on "
        f"results — {float(top[out_col]):.1f}% below grade level against "
        f"{others:.1f}% across the others. "
        + ("Having the most resources is not translating into the best "
           "outcomes here." if rank > len(d) / 2 else
           "Resources and results line up at the top of this table."),
        f"highest {pick} vs outcome rank, {len(d)} {lvl}s — a direct "
        f"comparison, not a correlation",
        "x_context_contrast", n=len(d)),
    ]


def x_too_few_units(ctx, district=None, min_n=None):
    """
    Say plainly why no correlation is offered at this level.

    Silence would look like an oversight. This is the finding: the level the
    user selected cannot support the question they may be about to ask.
    """
    if not ctx.get("descriptive_only"):
        return []
    n = ctx.get("n_units", 0)
    crit = ctx.get("min_detectable_r", float("nan"))
    lvl = ctx.get("level", "District").lower()
    return [_find(
        "Too few units to correlate", 0.0,
        f"**No correlation is reported at {lvl} level, deliberately.** There "
        f"are only {n} {lvl}s, and with {n} points a relationship would have "
        f"to reach |r| ≥ {crit:.2f} before it could be told apart from "
        f"chance — near-perfect. Any number computed here would be an "
        f"artefact of having four points, not a finding. The comparisons "
        f"above are direct and need no sample size; switch to district level "
        f"for the correlation analysis.",
        f"minimum detectable |r| at n = {n}, alpha = 0.05",
        "x_too_few_units", n=n),
    ]


GENERATORS = [
    x_over_under,
    x_rank_shift,
    x_peer_comparison,
    x_context_explains,
    x_strongest_link,
    x_no_link,
    x_redundant_context,
    x_unit_spread,
    x_context_contrast,
    x_too_few_units,
]

THEME = "🔗 District Context"

REGISTRY = {
    "x_over_under": dict(
        name="Performance vs circumstances",
        answers="Is 68% good — for a district like this one?",
        formula="residual of outcome ~ context controls (least squares)",
        filters="|residual| > ~2 SD, context model must be usable"),
    "x_rank_shift": dict(
        name="Rank vs context",
        answers="Which districts are doing better than they look?",
        formula="raw rank − rank on the context-adjusted residual",
        filters="shift ≥ max(3, n/8), context model must be usable"),
    "x_peer_comparison": dict(
        name="Peer comparison",
        answers="How does it compare with districts in similar circumstances?",
        formula=f"{PEER_K} nearest districts in standardised context space",
        filters="≥3 peers, outcome must vary"),
    "x_context_explains": dict(
        name="Context explains",
        answers="How much of the gap between districts is circumstance?",
        formula="adjusted R² × outcome spread",
        filters="adj R² > 0.05, outcome must vary"),
    "x_strongest_link": dict(
        name="Context link",
        answers="Which district conditions relate to learning outcomes?",
        formula="Pearson r, Fisher z CI, Benjamini-Hochberg",
        filters="survives FDR correction"),
    "x_no_link": dict(
        name="No context link",
        answers="Which district conditions relate to learning outcomes?",
        formula="all indicators tested, none survives FDR",
        filters="reported with the minimum detectable |r|"),
    "x_redundant_context": dict(
        name="Redundant indicators",
        answers="How much independent information does the context file hold?",
        formula="pairwise r ≥ 0.7 among context indicators",
        filters="at least one redundant pair"),
    "x_unit_spread": dict(
        name="Spread across units",
        answers="How far apart are the units at this level?",
        formula="max − min of the size-weighted outcome",
        filters="≥2 units, spread > 0 — needs no sample size"),
    "x_context_contrast": dict(
        name="Resources vs results",
        answers="Does the best-resourced unit get the best results?",
        formula="rank on the leading indicator vs rank on the outcome",
        filters="≥3 units — a direct comparison, not a correlation"),
    "x_too_few_units": dict(
        name="Too few units to correlate",
        answers="Why is no correlation shown at this level?",
        formula="minimum detectable |r| at this number of units",
        filters=f"fires only when units < {MIN_DISTRICTS}"),
}


def generate(ctx, district=None, limit=4):
    """Cross-dataset findings for one district, strongest first."""
    if not ctx:
        return []
    found = []
    for fn in GENERATORS:
        try:
            found.extend(fn(ctx, district) or [])
        except Exception as exc:                   # noqa: BLE001 — surfaced below
            ERRORS.append({"generator": fn.__name__, "district": district,
                           "error": f"{type(exc).__name__}: {exc}"})
    found.sort(key=lambda x: -x["score"])
    return found[:limit]


def describe(ctx=None, district=None):
    """Registry rows, annotated with what fired this run."""
    rows = []
    for fn in GENERATORS:
        meta = REGISTRY.get(fn.__name__, {})
        row = {"Generator": meta.get("name", fn.__name__),
               "Answers": meta.get("answers", "—"),
               "Formula": meta.get("formula", "—"),
               "Filter": meta.get("filters", "—")}
        if ctx:
            try:
                hits = fn(ctx, district) or []
                row["Fired?"] = "✅ yes" if hits else "— nothing found"
                row["Score"] = round(hits[0]["score"], 1) if hits else None
            except Exception:
                row["Fired?"] = "⚠️ error"
                row["Score"] = None
        rows.append(row)
    return rows
