"""
Layers 2, 3, 6 — the scikit-learn parts (clustering, risk, what-if).
Small, fast, and easy to explain.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression


# ---------- Layer 3: Learning-Health status (rules) ----------
def health_status(below_pct):
    if below_pct >= 60: return "Critical"
    if below_pct >= 45: return "At-risk"
    return "Strong"


def trajectory(below_pct, prev_pct):
    if pd.isna(prev_pct): return "n/a"
    d = below_pct - prev_pct
    if d > 1: return "declining"
    if d < -1: return "improving"
    return "stagnant"


# ---------- Layer 2: cluster blocks into learning archetypes ----------
MIN_N_FOR_CLUSTER = 30


def _wpivot(d, index, columns, value="below_pct", weight="n"):
    """Size-weighted pivot. An unweighted mean lets one small grade dominate."""
    num = (d.assign(_p=pd.to_numeric(d[value], errors="coerce")
                    * pd.to_numeric(d[weight], errors="coerce"))
             .pivot_table(index=index, columns=columns, values="_p",
                          aggfunc="sum"))
    den = d.pivot_table(index=index, columns=columns, values=weight,
                        aggfunc="sum")
    return num / den.replace(0, np.nan)


def block_feature_matrix(agg, year=None, min_n=MIN_N_FOR_CLUSTER):
    """
    Block x competency matrix of size-weighted below%.

    TWO THINGS WERE WRONG HERE:

    `.fillna(0)` turned a competency a block was never measured on into 0%
    below grade level — a PERFECT score. KMeans then clustered that block as
    outstanding at a subject it has no data for. Missing is not zero, so
    blocks with an incomplete profile are dropped instead.

    `aggfunc="mean"` averaged grades unweighted, so a 20-child grade counted
    as much as a 1,500-child one.
    """
    if year is None:
        year = agg["year"].max()
    d = agg[agg["year"] == year]
    size = d.groupby(["district", "block"])["n"].sum()
    keep = set(size[size >= min_n].index)
    if keep:
        d = d[[t in keep for t in zip(d["district"], d["block"])]]
    piv = _wpivot(d, ["district", "block"], "competency")
    return piv.dropna(axis=0, how="any")        # complete profiles only


def cluster_blocks(agg, k=3, year=None, min_n=MIN_N_FOR_CLUSTER):
    piv = block_feature_matrix(agg, year, min_n=min_n)
    if piv.empty or len(piv) < 2:
        return piv.reset_index(), {}
    k = min(k, len(piv))
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(piv.values)
    comp_cols = list(piv.columns)
    piv = piv.assign(archetype=labels)
    # name each archetype by the competency where it is MOST distinct (above global avg)
    gmean = piv[comp_cols].mean()
    names = {}
    for lab in sorted(set(labels)):
        centroid = piv[piv["archetype"] == lab][comp_cols].mean()
        distinct = (centroid - gmean).idxmax()
        names[lab] = f"Distinctly-weak: {distinct}"
    piv["archetype_name"] = piv["archetype"].map(names)
    return piv.reset_index(), names


# ---------- Layer 6a: risk model (predict staying below grade) ----------
def train_risk(agg):
    """
    DEPRECATED — CIRCULAR. Do not use; kept only so older notebooks import.

    It predicts `below_pct >= 50` while using `below_pct` as a feature, so it
    re-describes the present and scores near-perfectly for no reason. The
    dashboard does not call it. train_early_warning() is the honest
    replacement: it predicts NEXT year and is validated on a transition it
    never saw.
    """
    import warnings
    warnings.warn("train_risk() is circular (below_pct predicts "
                  "below_pct >= 50); use train_early_warning() instead",
                  DeprecationWarning, stacklevel=2)
    d = agg.dropna(subset=["prev_pct"]).copy()
    d["yoy"] = d["below_pct"] - d["prev_pct"]
    X = d[["below_pct", "yoy"]].values
    y = (d["below_pct"] >= 50).astype(int).values
    if len(set(y)) < 2:                      # guard for degenerate demo data
        return None, d.assign(risk=(d["below_pct"] / 100.0))
    model = LogisticRegression().fit(X, y)
    d["risk"] = model.predict_proba(X)[:, 1]
    return model, d


# ---------- Layer 6a-v2: genuine temporal early-warning model ----------
#
# The old train_risk() was circular: it predicted `below_pct >= 50` while using
# `below_pct` as a feature, so it just re-described the present. This version
# predicts NEXT year from THIS year and is validated on a year it never saw.
#
# With three years of data (T0, T1, T2) there are two transitions:
#     train on  T0 -> T1
#     test  on  T1 -> T2      <- held out, never seen during fitting
#     deploy    T2 -> T3      <- the actual forecast
#
# That is why three years is enough here, even though a 3-point trend line is
# not: we are not extrapolating a curve through time, we are learning a
# cross-sectional relationship across thousands of unit x competency rows.

EW_FEATURES = ["below_pct", "above_pct", "gender_gap", "log_n",
               "unit_mean", "comp_mean"]


def _early_warning_frame(agg, min_n=30):
    """One row per (unit, competency, base year) with next year's outcome."""
    d = agg.copy()
    d = d[d["n"] >= min_n] if (d["n"] >= min_n).sum() >= 50 else d

    d["log_n"] = np.log1p(d["n"])
    d["gender_gap"] = d["gender_gap"].fillna(0.0)
    # context available at the base year, no leakage from the future
    d["unit_mean"] = d.groupby(["district", "block", "grade", "year"])[
        "below_pct"].transform("mean")
    d["comp_mean"] = d.groupby(["competency", "year"])["below_pct"].transform("mean")

    key = ["district", "block", "grade", "competency"]
    nxt = d[key + ["year", "below_pct"]].rename(
        columns={"below_pct": "next_below_pct"})
    nxt["year"] = nxt["year"] - 1            # align next year onto the base year
    m = d.merge(nxt, on=key + ["year"], how="inner")
    return m.dropna(subset=EW_FEATURES + ["next_below_pct"])


def train_early_warning(agg, min_n=30, risk_cut=50.0):
    """
    Predict next year's below-grade % per unit x competency.

    Honest evaluation: trained on the earliest transition, scored on the latest
    one, and compared against the naive 'nothing changes' baseline. If it cannot
    beat that baseline it is not worth presenting, and the caller is told so.
    """
    from sklearn.linear_model import Ridge

    m = _early_warning_frame(agg, min_n)
    if m.empty:
        return {"ok": False, "reason": "No unit has two consecutive years of data."}

    base_years = sorted(m["year"].unique())
    if len(base_years) < 2:
        return {"ok": False,
                "reason": f"Only one usable year transition "
                          f"({base_years[0]}→{base_years[0]+1}). Honest validation "
                          f"needs two, i.e. at least three years of data.",
                "single_transition": True}

    test_year = base_years[-1]
    train = m[m["year"] < test_year]
    test = m[m["year"] == test_year]
    if len(train) < 30 or len(test) < 10:
        return {"ok": False, "reason": "Too few rows to train and validate."}

    Xtr, ytr = train[EW_FEATURES].values, train["next_below_pct"].values
    Xte, yte = test[EW_FEATURES].values, test["next_below_pct"].values

    model = Ridge(alpha=1.0).fit(Xtr, ytr)
    pred = model.predict(Xte)

    mae = float(np.mean(np.abs(pred - yte)))
    naive = float(np.mean(np.abs(test["below_pct"].values - yte)))   # persistence
    ss_res = float(np.sum((yte - pred) ** 2))
    ss_tot = float(np.sum((yte - yte.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # early-warning framing: did we flag the units that actually ended up bad?
    real_bad = yte >= risk_cut
    flagged = pred >= risk_cut
    tp = int((real_bad & flagged).sum())
    precision = tp / max(int(flagged.sum()), 1)
    recall = tp / max(int(real_bad.sum()), 1)

    # The SAME early-warning question asked of the naive rule: "flag whatever is
    # already above the line." When the model loses on MAE this is the better
    # predictor, so it needs its own precision/recall to be offered in place of
    # the model rather than as an apology for it.
    n_flag = test["below_pct"].values >= risk_cut
    n_tp = int((real_bad & n_flag).sum())
    naive_precision = n_tp / max(int(n_flag.sum()), 1)
    naive_recall = n_tp / max(int(real_bad.sum()), 1)

    # forecast for the year after the last observed one
    latest = m[m["year"] == m["year"].max()].copy()
    live = _early_warning_frame(agg, min_n)          # rows that have a next year
    last_obs = agg["year"].max()
    fc_src = agg[agg["year"] == last_obs].copy()
    fc_src["log_n"] = np.log1p(fc_src["n"])
    fc_src["gender_gap"] = fc_src["gender_gap"].fillna(0.0)
    fc_src["unit_mean"] = fc_src.groupby(["district", "block", "grade"])[
        "below_pct"].transform("mean")
    fc_src["comp_mean"] = fc_src.groupby("competency")["below_pct"].transform("mean")
    fc_src = fc_src.dropna(subset=EW_FEATURES)
    fc_src["predicted_next"] = model.predict(fc_src[EW_FEATURES].values).clip(0, 100)
    fc_src["change"] = fc_src["predicted_next"] - fc_src["below_pct"]

    return {
        "ok": True,
        "model": model,
        "train_years": [int(y) for y in sorted(train["year"].unique())],
        "test_year": int(test_year),
        "forecast_year": int(last_obs) + 1,
        "n_train": int(len(train)), "n_test": int(len(test)),
        "mae": round(mae, 2),
        "naive_mae": round(naive, 2),
        "beats_naive": bool(mae < naive),
        "improvement_pct": round(100 * (naive - mae) / naive, 1) if naive else 0.0,
        "r2": round(r2, 3),
        "precision": round(precision, 3), "recall": round(recall, 3),
        "naive_precision": round(naive_precision, 3),
        "naive_recall": round(naive_recall, 3),
        "n_flagged": int(flagged.sum()), "n_real_bad": int(real_bad.sum()),
        "risk_cut": float(risk_cut),
        "coefficients": dict(zip(EW_FEATURES, model.coef_.round(3))),
        "forecast": fc_src[["district", "block", "grade", "competency", "n",
                            "below_pct", "predicted_next", "change"]]
                      .sort_values("predicted_next", ascending=False)
                      .reset_index(drop=True),
        # What to show when the model loses to persistence: the units already
        # above the risk line, which IS the better predictor here. Same rows,
        # ranked by something we can defend.
        "naive_watchlist": fc_src[["district", "block", "grade", "competency",
                                   "n", "below_pct"]]
                             .rename(columns={"below_pct":
                                              f"below_pct_{int(last_obs)}"})
                             .sort_values(f"below_pct_{int(last_obs)}",
                                          ascending=False)
                             .reset_index(drop=True),
        "last_observed_year": int(last_obs),
    }


# ---------- Layer 6b: What-If simulator (evidence-based) ----------
MIN_N_FOR_BENCHMARK = 30


def _stable(agg, min_n=MIN_N_FOR_BENCHMARK):
    """
    Rows with enough students that a year-over-year change means something.

    Without this, a group of 2 students where one child improves registers as a
    "50 point gain", and the benchmark quantiles become meaningless. Falls back
    to the largest available groups when nothing clears the bar.
    """
    d = agg.dropna(subset=["prev_pct"])
    ok = d[d["n"] >= min_n]
    if len(ok) >= 20:
        return ok, min_n
    if d.empty:
        return d, 0
    cut = float(d["n"].quantile(0.75))          # best we can do on thin data
    return d[d["n"] >= cut], cut


def improvement_benchmarks(agg, min_n=MIN_N_FOR_BENCHMARK):
    """
    Derive realistic improvement rates FROM THE DATA, not from a guess.
    Only groups large enough for the change to be real are used.
    """
    d, used_cut = _stable(agg, min_n)
    d = d.copy()
    d["gain"] = d["prev_pct"] - d["below_pct"]          # positive = improved
    gains = d[d["gain"] > 0]["gain"]
    if gains.empty:
        return {"typical": 0.0, "strong": 0.0, "best": 0.0,
                "n_observed": 0, "n_groups": int(len(d)),
                "share_improving": 0.0, "net_change_pts": 0.0,
                "conditioned_on": "groups that improved",
                "min_n_used": used_cut, "reliable": False}
    # These quantiles are conditioned on IMPROVING groups. Reporting the median
    # of that subset as "typical" without saying so is misleading: where half
    # the blocks improved 2 points and half worsened 10, it returns "typical
    # improvement 2.0" while the population actually moved -4.0. Callers get
    # the share and the net change so they can tell the two apart.
    return {
        "typical":    float(gains.quantile(0.50)),
        "strong":     float(gains.quantile(0.75)),
        "best":       float(gains.quantile(0.90)),
        "n_observed": int(len(gains)),
        "n_groups":   int(len(d)),
        "share_improving": round(float(len(gains) / max(len(d), 1)), 3),
        "net_change_pts": round(float(
            (d["gain"] * d["n"]).sum() / max(d["n"].sum(), 1)), 2),
        "conditioned_on": "groups that improved",
        "min_n_used": used_cut,
        "reliable":   bool(used_cut >= min_n and len(gains) >= 20),
    }


def natural_rebound(agg, worst_quintile=0.20, min_n=MIN_N_FOR_BENCHMARK):
    """
    Regression-to-the-mean baseline.

    We deliberately target the WORST blocks. Some are worst partly by bad luck
    and would improve next year with no intervention at all. This measures that
    natural rebound so we can subtract it and avoid overstating our impact.

    Restricted to adequately sized groups — in tiny groups the "rebound" is just
    sampling noise, which would inflate the correction absurdly.
    """
    d, _ = _stable(agg, min_n)
    if d.empty:
        return 0.0
    thresh = d["prev_pct"].quantile(1 - worst_quintile)
    worst = d[d["prev_pct"] >= thresh]
    if worst.empty or worst["n"].sum() <= 0:
        return 0.0
    # child-weighted, like every other rate in the system
    return float(((worst["prev_pct"] - worst["below_pct"]) * worst["n"]).sum()
                 / worst["n"].sum())


def what_if(agg, district, competency, n_blocks=10, year=None,
            min_n=MIN_N_FOR_BENCHMARK):
    """
    Scenario planner, NOT a prediction.

    Fixes vs the earlier version:
      - actually filters to the selected district (it was ignored before)
      - rolls grades up so "blocks" really means blocks (not block x grade)
      - improvement rates come from observed history, not an invented 0.6
      - subtracts the regression-to-the-mean rebound
      - reports a RANGE, never a single false-precision number
    """
    if year is None:
        year = agg["year"].max()

    d = agg[(agg["year"] == year) &
            (agg["competency"] == competency) &
            (agg["district"] == district)]           # <-- district now respected
    if d.empty:
        return None

    # Roll grades up so each row is one real block — WEIGHTED. An unweighted
    # mean of grade rates let a 6-child grade set the block's figure.
    blocks = (d.groupby("block")
                .apply(lambda g: pd.Series({
                    "below_pct": float((g["below_pct"] * g["n"]).sum()
                                       / g["n"].sum()) if g["n"].sum() else np.nan,
                    "n": g["n"].sum()}), include_groups=False)
                .reset_index().dropna(subset=["below_pct"]))

    # Never plan an intervention into a group too small to measure. Targeting
    # the highest below% with no floor picked the smallest block every time —
    # a 4-child block at 100% outranked six blocks of 1,200 children.
    eligible = blocks[blocks["n"] >= min_n]
    skipped = int(len(blocks) - len(eligible))
    if eligible.empty:
        return {"competency": competency, "district": district,
                "blocks_targeted": 0, "block_names": [], "students_covered": 0,
                "before_below_pct": None, "scenarios": [],
                "blocks_below_min_n": skipped,
                "reason": f"No block in {district} has {min_n}+ students "
                          f"assessed on {competency}."}
    target = eligible.nlargest(min(n_blocks, len(eligible)), "below_pct")

    students = int(target["n"].sum())
    # the rate across the CHILDREN being targeted, not the mean of block means
    before = float((target["below_pct"] * target["n"]).sum() / students)
    bench = improvement_benchmarks(agg)
    rebound = natural_rebound(agg)

    scenarios = []
    for label, gain in [("Conservative", bench["typical"]),
                        ("Realistic",    bench["strong"]),
                        ("Optimistic",   bench["best"])]:
        net = max(gain - rebound, 0.0)               # credit only what we add
        after = max(before - net, 0.0)
        scenarios.append({
            "scenario": label,
            "assumed_gain_pts": round(gain, 1),
            "net_of_rebound_pts": round(net, 1),
            "after_below_pct": round(after, 1),
            "children_moved": int(round(students * net / 100.0)),
        })

    return {
        "competency": competency,
        "district": district,
        "blocks_targeted": int(len(target)),
        "block_names": list(target["block"]),
        "students_covered": students,
        "before_below_pct": round(before, 1),
        "natural_rebound_pts": round(rebound, 2),
        "benchmark_sample": bench["n_observed"],
        "scenarios": scenarios,
    }
