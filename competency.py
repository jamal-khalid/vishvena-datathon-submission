"""
Competency Intelligence Report — deterministic version.

This reproduces the report structure from the original Groq-based backend
(app/utils.py -> generate_competency_report_prompt), but computes every
section instead of asking a model to write it.

Original spec              -> how we compute it
-------------------------------------------------------------------------
1 Overall performance      -> mean below/at/above %, rank vs other competencies
2 Performance distribution -> counts by level (below / at / above)
3 Demographic analysis     -> groupby gender + grade + block, WITH z-tests
4 Learning patterns        -> variance, spread, cross-competency correlation
5 Risk analysis            -> threshold rules + children affected
6 Summary                  -> template

The LLM version could not do the z-tests or the correlation. This one can.
"""
import numpy as np
import pandas as pd
from stats_tests import (two_proportion_z, proportion_test, cohens_h,
                         effect_label, sig_marker)


def _level(below_pct):
    if below_pct >= 60: return "Critical"
    if below_pct >= 45: return "Needs attention"
    if below_pct >= 30: return "Developing"
    return "Healthy"


def report(agg, district, competency, year=None):
    """Full competency report for one competency inside one district."""
    d = agg[(agg["district"] == district) & (agg["competency"] == competency)]
    if d.empty:
        return None
    if year is None:
        year = d["year"].max()
    cur = d[d["year"] == year]
    if cur.empty:
        return None

    students = int(cur["n"].sum())
    below = float((cur["below_pct"] * cur["n"]).sum() / students)
    above = float((cur["above_pct"] * cur["n"]).sum() / students)
    at = max(100 - below - above, 0)

    # ---- 1. rank against the other competencies in this district ----------
    all_comp = (agg[(agg["district"] == district) & (agg["year"] == year)]
                .groupby("competency")
                .apply(lambda g: (g["below_pct"] * g["n"]).sum() / g["n"].sum(),
                       include_groups=False)
                .sort_values(ascending=False))
    rank = int(list(all_comp.index).index(competency)) + 1

    overview = {
        "students_assessed": students,
        "below_pct": round(below, 1),
        "at_pct": round(at, 1),
        "above_pct": round(above, 1),
        "performance_level": _level(below),
        "rank": f"{rank} of {len(all_comp)} (1 = weakest)",
        "children_below": int(round(students * below / 100)),
    }

    # ---- 2. distribution --------------------------------------------------
    distribution = {
        "Below grade level": int(round(students * below / 100)),
        "At grade level":    int(round(students * at / 100)),
        "Above grade level": int(round(students * above / 100)),
    }

    # ---- 3a. gender (with significance test) ------------------------------
    f = float((cur["f_below"] * cur["n"]).sum() / students)
    m = float((cur["m_below"] * cur["n"]).sum() / students)
    # Real arm sizes where we have them; a test valid at small cell counts.
    if "f_n" in cur.columns and cur["f_n"].sum() > 0:
        nf = max(int(cur["f_n"].sum()), 1)
        nm = max(int(cur["m_n"].sum()), 1)
    else:
        nf = nm = max(students // 2, 1)
    z, p, _method = proportion_test(f, m, nf, nm)
    h = cohens_h(f, m)
    sig = p < 0.05 and abs(f - m) >= 4
    if sig:
        lag = "Girls" if f > m else "Boys"
        gtext = (f"{lag} perform worse by {abs(f-m):.1f} points "
                 f"({f:.1f}% girls vs {m:.1f}% boys below grade). "
                 f"Statistically significant ({sig_marker(p)}), {effect_label(h)} effect.")
    else:
        gtext = (f"No meaningful gender difference ({f:.1f}% girls vs {m:.1f}% boys, "
                 f"{sig_marker(p)}). Not reported as a finding.")
    gender = {"girls_below_pct": round(f, 1), "boys_below_pct": round(m, 1),
              "gap_pts": round(f - m, 1), "z": round(z, 2), "p_value": round(p, 4),
              "significant": bool(sig), "summary": gtext}

    # ---- 3b. grade --------------------------------------------------------
    gr = (cur.groupby("grade")
             .apply(lambda g: pd.Series({
                 "below_pct": (g["below_pct"] * g["n"]).sum() / g["n"].sum(),
                 "students": g["n"].sum()}), include_groups=False)
             .reset_index())
    gr["below_pct"] = gr["below_pct"].round(1)
    gr["students"] = gr["students"].astype(int)
    lo_g, hi_g = gr.loc[gr["below_pct"].idxmin()], gr.loc[gr["below_pct"].idxmax()]
    slope = float(np.polyfit(gr["grade"], gr["below_pct"], 1)[0]) if len(gr) > 1 else 0.0
    if slope > 0.5:
        gr_text = (f"Performance worsens as children advance "
                   f"(+{slope:.1f} points per grade) — the gap compounds.")
    elif slope < -0.5:
        gr_text = (f"Performance improves with grade level "
                   f"({slope:.1f} points per grade).")
    else:
        gr_text = "Performance is broadly flat across grades."

    grade = {"table": gr, "best": f"Grade {int(lo_g['grade'])} ({lo_g['below_pct']}%)",
             "worst": f"Grade {int(hi_g['grade'])} ({hi_g['below_pct']}%)",
             "slope_per_grade": round(slope, 2), "summary": gr_text}

    # ---- 3c. geography ----------------------------------------------------
    blk = (cur.groupby("block")
              .apply(lambda g: pd.Series({
                  "below_pct": (g["below_pct"] * g["n"]).sum() / g["n"].sum(),
                  "students": g["n"].sum()}), include_groups=False)
              .reset_index().sort_values("below_pct", ascending=False))
    blk["below_pct"] = blk["below_pct"].round(1)
    blk["students"] = blk["students"].astype(int)
    spread = float(blk["below_pct"].max() - blk["below_pct"].min())
    cv = float(blk["below_pct"].std() / blk["below_pct"].mean() * 100) if len(blk) > 1 else 0.0
    geography = {
        "table": blk,
        "worst_block": f"{blk.iloc[0]['block']} ({blk.iloc[0]['below_pct']}%)",
        "best_block": f"{blk.iloc[-1]['block']} ({blk.iloc[-1]['below_pct']}%)",
        "spread_pts": round(spread, 1),
        "coefficient_of_variation": round(cv, 1),
        "summary": (f"Performance varies {spread:.0f} points across blocks "
                    f"(CV {cv:.0f}%). "
                    + ("Highly uneven — a district average is misleading here."
                       if cv >= 20 else
                       "Fairly consistent across the district.")),
    }

    # ---- 4. trend ---------------------------------------------------------
    tr = (d.groupby("year")
            .apply(lambda g: (g["below_pct"] * g["n"]).sum() / g["n"].sum(),
                   include_groups=False)
            .round(1).reset_index(name="below_pct"))
    if len(tr) > 1:
        delta = float(tr.iloc[-1]["below_pct"] - tr.iloc[0]["below_pct"])
        word = "worsened" if delta > 1 else ("improved" if delta < -1 else "stayed flat")
        tr_text = (f"Over {len(tr)} years this competency has {word} "
                   f"({delta:+.1f} points overall).")
    else:
        delta, tr_text = 0.0, "Only one year of data — no trend can be determined."
    trend = {"table": tr, "total_change_pts": round(delta, 1), "summary": tr_text}

    # ---- 5. risk ----------------------------------------------------------
    n_blocks_critical = int((blk["below_pct"] >= 60).sum())
    widespread = n_blocks_critical >= max(1, len(blk) // 2)
    risk = {
        "level": _level(below),
        "children_below_grade": overview["children_below"],
        "blocks_critical": f"{n_blocks_critical} of {len(blk)}",
        "widespread": bool(widespread),
        "summary": ("A widespread challenge — the majority of blocks are critical, "
                    "so this needs a district-wide response."
                    if widespread else
                    "Concentrated in specific blocks rather than district-wide, "
                    "so targeted block-level action is appropriate."),
    }

    # ---- 6. summary paragraph --------------------------------------------
    summary = (
        f"{competency} is ranked {rank} of {len(all_comp)} competencies in {district}, "
        f"with {below:.0f}% of {students:,} assessed students below grade level "
        f"({overview['children_below']:,} children). {tr_text} "
        f"{geography['summary']} {gr_text} "
        f"{'' if not sig else gtext + ' '}"
        f"{risk['summary']}"
    )

    return {"competency": competency, "district": district, "year": int(year),
            "overview": overview, "distribution": distribution, "gender": gender,
            "grade": grade, "geography": geography, "trend": trend,
            "risk": risk, "summary": summary}


def correlation_matrix(agg, district=None, year=None):
    """
    Which competencies fail TOGETHER?

    Correlates below_pct across blocks. A high correlation means blocks weak in
    one are weak in the other — pointing at a shared root cause, so one
    intervention can address both. The LLM version could only guess at this.
    """
    d = agg if district is None else agg[agg["district"] == district]
    if year is None:
        year = d["year"].max()
    d = d[d["year"] == year]
    piv = d.pivot_table(index=["district", "block"], columns="competency",
                        values="below_pct", aggfunc="mean")
    if piv.shape[0] < 3:
        return None
    return piv.corr().round(2)


def strongest_pairs(corr, top=3):
    """Human-readable list of the most-linked competency pairs."""
    if corr is None:
        return []
    out = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            out.append({"pair": f"{cols[i]} ↔ {cols[j]}",
                        "r": float(corr.iloc[i, j])})
    out.sort(key=lambda x: -abs(x["r"]))
    return out[:top]
