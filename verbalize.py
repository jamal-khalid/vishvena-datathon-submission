"""
Layer 1 (text half) — turn aggregated numbers into sentences.

Deterministic templates, so every figure traces to an exact computation and the
wording is byte-identical on every run (which the fixed-seed reproducibility
rule requires). No model is involved at any point.
"""
import pandas as pd

# Below this many students a percentage is not worth stating as a fact:
# with n=1 every value is 0% or 100%, and a "100-point swing" is one child.
MIN_N_FOR_TREND = 10


def _sentence(r):
    n = int(r.n)
    plural = "student" if n == 1 else "students"
    verb = "is" if n == 1 else "are"
    parts = [
        f"In {r.block}" + (f" (Grade {int(r.grade)})" if int(r.grade) else "")
        + f", {r.competency}: "
        f"{r.below_pct:.0f}% of {n} {plural} {verb} below grade level in {int(r.year)}."
    ]
    # Trend and gender claims need enough students to be meaningful at all.
    if n >= MIN_N_FOR_TREND:
        if pd.notna(r.prev_pct):
            d = r.below_pct - r.prev_pct
            if abs(d) >= 1:
                word = "worsened" if d > 0 else "improved"
                parts.append(f"This {word} from {r.prev_pct:.0f}% the previous "
                             f"year ({d:+.0f} pts).")
        if pd.notna(r.gender_gap) and abs(r.gender_gap) >= 4:
            lag = "girls" if r.gender_gap > 0 else "boys"
            parts.append(f"Here {lag} trail by {abs(r.gender_gap):.0f} points.")

    affected = int(round(r.n * r.below_pct / 100.0))
    child = "child is" if affected == 1 else "children are"
    parts.append(f"About {affected} {child} affected.")

    if n < MIN_N_FOR_TREND:
        parts.append(f"(Only {n} {plural} — too few to read a trend or gender gap.)")
    return " ".join(parts)


def verbalize_district(agg, district, year=None):
    """Return (list_of_sentences, joined_text) for one district's latest year."""
    d = agg[agg["district"] == district]
    if year is None:
        year = d["year"].max()
    d = d[d["year"] == year]
    sents = [_sentence(r) for r in d.itertuples()]
    return sents, "\n".join(sents)


def competency_table(agg, district, year=None):
    d = agg[agg["district"] == district]
    if year is None:
        year = d["year"].max()
    d = d[d["year"] == year]
    t = (d.groupby("competency")
           .agg(below_pct=("below_pct", "mean"),
                gender_gap=("gender_gap", "mean"),
                students=("n", "sum"))
           .reset_index()
           .sort_values("below_pct", ascending=False))
    t["below_pct"] = t["below_pct"].round(0)
    t["gender_gap"] = t["gender_gap"].round(1)
    return t
