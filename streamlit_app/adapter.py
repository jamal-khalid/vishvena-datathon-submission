"""
Bridge: dashboard data  ->  the analysis layers' aggregate format.

Handles the real datathon shape natively:
    Year ("2022-23") | Grade | Division/District/Block/Cluster/GP | Gender | Q1..Q20 | Score

Three things this has to get right, because getting them wrong fails SILENTLY:
  1. Year arrives as "2022-23", not an integer -> parsed, else every trend is lost.
  2. Score is raw marks (0-20), not a percentage -> auto-scaled, else every
     student looks "below grade level".
  3. The 20 question columns ARE the competency dimension -> unpivoted, else
     there is only one meaningless competency to analyse.

Target schema (one row per geography x grade x competency x year):
    division district block grade competency year
    n below_pct above_pct f_below m_below gender_gap prev_pct
"""
import re

import duckdb

import numpy as np
import pandas as pd

REQUIRED = ["division", "district", "block", "grade", "competency", "year",
            "n", "students", "f_n", "m_n", "below_pct", "above_pct",
            "f_below", "m_below", "gender_gap", "prev_pct"]

# Cell = one geography x grade x year. `students` counts distinct children in
# that cell; `n` counts their assessment responses (20x larger when the question
# items form the competency dimension). See units.py for why both are needed.
CELL_KEYS = ["division", "district", "block", "grade", "year"]


def _student_counts(base_rows, comp=None):
    """
    Distinct children per cell — the honest denominator for any headcount.

    Two shapes reach this function and they must not be counted the same way:
      - WIDE  (comp=None): one row per child, so a row count IS a child count.
      - LONG  (comp set):  one row per child x competency. Here the children in
        a cell equal the rows for any ONE competency, so take the max across
        competencies (max, not mean, so partially-missing items don't deflate it).
    Counting rows in the long shape is what inflates every headcount 20x.
    """
    if comp is None or comp not in base_rows.columns:
        return (base_rows.groupby(CELL_KEYS, dropna=False)
                         .size().rename("students").reset_index())
    per = (base_rows.groupby(CELL_KEYS + [comp], dropna=False)
                    .size().rename("k").reset_index())
    return (per.groupby(CELL_KEYS, dropna=False)["k"].max()
               .rename("students").reset_index())


# ------------------------------------------------------------------ helpers
def _female_mask(s):
    v = s.astype(str).str.strip().str.lower()
    return v.isin(["f", "female", "girl", "girls", "women", "w"])


def parse_year(s):
    """'2022-23' -> 2022 ; 2024 -> 2024 ; '2024-25' -> 2024."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    txt = s.astype(str)
    got = txt.str.extract(r"((?:19|20)\d{2})", expand=False)
    return pd.to_numeric(got, errors="coerce")


def find_item_columns(df):
    """Binary question columns Q1..Qn with values in {0,1}."""
    items = []
    for c in df.columns:
        if re.fullmatch(r"[Qq]\s*_?\d+", str(c).strip()):
            v = pd.to_numeric(df[c], errors="coerce").dropna().unique()
            if len(v) and set(v).issubset({0, 1}):
                items.append(c)
    return sorted(items, key=lambda x: int(re.sub(r"\D", "", x)))


def rescale_score(series):
    """
    Put any score on a 0-100 scale and say what was assumed.
    Raw marks out of 20 are indistinguishable from percentages without this.
    """
    s = pd.to_numeric(series, errors="coerce")
    mx = float(s.max()) if s.notna().any() else np.nan
    if not np.isfinite(mx) or mx <= 0:
        return s, "could not read as numeric"
    if mx <= 1.5:
        return s * 100.0, "detected 0–1 proportion, ×100"
    if mx <= 100.0 and mx > 30:
        return s, "treated as a 0–100 percentage"
    return s / mx * 100.0, f"detected raw marks out of {mx:g}, converted to %"


# ------------------------------------------------------------------- builder
def build_agg(df, *, hierarchy, score_col=None, year_col=None, gender_col=None,
              comp_col=None, grade_col=None, below_cut=50.0, above_cut=75.0,
              use_items=True, finest=None, by_grade=True, qmap=None):
    """
    `finest` chooses the smallest unit findings are reported on. Going finer than
    the data supports produces confident nonsense ("improved 100 points" on two
    students), so this is the main lever for statistical honesty.
    """
    d = df.copy()
    notes = []

    # ---- geography ---------------------------------------------------------
    h = [c for c in (hierarchy or []) if c in d.columns]
    if not h:
        return None, "No hierarchy columns selected — pick at least one in the sidebar."
    if len(h) >= 3:
        div_c, dist_c, blk_c = h[0], h[1], h[2]
    elif len(h) == 2:
        div_c, dist_c, blk_c = h[0], h[0], h[1]
    else:
        div_c = dist_c = blk_c = h[0]

    if finest and finest in h:                  # override the reporting unit
        i = h.index(finest)
        blk_c = finest
        dist_c = h[max(i - 1, 0)] if i > 0 else finest
        div_c = h[0]

    base = pd.DataFrame({
        "division": d[div_c].astype(str),
        "district": d[dist_c].astype(str),
        "block":    d[blk_c].astype(str),
    }, index=d.index)

    if by_grade and grade_col and grade_col in d.columns:
        base["grade"] = pd.to_numeric(d[grade_col], errors="coerce").fillna(0).astype(int)
    else:
        base["grade"] = 0
        if grade_col:
            notes.append("grades pooled (not split) to keep group sizes usable")

    # ---- year (must handle "2022-23") --------------------------------------
    if year_col and year_col in d.columns:
        yr = parse_year(d[year_col])
        if yr.isna().all():
            notes.append(f"⚠️ could not read years from '{year_col}' — trends disabled")
            yr = pd.Series(0, index=d.index)
        else:
            notes.append(f"years parsed from '{year_col}': "
                         f"{', '.join(map(str, sorted(yr.dropna().unique().astype(int))))}")
    else:
        yr = pd.Series(0, index=d.index)
    base["year"] = yr.fillna(0).astype(int)

    if gender_col and gender_col in d.columns:
        base["_is_f"] = _female_mask(d[gender_col]).astype(float)
    else:
        base["_is_f"] = 0.0

    # ---- outcome: prefer the 20 question items over a single score ---------
    items = find_item_columns(d) if use_items else []
    real_comp = (comp_col and comp_col in d.columns
                 and d[comp_col].nunique() > 1)

    if items and not real_comp:
        # Each question becomes a competency; a wrong answer is "below grade".
        #
        # A pandas .melt() here materialises rows x items (2M x 20 = 40M rows)
        # and raises MemoryError at real scale. DuckDB does the UNPIVOT and the
        # GROUP BY in one streaming pass instead — measured at 1.2s / 0.4 GB on
        # 2M rows, where pandas could not complete at all.
        wide = base.join(d[items])
        qs = ", ".join(f'"{c}"' for c in items)
        con = duckdb.connect(config={"memory_limit": "3GB"})
        try:
            con.register("wide_tbl", wide)
            g = con.sql(f"""
                SELECT division, district, block, grade, competency, year,
                       COUNT(*)                                             AS n,
                       SUM(CASE WHEN correct = 0 THEN 1 ELSE 0 END)         AS below,
                       SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END)         AS above,
                       SUM(_is_f)                                           AS f_n,
                       SUM(CASE WHEN correct = 0 THEN _is_f ELSE 0 END)     AS f_below_n,
                       SUM(CASE WHEN correct = 0 THEN 1 - _is_f ELSE 0 END) AS m_below_n
                FROM (
                    SELECT division, district, block, grade, year, _is_f,
                           competency, correct
                    FROM wide_tbl
                    UNPIVOT (correct FOR competency IN ({qs}))
                )
                WHERE correct IS NOT NULL
                GROUP BY division, district, block, grade, competency, year
            """).df()
        finally:
            con.close()
        if qmap:
            # question map provided: fold the per-item rows into named
            # competencies by summing counts. Two key shapes are accepted:
            #   {"Q1": "addition"}                       — one paper for all
            #   {(2023, 5, "Q1"): "addition"}            — per year & grade
            #     (the real GP contest changes the paper every year/grade)
            _qm = dict(qmap)
            _multi = any(isinstance(k, tuple) for k in _qm)
            if _multi:
                _qm = {(int(k[0]), int(k[1]), str(k[2])): str(v)
                       for k, v in _qm.items()}
                g["competency"] = [
                    _qm.get((int(y), int(gr), str(q)), str(q))
                    for y, gr, q in zip(g["year"], g["grade"],
                                        g["competency"])]
            else:
                _qm = {str(k): str(v) for k, v in _qm.items()}
                g["competency"] = g["competency"].map(
                    lambda q: _qm.get(str(q), str(q)))
            g = (g.groupby(["division", "district", "block", "grade",
                            "competency", "year"], as_index=False)
                 [["n", "below", "above", "f_n", "f_below_n", "m_below_n"]]
                 .sum())
            notes.append(f"🗺️ question map applied: {len(items)} items folded "
                         f"into {g['competency'].nunique()} named competencies")
        else:
            notes.append(f"🧩 {len(items)} question items used as the competency "
                         f"dimension (Q1–Q{len(items)}); a wrong answer counts "
                         f"as below grade level")
        # One child = one row of `base` but `len(items)` rows of `g`. Count the
        # children now, while we still have the per-child frame.
        answered = base[d[items].notna().any(axis=1)]
        notes.append(f"👥 {len(answered):,} children → {len(answered) * len(items):,} "
                     f"responses; headcounts use children, percentages use responses")
        return _finalize(g, df, div_c, dist_c, blk_c, notes,
                         stu=_student_counts(answered))
    else:
        if not score_col or score_col not in d.columns:
            return None, "No score column and no Q1..Qn item columns found."
        scaled, how = rescale_score(d[score_col])
        notes.append(f"📏 '{score_col}': {how}")
        out = base.copy()
        out["competency"] = (d[comp_col].astype(str) if real_comp else "Overall")
        out["_below"] = (scaled < below_cut).astype(float)
        out["_above"] = (scaled >= above_cut).astype(float)
        out = out[scaled.notna()]
        notes.append(f"below grade = score < {below_cut:g}, above = ≥ {above_cut:g}")

    if out.empty:
        return None, "No usable rows after cleaning."

    out["_f_below"] = out["_below"] * out["_is_f"]
    out["_m_below"] = out["_below"] * (1 - out["_is_f"])

    keys = ["division", "district", "block", "grade", "competency", "year"]
    g = out.groupby(keys, dropna=False).agg(
        n=("_below", "size"), below=("_below", "sum"), above=("_above", "sum"),
        f_n=("_is_f", "sum"), f_below_n=("_f_below", "sum"),
        m_below_n=("_m_below", "sum"),
    ).reset_index()

    # `out` may be long (one row per child x competency) if the dashboard melted
    # the items before handing the frame over, so count per competency.
    return _finalize(g, df, div_c, dist_c, blk_c, notes,
                     stu=_student_counts(out, comp="competency"))


def _finalize(g, df, div_c, dist_c, blk_c, notes, stu=None):
    """Shared tail: derive percentages, gender gap and prior-year value."""
    g["m_n"] = g["n"] - g["f_n"]
    g["below_pct"] = (100.0 * g["below"] / g["n"]).round(1)
    g["above_pct"] = (100.0 * g["above"] / g["n"]).round(1)
    g["f_below"] = np.where(g["f_n"] > 0, 100.0 * g["f_below_n"] / g["f_n"], np.nan).round(1)
    g["m_below"] = np.where(g["m_n"] > 0, 100.0 * g["m_below_n"] / g["m_n"], np.nan).round(1)
    g["gender_gap"] = (g["f_below"] - g["m_below"]).round(1)

    if stu is not None:
        g = g.merge(stu, on=CELL_KEYS, how="left")
    if "students" not in g.columns or g["students"].isna().any():
        # Never guess: if the join missed, fall back to the response count so a
        # headcount is at worst too large, never silently absent.
        g["students"] = g.get("students", pd.Series(index=g.index, dtype=float)).fillna(g["n"])

    g = g.sort_values(["district", "block", "grade", "competency", "year"])
    g["prev_pct"] = (g.groupby(["district", "block", "grade", "competency"])
                       ["below_pct"].shift(1))

    agg = g[REQUIRED].reset_index(drop=True)
    for c in ("n", "students", "f_n", "m_n"):
        agg[c] = agg[c].fillna(0).astype(int)

    head = (f"{len(df):,} records → {len(agg):,} aggregated rows  ·  "
            f"geography: {div_c} → {dist_c} → {blk_c}")
    return agg, head + "  ·  " + "  ·  ".join(notes)


# ------------------------------------------------------------------- checks
def health(agg):
    if agg is None or agg.empty:
        return {}
    return {
        "rows": len(agg),
        "districts": int(agg["district"].nunique()),
        "blocks": int(agg["block"].nunique()),
        "competencies": int(agg["competency"].nunique()),
        "years": int(agg["year"].nunique()),
        "median_n_per_row": int(agg["n"].median()),
        "has_trend": bool(agg["prev_pct"].notna().any()),
        "has_gender": bool(agg["gender_gap"].notna().any()),
        "has_grades": bool(agg["grade"].nunique() > 1),
    }


def warnings(agg, min_n=30):
    """Statistical caveats worth showing the user before they trust the output."""
    if agg is None or agg.empty:
        return ["No aggregated data."]
    w = []
    thin = int((agg["n"] < min_n).sum())
    if thin:
        w.append(f"⚠️ {thin} of {len(agg):,} groups have fewer than {min_n} students — "
                 f"percentages there are unstable. Analyse at district level, or "
                 f"treat block figures as indicative only.")
    if not agg["prev_pct"].notna().any():
        w.append("⚠️ No year-over-year comparison available — trend, decline, "
                 "bright-spot and chronic-weakness findings are disabled.")
    if agg["year"].nunique() <= 1:
        w.append("⚠️ Only one year present in the current selection.")
    if not agg["gender_gap"].notna().any():
        w.append("⚠️ No gender column resolved — equity findings are disabled.")
    return