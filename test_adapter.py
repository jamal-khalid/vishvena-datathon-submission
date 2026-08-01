"""
Known-answer tests for adapter.py — and for the thing that makes this dataset
awkward: the competency set CHANGES between papers.

The real file has 7 core competencies present in all 9 papers plus 3 that
rotate — fraction disappears after 2023-24, data handling arrives in 2023-24,
place value is missing from Grade 6 in the first two years. Next year's papers
will rotate differently, so these tests ask "does it hold for ANY rotation",
not "does it hold for this file".

These drive the real adapter.build_agg, not a copy of its logic.
"""
import sys

import numpy as np
import pandas as pd

REPO = r"D:\Workspace\dev_release\askingindia_ai_engine\vishvena-datathon-submission"
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + r"\streamlit_app")

import adapter                                    # noqa: E402
import insights as L_ins                           # noqa: E402
import models_ml as L_mod                          # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'BUG '}  {name}")
    for line in str(detail).strip().split("\n"):
        if line:
            print(f"          {line}")
    if not ok:
        FAIL.append(name)


def wide(rows, n_per=120):
    """rows: (year, competency, pct_below). Builds a raw per-child frame."""
    out = []
    rng = np.random.default_rng(5)
    for year, comp, below in rows:
        for blk in ("B1", "B2", "B3"):
            k = n_per
            scores = np.where(rng.random(k) < (below / 100.0), 20.0, 80.0)
            out.append(pd.DataFrame({
                "Division": "DV", "District": "D1", "Block": blk,
                "Year": year, "Grade": 5, "Competency": comp,
                "Gender": ["female", "male"] * (k // 2),
                "Score": scores}))
    return pd.concat(out, ignore_index=True)


def build(df):
    agg, _note = adapter.build_agg(
        df, hierarchy=["Division", "District", "Block"], score_col="Score",
        year_col="Year", gender_col="Gender", comp_col="Competency",
        grade_col="Grade", below_cut=50.0, above_cut=75.0, use_items=False)
    return agg


def prev_of(agg, comp, year):
    hit = agg[(agg["competency"] == comp) & (agg["year"] == year)]
    return None if hit.empty else hit["prev_pct"].iloc[0]


print("=" * 74)
print("prev_pct WHEN THE COMPETENCY SET MOVES")
print("=" * 74)

# `fraction` is skipped in 2023 while the dataset clearly HAS 2023.
rows = [(2022, "fraction", 40.0), (2024, "fraction", 70.0)]
rows += [(y, c, 50.0) for c in ("addition", "division")
         for y in (2022, 2023, 2024)]
agg = build(wide(rows))
p = prev_of(agg, "fraction", 2024)
check("a competency that skips a year gets NO previous year",
      p is None or pd.isna(p),
      f"fraction: 40% in 2022, not tested 2023, 70% in 2024\n"
      f"prev_pct for 2024 = {p}\n"
      "shift(1) steps back one ROW, not one year — unguarded it compares\n"
      "2024 against 2022 and every 'deteriorated since last year' sentence\n"
      "downstream silently spans two years and an untested one between")

# a genuinely consecutive pair must still work
p = prev_of(agg, "addition", 2024)
check("a competency present every year keeps its previous year",
      p is not None and not pd.isna(p),
      f"addition prev_pct for 2024 = {p} (must not be blanked)")

# biennial file: the dataset itself only has 2022 and 2024
rows = [(y, c, 50.0) for c in ("addition", "division") for y in (2022, 2024)]
agg2 = build(wide(rows))
p = prev_of(agg2, "addition", 2024)
check("a biennial dataset still has trends",
      p is not None and not pd.isna(p),
      "dataset years are 2022 and 2024 with no 2023 at all, so 2022 IS the\n"
      "preceding year here; a naive year-1 rule would wipe out every trend\n"
      f"addition prev_pct for 2024 = {p}")

# a competency introduced mid-series has nothing before it
rows = [(2023, "data handling", 60.0), (2024, "data handling", 55.0)]
rows += [(y, c, 50.0) for c in ("addition",) for y in (2022, 2023, 2024)]
agg3 = build(wide(rows))
p = prev_of(agg3, "data handling", 2023)
check("a newly introduced competency has no prior year",
      p is None or pd.isna(p),
      f"data handling first appears in 2023; prev_pct = {p}")

print()
print("=" * 74)
print("A RAGGED COMPETENCY SET SURVIVES THE WHOLE PIPELINE")
print("=" * 74)
rows = []
for y in (2022, 2023, 2024):
    for c in ("addition", "division", "measurement"):
        rows.append((y, c, 45.0 + (y - 2022) * 3))
rows += [(y, "fraction", 55.0) for y in (2022, 2023)]        # leaves
rows += [(y, "data handling", 35.0) for y in (2023, 2024)]   # arrives
rows += [(2024, "place value", 60.0)]                        # one year only
agg = build(wide(rows))
present = (agg.pivot_table(index="competency", columns="year", values="n",
                           aggfunc="size").fillna(0).astype(int))
print(present.to_string().replace("\n", "\n    "))
check("the aggregate carries every competency that WAS tested",
      set(present.index) == {"addition", "division", "measurement",
                             "fraction", "data handling", "place value"},
      f"competencies in aggregate: {sorted(present.index)}")
check("a competency is absent, not zero, in a year it was not tested",
      len(agg[(agg["competency"] == "place value") & (agg["year"] < 2024)]) == 0,
      "place value appears only in 2024; earlier years must have NO ROW,\n"
      "never a row with 0 — 'not asked' and 'all correct' must not look alike")

ins = L_ins.generate(agg, "D1", limit=10, min_n=30)
# `place value` exists in 2024 only, and `fraction` vanished after 2023.
# Neither has a comparable prior year, so no finding may state a change for
# them — a trend sentence there would be manufactured from a missing paper.
_trendy = ("deteriorat", "improv", "from ", "rose", "fell", "points in one year")
_bad = [i["text"] for i in ins
        if any(c in i["text"].lower() for c in ("place value", "fraction"))
        and any(w in i["text"].lower() for w in _trendy)]
check("no trend is claimed for a competency without a comparable prior year",
      not _bad,
      f"{len(ins)} findings generated; place value is 2024-only and fraction\n"
      f"ended in 2023, so neither can carry a year-over-year claim\n"
      f"offending: {_bad[:1]}")
check("the weakest-competency finding counts only competencies tested THAT year",
      not [i for i in ins if "fraction" in i["text"].lower()],
      "fraction was not tested in 2024, so it cannot be the weakest "
      "competency of 2024")

fm = L_mod.block_feature_matrix(agg, min_n=30)
zeros = 0 if fm is None or fm.empty else int((fm == 0).sum().sum())
check("the clustering feature matrix has no fabricated zeros", zeros == 0,
      f"matrix {None if fm is None else fm.shape}, exact-zero cells {zeros}\n"
      "0 would read as 'nobody below grade level' — a PERFECT score — for a\n"
      "competency that was never asked")

print()
print("=" * 74)
print(f"{len(FAIL)} FAILURE(S): {FAIL}" if FAIL else "ALL ADAPTER CHECKS PASSED")
print("=" * 74)
sys.exit(1 if FAIL else 0)
