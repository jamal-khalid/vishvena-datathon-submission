"""
Known-answer tests for models_ml.py — clustering, benchmarks, what-if.

These feed three tabs: Archetypes & Risk, What-If, and Prediction.
"""
import sys, pathlib
import numpy as np
import pandas as pd

REPO = r"D:\Workspace\dev_release\askingindia_ai_engine\vishvena-datathon-submission"
sys.path.insert(0, REPO)
import models_ml as M

COLS = ["division", "district", "block", "grade", "competency", "year",
        "n", "students", "f_n", "m_n", "below_pct", "above_pct",
        "f_below", "m_below", "gender_gap", "prev_pct"]


def row(block="B1", comp="C1", grade=5, year=2024, n=100, below=50.0,
        prev=None, district="D1"):
    return {"division": "DV", "district": district, "block": block,
            "grade": grade, "competency": comp, "year": year,
            "n": n, "students": n, "f_n": n // 2, "m_n": n - n // 2,
            "below_pct": below, "above_pct": 100 - below,
            "f_below": below, "m_below": below, "gender_gap": 0.0,
            "prev_pct": prev}


def agg(rows):
    return pd.DataFrame(rows, columns=COLS)


BUGS = []
def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'BUG '}  {name}")
    for line in str(detail).strip().split("\n"):
        if line:
            print(f"          {line}")
    if not ok:
        BUGS.append(name)


print("=" * 78)
print("1. Missing competency data — treated as MISSING or as PERFECT?")
print("=" * 78)
# Two blocks. Both terrible at C1. "Gap" has NO data at all for C2.
a = agg([row(block="Gap",  comp="C1", below=90.0),
         row(block="Full", comp="C1", below=90.0),
         row(block="Full", comp="C2", below=88.0)])
piv = M.block_feature_matrix(a, min_n=1)
present = ("D1", "Gap") in set(piv.index)
val = float(piv.loc[("D1", "Gap"), "C2"]) if present else None
# correct = the block is dropped, or C2 is NaN. Wrong = C2 stored as 0.
report("a competency with no data is not scored as 0% below grade",
       (not present) or (val is None) or (val != 0.0 and not (val == val)),
       f"'Gap' has no C2 rows at all.\n"
       f"in the feature matrix: "
       + (f"C2 = {val}" if present else "the block is excluded (correct)")
       + "\n0 would mean 'nobody below grade level' — a PERFECT score — so "
         "KMeans\nwould cluster it as excellent at a subject never measured")

print()
print("=" * 78)
print("2. Are 3-student blocks clustered and named alongside real ones?")
print("=" * 78)
rows_ = []
for i in range(6):
    for c in ("C1", "C2", "C3"):
        rows_.append(row(block=f"Big{i}", comp=c, n=900, below=40.0 + i))
for c in ("C1", "C2", "C3"):
    rows_.append(row(block="Micro", comp=c, n=3, below=100.0))
piv, names = M.cluster_blocks(agg(rows_), k=3)
in_out = "Micro" in set(piv["block"])
report("a 3-student block is excluded from the archetypes", not in_out,
       f"blocks clustered: {sorted(set(piv['block']))}\n"
       "a 3-child block defines its own archetype and skews the centroids")

print()
print("=" * 78)
print("3. what_if — does it weight blocks by size?")
print("=" * 78)
# Block Big: 3,000 children at 55%.  Block Tiny: 6 children at 100%.
a = agg([row(block="Big",  comp="C1", grade=4, n=1500, below=55.0),
         row(block="Big",  comp="C1", grade=5, n=1500, below=55.0),
         row(block="Tiny", comp="C1", grade=4, n=6,    below=100.0)])
w = M.what_if(a, "D1", "C1", n_blocks=2)
before = w.get("before_below_pct") if isinstance(w, dict) else None
true_w = (55.0 * 3000 + 100.0 * 6) / 3006
report("'before' is the child-weighted rate",
       before is not None and abs(before - true_w) < 1.0,
       f"true child-weighted rate: {true_w:.2f}%\n"
       f"what_if reports 'before' = {before}\n"
       f"unweighted mean of the two block means = {(55.0 + 100.0) / 2:.2f}%")

print()
print("=" * 78)
print("4. what_if — does it target the smallest blocks?")
print("=" * 78)
rows_ = [row(block="Tiny", comp="C1", n=4, below=100.0)]
for i in range(6):
    rows_.append(row(block=f"Real{i}", comp="C1", n=1200, below=60.0 + i))
w = M.what_if(agg(rows_), "D1", "C1", n_blocks=3)
blocks = w.get("blocks") if isinstance(w, dict) else None
names_hit = str(w)[:600]
report("a 4-student block is not in the intervention target list",
       "Tiny" not in names_hit,
       f"what_if picks the n_blocks with the HIGHEST below%, which is the\n"
       f"smallest block every time\nreturned: {names_hit[:220]}")

print()
print("=" * 78)
print("5. improvement_benchmarks — 'typical' improvement of what population?")
print("=" * 78)
# 20 blocks improve by 2 pts, 20 blocks WORSEN by 10 pts.
rows_ = []
for i in range(20):
    rows_.append(row(block=f"Up{i}",   comp="C1", n=500, below=48.0, prev=50.0))
    rows_.append(row(block=f"Down{i}", comp="C1", n=500, below=60.0, prev=50.0))
b = M.improvement_benchmarks(agg(rows_))
print(f"    half the blocks improved 2 pts, half worsened 10 pts")
print(f"    net change across all blocks: {(20*2 + 20*-10)/40:+.1f} pts")
print(f"    benchmarks: typical={b['typical']}, strong={b['strong']}, "
      f"best={b['best']}, n_observed={b['n_observed']}")
report("the benchmark makes clear it is conditioned on improving blocks only",
       b.get("share_improving") is not None,
       "quantiles are taken over gains > 0 only, so 'typical' is the typical\n"
       "gain AMONG IMPROVERS, not what a randomly chosen block should expect;\n"
       "with no share_improving field the caller cannot tell the difference")

print()
print("=" * 78)
print("6. natural_rebound — weighted?")
print("=" * 78)
a = agg([row(block="Big",  comp="C1", n=4000, below=79.0, prev=80.0),
         row(block="Tiny", comp="C1", n=8,    below=20.0, prev=80.0)])
r = M.natural_rebound(a, min_n=1)
true_w = ((80 - 79) * 4000 + (80 - 20) * 8) / 4008
report("rebound is child-weighted", abs(r - true_w) < 0.5,
       f"true child-weighted rebound: {true_w:.2f} pts\n"
       f"natural_rebound returns: {r:.2f} pts\n"
       f"unweighted mean of the two = {((80-79)+(80-20))/2:.2f} pts")

print()
print("=" * 78)
print("7. Is the known-circular train_risk() still shipped?")
print("=" * 78)
import inspect
has = hasattr(M, "train_risk")
doc = (M.train_risk.__doc__ or "") if has else ""
used = False
try:
    src = open(REPO + r"\streamlit_app\streamlit_app.py", encoding="utf-8").read()
    used = "train_risk" in src
except Exception:
    pass
report("the circular model is not exposed as a usable function",
       (not has) or used is False and "DEPRECATED" in doc.upper(),
       f"train_risk exists: {has}; referenced by the dashboard: {used}\n"
       f"its own comment says it 'just re-described the present'")

print()
print("=" * 78)
print(f"{len(BUGS)} ISSUE(S): {BUGS}" if BUGS else "ALL CHECKS PASSED")
print("=" * 78)

import sys as _s
_s.exit(1 if BUGS else 0)
