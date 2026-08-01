"""
Known-answer tests for the recommendation engine.

Run:  python test_playbook.py     (exit code 1 if any rule misfires)

A wrong insight misinforms. A wrong recommendation sends a resource teacher to
the wrong block, so the bar is higher, not lower.
"""
import sys
import pandas as pd

REPO = r"D:\Workspace\dev_release\askingindia_ai_engine\vishvena-datathon-submission"
sys.path.insert(0, REPO)
import playbook as P

COLS = ["division", "district", "block", "grade", "competency", "year",
        "n", "students", "f_n", "m_n", "below_pct", "above_pct",
        "f_below", "m_below", "gender_gap", "prev_pct"]


def row(block="B1", comp="C1", grade=5, year=2024, n=100, below=50.0,
        prev=None, f_below=None, m_below=None, district="D1"):
    f_below = below if f_below is None else f_below
    m_below = below if m_below is None else m_below
    return {"division": "DV", "district": district, "block": block,
            "grade": grade, "competency": comp, "year": year,
            "n": n, "students": n, "f_n": n // 2, "m_n": n - n // 2,
            "below_pct": below, "above_pct": 100 - below,
            "f_below": f_below, "m_below": m_below,
            "gender_gap": f_below - m_below, "prev_pct": prev}


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
print("1. Does it recommend action for a 2-student block?")
print("=" * 78)
a = agg([row(block="Tiny", n=2, below=100.0, prev=0.0),
         row(block="Solid", n=5000, below=62.0, prev=61.0)])
recs = P.recommend(a, "D1", limit=10)
tiny = [r for r in recs if r["block"] == "Tiny"]
report("no P1 action from 2 children", not tiny,
       "\n".join(f"{r['priority']}: {r['recommendation'][:120]}" for r in tiny)
       or "correctly skipped")

print()
print("=" * 78)
print("2. Is 'Declining' evidence-based, or any move over 1 point?")
print("=" * 78)
# 40 children, 61% -> 62.5%. One-and-a-half points on 40 kids is noise.
a = agg([row(block="B1", n=40, below=62.5, prev=61.0)])
recs = P.recommend(a, "D1")
traj = recs[0]["trajectory"] if recs else "(none)"
report("a 1.5-point move on 40 children is not called Declining",
       traj != "Declining",
       f"classified as: {traj}\n"
       + (f"action: {recs[0]['priority']} — {recs[0]['recommendation'][:110]}"
          if recs else ""))

print()
print("=" * 78)
print("3. 'consecutive years' — are they actually consecutive?")
print("=" * 78)
# present in 2020 and 2024 only, four years apart
a = agg([row(block="B1", year=y, below=80.0, n=500) for y in (2020, 2024)])
recs = P.recommend(a, "D1")
txt = recs[0]["recommendation"] if recs else ""
report("does not claim consecutive years from a gap",
       "consecutive" not in txt.lower(),
       f"years present: 2020, 2024\nsaid: {txt[:200]}")

print()
print("=" * 78)
print("4. Worst grade — weighted by how many children are in it?")
print("=" * 78)
# Grade 4: 1000 kids at 55%.  Grade 6: 20 kids at 95%.
# Unweighted, Grade 6 looks 40 pts worse and gets the delivery focus.
a = agg([row(block="B1", grade=4, n=1000, below=55.0),
         row(block="B1", grade=6, n=20, below=95.0)])
recs = P.recommend(a, "D1")
txt = recs[0]["recommendation"] if recs else ""
report("does not target a 20-child grade over a 1000-child one",
       "Grade 6" not in txt,
       f"said: {txt[:200]}")

print()
print("=" * 78)
print("5. Gender clause — corrected for testing every cell?")
print("=" * 78)
import random
random.seed(5)
rows_ = []
for i in range(200):
    nf = nm = 80
    fb = 100.0 * sum(random.random() < 0.5 for _ in range(nf)) / nf
    mb = 100.0 * sum(random.random() < 0.5 for _ in range(nm)) / nm
    r = row(block=f"B{i}", comp="C1", n=160, below=62.0, prev=61.0,
            f_below=fb, m_below=mb)
    r["f_n"], r["m_n"] = nf, nm
    rows_.append(r)
a = agg(rows_)
recs = P.recommend(a, "D1", limit=200)
eq = [r for r in recs if "gender-responsive" in r["recommendation"]]
report("no gender clause when boys and girls are identical", not eq,
       f"200 blocks, girls and boys from the SAME 50% distribution\n"
       f"{len(eq)} recommendation(s) added a gender-equity clause")

print()
print("=" * 78)
print("6. Peer model — can it tell you to copy a tiny block?")
print("=" * 78)
a = agg([row(block="Big", comp="C1", n=4000, below=70.0),
         row(block="Big", comp="C2", n=4000, below=40.0),
         row(block="Micro", comp="C1", n=3, below=0.0),
         row(block="Micro", comp="C2", n=3, below=40.0),
         row(block="Other", comp="C1", n=900, below=68.0),
         row(block="Other", comp="C2", n=900, below=41.0)])
recs = P.recommend(a, "D1", limit=10)
copies = [r for r in recs if r.get("peer_model") == "Micro"]
report("never proposes copying a 3-student block", not copies,
       "\n".join(r["recommendation"][:150] for r in copies) or "none")

print()
print("=" * 78)
print("7. Bundle clause — is r >= 0.90 tested for significance?")
print("=" * 78)
# 4 blocks, two competencies of pure noise. r can easily exceed 0.9 on 4 points.
import numpy as np
rng = np.random.default_rng(2)
found = 0
for trial in range(40):
    rr = []
    for b in range(4):
        for c, v in (("C1", rng.uniform(40, 80)), ("C2", rng.uniform(40, 80))):
            rr.append(row(block=f"B{b}", comp=c, n=400, below=float(v),
                          prev=float(v) - 0.2))
    p, r_, pv, nb = P._bundle_partner(agg(rr), "C1", 2024)
    if p:
        found += 1
report("random noise across 4 blocks is not called a shared root cause",
       found == 0,
       f"{found} of 40 random datasets produced a 'shares a root cause' clause")

print()
print("=" * 78)
print("8. Does the file agree with itself on how many outputs exist?")
print("=" * 78)
import math, re, io as _io
space = P.combination_space()
real = len(P.BASE_ACTIONS) * sum(math.comb(len(P.MODIFIERS), i)
                                 for i in range(P.MAX_CLAUSES + 1))
doc = P.__doc__ or ""
claimed = [int(x.replace(",", "")) for x in re.findall(r"([\d,]{3,})\s*distinct", doc)]
print(f"    docstring claims          : {claimed or 'none'}")
print(f"    combination_space() says  : {space['distinct_outputs']:,}")
print(f"    independently recomputed  : {real:,}   (MAX_CLAUSES={P.MAX_CLAUSES})")
report("the reported count matches what can actually be produced",
       space["distinct_outputs"] == real,
       f"{space['base_actions']} base actions x {space['clause_subsets']} "
       f"clause subsets")
report("the docstring agrees with the code",
       bool(claimed) and all(c == real for c in claimed),
       "a hardcoded number in prose drifts the moment a clause is added")

print()
print("=" * 78)
print(f"{len(BUGS)} ISSUE(S): {BUGS}" if BUGS else "ALL CHECKS PASSED")
print("=" * 78)

import sys as _s
_s.exit(1 if BUGS else 0)
