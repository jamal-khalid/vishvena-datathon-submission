"""
Known-answer tests for competency.py (Competency Report) and verbalize.py
(Facts & Health) — the last two unaudited modules that write user-facing text.
"""
import sys
import numpy as np
import pandas as pd

REPO = r"D:\Workspace\dev_release\askingindia_ai_engine\vishvena-datathon-submission"
sys.path.insert(0, REPO)
import competency as C
import verbalize as V

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
print("competency.py — Competency Report")
print("=" * 78)

print("\n1. Worst block — is it just the smallest block?")
a = agg([row(block="Big",   n=4000, below=55.0),
         row(block="Mid",   n=1200, below=52.0),
         row(block="Micro", n=3,    below=100.0)])
r = C.report(a, "D1", "C1")
worst = r["geography"]["worst_block"] if r else ""
report("the worst block is not a 3-student block", "Micro" not in worst,
       f"worst_block reported as: {worst}\n"
       f"the district's real problem is Big (4,000 children at 55%)")

print("\n2. Spread and CV — inflated by tiny blocks?")
sp = r["geography"]["spread_pts"] if r else None
cv = r["geography"]["coefficient_of_variation"] if r else None
big_only = agg([row(block="Big", n=4000, below=55.0),
                row(block="Mid", n=1200, below=52.0)])
r2 = C.report(big_only, "D1", "C1")
report("spread is not driven by a 3-child block",
       sp is not None and abs(sp - r2["geography"]["spread_pts"]) < 2,
       f"with the 3-child block: spread {sp} pts, CV {cv}%\n"
       f"without it:            spread {r2['geography']['spread_pts']} pts, "
       f"CV {r2['geography']['coefficient_of_variation']}%\n"
       f"the summary sentence keys off CV >= 20 to call a district 'highly uneven'")

print("\n3. Grade slope — weighted, and is 'compounds' a cohort claim?")
a = agg([row(grade=4, n=3000, below=50.0),
         row(grade=5, n=3000, below=50.0),
         row(grade=6, n=20,   below=95.0)])
r3 = C.report(a, "D1", "C1")
txt = r3["grade"]["summary"] if r3 else ""
slope = r3["grade"]["slope_per_grade"] if r3 else None
report("a 20-child grade does not create a 'gap compounds' finding",
       "compounds" not in txt,
       f"grades: 4 = 50% (3,000 kids), 5 = 50% (3,000), 6 = 95% (20)\n"
       f"slope {slope} pts/grade — said: {txt}\n"
       "also: these are different children in the same year, not a cohort")

print("\n4. Blocks-critical count — size gated?")
rows_ = [row(block=f"Real{i}", n=1500, below=40.0) for i in range(4)]
rows_ += [row(block=f"Micro{i}", n=2, below=100.0) for i in range(4)]
r4 = C.report(agg(rows_), "D1", "C1")
crit = r4["risk"]["blocks_critical"] if r4 else ""
wide = r4["risk"]["widespread"] if r4 else None
report("2-student blocks do not trigger a district-wide response",
       not wide,
       f"4 real blocks at 40%, 4 blocks of 2 children at 100%\n"
       f"blocks_critical = {crit}, widespread = {wide}\n"
       f"said: {r4['risk']['summary'][:110] if r4 else ''}")

print("\n5. correlation_matrix — any significance test or size floor?")
rng = np.random.default_rng(3)
hits = 0
for _ in range(30):
    rr = []
    for b in range(4):                      # only 4 blocks — r is unstable
        for c in ("C1", "C2"):
            rr.append(row(block=f"B{b}", comp=c, n=500,
                          below=float(rng.uniform(30, 70))))
    m = C.correlation_matrix(agg(rr), "D1")
    if m is not None:
        pairs = C.strongest_pairs(m, top=1)
        if pairs and abs(pairs[0]["r"]) >= 0.9:
            hits += 1
report("random noise across 4 blocks is not reported as a strong pairing",
       hits == 0,
       f"{hits} of 30 random datasets produced a |r| >= 0.90 'strongest pair'\n"
       "strongest_pairs() returns r with no p-value and no block count")

print()
print("=" * 78)
print("verbalize.py — Facts & Health")
print("=" * 78)

print("\n6. competency_table — weighted rate?")
a = agg([row(block="Big",  comp="C1", n=4000, below=40.0),
         row(block="Tiny", comp="C1", n=10,   below=100.0)])
t = V.competency_table(a, "D1")
got = float(t.loc[t["competency"] == "C1", "below_pct"].iloc[0])
true_w = (40.0 * 4000 + 100.0 * 10) / 4010
report("the competency rate is child-weighted", abs(got - true_w) < 1.0,
       f"true child-weighted: {true_w:.2f}%\n"
       f"competency_table reports: {got}%   (unweighted mean = "
       f"{(40.0 + 100.0) / 2:.1f}%)\n"
       f"'students' in the same row IS summed, so the table mixes an "
       f"unweighted rate\nwith a weighted headcount")

print("\n7. Does Facts & Health agree with Insights on what is reportable?")
import insights as I
print(f"    verbalize MIN_N_FOR_TREND = {V.MIN_N_FOR_TREND}")
print(f"    insights  MIN_N           = {I.MIN_N}")
a = agg([row(block="Small", n=12, below=90.0, prev=40.0)])
s, _ = V.verbalize_district(a, "D1")
states_trend = "worsened" in s[0] or "improved" in s[0]
ins = I.generate(a, "D1", limit=5, min_n=I.MIN_N)
ins_says = any("Small" in i["text"] for i in ins)
report("the two layers apply the same reporting floor",
       states_trend == ins_says,
       f"12 children, 40% -> 90%\n"
       f"Facts & Health states a trend : {states_trend}\n"
       f"Insights reports it           : {ins_says}\n"
       f"said: {s[0][:160]}")

print()
print("=" * 78)
print(f"{len(BUGS)} ISSUE(S): {BUGS}" if BUGS else "ALL CHECKS PASSED")
print("=" * 78)

import sys as _s
_s.exit(1 if BUGS else 0)
