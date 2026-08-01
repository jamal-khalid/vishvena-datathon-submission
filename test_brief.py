"""
Known-answer tests for the role-based briefs.

Run:  python test_brief.py     (exit code 1 if a brief overstates anything)

A brief is the thing an officer actually reads. If it quotes a figure the
Insights tab has already suppressed as noise, the submission contradicts
itself in front of a judge.
"""
import sys
import pandas as pd

REPO = r"D:\Workspace\dev_release\askingindia_ai_engine\vishvena-datathon-submission"
sys.path.insert(0, REPO)
import brief as B

COLS = ["division", "district", "block", "grade", "competency", "year",
        "n", "students", "f_n", "m_n", "below_pct", "above_pct",
        "f_below", "m_below", "gender_gap", "prev_pct"]


def row(block="B1", comp="C1", grade=5, year=2024, n=100, below=50.0,
        prev=None, f_below=None, m_below=None, district="D1", division="DV"):
    f_below = below if f_below is None else f_below
    m_below = below if m_below is None else m_below
    return {"division": division, "district": district, "block": block,
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
print("1. Does the block brief describe a 3-child block?")
print("=" * 78)
a = agg([row(block="Solid", n=4000, below=40.0),
         row(block="Micro", n=3, below=100.0)])
b = B.build(a, "D1", role="block")
report("does not write a brief about 3 children",
       not b or b["metrics"].get("children", 0) > 3,
       f"chose block: {b['scope'] if b else '(none)'}\n"
       + (f"said: {b['text'][:170]}" if b else ""))

print()
print("=" * 78)
print("2. Weakest competency — measured on how many children?")
print("=" * 78)
# C_big: 5,000 children at 55%.  C_tiny: 4 children at 95%.
a = agg([row(block="B1", comp="C_big", n=5000, below=55.0),
         row(block="B1", comp="C_tiny", n=4, below=95.0)])
b = B._block_brief(a, "D1", "B1")
txt = b["text"] if b else ""
report("does not call a 4-child competency the weakest area",
       "C_tiny" not in txt,
       f"said: {txt[:200]}")

print()
print("=" * 78)
print("3. Year-on-year delta — weighted by block size?")
print("=" * 78)
# A 4,000-child block improved 1 pt. A 10-child block collapsed 60 pts.
# Unweighted, the district reads as 'deteriorated by 30 points'.
a = agg([row(block="Big", n=4000, below=40.0, prev=41.0),
         row(block="Tiny", n=10, below=95.0, prev=35.0)])
b = B._district_brief(a, "D1")
txt = b["text"] if b else ""
true_w = ((40 - 41) * 4000 + (95 - 35) * 10) / 4010
import re
m = re.search(r"by (\d+) points year-on-year", txt)
said = int(m.group(1)) if m else None
report("reports the size-weighted change",
       said is None or abs(said - abs(true_w)) <= 1,
       f"true weighted change: {true_w:+.2f} pts\n"
       f"brief says: {said if said is not None else 'no change mentioned'} pts\n"
       f"text: {txt[-170:]}")

print()
print("=" * 78)
print("4. 'Significant gender gaps in N combinations' — corrected?")
print("=" * 78)
import random
random.seed(11)
rows_ = []
for i in range(150):
    nf = nm = 90
    fb = 100.0 * sum(random.random() < 0.5 for _ in range(nf)) / nf
    mb = 100.0 * sum(random.random() < 0.5 for _ in range(nm)) / nm
    r = row(block=f"B{i}", comp="C1", n=180, below=50.0, f_below=fb, m_below=mb)
    r["f_n"], r["m_n"] = nf, nm
    rows_.append(r)
b = B._district_brief(agg(rows_), "D1")
n_sig = b["metrics"]["significant_gender_gaps"] if b else -1
report("finds no gender gaps when there are none", n_sig == 0,
       f"150 blocks, girls and boys from the SAME 50% distribution\n"
       f"brief reports {n_sig} significant gender gap(s)")

print()
print("=" * 78)
print("5. Does the policy brief still ASK for a test we already run?")
print("=" * 78)
import numpy as np
import insights_cross as X
NAMES = [f"D{i}" for i in range(14)]
rng = np.random.default_rng(4)
a = agg([row(block=f"B{i}", district=NAMES[i], n=900,
             below=28.0 + 1.4 * i, prev=28.0 + 1.4 * i) for i in range(14)])
sec = pd.DataFrame({"District": NAMES,
                    "Per Capita Income": rng.uniform(60, 240, 14),
                    "Literacy": rng.uniform(55, 92, 14),
                    "Teachers": rng.uniform(400, 4000, 14)})
ctx = X.prepare(a, sec)

# WITHOUT context the honest thing is still to say the test is needed
b0 = B._policy_brief(a)
kept = "should be tested against socio-economic" in (b0["text"] if b0 else "")
report("without context, still says the test is needed", kept,
       "silence would imply the question had been settled")

# WITH context it must report the ANSWER instead
b1 = B._policy_brief(a, context=ctx)
txt = b1["text"] if b1 else ""
asks = "should be tested against socio-economic" in txt
answers = ("Tested against" in txt) or ("too few to test" in txt)
report("with context, reports the result instead of asking for the test",
       answers and not asks,
       f"context prepared: {ctx is not None}"
       + (f", {ctx['n_units']} units" if ctx else "")
       + "\nsaid: ..." + txt[max(txt.find("-point spread"), 0):][:230])

print()
print("=" * 78)
print("6. Can a brief use district context at all?")
print("=" * 78)
import inspect
sig = inspect.signature(B.build)
report("build() accepts a cross-dataset context", "context" in sig.parameters,
       f"build{sig}\n"
       "without it a policy brief cannot say whether a district beats or "
       "falls short of its circumstances")

print()
print("=" * 78)
print("7. Do briefs agree with the Insights tab on what counts as real?")
print("=" * 78)
import insights as I
a = agg([row(block="Tiny", n=6, below=100.0, prev=0.0, comp="C1"),
         row(block="Tiny", n=6, below=90.0, prev=10.0, comp="C2"),
         row(block="Big", n=5000, below=42.0, prev=41.0, comp="C1"),
         row(block="Big", n=5000, below=41.0, prev=41.0, comp="C2")])
ins = I.generate(a, "D1", limit=6, min_n=30)
ins_mentions_tiny = any("Tiny" in i["text"] for i in ins)
br = B._district_brief(a, "D1")
br_mentions_tiny = "Tiny" in (br["text"] if br else "")
report("brief and insights agree about the 6-child block",
       ins_mentions_tiny == br_mentions_tiny,
       f"Insights mention it : {ins_mentions_tiny}\n"
       f"Brief mentions it   : {br_mentions_tiny}\n"
       + (f"brief: {br['text'][:170]}" if br else ""))

print()
print("=" * 78)
print(f"{len(BUGS)} ISSUE(S): {BUGS}" if BUGS else "ALL CHECKS PASSED")
print("=" * 78)

import sys as _s
_s.exit(1 if BUGS else 0)
