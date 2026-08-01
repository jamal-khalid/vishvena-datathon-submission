"""
Known-answer tests for every insight generator.

Run:  python test_insights.py     (exit code 1 if any generator lies)

Each case builds a tiny AGG where the TRUE answer is known by construction,
runs the generator, and checks the sentence against the truth. A generator
that says something false on data we designed will say something false on
the judges' data too.
"""
import sys, itertools
import pandas as pd

SHARE = r"D:\Workspace\dev_release\askingindia_ai_engine\datathon_share"
sys.path.insert(0, SHARE)
import insights as I
MN = 1   # gate opened; case 10 tests the gate itself

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
def report(name, ok, detail):
    print(f"  {'PASS' if ok else 'BUG '}  {name}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"          {line}")
    if not ok:
        BUGS.append(name)


print("=" * 78)
print("1. g_worst_competency — is the district rate weighted by block size?")
print("=" * 78)
# C1: one huge block at 10% + one tiny block at 90%  -> true district rate ~10.8%
# C2: every block at 40%                             -> true district rate 40%
# C2 is genuinely worse. An UNWEIGHTED mean makes C1 look like 50%.
a = agg([row(block="Big", comp="C1", n=1000, below=10.0),
         row(block="Tiny", comp="C1", n=10, below=90.0),
         row(block="Big", comp="C2", n=1000, below=40.0),
         row(block="Tiny", comp="C2", n=10, below=40.0)])
true_c1 = (1000 * 10 + 10 * 90) / 1010
true_c2 = 40.0
out = I.g_worst_competency(a, "D1", min_n=MN)
said = out[0]["text"] if out else "(nothing)"
picked = "C1" if "**C1**" in said else ("C2" if "**C2**" in said else "?")
report("weights blocks by size", picked == "C2",
       f"true rates: C1 = {true_c1:.1f}%, C2 = {true_c2:.1f}%  -> C2 is worse\n"
       f"generator picked: {picked}\n"
       f"said: {said}")

print()
print("=" * 78)
print("2. g_outlier_block — is the z-score computed against the OTHER blocks?")
print("=" * 78)
# Including the candidate in its own mean and SD caps the achievable z at
# (k-1)/sqrt(k). With 3 blocks that is 1.155, so a `z >= 1.5` filter could
# never fire however extreme the outlier was. Leaving the candidate out
# removes the ceiling.
for k in (3, 4, 5, 6):
    vals = [0.0] * (k - 1) + [100.0]          # the most extreme case there is
    s = pd.Series(vals)
    zin = (s.max() - s.mean()) / s.std()
    peers = pd.Series(vals[:-1])
    zout = ((s.max() - peers.mean()) / peers.std()
            if peers.std() else float("inf"))
    print(f"    {k} blocks -> z including self = {zin:.3f}"
          f"   leave-one-out = {zout:.3f}")
# only 2 peers: not enough to say what "normal" is, so silence is correct
a3 = agg([row(block=f"B{i}", below=v)
          for i, v in enumerate([0.0, 0.0, 100.0])])
out3 = I.g_outlier_block(a3, "D1", min_n=MN)
report("stays silent with only 2 peer blocks", not out3,
       "two peers cannot establish a district norm to be an outlier from")
# 4 peers around 11%, one at 90%: unmistakably an outlier, must be reported
a5 = agg([row(block=f"B{i}", below=v)
          for i, v in enumerate([10.0, 12.0, 11.0, 13.0, 90.0])])
out5 = I.g_outlier_block(a5, "D1", min_n=MN)
report("fires once there are enough peers", bool(out5),
       (out5[0]["text"][:130] if out5 else "MISSED a genuine outlier"))

print()
print("=" * 78)
print("3. g_persistence — does 'every one of the N years' actually check years?")
print("=" * 78)
# ONE year of data, but three grades. seen = 3 rows, years = 1.
a = agg([row(grade=g, year=2024, below=80.0) for g in (4, 5, 6)])
out = I.g_persistence(a, "D1", min_n=MN)
print(f"    data: 1 year, 3 grades -> {len(a)} rows")
report("does not claim multi-year evidence from one year", not out,
       (f"said: {out[0]['text']}" if out else "correctly stayed silent"))

# now the reverse: a block seen in every year but with grades, 3 years
a = agg([row(grade=g, year=y, below=80.0)
         for y in (2022, 2023, 2024) for g in (4, 5, 6)])
out = I.g_persistence(a, "D1", min_n=MN)
report("still fires on genuine 3-year chronic weakness", bool(out),
       (f"said: {out[0]['text'][:100]}" if out else "MISSED a real finding"))

print()
print("=" * 78)
print("4. g_grade_progression — does it look at the middle grades?")
print("=" * 78)
# Grade 4 = 50, Grade 5 = 90, Grade 6 = 52.  Endpoints differ by only +2, but
# the real story is a catastrophic Grade 5. Comparing only min and max grade
# reports "gaps widen by 2 points" and misses it entirely.
a = agg([row(grade=4, below=50.0), row(grade=5, below=90.0),
         row(grade=6, below=52.0)])
out = I.g_grade_progression(a, "D1", min_n=MN)
said = out[0]["text"] if out else "(nothing)"
report("notices the worst grade is in the middle", "Grade 5" in said or not out,
       f"grades: 4 = 50%, 5 = 90%, 6 = 52%\nsaid: {said}")

print()
print("=" * 78)
print("5. g_grade_progression — is 'compounding' a cohort claim?")
print("=" * 78)
a = agg([row(grade=4, below=30.0), row(grade=6, below=70.0)])
out = I.g_grade_progression(a, "D1", min_n=MN)
said = out[0]["text"] if out else ""
claims_cohort = "compounding" in said.lower()
report("avoids implying the same children were followed", not claims_cohort,
       "Grade 4 and Grade 6 are DIFFERENT children measured in the same year.\n"
       "'Gaps are compounding' is a longitudinal claim from cross-sectional data.\n"
       f"said: {said}")

print()
print("=" * 78)
print("6. g_scale — does its score reflect how big the burden is?")
print("=" * 78)
small = agg([row(block="B1", n=50, below=20.0), row(block="B2", n=40, below=10.0)])
huge = agg([row(block="B1", n=100000, below=95.0), row(block="B2", n=40, below=10.0)])
s1 = I.g_scale(small, "D1", min_n=MN)[0]["score"]
s2 = I.g_scale(huge, "D1", min_n=MN)[0]["score"]
report("score scales with the number of children affected", s2 > s1,
       f"10 children affected  -> score {s1}\n"
       f"95,000 children affected -> score {s2}\n"
       "score is hardcoded, so this generator always ranks in the same place")

print()
print("=" * 78)
print("7. g_gender_gap — is it corrected for testing hundreds of cells?")
print("=" * 78)
# 300 cells, NO real gender difference anywhere: girls and boys identical
# at 50% in expectation. Sampling noise alone will push some cells past
# p < 0.05, and the generator reports the most extreme one it finds.
import random
random.seed(7)
rows_ = []
for i in range(300):
    nf = nm = 60
    fb = 100.0 * sum(random.random() < 0.5 for _ in range(nf)) / nf
    mb = 100.0 * sum(random.random() < 0.5 for _ in range(nm)) / nm
    r = row(block=f"B{i}", comp=f"C{i%20}", n=120, below=50.0,
            f_below=fb, m_below=mb)
    r["f_n"], r["m_n"] = nf, nm
    rows_.append(r)
a = agg(rows_)
out = I.g_gender_gap(a, "D1", min_n=MN)
report("stays silent when there is no real gender difference", not out,
       "300 cells, girls and boys drawn from the SAME 50% distribution.\n"
       + (f"said: {out[0]['text'][:150]}" if out
          else "correctly found nothing"))

print()
print("=" * 78)
print("8. Do generators fail loudly, or silently look like 'nothing found'?")
print("=" * 78)
def _boom(agg, district, year=None, min_n=MN):
    raise ValueError("simulated bug inside a generator")
_boom.__name__ = "g_worst_block"
_i = [f.__name__ for f in I.GENERATORS].index("g_worst_block")
_keep = I.GENERATORS[_i]
I.GENERATORS[_i] = _boom
_desc = I.describe(agg([row(below=90.0)]), "D1", min_n=MN)
_shown = [r["Fired?"] for r in _desc if r["Generator"] == "Worst block"]
I.GENERATORS[_i] = _keep
report("a crashing generator is distinguishable from an empty one",
       _shown == ["⚠️ error"],
       f"reported as {_shown}; ERRORS log: "
       f"{I.ERRORS[-1] if I.ERRORS else 'empty'}")

print()
print("=" * 78)
print("9. g_within_district_spread — does it have any threshold at all?")
print("=" * 78)
a = agg([row(block="B1", below=50.0), row(block="B2", below=50.4)])
out = I.g_within_district_spread(a, "D1", min_n=MN)
report("suppresses a spread of well under one point", not out,
       f"blocks differ by 0.4 points\n"
       + (f"said: {out[0]['text'][:130]}" if out else "stayed silent"))

print()
print("=" * 78)
print("10. Is any generator gated on sample size?")
print("=" * 78)
a = agg([row(block="Tiny", n=2, below=100.0, prev=0.0),
         row(block="Solid", n=5000, below=61.0, prev=60.0)])
res = I.generate(a, "D1", limit=7, min_n=30)
tiny = [r for r in res if "Tiny" in r["text"]]
report("does not headline a 2-student block", not tiny,
       f"Tiny: 2 students, 0% -> 100%   Solid: 5,000 students, 60% -> 61%\n"
       + ("\n".join(f"reported: {r['text'][:110]}" for r in res[:3])))

print()
print("=" * 78)
print(f"{len(BUGS)} FAILURE(S)" if BUGS else "ALL GENERATOR CHECKS PASSED")
for b in BUGS:
    print(f"  - {b}")
print("=" * 78)

import sys as _s
_s.exit(1 if BUGS else 0)
