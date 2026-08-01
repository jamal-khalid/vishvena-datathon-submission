"""
Cross-dataset insight family: does it speak when it should and stay silent
when it shouldn't?

The critical case is the FLAT one. On the merged file we were given, the
district outcome spans 0.40 points across all 31 districts. Every correlation
against it is an artefact, and a generator that produces confident sentences
there is worse than useless.
"""
import sys
import numpy as np
import pandas as pd

REPO = r"D:\Workspace\dev_release\askingindia_ai_engine\vishvena-datathon-submission"
sys.path.insert(0, REPO)
import insights as I
import insights_cross as X

FAIL = []
def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    for line in str(detail).strip().split("\n"):
        if line:
            print(f"          {line}")
    if not ok:
        FAIL.append(name)


def fake_agg(district_rates, n_per=300, rich=False):
    """
    AGG for the cross-dataset tests.

    rich=True adds blocks, grades, years and a gender split so the PRIMARY
    generators have something to find too — without that, "cross-dataset
    findings don't crowd out primary ones" passes vacuously because there
    were no primary findings to crowd out.
    """
    g = np.random.default_rng(3)
    rows = []
    blocks = [f"{{}}-B{i}" for i in range(4)] if rich else ["{}-B1"]
    grades = (4, 5, 6) if rich else (5,)
    years = (2023, 2024) if rich else (2024,)
    for d, rate in district_rates.items():
        for bi, bt in enumerate(blocks):
            for gr in grades:
                for yr in years:
                    for c in ("C1", "C2"):
                        jit = float(g.normal(0, 6)) if rich else 0.0
                        v = min(max(rate + jit + (4 * bi if rich else 0), 1), 99)
                        prev = (min(max(v + float(g.normal(0, 3)), 1), 99)
                                if rich and yr == 2024 else None)
                        fb = min(max(v + (float(g.normal(0, 4)) if rich else 0), 1), 99)
                        mb = min(max(2 * v - fb, 1), 99)
                        rows.append({"division": "DV", "district": d,
                                     "block": bt.format(d), "grade": gr,
                                     "year": yr, "competency": c,
                                     "n": n_per, "students": n_per,
                                     "f_n": n_per // 2, "m_n": n_per - n_per // 2,
                                     "below_pct": v, "above_pct": 100 - v,
                                     "f_below": fb, "m_below": mb,
                                     "gender_gap": fb - mb, "prev_pct": prev})
    return pd.DataFrame(rows)


NAMES = [f"D{i:02d}" for i in range(31)]
rng = np.random.default_rng(11)
income = rng.uniform(50, 250, 31)
literacy = rng.uniform(55, 92, 31)

print("=" * 78)
print("CASE A — outcome is FLAT (the merged_dataset.xlsx situation)")
print("=" * 78)
flat = {n: 65.5 + rng.normal(0, 0.06) for n in NAMES}
sec = pd.DataFrame({"District": NAMES, "Per Capita Income": income,
                    "Literacy": literacy,
                    "Teachers": rng.uniform(500, 5000, 31),
                    "Libraries": rng.uniform(20, 600, 31)})
ctxA = X.prepare(fake_agg(flat), sec)
spreadA = max(flat.values()) - min(flat.values())
print(f"    outcome spread across 31 districts: {spreadA:.2f} points")
outA = X.generate(ctxA, "D00")
claims = [f for f in outA if f["source"] in
          ("x_over_under", "x_rank_shift", "x_peer_comparison",
           "x_context_explains", "x_strongest_link")]
report("makes no district claim against a flat outcome", not claims,
       "\n".join(f"said: {c['text'][:120]}" for c in claims) or
       "no over/under, rank, peer or link claims")
warned = [f for f in outA if f["source"] == "x_no_link"]
report("says out loud that there is nothing to explain", bool(warned),
       (warned[0]["text"][:150] if warned else "silent — the reader gets no "
        "warning that the outcome is flat"))

print()
print("=" * 78)
print("CASE B — outcome genuinely driven by context")
print("=" * 78)
# below% falls as income rises, plus real noise
line = 40 - 0.08 * income                      # the deterministic relationship
truth = line + rng.normal(0, 3.0, 31)          # districts scatter around it
drive = {n: float(v) for n, v in zip(NAMES, truth)}
# Plant the over-performer against the LINE, not against its own noisy value.
# Offsetting truth[0] by -9 only moves it -9 from where noise had already put
# it, which is not -9 from the model — and the generator is right to say so.
drive["D00"] = float(line[0] - 9.0)
ctxB = X.prepare(fake_agg(drive), sec)
print(f"    outcome spread: {max(drive.values()) - min(drive.values()):.1f} points")
print(f"    context model usable: {(ctxB.get('fit') or {}).get('usable')}, "
      f"adj R² = {(ctxB.get('fit') or {}).get('adj_r2'):.2f}")
outB = X.generate(ctxB, "D00", limit=6)
report("finds the planted over-performer",
       any(f["source"] == "x_over_under" for f in outB),
       "\n".join(f"[{f['source']}] {f['text'][:135]}" for f in outB))
report("detects the real income link",
       any(f["source"] == "x_strongest_link" for f in outB),
       "" if any(f["source"] == "x_strongest_link" for f in outB)
       else "x_strongest_link did not fire despite a planted relationship")

print()
print("=" * 78)
print("CASE C — an AVERAGE district must not be called exceptional")
print("=" * 78)
r = ctxB["residuals"]
mid = r.iloc[len(r) // 2][ctxB["key"]]
outC = X.generate(ctxB, mid)
ou = [f for f in outC if f["source"] == "x_over_under"]
print(f"    {mid} residual = {float(r.iloc[len(r)//2]['over_under']):+.2f} "
      f"(SD of residuals = {r['over_under'].std():.2f})")
report("no over/under claim for a middle-of-the-pack district", not ou,
       (ou[0]["text"][:130] if ou else "correctly silent"))

print()
print("=" * 78)
print("CASE D — merged into the main ranked list")
print("=" * 78)
agg = fake_agg(drive, rich=True)
plain = I.generate(agg, "D00", limit=8)
withctx = I.generate(agg, "D00", limit=8, context=ctxB)
srcs = {f["source"] for f in withctx}
report("primary generators fire on this data at all", bool(plain),
       f"primary-only sources: {sorted({f['source'] for f in plain})}")
report("context=None leaves behaviour unchanged",
       bool(plain) and all(f["source"].startswith("g_") for f in plain),
       "no x_* finding leaks in when no context is supplied")
report("cross-dataset findings join the same list",
       any(s.startswith("x_") for s in srcs),
       "\n".join(f"[{f['source']}] {f['text'][:110]}" for f in withctx))
report("primary findings are not crowded out",
       any(t.startswith("g_") for t in srcs) and any(t.startswith("x_") for t in srcs),
       "primary: " + str(sorted(t for t in srcs if t.startswith("g_")))
       + "\ncross  : " + str(sorted(t for t in srcs if t.startswith("x_"))))
report("no generator errors",
       not I.ERRORS and not X.ERRORS,
       str(I.ERRORS[:2] + X.ERRORS[:2]) or "none")

print()
print("=" * 78)
print("CASE E — no context supplied at all")
print("=" * 78)
report("prepare() returns None rather than raising",
       X.prepare(agg, None) is None and X.prepare(None, sec) is None,
       "degrades to primary-only insights")
report("generate() tolerates a None context",
       bool(I.generate(agg, "D00", limit=5, context=None)),
       "still returns primary findings")

print()
print("=" * 78)
print(f"{len(FAIL)} FAILURE(S): {FAIL}" if FAIL else "ALL CROSS-DATASET CHECKS PASSED")
print("=" * 78)
sys.exit(1 if FAIL else 0)
