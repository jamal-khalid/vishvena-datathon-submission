"""
The indicator-family and competency-predictability layer.

Two claims are being made on screen and both are easy to fake:

  1. "things about people track results more than things about buildings" —
     which is only meaningful if the RATES are actually computed from the
     right denominators, and if a genuinely strong infrastructure signal
     would still be reported when one exists.

  2. "context explains X% of this competency" — which is only honest if the
     number is out-of-sample. An in-sample R^2 on ~29 districts with 4
     predictors is inflated by construction, so the test plants pure noise
     and requires the reported figure to collapse.
"""
import sys
import numpy as np
import pandas as pd

REPO = r"D:\Workspace\dev_release\askingindia_ai_engine\vishvena-datathon-submission"
sys.path.insert(0, REPO)
import secondary as S

FAIL = []
def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    for line in str(detail).strip().split("\n"):
        if line:
            print(f"          {line}")
    if not ok:
        FAIL.append(name)


def context_frame(n=29, seed=0):
    """A district context file with the same column names as the real one."""
    g = np.random.default_rng(seed)
    hh = g.integers(200_000, 1_200_000, n)
    enrol = g.integers(50_000, 700_000, n)
    return pd.DataFrame({
        "District": [f"D{i:02d}" for i in range(n)],
        "Per Capita Income": g.integers(120_000, 780_000, n),
        "Total Libraries": g.integers(80, 600, n).astype(float),
        "Total Household": hh,
        "Rural Household": (hh * g.uniform(0.3, 0.8, n)).astype(int),
        "Urban Household": (hh * g.uniform(0.2, 0.6, n)).astype(int),
        "Rural Total Literacy": g.integers(200_000, 2_000_000, n),
        "Rural Female Literacy": g.integers(100_000, 900_000, n),
        "Urban Total Literacy": g.integers(200_000, 6_000_000, n),
        "Urban Female Literacy": g.integers(90_000, 3_000_000, n),
        "Student Enrolment": enrol,
        "Primary Teachers Total": g.integers(2_000, 22_000, n),
        "Primary Teachers Female": g.integers(1_000, 11_000, n),
        "Student Teacher Ratio": g.integers(14, 66, n),
        "Internet": g.integers(200, 1_100, n),
        "Computer Available": g.integers(400, 2_300, n),
        "Primary+Upper Primary Teachers (Total)": g.integers(1_500, 6_000, n),
    })


print("=" * 78)
print("CASE A — rates are built from the RIGHT denominator")
print("=" * 78)
df = context_frame()
ind, fam, skipped = S.build_indicators(df)
expect = (df["Total Libraries"] / df["Total Household"] * 1e5)
got = ind["Libraries per 100k households"]
report("libraries are divided by households, not by students",
       np.allclose(got, expect),
       f"first district: {got.iloc[0]:.2f} vs hand-computed {expect.iloc[0]:.2f}")
expect2 = df["Internet"] / df["Computer Available"] * 100
report("connected-computer share is internet / computers",
       np.allclose(ind["Connected computer share (%)"], expect2),
       f"{ind['Connected computer share (%)'].min():.0f}% to "
       f"{ind['Connected computer share (%)'].max():.0f}%")
report("nothing was silently skipped on a complete file", not skipped,
       f"skipped: {skipped or '(none)'}")
report("every indicator carries a family",
       all(k in fam for k in ind.columns),
       f"families present: {sorted(set(fam.values()))}")

print()
print("=" * 78)
print("CASE B — a zero denominator must not become an infinite rate")
print("=" * 78)
z = context_frame()
z.loc[0, "Computer Available"] = 0
z.loc[1, "Total Household"] = 0
iz, _, _ = S.build_indicators(z)
bad = [c for c in iz.columns if np.isinf(iz[c]).any()]
report("no infinities anywhere", not bad, f"columns with inf: {bad or '(none)'}")
report("the zero-denominator district became missing, not huge",
       pd.isna(iz["Connected computer share (%)"].iloc[0]),
       f"value = {iz['Connected computer share (%)'].iloc[0]}")

print()
print("=" * 78)
print("CASE C — a missing column is REPORTED, not silently dropped")
print("=" * 78)
miss = context_frame().drop(columns=["Total Libraries", "Internet"])
im, _, sk = S.build_indicators(miss)
report("the affected indicators are named in `skipped`",
       "Libraries per 100k households" in sk
       and "Internet per 1000 students" in sk,
       f"skipped: {sk}")
report("they do not appear as all-NaN columns",
       "Libraries per 100k households" not in im.columns,
       "a NaN column would quietly shrink n for that row only")

print()
print("=" * 78)
print("CASE D — the headline claim is EARNED, not hard-coded")
print("=" * 78)
# Plant the OPPOSITE of the real-world finding: infrastructure drives the
# outcome and human capital does not. If the layer still reports human
# capital as the stronger family, it is asserting a conclusion rather than
# measuring one.
flip = context_frame(seed=7)
rate = flip["Computer Available"] / flip["Student Enrolment"] * 1000
flip["Below grade level (%)"] = 60 - 3.0 * (rate - rate.mean()) / rate.std()
res = S.indicator_families(flip, "Below grade level (%)")
byf = res["by_family"].set_index("family")["mean_abs_r"]
report("a planted INFRASTRUCTURE effect is reported as infrastructure",
       byf.get("Infrastructure", 0) > byf.get("Human capital", 0),
       "\n".join(f"{k}: {v:.2f}" for k, v in byf.items()))
top = res["table"].iloc[0]
report("the specific planted indicator ranks first",
       "Computers per 1000 students" in str(top["indicator"]),
       f"top indicator was {top['indicator']} (r={top['r']:+.2f})")

print()
print("=" * 78)
print("CASE E — pure noise must not produce a confirmed finding")
print("=" * 78)
noise = context_frame(seed=3)
noise["Below grade level (%)"] = np.random.default_rng(5).normal(55, 8,
                                                                len(noise))
rn = S.indicator_families(noise, "Below grade level (%)")
nsurv = int(rn["table"]["survives"].sum())
report("no indicator survives FDR against random noise", nsurv == 0,
       f"{nsurv} survived; smallest corrected p = "
       f"{rn['table']['p_adj'].min():.3f}")

print()
print("=" * 78)
print("CASE F — the teacher-column contradiction is detected")
print("=" * 78)
tc = rn.get("teacher_conflict")
report("the impossible ordering is counted",
       tc is not None and tc["n_bad"] > 0,
       f"{tc['n_bad']} of {tc['n']} districts flagged" if tc else "not checked")
clean = context_frame()
clean["Primary+Upper Primary Teachers (Total)"] = (
    clean["Primary Teachers Total"] + 500)
tc2 = S._teacher_column_conflict(clean)
report("a CONSISTENT file raises no flag", tc2 is not None and tc2["n_bad"] == 0,
       f"n_bad = {tc2['n_bad'] if tc2 else 'n/a'} when the totals are coherent")

print()
print("=" * 78)
print("CASE G — competency predictability is genuinely out-of-sample")
print("=" * 78)
cp = context_frame(seed=11)
ind2, _, _ = S.build_indicators(cp)
for c in ind2.columns:
    cp[c] = ind2[c]
ctrl = ["Per capita income", "Rural literates per household",
        "Student-teacher ratio", "Teachers per 1000 students"]
g = np.random.default_rng(2)
# "real" is genuinely driven by income; "noise" is not driven by anything.
inc = (cp["Per capita income"] - cp["Per capita income"].mean()) / \
      cp["Per capita income"].std()
cp["real"] = 50 + 9 * inc + g.normal(0, 1.5, len(cp))
cp["noise"] = g.normal(50, 9, len(cp))
pr = S.competency_predictability(cp, ["real", "noise"], controls=ctrl)
tab = pr["table"].set_index("competency")
report("the genuinely-driven competency is recovered",
       tab.loc["real", "loo_r2"] > 0.6,
       f"real: honest R2 = {tab.loc['real', 'loo_r2']:.2f}, "
       f"typical miss {tab.loc['real', 'loo_mae']:.1f} pts")
report("the noise competency does NOT get a real score",
       tab.loc["noise", "loo_r2"] < 0.15,
       f"noise: honest R2 = {tab.loc['noise', 'loo_r2']:.2f} "
       f"(in-sample it claimed {tab.loc['noise', 'in_sample_r2']:.2f})")
report("in-sample always flatters relative to leave-one-out",
       bool((pr["table"]["in_sample_r2"] >= pr["table"]["loo_r2"]).all()),
       "\n".join(f"{r.competency}: {r.in_sample_r2:.2f} -> {r.loo_r2:.2f}"
                 for r in pr["table"].itertuples()))
report("a failed model is allowed to go NEGATIVE, not clipped at zero",
       float(tab.loc["noise", "loo_r2"]) < 0.0,
       f"noise honest R2 = {tab.loc['noise', 'loo_r2']:.2f} — clipping this "
       f"to 0 would disguise a failed model as a weak one")

print()
print("=" * 78)
print("CASE H — guardrails")
print("=" * 78)
report("too few districts returns None rather than a fitted model",
       S.competency_predictability(cp.head(6), ["real"], controls=ctrl) is None,
       "6 districts, 4 predictors — refused")
report("no usable control column returns None",
       S.competency_predictability(cp, ["real"], controls=["nope"]) is None)
report("an absent competency column is ignored, not invented",
       S.competency_predictability(cp, ["real", "ghost"],
                                   controls=ctrl)["table"].shape[0] == 1)
report("indicator_families tolerates a missing outcome",
       S.indicator_families(df, "not_a_column") is None)
report("indicator_families tolerates None", S.indicator_families(None, "x") is None)

print()
print("=" * 78)
print(f"{len(FAIL)} FAILURE(S): {FAIL}" if FAIL else "ALL INDICATOR CHECKS PASSED")
print("=" * 78)
sys.exit(1 if FAIL else 0)
