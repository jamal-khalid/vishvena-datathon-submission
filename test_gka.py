"""
Known-answer tests for gka.py — the GKA impact layer.

Every case is a dataset whose correct answer is known by construction, and
most of them encode a bug that was actually shipped and caught:
paper difficulty leaking into a "learning" number, units with net growth
appearing in a danger table, a rebound described as a recovery while the unit
sat 31 points below where it started, and paper keys that silently missed.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\Workspace\dev_release\askingindia_ai_engine"
                   r"\vishvena-datathon-submission")
import gka  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'BUG '}  {name}")
    for line in str(detail).strip().split("\n"):
        if line:
            print(f"          {line}")
    if not ok:
        FAIL.append(name)


QS = [f"Q{i}" for i in range(1, 11)]


def make(rows):
    """rows: (year, grade, unit, gp, n_children, p_correct_per_item)."""
    out = []
    rng = np.random.default_rng(11)
    for year, grade, unit, gp, n, p in rows:
        blk = {"Year": [year] * n, "Grade": [grade] * n,
               "District": [unit] * n, "GP ID": [gp] * n}
        for q in QS:
            blk[q] = rng.binomial(1, p, n)
        out.append(pd.DataFrame(blk))
    return pd.concat(out, ignore_index=True)


def qmap(paper_comps):
    return {q: c for q, c in zip(QS, paper_comps)}


print("=" * 74)
print("COVERAGE")
print("=" * 74)
qm = {("2022-23", 4): qmap(["a"] * 5 + ["b"] * 3 + ["c"] * 2),
      ("2023-24", 4): qmap(["a"] * 5 + ["b"] * 5),
      ("2024-25", 4): qmap(["a"] * 7 + ["b"] * 3)}
cov = gka.coverage(qm)
check("core = competencies in every paper", cov["core"] == ["a", "b"],
      f"core={cov['core']}  rotating={sorted(cov['rotating'])}")
check("a competency missing from one paper is rotating, not core",
      "c" in cov["rotating"] and cov["rotating"]["c"]["n_papers"] == 1,
      f"c tested in {cov['rotating']['c']['n_papers']} of 3 papers")

qm2 = {("2022-23", 4): qmap(["a"] * 10), ("2022-23", 5): qmap(["a"] * 10),
       ("2023-24", 4): qmap(["a"] * 10),
       ("2023-24", 5): qmap(["a"] * 5 + ["d"] * 5)}
note = gka.coverage_note(qm2)
check("the note names the GRADE a competency is missing from, not just the year",
      note is not None and "Grade 4" in note,
      "a year-level note calls 'd' present in 2023-24 and never mentions that\n"
      "Grade 4 did not test it — the gap the earlier version missed\n"
      f"note: {(note or '')[:150]}")

print()
print("=" * 74)
print("PAPER DIFFICULTY")
print("=" * 74)
# identical children (p=0.5 both years); the second paper is simply harder.
easy = make([("2022-23", 4, "D1", "G1", 4000, 0.60)])
hard = make([("2023-24", 5, "D1", "G1", 4000, 0.40)])
df = pd.concat([easy, hard], ignore_index=True)
names_a = {q: f"skill{i}" for i, q in enumerate(QS)}
names_b = {q: f"skill{i}" for i, q in enumerate(QS)}
grid = gka.paper_grid(df)
check("the raw grid shows a fall that is purely the paper",
      abs(float(grid[grid["Grade"] == 5]["pct"].iloc[0])
          - float(grid[grid["Grade"] == 4]["pct"].iloc[0])) > 15,
      "same ability, different paper — raw change is ~20 points")

qn = {("2022-23", 4): names_a, ("2023-24", 5): names_b}
t, meta = gka.anchored_step(df, qn, ("2022-23", 4), ("2023-24", 5), min_n=30)
check("anchoring on shared skills still reports the real per-skill change",
      meta["ok"] and len(t) == 1,
      f"anchors={meta['n_anchors']} drift={float(t.iloc[0]['drift']):+.1f}\n"
      "(all 10 items are shared here, so anchored == raw by construction —\n"
      "the point of the case is that the machinery runs and reports honestly)")

# only 2 skills in common -> must refuse
qn_few = {("2022-23", 4): {q: f"s{i}" for i, q in enumerate(QS)},
          ("2023-24", 5): {q: (f"s{i}" if i < 2 else f"other{i}")
                           for i, q in enumerate(QS)}}
t2, meta2 = gka.anchored_step(df, qn_few, ("2022-23", 4), ("2023-24", 5))
check("a step with too few shared skills is refused, not estimated",
      (not meta2["ok"]) and meta2["n_anchors"] == 2 and t2.empty,
      f"anchors={meta2['n_anchors']}, MIN_ANCHORS={gka.MIN_ANCHORS}\n"
      f"reason: {(meta2['reason'] or '')[:90]}")

print()
print("=" * 74)
print("GRID ORDERING FLIP")
print("=" * 74)
flip_df = pd.concat([
    make([("2022-23", 4, "D1", "G1", 800, 0.45),
          ("2022-23", 6, "D1", "G1", 800, 0.65),
          ("2024-25", 4, "D1", "G1", 800, 0.65),
          ("2024-25", 6, "D1", "G1", 800, 0.45)])], ignore_index=True)
fl = gka.grid_is_inconsistent(gka.paper_grid(flip_df))
check("a reversed grade ordering between years is detected", fl["flipped"],
      f"rises in {fl.get('rises_in')}, falls in {fl.get('falls_in')}\n"
      "children cannot get better in G4 and worse in G6 in the same year")

print()
print("=" * 74)
print("TRAJECTORY SHAPE")
print("=" * 74)
check("a step inside its own noise is flat, not a trend",
      gka.step_dir(0.8, se=2.0) == "flat",
      "0.8 points with SE 2.0 — significance and size both fail")
check("a big step on a huge sample is still flat if under the floor",
      gka.step_dir(0.4, se=0.01) == "flat",
      "0.4 points is 'significant' on 40,000 children and means nothing;\n"
      f"the practical floor is {gka.FLAT_PTS} points")
sh = gka.classify([50.0, 40.0, 48.0], [1.0, 1.0, 1.0])
check("down-then-up is classified as a V-shape recovery",
      sh["key"] == "down_up", f"got {sh['key']} — {sh['label']}")

print()
print("=" * 74)
print("DANGER TABLE")
print("=" * 74)
# 12 units so ranking is meaningful. One grows overall via a spike then dip;
# one declines in both steps; the rest are stable.
rows = []
for i in range(10):
    for y, p in (("2022-23", 0.50), ("2023-24", 0.50), ("2024-25", 0.50)):
        rows.append((y, 4, f"Stable{i}", f"Stable{i}", 400, p))
# net GROWTH, but shape is up_down
for y, p in (("2022-23", 0.30), ("2023-24", 0.85), ("2024-25", 0.70)):
    rows.append((y, 4, "Grower", "Grower", 400, p))
# strict decline
for y, p in (("2022-23", 0.70), ("2023-24", 0.50), ("2024-25", 0.30)):
    rows.append((y, 4, "Faller", "Faller", 400, p))
dd = make(rows)
d = gka.danger(dd, "District", limit=10)
units = [] if d is None or d.empty else d["unit"].tolist()
check("a unit with net GROWTH is not listed as in danger",
      "Grower" not in units,
      "Grower went 30% -> 85% -> 70%: up 40 points overall, shape up_down\n"
      f"table: {units}")
check("a unit falling in both steps IS listed, and is flagged strict",
      "Faller" in units
      and bool(d[d["unit"] == "Faller"]["strict_decline"].iloc[0]),
      f"table: {units}")

print()
print("=" * 74)
print("CONSTANT PANEL AND COMMON GRADES")
print("=" * 74)
rows = []
for i in range(10):
    for y in ("2022-23", "2023-24", "2024-25"):
        rows.append((y, 4, f"U{i}", f"U{i}", 300, 0.5))
rows.append(("2024-25", 4, "Newcomer", "Newcomer", 300, 0.9))   # one year only
per = gka.percentiles(make(rows), "District")
check("a unit present in only one year is excluded from the panel",
      per is not None and "Newcomer" not in set(per["unit"]),
      "otherwise the comparison set changes underneath every trend\n"
      f"units in panel: {sorted(set(per['unit']))[:4]} …")

rows = []
for i in range(10):
    for y in ("2022-23", "2023-24", "2024-25"):
        for g in (4, 5):
            rows.append((y, g, f"U{i}", f"U{i}", 300, 0.5))
# this unit clears the floor in G5 only in the last year
rows += [("2022-23", 4, "Partial", "Partial", 300, 0.5),
         ("2023-24", 4, "Partial", "Partial", 300, 0.5),
         ("2024-25", 4, "Partial", "Partial", 300, 0.5),
         ("2024-25", 5, "Partial", "Partial", 300, 0.9)]
tr = gka.unit_trajectory(gka.percentiles(make(rows), "District"))
pg = tr[tr["unit"] == "Partial"]["grades"].tolist()
check("only grades a unit holds in EVERY year are averaged",
      pg and set(pg) == {1},
      f"'Partial' has G4 in all years and G5 only in the last; grades used "
      f"per year: {pg}\n"
      "mixing them would compare a Grade 5 standing against a Grade 4 one")

print()
print("=" * 74)
print("RECOMMENDATIONS")
print("=" * 74)
missing = [(k, s) for k in gka.SHAPES
           for s in ("critical", "weak", "middling", "strong")
           if (f"{k[0]}_{k[1]}", s) not in gka.BASE_ACTIONS]
check("every shape x severity cell has an action", not missing,
      f"{len(gka.BASE_ACTIONS)} cells filled; missing: {missing}")

space = gka.combination_space()
check("the combination count is computed from the rules, not hardcoded",
      space["distinct_outputs"]
      == space["base_actions"] * space["clause_subsets"],
      f"{space['base_actions']} base x {space['clause_subsets']} clause "
      f"subsets = {space['distinct_outputs']:,}")

fake = {"danger_district": pd.DataFrame([{
            "unit": "X", "start": 90.0, "mid": 40.0, "end": 59.0,
            "change": -31.0, "z_change": -0.8, "shape": "Recovery (V-shape)",
            "shape_key": "down_up", "children": 40000, "headroom": 41.0,
            "p": 0.2, "evidence": 0.0, "typical_move_sd": 0.4,
            "strict_decline": False, "beyond_normal_movement": False}]),
        "competency": None, "unit_coverage": None, "min_n": 30}
rec = gka.recommendations(fake, "district", limit=1)[0]
txt = rec["recommendation"].lower()
check("a unit 31 points DOWN is never told 'the recovery is underway'",
      "recovery is underway" not in txt and "still" in txt,
      "every row in a danger table is negative on net, so a shape ending\n"
      "upward must say so as a partial rebound\n"
      f"said: {rec['recommendation'][:150]}")

print()
print("=" * 74)
print("PAPER KEY NORMALISATION")
print("=" * 74)
df2 = make([("2022", 4, "D1", "G1", 500, 0.5), ("2023", 5, "D1", "G1", 500, 0.5)])
res_int = gka.analyse(df2, {(2022, 4): qmap(["a"] * 10),
                            (2023, 5): qmap(["a"] * 10)},
                      {(2022, 4): names_a, (2023, 5): names_b},
                      unit_col="District")
ok_steps = [s for c in (res_int or {}).get("cohorts", []) for s in c["steps"]
            if s["meta"].get("ok")]
check("integer paper keys still match a string year column",
      len(ok_steps) >= 1,
      "the workbooks key papers '2022-23' while the dashboard converts Year\n"
      "to 2022; unreconciled, every anchor lookup missed and the tab reported\n"
      "that no two papers shared a skill\n"
      f"comparable steps: {len(ok_steps)}")

print()
print("=" * 74)
print(f"{len(FAIL)} FAILURE(S): {FAIL}" if FAIL else "ALL GKA CHECKS PASSED")
print("=" * 74)
sys.exit(1 if FAIL else 0)
