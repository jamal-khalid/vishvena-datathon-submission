"""
Generate a synthetic school-assessment dataset that mimics the datathon data:
Grades 4/5/6, three years, disaggregated by gender and geography
(Division > District > Block > Cluster), across several competencies.

Real datathon data will replace this file — the rest of the pipeline is unchanged.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

COMPETENCIES = ["Subtraction", "Fractions", "Word problems", "Reading fluency", "Shapes"]
NUMERACY = {"Subtraction", "Fractions", "Word problems"}
GRADES = [4, 5, 6]
YEARS = [2024, 2025, 2026]

# Division -> District -> Blocks
GEO = {
    "Bangalore Division": {
        "Bangalore Rural": ["Anekal", "Hoskote", "Devanahalli", "Nelamangala"],
        "Ramanagara":      ["Channapatna", "Kanakapura", "Magadi"],
    },
    "Mysore Division": {
        "Mandya":          ["Maddur", "Malavalli", "Srirangapatna"],
        "Hassan":          ["Arsikere", "Belur", "Sakleshpur"],
    },
}

# base probability of being "below grade" per competency (higher = harder)
COMP_DIFF = {"Subtraction": 0.50, "Fractions": 0.55, "Word problems": 0.52,
             "Reading fluency": 0.32, "Shapes": 0.28}


def generate(students_per_group=300, seed=7):
    rng = np.random.default_rng(seed)
    # a fixed random "difficulty offset" per block so blocks differ
    blocks = [(dv, ds, bl) for dv, dd in GEO.items() for ds, bls in dd.items() for bl in bls]
    block_offset = {bl: rng.uniform(-0.12, 0.18) for (_, _, bl) in blocks}
    # each block moves at its own pace: some improve, some stagnate, some slide back.
    # (a single uniform trend made every block look "Improving" and collapsed the
    #  variety of the intervention playbook)
    block_trend = {bl: rng.uniform(-0.05, 0.035) for (_, _, bl) in blocks}
    # gender gap is not uniform either — it is wide in some blocks, absent in others
    block_gender = {bl: rng.uniform(0.0, 0.10) for (_, _, bl) in blocks}
    # a few blocks are specifically weak in ONE competency (gives KMeans real structure)
    comp_quirk = {bl: (rng.choice(COMPETENCIES), rng.uniform(0.05, 0.15))
                  for (_, _, bl) in blocks}

    frames = []
    for (division, district, block) in blocks:
        for grade in GRADES:
            n = students_per_group
            sid = np.array([f"{block[:3].upper()}{grade}-{i:04d}" for i in range(n)])
            gender = rng.choice(["F", "M"], size=n)
            cluster = rng.choice([f"{block}-C{k}" for k in range(1, 4)], size=n)
            # expand over years x competencies
            for year in YEARS:
                for comp in COMPETENCIES:
                    p_below = COMP_DIFF[comp] + block_offset[block]
                    p_below += block_trend[block] * (year - 2024)     # per-block direction
                    if comp == comp_quirk[block][0]:                 # block-specific weak spot
                        p_below += comp_quirk[block][1]
                    p_below += np.where((gender == "F") & (comp in NUMERACY),
                                        block_gender[block], 0.0)     # gap varies by block
                    p_below += 0.02 * (grade - 5)                    # older grades slightly harder
                    p_below = np.clip(p_below, 0.05, 0.9)
                    u = rng.random(n)
                    level = np.where(u < p_below, "below",
                             np.where(u < p_below + 0.28, "at", "above"))
                    frames.append(pd.DataFrame({
                        "student_id": sid, "gender": gender,
                        "division": division, "district": district,
                        "block": block, "cluster": cluster,
                        "grade": grade, "year": year,
                        "competency": comp, "level": level,
                    }))
    df = pd.concat(frames, ignore_index=True)
    return df


def build(students_per_group=300, seed=7):
    os.makedirs(DATA_DIR, exist_ok=True)
    df = generate(students_per_group, seed)
    csv_path = os.path.join(DATA_DIR, "assessment.csv")
    df.to_csv(csv_path, index=False)
    return csv_path, len(df)


if __name__ == "__main__":
    path, n = build()
    print(f"Wrote {n:,} rows -> {path}")
