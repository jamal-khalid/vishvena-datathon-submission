"""
Generate a synthetic education dataset matching the data-thon description:
- Grades 4, 5, 6 | 3 years | multiple competencies
- Gender + hierarchy: Division > District > Block > Cluster
- Objective marks + subjective text columns
Run:  python generate_sample_data.py  ->  sample_education_data.xlsx
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

DIVISIONS = {  # real Karnataka divisions/districts (2011-map spellings,
               # matching karnataka_districts.geojson -> choropleth works)
    "Belagavi Division": ["Bagalkote", "Belagavi", "Vijayapura", "Dharwad",
                          "Gadag", "Haveri", "Uttara Kannada"],
    "Bengaluru Division": ["Bengaluru Urban", "Bengaluru Rural",
                           "Chikkaballapura", "Chitradurga", "Davanagere",
                           "Kolar", "Ramanagara", "Shivamogga", "Tumakuru"],
    "Kalaburagi Division": ["Ballari", "Bidar", "Kalaburagi", "Koppal",
                            "Raichur", "Vijayanagara", "Yadgir"],
    "Mysuru Division": ["Chamarajanagara", "Chikkamagaluru",
                        "Dakshina Kannada", "Hassan", "Kodagu", "Mandya",
                        "Mysuru", "Udupi"],
}
BLOCKS_PER_DISTRICT = 3
CLUSTERS_PER_BLOCK = 4
STUDENTS_PER_CLUSTER = 30

YEARS = [2023, 2024, 2025]
GRADES = [4, 5, 6]
COMPETENCIES = ["Reading", "Math", "Science", "Language", "Logical Reasoning"]

TEACHER_QUALITY = ["well qualified", "adequately qualified", "under qualified"]
INCOME_GROUP = ["low income group", "middle income group", "high income group"]
INFRA = ["good infrastructure", "average infrastructure", "poor infrastructure"]

rows = []
student_id = 1000

for division, districts in DIVISIONS.items():
    for district in districts:
        # District-level effect so geographies genuinely differ
        district_effect = rng.normal(0, 8)
        for b in range(1, BLOCKS_PER_DISTRICT + 1):
            block = f"{district}-Block{b}"
            block_effect = rng.normal(0, 5)
            for c in range(1, CLUSTERS_PER_BLOCK + 1):
                cluster = f"{block}-C{c}"
                cluster_effect = rng.normal(0, 4)
                teacher = rng.choice(TEACHER_QUALITY, p=[0.3, 0.5, 0.2])
                income = rng.choice(INCOME_GROUP, p=[0.4, 0.45, 0.15])
                infra = rng.choice(INFRA, p=[0.25, 0.5, 0.25])
                teacher_bonus = {"well qualified": 6, "adequately qualified": 0, "under qualified": -6}[teacher]
                income_bonus = {"low income group": -5, "middle income group": 0, "high income group": 4}[income]

                for _ in range(STUDENTS_PER_CLUSTER):
                    student_id += 1
                    gender = rng.choice(["Female", "Male"])
                    ability = rng.normal(0, 10)
                    grade = int(rng.choice(GRADES))
                    for year in YEARS:
                        # Slight improvement trend over years + noise
                        year_effect = (year - YEARS[0]) * rng.normal(1.5, 1)
                        for comp in COMPETENCIES:
                            base = 55
                            gender_gap = -4 if (comp == "Math" and gender == "Female") else (
                                3 if (comp in ("Reading", "Language") and gender == "Female") else 0)
                            score = (base + ability + district_effect + block_effect +
                                     cluster_effect + teacher_bonus + income_bonus +
                                     gender_gap + year_effect + rng.normal(0, 7))
                            rows.append({
                                "Student_ID": f"S{student_id}",
                                "Year": year,
                                "Grade": grade,
                                "Gender": gender,
                                "Division": division,
                                "District": district,
                                "Block": block,
                                "Cluster": cluster,
                                "Competency": comp,
                                "Score": round(float(np.clip(score, 0, 100)), 1),
                                "Teacher_Quality": f"Teachers are {teacher}",
                                "Parent_Background": f"Parents are from {income}",
                                "School_Infrastructure": f"School has {infra}",
                            })

df = pd.DataFrame(rows)
df.to_excel("sample_education_data.xlsx", index=False)
print(f"Wrote sample_education_data.xlsx  ({len(df):,} rows, {df.Student_ID.nunique():,} students)")