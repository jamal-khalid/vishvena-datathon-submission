"""
Layer 1 (compute half) — the "reduce" step.
Load the big CSV via DuckDB, convert to Parquet once, and aggregate
1M+ rows down to a few hundred summary rows entirely on the laptop.
"""
import os
import duckdb
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
CSV = os.path.join(DATA_DIR, "assessment.csv")
PARQUET = os.path.join(DATA_DIR, "assessment.parquet")


def to_parquet():
    """One-time: CSV -> Parquet (smaller + faster). Returns the parquet path."""
    duckdb.sql(f"COPY (SELECT * FROM read_csv_auto('{CSV}')) TO '{PARQUET}' (FORMAT PARQUET)")
    return PARQUET


def aggregate():
    """Reduce every row to per (district, block, grade, competency, year) stats."""
    src = PARQUET if os.path.exists(PARQUET) else CSV
    reader = f"'{src}'" if src.endswith(".parquet") else f"read_csv_auto('{src}')"
    agg = duckdb.sql(f"""
        SELECT division, district, block, grade, competency, year,
               COUNT(*)                                                   AS n,
               100.0*AVG(CASE WHEN level='below' THEN 1 ELSE 0 END)       AS below_pct,
               100.0*AVG(CASE WHEN level='above' THEN 1 ELSE 0 END)       AS above_pct,
               100.0*AVG(CASE WHEN gender='F' AND level='below' THEN 1.0
                             WHEN gender='F' THEN 0.0 END)                AS f_below,
               100.0*AVG(CASE WHEN gender='M' AND level='below' THEN 1.0
                             WHEN gender='M' THEN 0.0 END)                AS m_below
        FROM {reader}
        GROUP BY division, district, block, grade, competency, year
        ORDER BY district, block, grade, competency, year
    """).df()

    # derive gender gap + previous-year value (for trend) in pandas
    agg["gender_gap"] = (agg["f_below"] - agg["m_below"]).round(1)
    agg = agg.sort_values(["district", "block", "grade", "competency", "year"])
    agg["prev_pct"] = agg.groupby(["district", "block", "grade", "competency"])["below_pct"].shift(1)
    for c in ["below_pct", "above_pct", "f_below", "m_below", "prev_pct"]:
        agg[c] = agg[c].round(1)
    return agg


def counts():
    """Row count of the raw dataset (for the demo header)."""
    src = PARQUET if os.path.exists(PARQUET) else CSV
    reader = f"'{src}'" if src.endswith(".parquet") else f"read_csv_auto('{src}')"
    return int(duckdb.sql(f"SELECT COUNT(*) FROM {reader}").fetchone()[0])


if __name__ == "__main__":
    to_parquet()
    a = aggregate()
    print(a.head(12).to_string(index=False))
    print("rows:", counts(), "-> aggregated to", len(a))
