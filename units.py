"""
The response-vs-child distinction.

When the 20 question columns become the competency dimension, one child produces
20 rows in the aggregate. So `n` summed across competencies counts every child
20 times. Three things break if that goes unnoticed:

  1. Headcounts inflate 20x   ("6,420 assessed children" for 321 children).
  2. Significance tests over-power. The 20 answers from one child are not 20
     independent observations — they are one child, measured 20 times. Feeding
     the response count to a z-test shrinks the standard error by sqrt(20) and
     turns p=0.35 into p<0.0001. That is textbook pseudo-replication.
  3. Any absolute-size band ("Large / Medium / Small") is calibrated wrong.

Percentages are NOT affected: the factor cancels in (below_pct * n).sum() / n.sum().
So weighted means may keep using `n`. Headcounts and significance tests must
come through here.

    n         -> assessment responses  -> denominator for a PERCENTAGE
    students  -> distinct children     -> denominator for a HEADCOUNT / a Z-TEST
"""

CELL_KEYS = ["division", "district", "block", "grade", "year"]


def headcount(rows):
    """Distinct children behind `rows` (not their assessment responses)."""
    if rows is None or len(rows) == 0:
        return 0
    if "students" in rows.columns:
        keys = [k for k in CELL_KEYS if k in rows.columns]
        # `students` is constant within a cell and repeated on every competency
        # row of that cell, so de-duplicate before summing.
        return int(rows.drop_duplicates(keys)["students"].sum())
    # Legacy aggregates (one row per student already) have no inflation.
    return int(rows["n"].sum())


def children_below(rows, below_pct):
    """How many actual children the percentage corresponds to."""
    return int(round(headcount(rows) * float(below_pct) / 100.0))


def eff_n(rows):
    """
    Sample size to use in a significance test: the number of independent
    children, never the number of responses.
    """
    return max(headcount(rows), 1)
